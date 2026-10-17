"""Step 04: bin events into fixed-width time steps.

ICU counterpart: :mod:`clin_jepa.data.step04_discretization`, which loops over
stays in pandas and writes one pickle per stay. Two changes here:

**Bin width is 6 hours, not 1.** Measured occupancy on this cohort is 8.7% at
1h against the ICU pipeline's ~99.9%; see
``reports/hosp_pipeline/hosp_density_probe.md``.

**Binning runs in DuckDB and output is bucketed parquet, not per-stay pickle.**
388k admissions would mean 388k pickle files, and the per-stay pandas loop is
the ICU pipeline's slowest step. Expressing the bin assignment, the continuous
expansion, and the within-bin collapse as SQL turns it into three set
operations. The logical output is identical: per-admission ``n_steps``,
``delta_t``, ``has_obs``, ``is_no_action``, and binned obs/act event tables.

Output (under {output.discretized}):
  obs/bucket=NN/*.parquet   binned observation events (+ step_idx)
  act/bucket=NN/*.parquet   binned action events      (+ step_idx)
  meta/bucket=NN/*.parquet  per-admission n_steps, delta_t, has_obs, is_no_action
  discretize_stats.json

Usage:
    python -m clin_jepa.data_hosp.step04_hosp_discretization
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from clin_jepa.data_hosp import db

logger = logging.getLogger(__name__)

# Sources whose text value is part of an event's identity, so that two
# different values in one bin are two events rather than an overwrite.
# A blood culture growing two organisms is two findings; a drug re-ordered at a
# new dose in the same bin is one order at its latest dose.
TEXT_IDENTITY_SOURCES = ("micro",)


def _bin_and_collapse(
    src_glob: str,
    step_size: float,
    caps: dict[str, int],
    domain_label: str,
) -> str:
    """SQL: assign step_idx, expand continuous events, collapse within bin.

    Args:
        src_glob: parquet glob for the step02/step03 output.
        step_size: bin width in hours.
        caps: source_table -> max rows kept per (admission, bin).
        domain_label: 'observation' or 'action', for logging only.
    """
    text_id = ", ".join(f"'{s}'" for s in TEXT_IDENTITY_SOURCES)

    cap_when = "\n            ".join(
        f"WHEN source_table = '{src}' THEN {int(n)}" for src, n in caps.items()
    )
    cap_expr = (
        f"CASE\n            {cap_when}\n            ELSE 1000000 END"
        if caps
        else "1000000"
    )

    return f"""
    WITH raw AS (
        SELECT e.*, m.n_steps
        FROM read_parquet('{src_glob}', hive_partitioning = true, union_by_name = true) e
        JOIN stay_steps m USING (hadm_id)
        WHERE e.timestamp_hours < m.n_steps * {step_size}
    ),
    -- Point events land in the bin containing their timestamp. The rounding
    -- guards against float error putting an exact boundary in the prior bin.
    point AS (
        SELECT
            hadm_id, bucket, source_table, category, variable_name,
            numeric_value, text_value, unit, is_continuous, timestamp,
            least(
                greatest(floor(round(timestamp_hours, 6) / {step_size}), 0),
                n_steps - 1
            )::INTEGER AS step_idx
        FROM raw
        WHERE NOT is_continuous
    ),
    -- Continuous events occupy every bin they overlap, so they are expanded
    -- to one row per bin. Same rule as the ICU pipeline's infusion handling.
    cont_bounds AS (
        SELECT
            hadm_id, bucket, source_table, category, variable_name,
            numeric_value, text_value, unit, is_continuous, timestamp,
            least(greatest(floor(timestamp_hours / {step_size}), 0),
                  n_steps - 1)::INTEGER AS start_bin,
            CASE
                WHEN end_timestamp_hours IS NULL
                     OR end_timestamp_hours <= timestamp_hours
                THEN least(greatest(floor(timestamp_hours / {step_size}), 0),
                           n_steps - 1)::INTEGER
                ELSE least(
                        greatest(ceil(end_timestamp_hours / {step_size}) - 1,
                                 floor(timestamp_hours / {step_size})),
                        n_steps - 1
                     )::INTEGER
            END AS end_bin
        FROM raw
        WHERE is_continuous
    ),
    cont AS (
        SELECT
            hadm_id, bucket, source_table, category, variable_name,
            numeric_value, text_value, unit, is_continuous, timestamp,
            unnest(generate_series(start_bin, end_bin))::INTEGER AS step_idx
        FROM cont_bounds
        WHERE end_bin >= start_bin
    ),
    allev AS (
        SELECT * FROM point
        UNION ALL
        SELECT * FROM cont
    ),
    -- Within-bin collapse: keep the latest reading per variable. This is what
    -- bounds state_text length by the variable count rather than the event
    -- count, which is necessary once a bin is 6 hours wide.
    ranked AS (
        SELECT *,
            row_number() OVER (
                PARTITION BY hadm_id, step_idx, source_table, variable_name,
                    CASE WHEN source_table IN ({text_id})
                         THEN text_value ELSE NULL END
                ORDER BY timestamp DESC
            ) AS rn
        FROM allev
    ),
    collapsed AS (
        SELECT * FROM ranked WHERE rn = 1
    ),
    -- Per-source cap, so that one verbose source cannot crowd a bin out of
    -- the encoder's token budget.
    capped AS (
        SELECT *,
            row_number() OVER (
                PARTITION BY hadm_id, step_idx, source_table
                ORDER BY timestamp DESC
            ) AS src_rn
        FROM collapsed
    )
    SELECT
        hadm_id, step_idx, variable_name, numeric_value, text_value,
        unit, category, source_table, is_continuous, bucket
    FROM capped
    WHERE src_rn <= ({cap_expr})
    """


def _caps_from_features(features: dict, domain: str) -> dict[str, int]:
    """Read per-source within-bin caps out of the features config."""
    caps: dict[str, int] = {}
    if domain == "observation":
        obs = features["observations"]
        if "max_per_bin" in obs.get("microbiology", {}):
            caps["micro"] = obs["microbiology"]["max_per_bin"]
    else:
        acts = features["actions"]
        if "max_active_per_bin" in acts.get("prescriptions", {}):
            caps["rx"] = acts["prescriptions"]["max_active_per_bin"]
        if "max_per_bin" in acts.get("emar", {}):
            caps["emar_admin"] = acts["emar"]["max_per_bin"]
    return caps


META_SQL = """
WITH obs_steps AS (
    SELECT hadm_id, list_sort(list_distinct(list(step_idx))) AS obs_bins
    FROM read_parquet('{obs_glob}', hive_partitioning = true, union_by_name = true)
    GROUP BY hadm_id
),
act_steps AS (
    SELECT hadm_id, list_sort(list_distinct(list(step_idx))) AS act_bins
    FROM read_parquet('{act_glob}', hive_partitioning = true, union_by_name = true)
    GROUP BY hadm_id
),
joined AS (
    SELECT
        s.hadm_id, s.bucket, s.n_steps,
        coalesce(o.obs_bins, []::INTEGER[]) AS obs_bins,
        coalesce(a.act_bins, []::INTEGER[]) AS act_bins
    FROM stay_steps s
    LEFT JOIN obs_steps o USING (hadm_id)
    LEFT JOIN act_steps a USING (hadm_id)
)
SELECT
    hadm_id, bucket, n_steps, obs_bins, act_bins,
    len(obs_bins) AS n_obs_bins,
    len(act_bins) AS n_act_bins,
    -- has_obs / is_no_action as dense per-bin flags, matching the ICU
    -- pipeline's arrays of the same name.
    list_transform(
        range(0, n_steps),
        b -> list_contains(obs_bins, b)
    ) AS has_obs,
    list_transform(
        range(0, n_steps),
        b -> NOT list_contains(act_bins, b)
    ) AS is_no_action,
    -- delta_t: bins until the next bin holding an observation, NULL past the
    -- last one. Expressed in hours so it is comparable to the ICU pipeline.
    list_transform(
        range(0, n_steps),
        b -> (
            list_min(list_filter(obs_bins, x -> x > b)) - b
        ) * {step_size}
    ) AS delta_t
FROM joined
WHERE len(obs_bins) >= {min_obs_bins}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 04: hospital discretization")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--features_config", default="hosp_features.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    paths = db.load_yaml(args.paths_config)
    features = db.load_yaml(args.features_config)
    temporal = features["temporal"]
    step_size = float(temporal["step_size_hours"])
    max_steps = int(temporal["max_steps"])
    min_obs = int(temporal["min_obs_per_stay"])
    complete_bins_only = bool(temporal.get("complete_bins_only", False))
    require_all_obs_bins = bool(temporal.get("require_all_obs_bins", False))

    logger.info("=" * 62)
    logger.info("Step 04: discretization (step=%.0fh, max_steps=%d)",
                step_size, max_steps)
    logger.info("=" * 62)

    n_buckets = int(paths["sharding"]["n_buckets"])
    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"
    obs_in = Path(paths["output"]["observations"])
    act_in = Path(paths["output"]["actions"])
    out_root = Path(paths["output"]["discretized"])

    if out_root.exists():
        logger.info("Clearing %s", out_root)
        shutil.rmtree(out_root)
    for sub in ("obs", "act", "meta"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    con = db.connect(paths)
    db.register_cohort(con, cohort_path, n_buckets)

    # The legacy path retains a final partial bin via ceil(). Dense ward
    # experiments can opt into floor() so every model node is a complete bin.
    step_count_expr = (
        f"floor(los_hours / {step_size})"
        if complete_bins_only
        else f"ceil(los_hours / {step_size})"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stay_steps AS
        SELECT hadm_id, bucket,
               least({max_steps},
                     greatest(0, {step_count_expr})::INTEGER
               ) AS n_steps
        FROM cohort
        WHERE {step_count_expr} >= 1
        """
    )
    n_stays = int(con.execute("SELECT count(*) FROM stay_steps").fetchone()[0])
    logger.info("Admissions to discretize: %s", f"{n_stays:,}")

    # ── Observations ────────────────────────────────────────────────────
    logger.info("Binning observations ...")
    obs_sql = _bin_and_collapse(
        f"{obs_in}/bucket=*/*.parquet", step_size,
        _caps_from_features(features, "observation"), "observation",
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE _obs AS {obs_sql}")
    n_obs = int(con.execute("SELECT count(*) FROM _obs").fetchone()[0])
    db.copy_to_buckets(con, "SELECT * FROM _obs", out_root / "obs", overwrite=True)
    logger.info("  binned observation rows: %s", f"{n_obs:,}")

    # ── Actions ─────────────────────────────────────────────────────────
    logger.info("Binning actions ...")
    act_sql = _bin_and_collapse(
        f"{act_in}/bucket=*/*.parquet", step_size,
        _caps_from_features(features, "action"), "action",
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE _act AS {act_sql}")
    n_act = int(con.execute("SELECT count(*) FROM _act").fetchone()[0])
    db.copy_to_buckets(con, "SELECT * FROM _act", out_root / "act", overwrite=True)
    logger.info("  binned action rows: %s", f"{n_act:,}")

    # ── Per-admission metadata ──────────────────────────────────────────
    logger.info("Building per-admission metadata ...")
    meta_sql = META_SQL.format(
        obs_glob=f"{out_root}/obs/bucket=*/*.parquet",
        act_glob=f"{out_root}/act/bucket=*/*.parquet",
        step_size=step_size,
        min_obs_bins=1,
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE _meta AS {meta_sql}")

    if require_all_obs_bins:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _meta AS
            SELECT * FROM _meta WHERE n_obs_bins = n_steps
            """
        )

    # min_obs_per_stay is a floor on events, matching the ICU pipeline's
    # min_obs_per_stay, so it is applied against the event count not bin count.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _meta AS
        SELECT m.* FROM _meta m
        JOIN (
            SELECT hadm_id, count(*) AS n_events FROM _obs GROUP BY hadm_id
        ) e USING (hadm_id)
        WHERE e.n_events >= {min_obs}
        """
    )
    n_meta = int(con.execute("SELECT count(*) FROM _meta").fetchone()[0])
    db.copy_to_buckets(con, "SELECT * FROM _meta", out_root / "meta", overwrite=True)
    logger.info("  admissions retained: %s (%.1f%% of cohort)",
                f"{n_meta:,}", 100.0 * n_meta / n_stays)

    # ── Statistics ──────────────────────────────────────────────────────
    row = con.execute(
        """
        SELECT
            sum(n_steps), median(n_steps),
            quantile_cont(n_steps, 0.05), quantile_cont(n_steps, 0.95),
            max(n_steps),
            sum(n_obs_bins), sum(n_act_bins),
            avg(n_obs_bins * 1.0 / n_steps), avg(n_act_bins * 1.0 / n_steps)
        FROM _meta
        """
    ).fetchone()
    total_bins = int(row[0])
    filled_obs = int(row[5])
    filled_act = int(row[6])

    # Care setting is a continuous event covering the whole stay, so it puts a
    # row in every bin and makes raw occupancy ~100% by construction. The
    # number that actually says whether the bin width is right is how many
    # bins hold a real measurement, so both are reported.
    measured_bins = int(con.execute(
        """
        SELECT count(*) FROM (
            SELECT DISTINCT o.hadm_id, o.step_idx
            FROM _obs o JOIN _meta m USING (hadm_id)
            WHERE o.category NOT IN ('care_setting', 'diagnosis')
        )
        """
    ).fetchone()[0])
    measured_obs_rows = int(con.execute(
        "SELECT count(*) FROM _obs o JOIN _meta m USING (hadm_id) "
        "WHERE o.category NOT IN ('care_setting', 'diagnosis')"
    ).fetchone()[0])

    by_source = {
        f"{dom}:{src}": int(n)
        for dom, tbl in (("obs", "_obs"), ("act", "_act"))
        for src, n in con.execute(
            f"SELECT source_table, count(*) FROM {tbl} GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }
    by_category = {
        f"{dom}:{cat}": int(n)
        for dom, tbl in (("obs", "_obs"), ("act", "_act"))
        for cat, n in con.execute(
            f"SELECT category, count(*) FROM {tbl} GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }

    stats = {
        "step_size_hours": step_size,
        "max_steps": max_steps,
        "complete_bins_only": complete_bins_only,
        "require_all_obs_bins": require_all_obs_bins,
        "n_cohort": n_stays,
        "n_retained": n_meta,
        "total_bins": total_bins,
        "n_steps_median": float(row[1]),
        "n_steps_p5": float(row[2]),
        "n_steps_p95": float(row[3]),
        "n_steps_max": int(row[4]),
        "binned_obs_rows": n_obs,
        "binned_act_rows": n_act,
        "obs_bin_occupancy": round(filled_obs / total_bins, 4),
        "act_bin_occupancy": round(filled_act / total_bins, 4),
        # Excludes care_setting and diagnosis: the honest measure of density.
        "measured_bin_occupancy": round(measured_bins / total_bins, 4),
        "measured_obs_rows": measured_obs_rows,
        "obs_occupancy_per_stay_mean": round(float(row[7]), 4),
        "act_occupancy_per_stay_mean": round(float(row[8]), 4),
        "obs_events_per_filled_bin": round(n_obs / filled_obs, 2) if filled_obs else 0,
        "act_events_per_filled_bin": round(n_act / filled_act, 2) if filled_act else 0,
        "measured_events_per_measured_bin": (
            round(measured_obs_rows / measured_bins, 2) if measured_bins else 0
        ),
        "by_source": by_source,
        "by_category": by_category,
    }
    (out_root / "discretize_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    logger.info("-" * 62)
    logger.info("  total bins              %14s", f"{total_bins:,}")
    logger.info("  n_steps median / P95    %8.0f / %.0f", row[1], row[3])
    logger.info("  observation occupancy   %13.1f%%  (incl. care setting)",
                stats["obs_bin_occupancy"] * 100)
    logger.info("  measured occupancy      %13.1f%%  (labs/vitals/micro only)",
                stats["measured_bin_occupancy"] * 100)
    logger.info("  action occupancy        %13.1f%%",
                stats["act_bin_occupancy"] * 100)
    logger.info("  obs events / filled bin %14.2f",
                stats["obs_events_per_filled_bin"])
    logger.info("  act events / filled bin %14.2f",
                stats["act_events_per_filled_bin"])
    logger.info("Wrote %s", out_root / "discretize_stats.json")


if __name__ == "__main__":
    main()
