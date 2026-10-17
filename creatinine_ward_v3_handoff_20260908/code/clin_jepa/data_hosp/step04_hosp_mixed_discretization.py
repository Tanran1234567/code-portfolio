"""Step 04 (mixed resolution): 24h ward bins and 1h ICU bins.

The timeline is segmented by the real ICU intervals in ``mimiciv_icu.icustays``.
Ward segments before, between, and after ICU stays are split into 24-hour
intervals. ICU segments are split into one-hour intervals. Short boundary
intervals are retained so no event is duplicated or dropped.

Outputs have the same ``obs/act/meta`` layout as the fixed-grid hospital
pipeline, plus ``nodes`` and explicit per-node physical-time metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import pandas as pd

from clin_jepa.data_hosp import db
from clin_jepa.data_hosp.step04_hosp_discretization import (
    TEXT_IDENTITY_SOURCES,
    _caps_from_features,
)

logger = logging.getLogger(__name__)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _append_segment(
    rows: list[dict],
    hadm_id: int,
    bucket: int,
    start: float,
    end: float,
    width: float,
    is_icu: bool,
    step_idx: int,
) -> int:
    cursor = start
    while cursor < end - 1e-6:
        node_end = min(cursor + width, end)
        rows.append({
            "hadm_id": int(hadm_id),
            "bucket": int(bucket),
            "step_idx": int(step_idx),
            "node_start_hours": float(cursor),
            "node_end_hours": float(node_end),
            "duration_hours": float(node_end - cursor),
            "time_hours": float(cursor),
            "is_icu": bool(is_icu),
        })
        step_idx += 1
        cursor = node_end
    return step_idx


def build_nodes(
    cohort: pd.DataFrame,
    icu_intervals: pd.DataFrame,
    ward_bin_hours: float,
    icu_bin_hours: float,
) -> pd.DataFrame:
    """Build non-overlapping, gap-free mixed-resolution nodes."""
    by_hadm: dict[int, list[tuple[float, float]]] = {}
    for row in icu_intervals.itertuples(index=False):
        by_hadm.setdefault(int(row.hadm_id), []).append(
            (float(row.icu_start_hours), float(row.icu_end_hours))
        )

    rows: list[dict] = []
    for stay in cohort.itertuples(index=False):
        hadm_id = int(stay.hadm_id)
        los = float(stay.los_hours)
        bucket = int(stay.bucket)
        intervals = _merge_intervals([
            (max(0.0, start), min(los, end))
            for start, end in by_hadm.get(hadm_id, [])
            if min(los, end) > max(0.0, start)
        ])

        cursor = 0.0
        step_idx = 0
        for icu_start, icu_end in intervals:
            if icu_start > cursor + 1e-6:
                step_idx = _append_segment(
                    rows, hadm_id, bucket, cursor, icu_start,
                    ward_bin_hours, False, step_idx,
                )
            step_idx = _append_segment(
                rows, hadm_id, bucket, icu_start, icu_end,
                icu_bin_hours, True, step_idx,
            )
            cursor = max(cursor, icu_end)

        if cursor < los - 1e-6:
            _append_segment(
                rows, hadm_id, bucket, cursor, los,
                ward_bin_hours, False, step_idx,
            )

    nodes = pd.DataFrame(rows)
    if nodes.empty:
        raise RuntimeError("Mixed-resolution node construction produced no nodes")
    return nodes


def _cap_expression(caps: dict[str, int]) -> str:
    if not caps:
        return "1000000"
    branches = "\n                ".join(
        f"WHEN source_table = '{source}' THEN {int(limit)}"
        for source, limit in caps.items()
    )
    return f"CASE\n                {branches}\n                ELSE 1000000 END"


def _bin_and_collapse(
    src_glob: str,
    caps: dict[str, int],
) -> str:
    text_sources = ", ".join(f"'{source}'" for source in TEXT_IDENTITY_SOURCES)
    cap_expr = _cap_expression(caps)
    return f"""
    WITH raw AS (
        SELECT e.*
        FROM read_parquet(
            '{src_glob}', hive_partitioning = true, union_by_name = true
        ) e
        JOIN (SELECT DISTINCT hadm_id FROM nodes) n USING (hadm_id)
    ),
    point AS (
        SELECT
            e.hadm_id, n.step_idx, n.node_start_hours, n.node_end_hours,
            n.duration_hours, n.time_hours, n.is_icu,
            e.source_table, e.category, e.variable_name,
            e.numeric_value, e.text_value, e.unit, e.is_continuous,
            e.timestamp, e.timestamp_hours, n.bucket
        FROM raw e
        JOIN nodes n
          ON n.hadm_id = e.hadm_id
         AND e.timestamp_hours >= n.node_start_hours
         AND e.timestamp_hours < n.node_end_hours
        WHERE NOT e.is_continuous
    ),
    continuous AS (
        SELECT
            e.hadm_id, n.step_idx, n.node_start_hours, n.node_end_hours,
            n.duration_hours, n.time_hours, n.is_icu,
            e.source_table, e.category, e.variable_name,
            e.numeric_value, e.text_value, e.unit, e.is_continuous,
            e.timestamp, e.timestamp_hours, n.bucket
        FROM raw e
        JOIN nodes n
          ON n.hadm_id = e.hadm_id
         AND e.timestamp_hours < n.node_end_hours
         AND coalesce(e.end_timestamp_hours, e.timestamp_hours)
             > n.node_start_hours
        WHERE e.is_continuous
          AND coalesce(e.end_timestamp_hours, e.timestamp_hours)
              > e.timestamp_hours
    ),
    all_events AS (
        SELECT * FROM point
        UNION ALL
        SELECT * FROM continuous
    ),
    ranked AS (
        SELECT *,
            row_number() OVER (
                PARTITION BY hadm_id, step_idx, source_table, variable_name,
                    CASE WHEN source_table IN ({text_sources})
                         THEN text_value ELSE NULL END
                ORDER BY timestamp DESC
            ) AS rn
        FROM all_events
    ),
    capped AS (
        SELECT *,
            row_number() OVER (
                PARTITION BY hadm_id, step_idx, source_table
                ORDER BY timestamp DESC
            ) AS source_rank
        FROM ranked
        WHERE rn = 1
    )
    SELECT
        hadm_id, step_idx, node_start_hours, node_end_hours,
        duration_hours, time_hours, is_icu,
        variable_name, numeric_value, text_value, unit, category,
        source_table, is_continuous, timestamp_hours, bucket
    FROM capped
    WHERE source_rank <= ({cap_expr})
    """


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed 24h ward / 1h ICU hospital discretization"
    )
    parser.add_argument("--paths_config", default="hosp_paths_v1_mixed.yaml")
    parser.add_argument("--features_config", default="hosp_features_v1_mixed.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    paths = db.load_yaml(args.paths_config)
    features = db.load_yaml(args.features_config)
    temporal = features["temporal"]
    ward_bin_hours = float(temporal.get("ward_bin_hours", 24))
    icu_bin_hours = float(temporal.get("icu_bin_hours", 1))
    min_obs = int(temporal.get("min_obs_per_stay", 4))

    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"
    obs_in = Path(paths["output"]["observations"])
    act_in = Path(paths["output"]["actions"])
    out_root = Path(paths["output"]["discretized"])

    if out_root.exists():
        shutil.rmtree(out_root)
    for subdir in ("nodes", "obs", "act", "meta"):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    con = db.connect(paths)
    n_buckets = int(paths["sharding"]["n_buckets"])
    db.register_cohort(con, cohort_path, n_buckets)

    cohort = con.execute(
        "SELECT hadm_id, bucket, los_hours FROM cohort ORDER BY hadm_id"
    ).fetchdf()
    icu = con.execute(
        """
        SELECT
            c.hadm_id,
            greatest(
                date_diff('second', c.admittime, s.intime) / 3600.0, 0.0
            ) AS icu_start_hours,
            least(
                date_diff('second', c.admittime, s.outtime) / 3600.0,
                c.los_hours
            ) AS icu_end_hours
        FROM mimiciv_icu.icustays s
        JOIN cohort c USING (hadm_id)
        WHERE s.intime IS NOT NULL
          AND s.outtime IS NOT NULL
          AND s.outtime > c.admittime
          AND s.intime < c.admittime + c.los_hours * INTERVAL 1 HOUR
        ORDER BY c.hadm_id, s.intime
        """
    ).fetchdf()
    nodes = build_nodes(cohort, icu, ward_bin_hours, icu_bin_hours)
    con.register("_nodes_df", nodes)
    con.execute("CREATE OR REPLACE TEMP TABLE nodes AS SELECT * FROM _nodes_df")
    db.copy_to_buckets(con, "SELECT * FROM nodes", out_root / "nodes", overwrite=True)

    logger.info(
        "Built %s nodes: %s ward, %s ICU",
        f"{len(nodes):,}",
        f"{int((~nodes['is_icu']).sum()):,}",
        f"{int(nodes['is_icu'].sum()):,}",
    )

    obs_sql = _bin_and_collapse(
        f"{obs_in}/bucket=*/*.parquet",
        _caps_from_features(features, "observation"),
    )
    act_sql = _bin_and_collapse(
        f"{act_in}/bucket=*/*.parquet",
        _caps_from_features(features, "action"),
    )
    con.execute(f"CREATE OR REPLACE TEMP TABLE _obs AS {obs_sql}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE _act AS {act_sql}")

    db.copy_to_buckets(con, "SELECT * FROM _obs", out_root / "obs", overwrite=True)
    db.copy_to_buckets(con, "SELECT * FROM _act", out_root / "act", overwrite=True)

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE node_measurements AS
        SELECT
            n.hadm_id, n.step_idx,
            max(o.timestamp_hours) FILTER (
                WHERE o.category NOT IN (
                    'care_setting', 'diagnosis', 'admission_context',
                    'history_diagnosis'
                )
            ) AS measured_at,
            list_sort(list_distinct(
                list(o.category) FILTER (
                    WHERE o.category NOT IN (
                        'care_setting', 'diagnosis', 'admission_context',
                        'history_diagnosis'
                    )
                )
            )) AS observed_groups
        FROM nodes n
        LEFT JOIN _obs o USING (hadm_id, step_idx)
        GROUP BY n.hadm_id, n.step_idx
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE node_meta AS
        WITH joined AS (
            SELECT
                n.*,
                m.measured_at,
                coalesce(m.observed_groups, []::VARCHAR[]) AS observed_groups,
                count(a.variable_name) > 0 AS has_action
            FROM nodes n
            LEFT JOIN node_measurements m USING (hadm_id, step_idx)
            LEFT JOIN _act a USING (hadm_id, step_idx)
            GROUP BY ALL
        ),
        carried AS (
            SELECT *,
                max(measured_at) OVER (
                    PARTITION BY hadm_id ORDER BY step_idx
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS last_measurement_at
            FROM joined
        )
        SELECT *,
            measured_at IS NOT NULL AS has_measured,
            CASE
                WHEN last_measurement_at IS NULL THEN NULL
                ELSE greatest(node_end_hours - last_measurement_at, 0.0)
            END AS hours_since_last_measurement
        FROM carried
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE retained_hadm AS
        SELECT hadm_id
        FROM _obs
        WHERE category NOT IN (
            'care_setting', 'diagnosis', 'admission_context',
            'history_diagnosis'
        )
        GROUP BY hadm_id
        HAVING count(*) >= {min_obs}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _meta AS
        SELECT
            n.hadm_id, any_value(n.bucket) AS bucket, count(*) AS n_steps,
            list(n.node_start_hours ORDER BY n.step_idx) AS node_start_hours,
            list(n.node_end_hours ORDER BY n.step_idx) AS node_end_hours,
            list(n.duration_hours ORDER BY n.step_idx) AS duration_hours,
            list(n.time_hours ORDER BY n.step_idx) AS time_hours,
            list(n.is_icu ORDER BY n.step_idx) AS is_icu,
            list(n.has_measured ORDER BY n.step_idx) AS has_measured,
            list(n.has_action ORDER BY n.step_idx) AS has_action,
            list(n.hours_since_last_measurement ORDER BY n.step_idx)
                AS hours_since_last_measurement,
            list(n.observed_groups ORDER BY n.step_idx) AS observed_groups
        FROM node_meta n
        JOIN retained_hadm r USING (hadm_id)
        GROUP BY n.hadm_id
        """
    )
    db.copy_to_buckets(con, "SELECT * FROM _meta", out_root / "meta", overwrite=True)

    n_obs = int(con.execute("SELECT count(*) FROM _obs").fetchone()[0])
    n_act = int(con.execute("SELECT count(*) FROM _act").fetchone()[0])
    n_retained = int(con.execute("SELECT count(*) FROM _meta").fetchone()[0])
    n_measured = int(con.execute(
        "SELECT count(*) FROM node_meta WHERE has_measured"
    ).fetchone()[0])
    duration_counts = {
        f"{float(duration):g}h": int(count)
        for duration, count in con.execute(
            """
            SELECT round(duration_hours, 3), count(*)
            FROM nodes GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    }
    stats = {
        "mode": "mixed_ward_24h_icu_1h",
        "ward_bin_hours": ward_bin_hours,
        "icu_bin_hours": icu_bin_hours,
        "n_cohort": int(len(cohort)),
        "n_retained": n_retained,
        "total_nodes": int(len(nodes)),
        "ward_nodes": int((~nodes["is_icu"]).sum()),
        "icu_nodes": int(nodes["is_icu"].sum()),
        "measured_nodes": n_measured,
        "measured_node_occupancy": round(n_measured / len(nodes), 4),
        "binned_obs_rows": n_obs,
        "binned_act_rows": n_act,
        "duration_counts": duration_counts,
    }
    (out_root / "discretize_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    logger.info("Wrote mixed discretization to %s", out_root)


if __name__ == "__main__":
    main()
