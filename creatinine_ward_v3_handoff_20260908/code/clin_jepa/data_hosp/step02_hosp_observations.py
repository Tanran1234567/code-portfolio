"""Step 02: extract admission-wide observation (state) events.

ICU counterpart: :mod:`clin_jepa.data.step02_observations`, which draws ~90% of
its volume from ``chartevents`` via mimic-code concept views keyed on ICU
``stay_id``. ``chartevents`` exists for only 21.2% of this cohort, so the
hospital state is rebuilt from sources that span the whole admission:

* the mimic-code lab concepts, re-derived on ``hadm_id`` (they are built from
  ``mimiciv_hosp.labevents``, which is admission-wide, so the itemid
  whitelists and physiological bounds carry over unchanged);
* ``omr`` ward vitals, pinned to noon of their chart date;
* ``microbiologyevents`` culture results;
* ``transfers`` care-unit occupancy, which has no ICU-pipeline counterpart and
  is what lets the model represent escalation of care;
* the ICU concept tables, for the ICU segments of admissions that have them.

Every event is emitted in the long-format schema of
``helpers/hosp_schema.py`` and written to ``{output.observations}/bucket=NN/``.

Usage:
    python -m clin_jepa.data_hosp.step02_hosp_observations
    python -m clin_jepa.data_hosp.step02_hosp_observations --sources lab_concepts omr
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from clin_jepa.data_hosp import db
from clin_jepa.data_hosp.helpers.hosp_schema import SQL_EVENT_PROJECTION

logger = logging.getLogger(__name__)

# ── Time helpers ────────────────────────────────────────────────────────
# Every source projects hours since admittime and is clipped to the
# truncated LOS window, mirroring clip_to_icu_window() in the ICU pipeline.
_HOURS = "date_diff('second', c.admittime, {t}) / 3600.0"


def _window_predicate(t: str) -> str:
    return (
        f"{_HOURS.format(t=t)} >= 0 AND {_HOURS.format(t=t)} < c.los_hours"
    )


# ── Lab concepts ────────────────────────────────────────────────────────


def _lab_mapping_values(lab_concepts: dict) -> tuple[str, list[int]]:
    """Build a VALUES table mapping itemid -> concept / variable / unit / bounds.

    One pass over labevents with a join against this mapping replaces the nine
    separate concept queries the mimic-code views would need.
    """
    rows: list[str] = []
    itemids: list[int] = []
    for concept, spec in lab_concepts.items():
        category = spec["category"]
        for itemid, item in spec["items"].items():
            itemid = int(itemid)
            lo = item.get("min")
            hi = item.get("max")
            # mimic-code drops non-positive values except where zero is a
            # legitimate reading (anion gap, base excess).
            lo_sql = "NULL" if lo is None else f"{float(lo)}"
            hi_sql = "NULL" if hi is None else f"{float(hi)}"
            allow_zero = "TRUE" if item.get("allow_zero", False) else "FALSE"
            rows.append(
                f"({itemid}, '{concept}', '{category}', '{item['name']}', "
                f"'{item.get('unit', '')}', {lo_sql}, {hi_sql}, {allow_zero})"
            )
            itemids.append(itemid)
    values = ",\n            ".join(rows)
    return values, sorted(set(itemids))


def sql_lab_concepts(lab_concepts: dict) -> str:
    values, itemids = _lab_mapping_values(lab_concepts)
    id_list = ", ".join(str(i) for i in itemids)
    return f"""
    WITH lab_map(itemid, concept, category, var_name, unit, lo, hi, allow_zero) AS (
        VALUES
            {values}
    )
    SELECT
        c.hadm_id,
        l.charttime                              AS timestamp,
        {_HOURS.format(t='l.charttime')}         AS timestamp_hours,
        'observation'                            AS domain,
        m.concept                                AS source_table,
        m.category                               AS category,
        m.var_name                               AS variable_name,
        l.valuenum                               AS numeric_value,
        NULL::VARCHAR                            AS text_value,
        m.unit                                   AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.labevents l
    JOIN cohort c USING (hadm_id)
    JOIN lab_map m ON m.itemid = l.itemid
    WHERE l.itemid IN ({id_list})
      AND l.valuenum IS NOT NULL
      AND (l.valuenum > 0 OR m.allow_zero)
      AND (m.lo IS NULL OR l.valuenum >= m.lo)
      AND (m.hi IS NULL OR l.valuenum <= m.hi)
      AND {_window_predicate('l.charttime')}
    """


# ── OMR ward vitals ─────────────────────────────────────────────────────


def sql_omr(omr_cfg: dict) -> str:
    """Ward vitals from omr.

    omr carries ``chartdate`` only and no ``hadm_id``, so readings are pinned
    to a fixed hour of their date and matched into the admission window by
    ``subject_id``. Daily resolution, which a 6-hour bin tolerates and which
    is the only admission-wide vitals source MIMIC-IV offers.

    "Blood Pressure" arrives as the string "120/80" and is split into two
    numeric variables.
    """
    pin = int(omr_cfg.get("pin_hour", 12))
    names = omr_cfg["result_names"]

    branches = []
    for result_name, spec in names.items():
        esc = result_name.replace("'", "''")
        if spec.get("split_bp"):
            for idx, (var, part) in enumerate(
                [("sbp", 1), ("dbp", 2)], start=0,
            ):
                branches.append(f"""
        SELECT c.hadm_id, ts, '{var}' AS variable_name,
               try_cast(split_part(o.result_value, '/', {part}) AS DOUBLE) AS numeric_value,
               '{spec.get('unit', '')}' AS unit, c.bucket, c.admittime, c.los_hours
        FROM omr_pinned o JOIN cohort c
          ON c.subject_id = o.subject_id
         AND o.ts >= c.admittime AND o.ts < c.dischtime
        WHERE o.result_name = '{esc}'
                """)
        else:
            scale = float(spec.get("scale", 1.0))
            expr = f"try_cast(o.result_value AS DOUBLE)"
            if scale != 1.0:
                expr = f"({expr}) * {scale}"
            branches.append(f"""
        SELECT c.hadm_id, ts, '{spec['name']}' AS variable_name,
               {expr} AS numeric_value,
               '{spec.get('unit', '')}' AS unit, c.bucket, c.admittime, c.los_hours
        FROM omr_pinned o JOIN cohort c
          ON c.subject_id = o.subject_id
         AND o.ts >= c.admittime AND o.ts < c.dischtime
        WHERE o.result_name = '{esc}'
            """)

    union = "\n        UNION ALL\n".join(branches)
    result_name_list = ", ".join(
        "'" + n.replace("'", "''") + "'" for n in names
    )

    return f"""
    WITH omr_pinned AS (
        SELECT subject_id, result_name, result_value,
               (chartdate::TIMESTAMP + INTERVAL {pin} HOUR) AS ts
        FROM mimiciv_hosp.omr
        WHERE result_name IN ({result_name_list})
    ),
    joined AS (
        {union}
    )
    SELECT
        hadm_id,
        ts                                       AS timestamp,
        date_diff('second', admittime, ts) / 3600.0 AS timestamp_hours,
        'observation'                            AS domain,
        'omr'                                    AS source_table,
        '{omr_cfg['category']}'                  AS category,
        variable_name,
        numeric_value,
        NULL::VARCHAR                            AS text_value,
        unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        bucket
    FROM joined
    WHERE numeric_value IS NOT NULL
      AND date_diff('second', admittime, ts) / 3600.0 >= 0
      AND date_diff('second', admittime, ts) / 3600.0 < los_hours
    """


# ── Microbiology ────────────────────────────────────────────────────────


def sql_microbiology(cfg: dict) -> str:
    """Culture results as categorical observations.

    One row per (specimen, organism); a specimen with no organism is emitted
    as a negative result, which is clinically informative and would otherwise
    be silently dropped.
    """
    return f"""
    SELECT DISTINCT
        c.hadm_id,
        coalesce(m.charttime, m.chartdate)       AS timestamp,
        {_HOURS.format(t='coalesce(m.charttime, m.chartdate)')} AS timestamp_hours,
        'observation'                            AS domain,
        'micro'                                  AS source_table,
        '{cfg['category']}'                      AS category,
        lower(replace(coalesce(m.spec_type_desc, 'unknown'), ' ', '_'))
                                                 AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        coalesce(m.org_name, 'no growth')        AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.microbiologyevents m
    JOIN cohort c USING (hadm_id)
    WHERE coalesce(m.charttime, m.chartdate) IS NOT NULL
      AND {_window_predicate('coalesce(m.charttime, m.chartdate)')}
    """


# ── Care setting ────────────────────────────────────────────────────────


def sql_careunit(cfg: dict) -> str:
    """Ward / ICU / ED occupancy from transfers, as a continuous state.

    Emitted as a continuous event so that step04 expands it across every bin
    of the stay in that unit: care setting is a property of a time interval,
    not an instantaneous reading.
    """
    excluded = cfg.get("exclude_units", [])
    excl_sql = (
        "AND t.careunit NOT IN ("
        + ", ".join("'" + u.replace("'", "''") + "'" for u in excluded)
        + ")"
        if excluded
        else ""
    )
    return f"""
    SELECT
        c.hadm_id,
        t.intime                                 AS timestamp,
        greatest({_HOURS.format(t='t.intime')}, 0.0) AS timestamp_hours,
        'observation'                            AS domain,
        'careunit'                               AS source_table,
        '{cfg['category']}'                      AS category,
        'care_unit'                              AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        t.careunit                               AS text_value,
        ''                                       AS unit,
        coalesce(t.outtime, c.dischtime)         AS end_timestamp,
        least(
            {_HOURS.format(t='coalesce(t.outtime, c.dischtime)')},
            c.los_hours
        )                                        AS end_timestamp_hours,
        TRUE                                     AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.transfers t
    JOIN cohort c USING (hadm_id)
    WHERE t.intime IS NOT NULL
      AND t.careunit IS NOT NULL
      {excl_sql}
      AND {_HOURS.format(t='t.intime')} < c.los_hours
      AND {_HOURS.format(t="coalesce(t.outtime, c.dischtime)")} > 0
    """


# ── Admission-time context ─────────────────────────────────────────────


def sql_admission_context(cfg: dict) -> str:
    """Emit admission-time administrative context without future information.

    This local MIMIC-IV build has no ``admissions.diagnosis`` or ED diagnosis
    table, so admission type and source are the available admission-time
    descriptors. Current-admission ``diagnoses_icd`` remains label-only.
    """
    return f"""
    SELECT
        c.hadm_id,
        c.admittime                              AS timestamp,
        0.0                                      AS timestamp_hours,
        'observation'                            AS domain,
        'admission_context'                      AS source_table,
        'admission_context'                      AS category,
        'admission_profile'                      AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        concat(
            'type=', coalesce(a.admission_type, 'unknown'),
            '; location=', coalesce(a.admission_location, 'unknown')
        )                                        AS text_value,
        ''                                       AS unit,
        c.dischtime                              AS end_timestamp,
        c.los_hours                              AS end_timestamp_hours,
        TRUE                                     AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.admissions a
    JOIN cohort c USING (hadm_id)
    """


def sql_prior_diagnoses(cfg: dict) -> str:
    """Emit diagnoses from prior completed admissions as known history."""
    max_codes = int(cfg.get("max_codes", 20))
    return f"""
    WITH ranked AS (
        SELECT
            c.hadm_id,
            c.admittime,
            c.dischtime,
            c.los_hours,
            c.bucket,
            d.icd_code,
            d.icd_version,
            dd.long_title,
            row_number() OVER (
                PARTITION BY c.hadm_id
                ORDER BY p.dischtime DESC, d.seq_num ASC, d.icd_code
            ) AS history_rank
        FROM cohort c
        JOIN mimiciv_hosp.admissions p
          ON p.subject_id = c.subject_id
         AND p.hadm_id <> c.hadm_id
         AND p.dischtime IS NOT NULL
         AND p.dischtime <= c.admittime
        JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = p.hadm_id
        LEFT JOIN mimiciv_hosp.d_icd_diagnoses dd
          ON dd.icd_code = d.icd_code
         AND dd.icd_version = d.icd_version
    )
    SELECT
        hadm_id,
        admittime                                AS timestamp,
        0.0                                      AS timestamp_hours,
        'observation'                            AS domain,
        'prior_diagnoses'                        AS source_table,
        'history_diagnosis'                      AS category,
        icd_code                                 AS variable_name,
        icd_version::DOUBLE                      AS numeric_value,
        long_title                               AS text_value,
        ''                                       AS unit,
        dischtime                                AS end_timestamp,
        los_hours                                AS end_timestamp_hours,
        TRUE                                     AS is_continuous,
        bucket
    FROM ranked
    WHERE history_rank <= {max_codes}
    """


# ── ICU derived concepts ────────────────────────────────────────────────


def sql_icu_concept(table: str, spec: dict, category: str) -> str:
    """Unpivot one ICU concept table into long-format events.

    Joined to the cohort through ``icustays`` so events land on the right
    admission, and clipped to the admission window like every other source.
    """
    ts_col = spec.get("timestamp_col", "charttime")
    value_cols = spec["value_cols"]
    col_list = ", ".join(f'"{c}"' for c in value_cols)

    return f"""
    WITH src AS (
        SELECT c.hadm_id, c.bucket, c.admittime, c.los_hours,
               d.{ts_col} AS ts, {col_list}
        FROM {db.derived_table(table)} d
        JOIN mimiciv_icu.icustays s ON s.stay_id = d.stay_id
        JOIN cohort c ON c.hadm_id = s.hadm_id
        WHERE d.{ts_col} IS NOT NULL
          AND date_diff('second', c.admittime, d.{ts_col}) / 3600.0 >= 0
          AND date_diff('second', c.admittime, d.{ts_col}) / 3600.0 < c.los_hours
    ),
    long AS (
        UNPIVOT src
        ON {col_list}
        INTO NAME variable_name VALUE numeric_value
    )
    SELECT
        hadm_id,
        ts                                       AS timestamp,
        date_diff('second', admittime, ts) / 3600.0 AS timestamp_hours,
        'observation'                            AS domain,
        '{table}'                                AS source_table,
        '{category}'                             AS category,
        variable_name,
        numeric_value::DOUBLE                    AS numeric_value,
        NULL::VARCHAR                            AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        bucket
    FROM long
    WHERE numeric_value IS NOT NULL
    """


# ── Diagnoses (bin 0, label source only) ────────────────────────────────


def sql_diagnoses() -> str:
    """ICD diagnoses pinned to bin 0.

    Excluded from state_text by ``hosp_trajectories.yaml`` (they are the label
    source for downstream new-onset tasks), but carried through the pipeline so
    that label construction does not need to re-query MIMIC.
    """
    return f"""
    SELECT
        c.hadm_id,
        c.admittime                              AS timestamp,
        0.0                                      AS timestamp_hours,
        'observation'                            AS domain,
        'diagnoses'                              AS source_table,
        'diagnosis'                              AS category,
        d.icd_code                               AS variable_name,
        d.icd_version::DOUBLE                    AS numeric_value,
        dd.long_title                            AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN cohort c USING (hadm_id)
    LEFT JOIN mimiciv_hosp.d_icd_diagnoses dd
           ON dd.icd_code = d.icd_code AND dd.icd_version = d.icd_version
    """


# ── Driver ──────────────────────────────────────────────────────────────


def build_source_queries(con, features: dict) -> list[tuple[str, str]]:
    """Assemble the (name, sql) list of enabled observation sources."""
    obs = features["observations"]
    queries: list[tuple[str, str]] = [
        ("lab_concepts", sql_lab_concepts(obs["lab_concepts"])),
    ]

    if obs.get("omr", {}).get("enabled", False):
        queries.append(("omr", sql_omr(obs["omr"])))
    if obs.get("microbiology", {}).get("enabled", False):
        queries.append(("microbiology", sql_microbiology(obs["microbiology"])))
    if obs.get("careunit", {}).get("enabled", False):
        queries.append(("careunit", sql_careunit(obs["careunit"])))
    if obs.get("admission_context", {}).get("enabled", False):
        queries.append((
            "admission_context",
            sql_admission_context(obs["admission_context"]),
        ))
    if obs.get("prior_diagnoses", {}).get("enabled", False):
        queries.append((
            "prior_diagnoses",
            sql_prior_diagnoses(obs["prior_diagnoses"]),
        ))
    queries.append(("diagnoses", sql_diagnoses()))

    icu_cfg = obs.get("icu_concepts", {})
    if icu_cfg.get("enabled", False):
        available = db.derived_tables(con)
        if not available:
            logger.warning("ICU concepts requested but derived DB unavailable")
        for table, spec in icu_cfg["tables"].items():
            if table not in available:
                logger.warning("  skipping ICU concept %s (not in derived DB)", table)
                continue
            queries.append((
                f"icu_{table}",
                sql_icu_concept(table, spec, spec["category"]),
            ))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 02: hospital observations")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--features_config", default="hosp_features.yaml")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Restrict to these source names")
    parser.add_argument("--limit_admissions", type=int, default=None,
                        help="Subsample the cohort (smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    logger.info("=" * 62)
    logger.info("Step 02: hospital observation extraction")
    logger.info("=" * 62)

    paths = db.load_yaml(args.paths_config)
    features = db.load_yaml(args.features_config)
    n_buckets = int(paths["sharding"]["n_buckets"])

    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"
    out_dir = Path(paths["output"]["observations"])

    con = db.connect(paths)
    n_cohort = db.register_cohort(con, cohort_path, n_buckets)
    if args.limit_admissions:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE cohort AS "
            f"SELECT * FROM cohort ORDER BY hadm_id LIMIT {args.limit_admissions}"
        )
        n_cohort = int(con.execute("SELECT count(*) FROM cohort").fetchone()[0])
    logger.info("Cohort registered: %d admissions, %d buckets", n_cohort, n_buckets)

    queries = build_source_queries(con, features)
    if args.sources:
        queries = [(n, q) for n, q in queries if n in set(args.sources)]
        logger.info("Restricted to sources: %s", [n for n, _ in queries])

    if out_dir.exists() and not args.sources:
        logger.info("Clearing %s", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    for name, sql in queries:
        logger.info("Extracting %s ...", name)
        con.execute(f"CREATE OR REPLACE TEMP TABLE _src AS {sql}")
        n = int(con.execute("SELECT count(*) FROM _src").fetchone()[0])
        stats[name] = n
        if n == 0:
            logger.warning("  %s produced 0 rows", name)
            continue
        db.copy_to_buckets(
            con,
            f"SELECT {SQL_EVENT_PROJECTION} FROM _src",
            out_dir,
        )
        logger.info("  %s: %s events", name, f"{n:,}")

    total = sum(stats.values())
    logger.info("-" * 62)
    logger.info("Total observation events: %s", f"{total:,}")
    for name, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        logger.info("  %-22s %14s  (%5.1f%%)", name, f"{n:,}",
                    100.0 * n / total if total else 0.0)

    stats_path = out_dir / "_step02_stats.json"
    stats_path.write_text(json.dumps(
        {"n_cohort": n_cohort, "total_events": total, "by_source": stats},
        indent=2,
    ))
    logger.info("Wrote %s", stats_path)


if __name__ == "__main__":
    main()
