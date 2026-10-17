"""Step 03: extract admission-wide action (intervention) events.

ICU counterpart: :mod:`clin_jepa.data.step03_actions`, whose volume is
dominated by ``inputevents`` and ``procedureevents`` — both ICU-only tables.
The hospital analogue is built from order and administration records that span
the whole admission:

* ``prescriptions`` — medication orders, continuous over [starttime, stoptime);
* ``emar`` — actual administrations, point events, filtered to the statuses
  that mean a drug reached the patient;
* ``procedures_icd`` — coded procedures, pinned to noon of ``chartdate``;
* the mimic-code ``antibiotic`` concept, which is keyed on ``hadm_id`` and so
  is already admission-wide;
* ICU ventilation / vasoactive / RRT concepts for the ICU segments.

Carrier fluids and flushes are excluded here rather than in step06. The ICU
pipeline defers that filtering to ``helpers/action_filter.py`` because its
volume is manageable; at hospital scale ``prescriptions`` and ``emar`` together
are ~59M rows, and filtering at extraction avoids writing tens of millions of
rows that would only be dropped later.

Usage:
    python -m clin_jepa.data_hosp.step03_hosp_actions
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

_HOURS = "date_diff('second', c.admittime, {t}) / 3600.0"


def _quote_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


# ── Prescriptions ───────────────────────────────────────────────────────


def sql_prescriptions(cfg: dict) -> str:
    """Medication orders as continuous events.

    An order is active from starttime to stoptime, so step04 expands it across
    every bin it overlaps — the same treatment the ICU pipeline gives infusions.
    Orders with no stoptime collapse to a point event at starttime.
    """
    patterns = cfg.get("exclude_drug_patterns", [])
    excl = ""
    if patterns:
        clauses = " OR ".join(
            f"upper(r.drug) LIKE upper('{p}')" for p in patterns
        )
        excl = f"AND NOT ({clauses})"

    return f"""
    SELECT
        c.hadm_id,
        r.starttime                              AS timestamp,
        greatest({_HOURS.format(t='r.starttime')}, 0.0) AS timestamp_hours,
        'action'                                 AS domain,
        'rx'                                     AS source_table,
        '{cfg['category']}'                      AS category,
        r.drug                                   AS variable_name,
        try_cast(r.dose_val_rx AS DOUBLE)        AS numeric_value,
        trim(concat_ws(' ',
            nullif(r.dose_val_rx, ''),
            nullif(r.dose_unit_rx, ''),
            nullif(r.route, '')))                AS text_value,
        coalesce(r.dose_unit_rx, '')             AS unit,
        r.stoptime                               AS end_timestamp,
        least({_HOURS.format(t='r.stoptime')}, c.los_hours) AS end_timestamp_hours,
        (r.stoptime IS NOT NULL AND r.stoptime > r.starttime) AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.prescriptions r
    JOIN cohort c USING (hadm_id)
    WHERE r.starttime IS NOT NULL
      AND r.drug IS NOT NULL
      {excl}
      AND {_HOURS.format(t='r.starttime')} < c.los_hours
      AND {_HOURS.format(t="coalesce(r.stoptime, r.starttime)")} >= 0
    """


# ── eMAR administrations ────────────────────────────────────────────────


def sql_emar(cfg: dict) -> str:
    """Actual administrations as point events.

    ``event_txt`` distinguishes an administration from a flush, a refusal, or a
    documentation-only entry; only the former are interventions.
    """
    keep = cfg.get("keep_event_txt", ["Administered"])
    return f"""
    SELECT
        c.hadm_id,
        e.charttime                              AS timestamp,
        {_HOURS.format(t='e.charttime')}         AS timestamp_hours,
        'action'                                 AS domain,
        'emar_admin'                             AS source_table,
        '{cfg['category']}'                      AS category,
        e.medication                             AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        e.event_txt                              AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.emar e
    JOIN cohort c USING (hadm_id)
    WHERE e.charttime IS NOT NULL
      AND e.medication IS NOT NULL
      AND e.medication <> ''
      AND e.event_txt IN ({_quote_list(keep)})
      AND {_HOURS.format(t='e.charttime')} >= 0
      AND {_HOURS.format(t='e.charttime')} < c.los_hours
    """


def sql_poe(cfg: dict) -> str:
    """Provider orders visible before the ward cutoff as point actions."""
    return f"""
    SELECT
        c.hadm_id,
        p.ordertime                              AS timestamp,
        {_HOURS.format(t='p.ordertime')}         AS timestamp_hours,
        'action'                                 AS domain,
        'poe'                                    AS source_table,
        '{cfg['category']}'                      AS category,
        coalesce(p.order_type, 'Provider order') AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        concat_ws(' | ', p.order_subtype, p.order_status) AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.poe p
    JOIN cohort c USING (hadm_id)
    WHERE p.ordertime IS NOT NULL
      AND {_HOURS.format(t='p.ordertime')} >= 0
      AND {_HOURS.format(t='p.ordertime')} < c.los_hours
    """


# ── Coded procedures ────────────────────────────────────────────────────


def sql_procedures(cfg: dict) -> str:
    """ICD procedures, pinned to a fixed hour of chartdate (date-only source)."""
    pin = int(cfg.get("pin_hour", 12))
    ts = f"(p.chartdate::TIMESTAMP + INTERVAL {pin} HOUR)"
    return f"""
    SELECT
        c.hadm_id,
        {ts}                                     AS timestamp,
        {_HOURS.format(t=ts)}                    AS timestamp_hours,
        'action'                                 AS domain,
        'proc_icd'                               AS source_table,
        '{cfg['category']}'                      AS category,
        p.icd_code                               AS variable_name,
        p.icd_version::DOUBLE                    AS numeric_value,
        dp.long_title                            AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM mimiciv_hosp.procedures_icd p
    JOIN cohort c USING (hadm_id)
    LEFT JOIN mimiciv_hosp.d_icd_procedures dp
           ON dp.icd_code = p.icd_code AND dp.icd_version = p.icd_version
    WHERE p.chartdate IS NOT NULL
      AND {_HOURS.format(t=ts)} >= 0
      AND {_HOURS.format(t=ts)} < c.los_hours
    """


# ── Antibiotics (hospital-wide derived concept) ─────────────────────────


def sql_antibiotic(cfg: dict) -> str:
    return f"""
    SELECT
        c.hadm_id,
        a.starttime                              AS timestamp,
        greatest({_HOURS.format(t='a.starttime')}, 0.0) AS timestamp_hours,
        'action'                                 AS domain,
        'antibiotic'                             AS source_table,
        '{cfg['category']}'                      AS category,
        a.antibiotic                             AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        coalesce(a.route, '')                    AS text_value,
        ''                                       AS unit,
        a.stoptime                               AS end_timestamp,
        least({_HOURS.format(t='a.stoptime')}, c.los_hours) AS end_timestamp_hours,
        (a.stoptime IS NOT NULL AND a.stoptime > a.starttime) AS is_continuous,
        c.bucket                                 AS bucket
    FROM {db.derived_table('antibiotic')} a
    JOIN cohort c USING (hadm_id)
    WHERE a.starttime IS NOT NULL
      AND a.antibiotic IS NOT NULL
      AND {_HOURS.format(t='a.starttime')} < c.los_hours
      AND {_HOURS.format(t="coalesce(a.stoptime, a.starttime)")} >= 0
    """


# ── ICU intervention concepts ───────────────────────────────────────────


def sql_ventilation(spec: dict) -> str:
    """Ventilation episodes: a text-valued continuous action."""
    return f"""
    SELECT
        c.hadm_id,
        v.starttime                              AS timestamp,
        greatest(date_diff('second', c.admittime, v.starttime) / 3600.0, 0.0)
                                                 AS timestamp_hours,
        'action'                                 AS domain,
        'ventilation'                            AS source_table,
        '{spec['category']}'                     AS category,
        'ventilation_status'                     AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        v.ventilation_status                     AS text_value,
        ''                                       AS unit,
        v.endtime                                AS end_timestamp,
        least(date_diff('second', c.admittime, v.endtime) / 3600.0, c.los_hours)
                                                 AS end_timestamp_hours,
        TRUE                                     AS is_continuous,
        c.bucket                                 AS bucket
    FROM {db.derived_table('ventilation')} v
    JOIN mimiciv_icu.icustays s ON s.stay_id = v.stay_id
    JOIN cohort c ON c.hadm_id = s.hadm_id
    WHERE v.starttime IS NOT NULL
      AND v.ventilation_status IS NOT NULL
      AND v.ventilation_status <> 'None'
      AND date_diff('second', c.admittime, v.starttime) / 3600.0 < c.los_hours
      AND date_diff('second', c.admittime, coalesce(v.endtime, v.starttime)) / 3600.0 >= 0
    """


def sql_vasoactive(spec: dict) -> str:
    """Vasoactive infusions: one continuous event per agent with a rate."""
    cols = ", ".join(f'"{c}"' for c in spec["value_cols"])
    return f"""
    WITH src AS (
        SELECT c.hadm_id, c.bucket, c.admittime, c.los_hours,
               v.starttime, v.endtime, {cols}
        FROM {db.derived_table('vasoactive_agent')} v
        JOIN mimiciv_icu.icustays s ON s.stay_id = v.stay_id
        JOIN cohort c ON c.hadm_id = s.hadm_id
        WHERE v.starttime IS NOT NULL
          AND date_diff('second', c.admittime, v.starttime) / 3600.0 < c.los_hours
          AND date_diff('second', c.admittime, coalesce(v.endtime, v.starttime)) / 3600.0 >= 0
    ),
    long AS (
        UNPIVOT src ON {cols}
        INTO NAME variable_name VALUE rate
    )
    SELECT
        hadm_id,
        starttime                                AS timestamp,
        greatest(date_diff('second', admittime, starttime) / 3600.0, 0.0)
                                                 AS timestamp_hours,
        'action'                                 AS domain,
        'vasoactive_agent'                       AS source_table,
        '{spec['category']}'                     AS category,
        variable_name,
        rate::DOUBLE                             AS numeric_value,
        NULL::VARCHAR                            AS text_value,
        'mcg/kg/min'                             AS unit,
        endtime                                  AS end_timestamp,
        least(date_diff('second', admittime, endtime) / 3600.0, los_hours)
                                                 AS end_timestamp_hours,
        TRUE                                     AS is_continuous,
        bucket
    FROM long
    WHERE rate IS NOT NULL AND rate > 0
    """


def sql_rrt(spec: dict) -> str:
    """Renal replacement therapy as point events with a modality label."""
    return f"""
    SELECT
        c.hadm_id,
        r.charttime                              AS timestamp,
        date_diff('second', c.admittime, r.charttime) / 3600.0 AS timestamp_hours,
        'action'                                 AS domain,
        'rrt'                                    AS source_table,
        '{spec['category']}'                     AS category,
        'dialysis'                               AS variable_name,
        NULL::DOUBLE                             AS numeric_value,
        coalesce(r.dialysis_type, 'Dialysis')    AS text_value,
        ''                                       AS unit,
        NULL::TIMESTAMP                          AS end_timestamp,
        NULL::DOUBLE                             AS end_timestamp_hours,
        FALSE                                    AS is_continuous,
        c.bucket                                 AS bucket
    FROM {db.derived_table('rrt')} r
    JOIN mimiciv_icu.icustays s ON s.stay_id = r.stay_id
    JOIN cohort c ON c.hadm_id = s.hadm_id
    WHERE r.charttime IS NOT NULL
      AND r.dialysis_active = 1
      AND date_diff('second', c.admittime, r.charttime) / 3600.0 >= 0
      AND date_diff('second', c.admittime, r.charttime) / 3600.0 < c.los_hours
    """


# ── Driver ──────────────────────────────────────────────────────────────

_ICU_BUILDERS = {
    "ventilation": sql_ventilation,
    "vasoactive_agent": sql_vasoactive,
    "rrt": sql_rrt,
}


def build_source_queries(con, features: dict) -> list[tuple[str, str]]:
    acts = features["actions"]
    queries: list[tuple[str, str]] = []

    if acts.get("prescriptions", {}).get("enabled", False):
        queries.append(("prescriptions", sql_prescriptions(acts["prescriptions"])))
    if acts.get("emar", {}).get("enabled", False):
        queries.append(("emar", sql_emar(acts["emar"])))
    if acts.get("poe", {}).get("enabled", False):
        queries.append(("poe", sql_poe(acts["poe"])))
    if acts.get("procedures_icd", {}).get("enabled", False):
        queries.append(("procedures_icd", sql_procedures(acts["procedures_icd"])))

    available = db.derived_tables(con)

    if acts.get("antibiotic", {}).get("enabled", False):
        if "antibiotic" in available:
            queries.append(("antibiotic", sql_antibiotic(acts["antibiotic"])))
        else:
            logger.warning("  skipping antibiotic (not in derived DB)")

    icu_cfg = acts.get("icu_concepts", {})
    if icu_cfg.get("enabled", False):
        for table, spec in icu_cfg["tables"].items():
            if table not in available:
                logger.warning("  skipping ICU action %s (not in derived DB)", table)
                continue
            builder = _ICU_BUILDERS.get(table)
            if builder is None:
                logger.warning("  no builder for ICU action %s", table)
                continue
            queries.append((f"icu_{table}", builder(spec)))

    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03: hospital actions")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--features_config", default="hosp_features.yaml")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--limit_admissions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    logger.info("=" * 62)
    logger.info("Step 03: hospital action extraction")
    logger.info("=" * 62)

    paths = db.load_yaml(args.paths_config)
    features = db.load_yaml(args.features_config)
    n_buckets = int(paths["sharding"]["n_buckets"])

    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"
    out_dir = Path(paths["output"]["actions"])

    con = db.connect(paths)
    n_cohort = db.register_cohort(con, cohort_path, n_buckets)
    if args.limit_admissions:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE cohort AS "
            f"SELECT * FROM cohort ORDER BY hadm_id LIMIT {args.limit_admissions}"
        )
        n_cohort = int(con.execute("SELECT count(*) FROM cohort").fetchone()[0])
    logger.info("Cohort registered: %d admissions", n_cohort)

    queries = build_source_queries(con, features)
    if args.sources:
        queries = [(n, q) for n, q in queries if n in set(args.sources)]

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
            con, f"SELECT {SQL_EVENT_PROJECTION} FROM _src", out_dir,
        )
        logger.info("  %s: %s events", name, f"{n:,}")

    total = sum(stats.values())
    logger.info("-" * 62)
    logger.info("Total action events: %s", f"{total:,}")
    for name, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        logger.info("  %-22s %14s  (%5.1f%%)", name, f"{n:,}",
                    100.0 * n / total if total else 0.0)

    (out_dir / "_step03_stats.json").write_text(json.dumps(
        {"n_cohort": n_cohort, "total_events": total, "by_source": stats},
        indent=2,
    ))


if __name__ == "__main__":
    main()
