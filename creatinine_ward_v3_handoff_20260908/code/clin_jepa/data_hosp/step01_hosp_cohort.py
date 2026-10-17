"""Step 01: hospital admission cohort.

ICU counterpart: :mod:`clin_jepa.data.step01_cohort`, which selects the first
ICU stay per patient from the mimic-code ``icustay_detail`` view. Here the unit
is the hospital admission, the time origin is ``admittime``, and there is no
``icustay_detail`` equivalent, so demographics and comorbidity are assembled
from ``admissions`` / ``patients`` / ``diagnoses_icd`` directly.

Output:
  {output.cohort}/cohort.parquet
  {output.cohort}/cohort_stats.json

Usage:
    python -m clin_jepa.data_hosp.step01_hosp_cohort
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clin_jepa.data_hosp import db

logger = logging.getLogger(__name__)


# Elixhauser/Charlson-style comorbidity weighting needs the full mimic-code
# comorbidity view, which is not materialised on this machine. A diagnosis
# count is a defensible stand-in for the demographics text: it is monotone in
# comorbidity burden and, unlike a Charlson index, needs no ICD mapping table.
COHORT_SQL = """
WITH adm AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.race,
        a.hospital_expire_flag,
        p.gender,
        p.dod,
        p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year)
            AS admission_age,
        date_diff('second', a.admittime, a.dischtime) / 3600.0 AS los_hours_raw
    FROM mimiciv_hosp.admissions a
    JOIN mimiciv_hosp.patients p USING (subject_id)
    WHERE a.dischtime IS NOT NULL
      AND a.dischtime > a.admittime
),
filtered AS (
    SELECT * FROM adm
    WHERE admission_age >= {min_age}
      AND admission_age < 120
      AND los_hours_raw >= {min_los_hours}
),
dx AS (
    SELECT hadm_id, count(*) AS n_diagnoses
    FROM mimiciv_hosp.diagnoses_icd
    GROUP BY hadm_id
),
lab AS (
    SELECT l.hadm_id, count(*) AS n_lab_events
    FROM mimiciv_hosp.labevents l
    JOIN filtered f USING (hadm_id)
    WHERE l.valuenum IS NOT NULL
      AND l.charttime >= f.admittime
      AND l.charttime < f.dischtime
    GROUP BY l.hadm_id
),
icu AS (
    SELECT hadm_id,
           count(*)      AS n_icu_stays,
           sum(los) * 24 AS icu_hours
    FROM mimiciv_icu.icustays
    GROUP BY hadm_id
),
seq AS (
    SELECT hadm_id, subject_id, admittime, dischtime,
           row_number() OVER (PARTITION BY subject_id ORDER BY admittime)
               AS hadm_seq,
           count(*)     OVER (PARTITION BY subject_id) AS n_admissions_patient,
           lead(admittime) OVER (PARTITION BY subject_id ORDER BY admittime)
               AS next_admittime
    FROM filtered
),
-- Suspicion of infection: an antibiotic and a culture within 24h of each
-- other, which is the mimic-code `suspicion_of_infection` criterion. Used
-- because Sepsis-3 needs a SOFA trajectory and is therefore ICU-only, while
-- this is computable admission-wide. It fills the `sepsis3` slot of the
-- trajectory schema; see step06 for the aliasing note.
infection AS (
    SELECT DISTINCT ab.hadm_id
    FROM ({antibiotic_source}) ab
    JOIN mimiciv_hosp.microbiologyevents mb
      ON mb.hadm_id = ab.hadm_id
     AND abs(date_diff('hour', ab.starttime,
                       coalesce(mb.charttime, mb.chartdate))) <= 24
)
SELECT
    f.subject_id,
    f.hadm_id,
    f.admittime,
    f.dischtime,
    f.deathtime,
    f.gender,
    f.admission_age,
    f.race,
    f.admission_type,
    f.admission_location,
    f.discharge_location,
    f.insurance,
    f.los_hours_raw,
    least(f.los_hours_raw, {max_los_hours}) AS los_hours,
    f.hospital_expire_flag,
    coalesce(dx.n_diagnoses, 0)             AS n_diagnoses,
    coalesce(lab.n_lab_events, 0)           AS n_lab_events,
    coalesce(icu.n_icu_stays, 0)            AS n_icu_stays,
    coalesce(icu.icu_hours, 0.0)            AS icu_hours,
    coalesce(icu.n_icu_stays, 0) > 0        AS has_icu,
    s.hadm_seq,
    s.n_admissions_patient,
    coalesce(inf.hadm_id IS NOT NULL, FALSE) AS suspected_infection,
    -- Unplanned-readmission proxy: another qualifying admission starting
    -- within 30 days of this discharge.
    coalesce(
        date_diff('day', f.dischtime, s.next_admittime) BETWEEN 0 AND 30,
        FALSE
    )                                        AS readmission_30d,
    -- 30/90-day mortality from date of death, censored at the horizon.
    CASE WHEN f.dod IS NOT NULL
              AND date_diff('day', f.dischtime::DATE, f.dod) <= 30 THEN TRUE
         WHEN f.hospital_expire_flag = 1 THEN TRUE
         ELSE FALSE END                     AS mortality_30d,
    CASE WHEN f.dod IS NOT NULL
              AND date_diff('day', f.dischtime::DATE, f.dod) <= 90 THEN TRUE
         WHEN f.hospital_expire_flag = 1 THEN TRUE
         ELSE FALSE END                     AS mortality_90d,
    f.los_hours_raw > {prolonged_hours}     AS prolonged_stay
FROM filtered f
LEFT JOIN dx  USING (hadm_id)
LEFT JOIN lab USING (hadm_id)
LEFT JOIN icu USING (hadm_id)
LEFT JOIN infection inf ON inf.hadm_id = f.hadm_id
JOIN seq s    USING (hadm_id)
WHERE {lab_predicate}
ORDER BY f.hadm_id
"""

# The mimic-code antibiotic concept is keyed on hadm_id, so it works at
# hospital scope. Without the derived database the infection flag degrades to
# a prescriptions text match rather than disappearing.
ANTIBIOTIC_FROM_DERIVED = """
    SELECT hadm_id, starttime
    FROM derived_db.mimiciv_derived.antibiotic
    WHERE hadm_id IS NOT NULL AND starttime IS NOT NULL
"""

ANTIBIOTIC_FALLBACK = """
    SELECT hadm_id, starttime
    FROM mimiciv_hosp.prescriptions
    WHERE starttime IS NOT NULL
      AND (upper(drug) LIKE '%CILLIN%' OR upper(drug) LIKE '%CEF%'
           OR upper(drug) LIKE '%MYCIN%' OR upper(drug) LIKE '%FLOXACIN%'
           OR upper(drug) LIKE '%PENEM%'  OR upper(drug) LIKE '%CYCLINE%'
           OR upper(drug) LIKE '%METRONIDAZOLE%'
           OR upper(drug) LIKE '%VANCOMYCIN%')
"""


def build_cohort(con, cohort_cfg: dict) -> str:
    """Materialise the cohort as the temp table ``cohort_raw``.

    Returns the SQL used, for provenance in the stats file.
    """
    incl = cohort_cfg["inclusion"]
    excl = cohort_cfg["exclusion"]
    labels = cohort_cfg.get("labels", {})

    lab_predicate = (
        f"coalesce(lab.n_lab_events, 0) >= {int(excl.get('min_obs_events', 1))}"
        if excl.get("require_any_lab", True)
        else "TRUE"
    )

    antibiotic_source = (
        ANTIBIOTIC_FROM_DERIVED
        if "antibiotic" in db.derived_tables(con)
        else ANTIBIOTIC_FALLBACK
    )
    if antibiotic_source is ANTIBIOTIC_FALLBACK:
        logger.warning(
            "derived antibiotic table missing; using drug-name fallback for "
            "the suspected-infection flag",
        )

    sql = COHORT_SQL.format(
        min_age=int(incl["min_age"]),
        min_los_hours=float(excl["min_los_hours"]),
        max_los_hours=float(cohort_cfg["max_los_hours"]),
        prolonged_hours=float(labels.get("prolonged_stay_hours", 168)),
        lab_predicate=lab_predicate,
        antibiotic_source=antibiotic_source,
    )

    if not excl.get("keep_first_admission_only", False):
        con.execute(f"CREATE OR REPLACE TEMP TABLE cohort_raw AS {sql}")
    else:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE cohort_raw AS "
            f"SELECT * FROM ({sql}) WHERE hadm_seq = 1"
        )
    return sql


def log_funnel(con, cohort_cfg: dict) -> dict:
    """Report how many admissions each filter removes."""
    incl = cohort_cfg["inclusion"]
    excl = cohort_cfg["exclusion"]
    steps = [
        ("all_admissions", "SELECT count(*) FROM mimiciv_hosp.admissions"),
        (
            "with_valid_discharge",
            "SELECT count(*) FROM mimiciv_hosp.admissions "
            "WHERE dischtime IS NOT NULL AND dischtime > admittime",
        ),
        (
            f"age_ge_{incl['min_age']}",
            f"""SELECT count(*) FROM mimiciv_hosp.admissions a
                JOIN mimiciv_hosp.patients p USING (subject_id)
                WHERE a.dischtime > a.admittime
                  AND p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year)
                      BETWEEN {incl['min_age']} AND 119""",
        ),
        (
            f"los_ge_{excl['min_los_hours']}h",
            f"""SELECT count(*) FROM mimiciv_hosp.admissions a
                JOIN mimiciv_hosp.patients p USING (subject_id)
                WHERE a.dischtime > a.admittime
                  AND p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year)
                      BETWEEN {incl['min_age']} AND 119
                  AND date_diff('second', a.admittime, a.dischtime) / 3600.0
                      >= {excl['min_los_hours']}""",
        ),
        ("final_cohort", "SELECT count(*) FROM cohort_raw"),
    ]
    funnel = {}
    prev = None
    for name, q in steps:
        n = int(con.execute(q).fetchone()[0])
        funnel[name] = n
        delta = "" if prev is None else f"  (-{prev - n:,})"
        logger.info("  %-28s %10s%s", name, f"{n:,}", delta)
        prev = n
    return funnel


def compute_stats(con) -> dict:
    row = con.execute(
        """
        SELECT
            count(*), count(DISTINCT subject_id),
            avg(admission_age), median(admission_age),
            avg(los_hours), median(los_hours),
            quantile_cont(los_hours, 0.10), quantile_cont(los_hours, 0.90),
            quantile_cont(los_hours, 0.95), max(los_hours),
            avg(hospital_expire_flag), avg(CASE WHEN has_icu THEN 1 ELSE 0 END),
            avg(CASE WHEN mortality_30d THEN 1 ELSE 0 END),
            avg(CASE WHEN mortality_90d THEN 1 ELSE 0 END),
            avg(CASE WHEN prolonged_stay THEN 1 ELSE 0 END),
            avg(n_diagnoses), avg(n_lab_events),
            avg(n_admissions_patient),
            avg(CASE WHEN suspected_infection THEN 1 ELSE 0 END),
            avg(CASE WHEN readmission_30d THEN 1 ELSE 0 END)
        FROM cohort_raw
        """
    ).fetchone()
    keys = [
        "n_admissions", "n_patients", "age_mean", "age_median",
        "los_hours_mean", "los_hours_median", "los_hours_p10",
        "los_hours_p90", "los_hours_p95", "los_hours_max",
        "hospital_mortality_rate", "icu_overlap_rate",
        "mortality_30d_rate", "mortality_90d_rate", "prolonged_stay_rate",
        "n_diagnoses_mean", "n_lab_events_mean", "admissions_per_patient_mean",
        "suspected_infection_rate", "readmission_30d_rate",
    ]
    stats = {
        k: (round(float(v), 4) if v is not None else None)
        for k, v in zip(keys, row)
    }
    stats["n_admissions"] = int(stats["n_admissions"])
    stats["n_patients"] = int(stats["n_patients"])

    stats["gender"] = {
        g: int(n)
        for g, n in con.execute(
            "SELECT gender, count(*) FROM cohort_raw GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }
    stats["admission_type"] = {
        t: int(n)
        for t, n in con.execute(
            "SELECT admission_type, count(*) FROM cohort_raw "
            "GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 01: hospital cohort")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--cohort_config", default="hosp_cohort.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    logger.info("=" * 62)
    logger.info("Step 01: hospital admission cohort")
    logger.info("=" * 62)

    paths = db.load_yaml(args.paths_config)
    cohort_cfg = db.load_yaml(args.cohort_config)

    out_dir = Path(paths["output"]["cohort"])
    out_dir.mkdir(parents=True, exist_ok=True)

    con = db.connect(paths)

    logger.info("Building cohort ...")
    build_cohort(con, cohort_cfg)

    logger.info("Inclusion funnel:")
    funnel = log_funnel(con, cohort_cfg)

    out_path = out_dir / "cohort.parquet"
    con.execute(
        f"COPY (SELECT * FROM cohort_raw ORDER BY hadm_id) "
        f"TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    logger.info("Wrote %s", out_path)

    stats = compute_stats(con)
    stats["funnel"] = funnel
    stats["config"] = cohort_cfg
    stats_path = out_dir / "cohort_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    logger.info("-" * 62)
    logger.info("  admissions            %12s", f"{stats['n_admissions']:,}")
    logger.info("  patients              %12s", f"{stats['n_patients']:,}")
    logger.info("  LOS median            %12.1f h", stats["los_hours_median"])
    logger.info("  LOS P95               %12.1f h", stats["los_hours_p95"])
    logger.info("  ICU overlap           %12.1f %%", stats["icu_overlap_rate"] * 100)
    logger.info("  in-hospital mortality %12.2f %%",
                stats["hospital_mortality_rate"] * 100)
    logger.info("  90-day mortality      %12.2f %%",
                stats["mortality_90d_rate"] * 100)
    logger.info("Wrote %s", stats_path)


if __name__ == "__main__":
    main()
