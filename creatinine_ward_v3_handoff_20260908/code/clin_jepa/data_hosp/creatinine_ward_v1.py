"""Build the fixed 3000-patient, ICU-free 24-hour ward experiment cohort.

The cohort is intentionally narrow: one admission per patient, no ICU stay at
any point in the admission, only complete 24-hour bins, at least one
whitelisted state laboratory event in every modeled bin, and at least one
serum creatinine measurement in the modeled interval. Patients are split
before sampling. Node-level NOT_MEASURED support is preserved by explicitly
sampling patients who have at least one 24-hour bin without creatinine; no
never-measured patients are imported from outside the source pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data_hosp import db

LOG = logging.getLogger(__name__)
SCHEMA_VERSION = "creatinine_ward_v1"
DEFAULT_SEED = 20260828
CREATININE_ITEMID = 50912

# Same numeric lab whitelist used by the fixed-24h hospital feature config.
STATE_LAB_ITEMIDS = (
    50802, 50804, 50813, 50816, 50817, 50818, 50820, 50821, 50825,
    50861, 50862, 50863, 50867, 50868, 50878, 50882, 50883, 50885,
    50889, 50893, 50902, 50910, 50911, 50912, 50927, 50930, 50931,
    50954, 50963, 50971, 50976, 50983, 51003, 51006, 51144, 51146,
    51196, 51200, 51214, 51221, 51222, 51237, 51244, 51248, 51249,
    51250, 51254, 51256, 51265, 51274, 51275, 51277, 51279, 51301,
)


def stable_u64(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_sql(min_age: int, min_bins: int, max_bins: int) -> str:
    itemids = ",".join(map(str, STATE_LAB_ITEMIDS))
    return f"""
    WITH admissions_all AS (
        SELECT
            a.*,
            p.gender,
            p.dod,
            p.anchor_age + (year(a.admittime) - p.anchor_year) AS admission_age,
            date_diff('second', a.admittime, a.dischtime) / 3600.0 AS los_hours_raw,
            row_number() OVER (
                PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
            ) AS hadm_seq,
            count(*) OVER (PARTITION BY a.subject_id) AS n_admissions_patient,
            lead(a.admittime) OVER (
                PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
            ) AS next_admittime
        FROM mimiciv_hosp.admissions a
        JOIN mimiciv_hosp.patients p USING (subject_id)
        WHERE a.dischtime IS NOT NULL AND a.dischtime > a.admittime
    ),
    base AS (
        SELECT *, least({max_bins}, floor(los_hours_raw / 24.0))::INTEGER
            AS n_complete_bins
        FROM admissions_all a
        WHERE admission_age BETWEEN {min_age} AND 119
          AND floor(los_hours_raw / 24.0) >= {min_bins}
          AND NOT EXISTS (
              SELECT 1 FROM mimiciv_icu.icustays i WHERE i.hadm_id = a.hadm_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM mimiciv_hosp.transfers t
              WHERE t.hadm_id = a.hadm_id
                AND regexp_matches(
                    upper(coalesce(t.careunit, '')), 'ICU|INTENSIVE CARE'
                )
          )
    ),
    supported_labs AS (
        SELECT
            b.hadm_id,
            floor(
                date_diff('second', b.admittime, l.charttime) / 3600.0 / 24.0
            )::INTEGER AS step_idx,
            l.itemid,
            l.valuenum,
            l.charttime
        FROM base b
        JOIN mimiciv_hosp.labevents l USING (hadm_id)
        WHERE l.itemid IN ({itemids})
          AND l.valuenum IS NOT NULL
          AND isfinite(l.valuenum)
          AND l.charttime >= b.admittime
          AND l.charttime < b.admittime
              + b.n_complete_bins * INTERVAL 24 HOUR
    ),
    lab_stats AS (
        SELECT
            hadm_id,
            count(*) AS n_lab_events,
            count(DISTINCT step_idx) AS n_filled_bins,
            count(*) FILTER (WHERE itemid = {CREATININE_ITEMID}) AS n_creatinine,
            count(DISTINCT step_idx) FILTER (
                WHERE itemid = {CREATININE_ITEMID}
            ) AS n_creatinine_bins,
            min(valuenum) FILTER (WHERE itemid = {CREATININE_ITEMID}) AS creat_min,
            max(valuenum) FILTER (WHERE itemid = {CREATININE_ITEMID}) AS creat_max
        FROM supported_labs
        GROUP BY hadm_id
    ),
    dx AS (
        SELECT hadm_id, count(*) AS n_diagnoses
        FROM mimiciv_hosp.diagnoses_icd GROUP BY hadm_id
    )
    SELECT
        b.subject_id, b.hadm_id, b.admittime, b.dischtime, b.deathtime,
        b.gender, b.admission_age, b.race, b.admission_type,
        b.admission_location, b.discharge_location, b.insurance,
        b.los_hours_raw, b.n_complete_bins,
        b.n_complete_bins * 24.0 AS los_hours,
        b.hospital_expire_flag,
        coalesce(dx.n_diagnoses, 0) AS n_diagnoses,
        s.n_lab_events,
        0::INTEGER AS n_icu_stays,
        0.0::DOUBLE AS icu_hours,
        FALSE AS has_icu,
        b.hadm_seq,
        b.n_admissions_patient,
        FALSE AS suspected_infection,
        coalesce(
            date_diff('day', b.dischtime, b.next_admittime) BETWEEN 0 AND 30,
            FALSE
        ) AS readmission_30d,
        CASE WHEN b.dod IS NOT NULL
                  AND date_diff('day', b.dischtime::DATE, b.dod) <= 30
             THEN TRUE
             WHEN b.hospital_expire_flag = 1 THEN TRUE ELSE FALSE END
            AS mortality_30d,
        CASE WHEN b.dod IS NOT NULL
                  AND date_diff('day', b.dischtime::DATE, b.dod) <= 90
             THEN TRUE
             WHEN b.hospital_expire_flag = 1 THEN TRUE ELSE FALSE END
            AS mortality_90d,
        b.los_hours_raw > 168 AS prolonged_stay,
        s.n_creatinine, s.n_creatinine_bins, s.creat_min, s.creat_max,
        TRUE AS ever_creatinine,
        (s.n_creatinine_bins < b.n_complete_bins) AS has_missing_creatinine_bin
    FROM base b
    JOIN lab_stats s USING (hadm_id)
    LEFT JOIN dx USING (hadm_id)
    WHERE s.n_filled_bins = b.n_complete_bins
      AND s.n_creatinine > 0
    """


def select_cohort(
    candidates: pd.DataFrame,
    seed: int,
    split_sizes: dict[str, int],
    high_fraction: float = 0.33,
    any_missing_bin_fraction: float = 0.22,
) -> pd.DataFrame:
    """Choose one admission per patient, then sample crossed fixed quotas.

    Every candidate has at least one creatinine measurement in the modeled
    complete-bin interval. Sampling crosses two patient-level properties:
    ever HIGH (>1.5 mg/dL) versus never HIGH, and whether at least one 24-hour
    node is NOT_MEASURED. Within each crossed cell, higher measurement counts
    are preferred and ties are resolved by stable hashing.
    """
    frame = candidates.copy()
    if not bool(frame.ever_creatinine.all()):
        raise ValueError("candidate pool contains never-measured admissions")
    frame["_admission_key"] = [
        stable_u64(seed, int(subject), int(hadm), "admission")
        for subject, hadm in zip(frame.subject_id, frame.hadm_id)
    ]
    # Prefer the most frequently measured eligible admission for a patient;
    # deterministic hashing resolves ties.
    frame = (
        frame.sort_values(
            ["subject_id", "n_creatinine", "_admission_key"],
            ascending=[True, False, True],
        )
        .drop_duplicates("subject_id")
        .reset_index(drop=True)
    )
    split_hash = np.asarray([
        stable_u64(seed, int(subject), "split") % 100
        for subject in frame.subject_id
    ])
    frame["split"] = np.where(
        split_hash < 70, "train", np.where(split_hash < 85, "val", "test")
    )

    picked: list[pd.DataFrame] = []
    for split, total in split_sizes.items():
        pool = frame[frame.split == split].copy()
        n_high = int(round(total * high_fraction))
        n_missing = int(round(total * any_missing_bin_fraction))
        n_high_missing = int(round(n_high * any_missing_bin_fraction))
        quotas = {
            "high_with_missing_bin": n_high_missing,
            "high_full_coverage": n_high - n_high_missing,
            "not_high_with_missing_bin": n_missing - n_high_missing,
        }
        quotas["not_high_full_coverage"] = total - sum(quotas.values())

        ever_high = pool.creat_max.astype(float) > 1.5
        has_missing = pool.has_missing_creatinine_bin.astype(bool)
        masks = {
            "high_with_missing_bin": ever_high & has_missing,
            "high_full_coverage": ever_high & ~has_missing,
            "not_high_with_missing_bin": ~ever_high & has_missing,
            "not_high_full_coverage": ~ever_high & ~has_missing,
        }
        for stratum, quota in quotas.items():
            cell = pool[masks[stratum]].copy()
            if len(cell) < quota:
                raise RuntimeError(
                    f"insufficient {split}/{stratum}: "
                    f"need={quota}, available={len(cell)}"
                )
            cell["_sample_key"] = [
                stable_u64(seed, int(subject), stratum)
                for subject in cell.subject_id
            ]
            cell = cell.sort_values(
                ["n_creatinine", "_sample_key"], ascending=[False, True]
            ).head(quota)
            cell["creatinine_sampling_stratum"] = stratum
            picked.append(cell)

    result = pd.concat(picked, ignore_index=True)
    result["schema_version"] = SCHEMA_VERSION
    result["selection_seed"] = seed
    result = result.drop(columns=[c for c in result if c.startswith("_")])
    expected = sum(split_sizes.values())
    if result.subject_id.nunique() != expected or result.hadm_id.nunique() != expected:
        raise AssertionError("cohort is not one unique admission per patient")
    return result.sort_values(["split", "subject_id"]).reset_index(drop=True)


def build(paths_config: str, output_root: Path, config: dict) -> None:
    paths = db.load_yaml(paths_config)
    con = db.connect(paths)
    sql = _candidate_sql(
        int(config.get("min_age", 18)),
        int(config.get("min_complete_bins", 4)),
        int(config.get("max_complete_bins", 30)),
    )
    candidates = con.execute(sql).df()
    cohort = select_cohort(
        candidates,
        int(config.get("seed", DEFAULT_SEED)),
        config.get("split_sizes", {"train": 2100, "val": 450, "test": 450}),
        float(config.get("high_fraction", 0.33)),
        float(config.get("any_missing_bin_fraction", 0.22)),
    )

    cohort_dir = output_root / "cohort"
    split_dir = output_root / "splits"
    manifest_dir = output_root / "manifests"
    for directory in (cohort_dir, split_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cohort_path = cohort_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    cohort.to_parquet(manifest_dir / "cohort_3000.parquet", index=False)
    cohort.to_csv(manifest_dir / "cohort_3000.csv", index=False)
    cohort[["subject_id", "hadm_id", "split"]].to_csv(
        split_dir / "splits.csv", index=False
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COHORT_OK",
        "n_candidates_admissions": int(len(candidates)),
        "n_candidates_patients": int(candidates.subject_id.nunique()),
        "n_patients": int(cohort.subject_id.nunique()),
        "n_admissions": int(cohort.hadm_id.nunique()),
        "split_counts": cohort.split.value_counts().sort_index().to_dict(),
        "sampling_strata": (
            cohort.groupby(["split", "creatinine_sampling_stratum"])
            .size().rename("n").reset_index().to_dict("records")
        ),
        "creatinine_count": cohort.n_creatinine.describe().to_dict(),
        "complete_bins": cohort.n_complete_bins.describe().to_dict(),
        "icu_admissions": int(cohort.has_icu.sum()),
        "cohort_sha256": file_sha256(cohort_path),
        "selection_config": config,
    }
    (manifest_dir / "cohort_3000_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    LOG.info("CREATININE_WARD_COHORT_OK %s", cohort_path)


def verify(output_root: Path) -> None:
    cohort_path = output_root / "cohort" / "cohort.parquet"
    cohort = pd.read_parquet(cohort_path)
    expected = {"train": 2100, "val": 450, "test": 450}
    assert cohort.split.value_counts().to_dict() == expected
    assert cohort.subject_id.nunique() == 3000
    assert cohort.hadm_id.nunique() == 3000
    assert not cohort.has_icu.any()
    assert (cohort.n_icu_stays == 0).all()
    assert (cohort.icu_hours == 0).all()
    assert cohort.ever_creatinine.all()
    assert (cohort.n_creatinine > 0).all()
    assert (cohort.n_complete_bins >= 4).all()
    expected_missing = {"train": 462, "val": 99, "test": 99}
    observed_missing = (
        cohort.groupby("split").has_missing_creatinine_bin.sum().astype(int).to_dict()
    )
    assert observed_missing == expected_missing, (observed_missing, expected_missing)
    expected_high = {"train": 693, "val": 148, "test": 148}
    observed_high = (
        (cohort.creat_max.astype(float) > 1.5)
        .groupby(cohort.split).sum().astype(int).to_dict()
    )
    assert observed_high == expected_high, (observed_high, expected_high)
    sets = {
        split: set(cohort.loc[cohort.split == split, "subject_id"])
        for split in expected
    }
    assert not sets["train"].intersection(sets["val"])
    assert not sets["train"].intersection(sets["test"])
    assert not sets["val"].intersection(sets["test"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFY_OK",
        "n_patients": 3000,
        "icu_admissions": 0,
        "cohort_sha256": file_sha256(cohort_path),
    }
    (output_root / "verify_creatinine_ward_v1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    LOG.info("CREATININE_WARD_VERIFY_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cohort", "verify", "all"])
    parser.add_argument(
        "--paths-config", default="creatinine_ward_paths_v1.yaml"
    )
    parser.add_argument(
        "--output-root", default=os.environ.get("CREATININE_WARD_DATA_ROOT")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--high-fraction", type=float, default=0.33)
    parser.add_argument("--any-missing-bin-fraction", type=float, default=0.22)
    parser.add_argument("--min-complete-bins", type=int, default=4)
    parser.add_argument("--max-complete-bins", type=int, default=30)
    args = parser.parse_args()
    if not args.output_root:
        parser.error("set --output-root or CREATININE_WARD_DATA_ROOT")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": args.seed,
        "high_fraction": args.high_fraction,
        "any_missing_bin_fraction": args.any_missing_bin_fraction,
        "min_age": 18,
        "min_complete_bins": args.min_complete_bins,
        "max_complete_bins": args.max_complete_bins,
        "split_sizes": {"train": 2100, "val": 450, "test": 450},
    }
    if args.command in {"cohort", "all"}:
        build(args.paths_config, root, config)
    if args.command in {"verify", "all"}:
        verify(root)


if __name__ == "__main__":
    main()
