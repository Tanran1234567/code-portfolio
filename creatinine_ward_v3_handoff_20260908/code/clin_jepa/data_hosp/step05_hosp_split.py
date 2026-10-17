"""Step 05: patient-level train/val/test split.

Same 70/15/15 patient-level design as :mod:`clin_jepa.data.step05_split`, and
patient-level for the same reason: a hospital cohort keeps every admission per
patient (2.25 on average), so splitting on ``hadm_id`` would put different
admissions of one patient on both sides of the split and leak that patient's
baseline physiology.

Stratification differs. The ICU pipeline stratifies on mortality x sepsis3;
sepsis3 is not materialised here and is in any case an ICU concept, so the
strata are 90-day mortality x ICU exposure x a prolonged-stay flag, which are
the three stay-level outcomes the shards carry.

Output:
  {output.splits}/splits.csv
  {output.splits}/split_stats.json

Usage:
    python -m clin_jepa.data_hosp.step05_hosp_split --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from clin_jepa.data_hosp import db

logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def _stable_uniform(subject_id: int, seed: int) -> float:
    """Deterministic uniform draw per patient.

    Hashing rather than a shuffled permutation so that the split is stable
    when the cohort grows: adding admissions cannot move an existing patient
    between splits, which keeps evaluation comparable across pipeline reruns.
    """
    h = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def inherit_icu_split(assigned: pd.DataFrame, icu_split_csv: Path) -> pd.DataFrame:
    """Keep original Clin-JEPA ICU subject assignments; split only new subjects."""
    icu = pd.read_csv(icu_split_csv)
    if "subject_id" not in icu.columns or "split" not in icu.columns:
        raise ValueError(f"{icu_split_csv} must have subject_id and split columns")
    icu = icu[["subject_id", "split"]].drop_duplicates("subject_id")
    icu_map = dict(zip(icu["subject_id"].astype(int), icu["split"].astype(str)))
    out = assigned.copy()
    inherited = 0
    for i, row in out.iterrows():
        sid = int(row["subject_id"])
        if sid in icu_map:
            out.at[i, "split"] = icu_map[sid]
            inherited += 1
    logger.info(
        "Inherited ICU split for %d / %d patients from %s",
        inherited, len(out), icu_split_csv,
    )
    return out


def assign_splits(patients: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign each patient to train/val/test within its stratum."""
    patients = patients.copy()
    patients["u"] = [
        _stable_uniform(int(s), seed) for s in patients["subject_id"]
    ]

    out = []
    for stratum, grp in patients.groupby("stratum", sort=True):
        grp = grp.sort_values("u", kind="stable").reset_index(drop=True)
        n = len(grp)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        labels = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )
        grp["split"] = labels
        out.append(grp)
        logger.info(
            "  stratum %-14s n=%7d  train=%6d val=%5d test=%5d",
            stratum, n, n_train, n_val, n - n_train - n_val,
        )

    return pd.concat(out, ignore_index=True)[["subject_id", "split", "stratum"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 05: patient-level split")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--icu_split_csv",
        default=None,
        help="Optional Clin-JEPA ICU subject split CSV (subject_id,split) to inherit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    logger.info("=" * 62)
    logger.info("Step 05: patient-level train/val/test split (seed=%d)", args.seed)
    logger.info("=" * 62)

    paths = db.load_yaml(args.paths_config)
    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"
    out_dir = Path(paths["output"]["splits"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(
        cohort_path,
        columns=[
            "subject_id", "hadm_id", "mortality_90d", "has_icu",
            "prolonged_stay", "hospital_expire_flag",
        ],
    )
    logger.info("Cohort: %d admissions / %d patients",
                len(cohort), cohort["subject_id"].nunique())

    # A patient is positive for a stratum axis if any of their admissions is.
    patients = cohort.groupby("subject_id", as_index=False).agg(
        mortality_90d=("mortality_90d", "max"),
        has_icu=("has_icu", "max"),
        prolonged_stay=("prolonged_stay", "max"),
    )
    patients["stratum"] = (
        patients["mortality_90d"].astype(int).astype(str)
        + "_" + patients["has_icu"].astype(int).astype(str)
        + "_" + patients["prolonged_stay"].astype(int).astype(str)
    )

    logger.info("Assigning splits over %d strata:", patients["stratum"].nunique())
    assigned = assign_splits(patients, args.seed)
    if args.icu_split_csv:
        assigned = inherit_icu_split(assigned, Path(args.icu_split_csv))

    splits = cohort.merge(assigned, on="subject_id", how="left")
    splits = splits[["hadm_id", "subject_id", "split"]].sort_values("hadm_id")

    out_path = out_dir / "splits.csv"
    splits.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    stats = {
        "seed": args.seed,
        "n_admissions": int(len(splits)),
        "n_patients": int(splits["subject_id"].nunique()),
        "by_split": {},
    }
    logger.info("-" * 62)
    for split in ("train", "val", "test"):
        sub = splits[splits["split"] == split]
        sub_cohort = cohort[cohort["hadm_id"].isin(sub["hadm_id"])]
        entry = {
            "n_admissions": int(len(sub)),
            "n_patients": int(sub["subject_id"].nunique()),
            "admission_frac": round(len(sub) / len(splits), 4),
            "mortality_90d_rate": round(float(sub_cohort["mortality_90d"].mean()), 4),
            "hospital_mortality_rate": round(
                float(sub_cohort["hospital_expire_flag"].mean()), 4,
            ),
            "icu_overlap_rate": round(float(sub_cohort["has_icu"].mean()), 4),
        }
        stats["by_split"][split] = entry
        logger.info(
            "  %-6s adm=%7d (%.1f%%)  pat=%7d  mort90=%.2f%%  icu=%.1f%%",
            split, entry["n_admissions"], entry["admission_frac"] * 100,
            entry["n_patients"], entry["mortality_90d_rate"] * 100,
            entry["icu_overlap_rate"] * 100,
        )

    # A patient must never appear in two splits; assert it rather than trust it.
    overlap = (
        splits.groupby("subject_id")["split"].nunique().max()
    )
    if overlap > 1:
        raise RuntimeError(f"Patient leakage: {overlap} splits for one subject_id")
    logger.info("Leakage check passed: every patient is in exactly one split")

    (out_dir / "split_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
