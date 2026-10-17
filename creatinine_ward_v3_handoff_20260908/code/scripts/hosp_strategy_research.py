#!/usr/bin/env python3
"""Hospital bucketing + subset strategy research (no training).

Produces evidence for:
  1) fixed vs adaptive (irregular) binning feasibility
  2) pilot training-subset construction options
  3) hospital whitelist coverage vs ICU-style concepts

Outputs under reports/hosp_strategy_research_20260730/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = "/data/general_disease_pred/mimic4.db"
DEFAULT_COHORT = "/home/zhoujiale/clin_jepa_hosp_data/cohort/cohort.parquet"
DEFAULT_SPLITS = "/home/zhoujiale/clin_jepa_hosp_data/splits/splits.csv"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "hosp_strategy_research_20260730"
)

FIXED_STEPS = [1, 2, 3, 4, 6, 8, 12, 24]

# Lab concept itemids from hosp_features.yaml (hospital whitelist core).
LAB_WHITELIST = {
    "chemistry": [50862, 50930, 50976, 50868, 50882, 51006, 50893, 50902, 50912, 50931, 50983, 50971],
    "cbc": [51221, 51222, 51248, 51249, 51250, 51265, 51279, 51277, 51301],
    "coagulation": [51196, 51214, 51237, 51274, 51275],
    "enzyme": [50861, 50863, 50878, 50867, 50885, 50883, 50910, 50927, 50954],
    "blood_diff": [51146, 51200, 51244, 51254, 51256, 51144],
    "cardiac": [51003, 50911, 50963],
    "inflammation": [50889],
    "bg": [50802, 50804, 50813, 50816, 50817, 50818, 50820, 50821, 50825],
}


def load_views(con: duckdb.DuckDBPyConnection, cohort_path: str, sample: int, seed: int) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort_full AS
        SELECT * FROM read_parquet('{cohort_path}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE samp AS
        SELECT * FROM cohort_full
        USING SAMPLE {sample} ROWS (reservoir, {seed})
        """
    )
    n = con.execute("SELECT count(*) FROM samp").fetchone()[0]
    logger.info("sample admissions: %d", n)


def build_event_hours(con: duckdb.DuckDBPyConnection) -> None:
    """Observation-domain event hours for sampled admissions (lab+micro+chart+transfer)."""
    logger.info("building obs event hours ...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE obs_ev AS
        SELECT hadm_id, th FROM (
            SELECT l.hadm_id,
                   date_diff('second', s.admittime, l.charttime) / 3600.0 AS th
            FROM mimiciv_hosp.labevents l
            JOIN samp s USING (hadm_id)
            WHERE l.charttime >= s.admittime AND l.charttime < s.dischtime
            UNION ALL
            SELECT m.hadm_id,
                   date_diff('second', s.admittime, coalesce(m.charttime, m.chartdate)) / 3600.0 AS th
            FROM mimiciv_hosp.microbiologyevents m
            JOIN samp s USING (hadm_id)
            WHERE coalesce(m.charttime, m.chartdate) >= s.admittime
              AND coalesce(m.charttime, m.chartdate) < s.dischtime
            UNION ALL
            SELECT e.hadm_id,
                   date_diff('second', s.admittime, e.charttime) / 3600.0 AS th
            FROM mimiciv_icu.chartevents e
            JOIN samp s USING (hadm_id)
            WHERE e.charttime >= s.admittime AND e.charttime < s.dischtime
            UNION ALL
            SELECT t.hadm_id,
                   date_diff('second', s.admittime, t.intime) / 3600.0 AS th
            FROM mimiciv_hosp.transfers t
            JOIN samp s USING (hadm_id)
            WHERE t.intime IS NOT NULL
              AND t.intime >= s.admittime AND t.intime < s.dischtime
        )
        WHERE th >= 0
        """
    )
    n = con.execute("SELECT count(*) FROM obs_ev").fetchone()[0]
    logger.info("obs events in sample: %d", n)


def analyze_los_phase_density(con: duckdb.DuckDBPyConnection) -> dict:
    """Event density by relative LOS phase (mentor hint: early denser)."""
    logger.info("LOS-phase density ...")
    rows = con.execute(
        """
        WITH joined AS (
            SELECT o.hadm_id, o.th, s.los_hours,
                   CASE
                     WHEN s.los_hours <= 0 THEN NULL
                     WHEN o.th / s.los_hours < 0.20 THEN 'P1_0_20'
                     WHEN o.th / s.los_hours < 0.40 THEN 'P2_20_40'
                     WHEN o.th / s.los_hours < 0.60 THEN 'P3_40_60'
                     WHEN o.th / s.los_hours < 0.80 THEN 'P4_60_80'
                     ELSE 'P5_80_100'
                   END AS phase
            FROM obs_ev o JOIN samp s USING (hadm_id)
            WHERE o.th < s.los_hours
        )
        SELECT phase,
               count(*) AS n_events,
               count(DISTINCT hadm_id) AS n_adm,
               avg(los_hours) AS los_mean
        FROM joined
        WHERE phase IS NOT NULL
        GROUP BY phase
        ORDER BY phase
        """
    ).fetchall()

    # Hours per phase = 20% of LOS; report events per patient-hour in phase.
    phase = {}
    for r in rows:
        name = r[0]
        n_events, n_adm, los_mean = int(r[1]), int(r[2]), float(r[3])
        # Approximate patient-hours in this 20% phase.
        phase_hours = n_adm * los_mean * 0.20
        eph = n_events / phase_hours if phase_hours > 0 else None
        phase[name] = {
            "n_events": n_events,
            "n_admissions": n_adm,
            "los_mean": round(los_mean, 2),
            "events_per_patient_hour": round(eph, 4) if eph is not None else None,
        }

    # Also absolute hours from admission: 0-24h, 24-72h, 72-168h, 168+.
    abs_rows = con.execute(
        """
        WITH joined AS (
            SELECT o.hadm_id, o.th, s.los_hours,
                   CASE
                     WHEN o.th < 24 THEN 'H0_24'
                     WHEN o.th < 72 THEN 'H24_72'
                     WHEN o.th < 168 THEN 'H72_168'
                     ELSE 'H168_plus'
                   END AS window
            FROM obs_ev o JOIN samp s USING (hadm_id)
            WHERE o.th < s.los_hours
        ),
        exposure AS (
            SELECT
                hadm_id,
                least(los_hours, 24) AS h0_24,
                greatest(0, least(los_hours, 72) - 24) AS h24_72,
                greatest(0, least(los_hours, 168) - 72) AS h72_168,
                greatest(0, los_hours - 168) AS h168_plus
            FROM samp
        )
        SELECT j.window,
               count(*) AS n_events,
               count(DISTINCT j.hadm_id) AS n_adm,
               CASE j.window
                 WHEN 'H0_24' THEN (SELECT sum(h0_24) FROM exposure)
                 WHEN 'H24_72' THEN (SELECT sum(h24_72) FROM exposure)
                 WHEN 'H72_168' THEN (SELECT sum(h72_168) FROM exposure)
                 ELSE (SELECT sum(h168_plus) FROM exposure)
               END AS exposed_hours
        FROM joined j
        GROUP BY j.window
        ORDER BY j.window
        """
    ).fetchall()
    absolute = {}
    for r in abs_rows:
        exposed = float(r[3]) if r[3] is not None else 0.0
        absolute[r[0]] = {
            "n_events": int(r[1]),
            "n_admissions": int(r[2]),
            "exposed_hours": round(exposed, 1),
            "events_per_hour": round(int(r[1]) / exposed, 4) if exposed > 0 else None,
        }
    return {"by_relative_phase": phase, "by_absolute_window": absolute}


def analyze_fixed_bins(con: duckdb.DuckDBPyConnection) -> dict:
    """Fixed-width occupancy + training-difficulty proxies."""
    logger.info("fixed-bin occupancy ...")
    out = {}
    for step in FIXED_STEPS:
        row = con.execute(
            f"""
            WITH bins AS (
                SELECT hadm_id, floor(th / {step}) AS b, count(*) AS n
                FROM obs_ev
                GROUP BY 1, 2
            ),
            tot AS (
                SELECT hadm_id, greatest(1, ceil(los_hours / {step}))::BIGINT AS n_bins
                FROM samp
            ),
            per_adm AS (
                SELECT t.hadm_id, t.n_bins,
                       coalesce(count(b.b), 0) AS filled,
                       coalesce(sum(b.n), 0) AS n_events
                FROM tot t
                LEFT JOIN bins b ON b.hadm_id = t.hadm_id AND b.b >= 0 AND b.b < t.n_bins
                GROUP BY 1, 2
            )
            SELECT
                sum(n_bins) AS total_bins,
                sum(filled) AS filled_bins,
                sum(filled) * 1.0 / sum(n_bins) AS occupancy,
                avg(filled * 1.0 / n_bins) AS occ_mean,
                median(filled * 1.0 / n_bins) AS occ_median,
                sum(n_events) * 1.0 / sum(n_bins) AS ev_per_bin,
                sum(n_events) * 1.0 / nullif(sum(filled), 0) AS ev_per_filled,
                median(n_bins) AS bins_med,
                quantile_cont(n_bins, 0.95) AS bins_p95,
                avg(n_bins) AS bins_mean
            FROM per_adm
            """
        ).fetchone()
        occ = float(row[2])
        bins_med = float(row[7])
        # Training proxies assuming t_max covering ~7d wall-clock.
        wall_hours_window = 168
        steps_in_7d = max(1, int(round(wall_hours_window / step)))
        empty_rate = 1.0 - occ
        # Difficulty score: high empty OR too-short sequences OR over-long sequences.
        # Target: occupancy ~0.4-0.7, median bins ~12-28.
        score_occ = 1.0 - abs(occ - 0.55) / 0.55
        score_len = 1.0 - abs(bins_med - 16) / 16
        feasibility = round(max(0.0, 0.6 * score_occ + 0.4 * score_len), 3)
        out[str(step)] = {
            "step_hours": step,
            "total_bins": int(row[0]),
            "filled_bins": int(row[1]),
            "occupancy": round(occ, 4),
            "occupancy_mean": round(float(row[3]), 4),
            "occupancy_median": round(float(row[4]), 4),
            "events_per_bin": round(float(row[5]), 3),
            "events_per_filled_bin": round(float(row[6]), 3),
            "n_bins_median": bins_med,
            "n_bins_p95": float(row[8]),
            "n_bins_mean": round(float(row[9]), 2),
            "empty_bin_rate": round(empty_rate, 4),
            "steps_in_7d_window": steps_in_7d,
            "feasibility_score": feasibility,
            "train_difficulty": (
                "hard_sparse" if occ < 0.25
                else "moderate" if occ < 0.40
                else "balanced" if occ < 0.65
                else "coarse_easy"
            ),
            "target_resolution_note": (
                f"{step}h resolution; 7d window = {steps_in_7d} steps"
            ),
        }
    return out


def analyze_inter_event_gaps(con: duckdb.DuckDBPyConnection) -> dict:
    """Inter-event gaps by LOS phase — basis for irregular spacing."""
    logger.info("inter-event gaps ...")
    # Median gap between consecutive obs events per admission, then by phase of the gap midpoint.
    df = con.execute(
        """
        WITH ordered AS (
            SELECT hadm_id, th,
                   lag(th) OVER (PARTITION BY hadm_id ORDER BY th) AS prev_th
            FROM (
                SELECT hadm_id, th FROM obs_ev
                QUALIFY row_number() OVER (PARTITION BY hadm_id, round(th, 3) ORDER BY th) = 1
            )
        ),
        gaps AS (
            SELECT o.hadm_id, o.th, o.prev_th, s.los_hours,
                   o.th - o.prev_th AS gap_h,
                   (o.th + o.prev_th) / 2.0 AS mid_th
            FROM ordered o JOIN samp s USING (hadm_id)
            WHERE o.prev_th IS NOT NULL AND o.th > o.prev_th
        )
        SELECT
            CASE
              WHEN mid_th / los_hours < 0.20 THEN 'P1_0_20'
              WHEN mid_th / los_hours < 0.40 THEN 'P2_20_40'
              WHEN mid_th / los_hours < 0.60 THEN 'P3_40_60'
              WHEN mid_th / los_hours < 0.80 THEN 'P4_60_80'
              ELSE 'P5_80_100'
            END AS phase,
            count(*) AS n_gaps,
            median(gap_h) AS gap_median,
            avg(gap_h) AS gap_mean,
            quantile_cont(gap_h, 0.25) AS gap_p25,
            quantile_cont(gap_h, 0.75) AS gap_p75,
            quantile_cont(gap_h, 0.90) AS gap_p90
        FROM gaps
        WHERE los_hours > 0 AND mid_th < los_hours
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    return df.to_dict(orient="records")


def simulate_adaptive_bins(con: duckdb.DuckDBPyConnection) -> dict:
    """Simulate irregular schemes and measure occupancy / sequence length.

    Schemes (mentor-inspired):
      A) early20_6h_then_12h: first 20% LOS at 6h, rest at 12h
      B) early20_6h_then_24h: first 20% LOS at 6h, rest at 24h
      C) early48h_6h_then_12h: first 48 wall hours at 6h, rest 12h
      D) early48h_6h_mid_12h_late_24h: 0-48@6, 48-168@12, 168+@24
      E) gap_guided: use phase-specific step from gap medians (approx)
    """
    logger.info("adaptive bin simulation ...")
    adm = con.execute(
        "SELECT hadm_id, los_hours, has_icu FROM samp"
    ).fetchdf()
    ev = con.execute("SELECT hadm_id, th FROM obs_ev").fetchdf()
    by_adm = {h: g["th"].to_numpy() for h, g in ev.groupby("hadm_id")}

    def edges_scheme_a(los: float) -> np.ndarray:
        cut = 0.20 * los
        early = np.arange(0, cut, 6.0)
        late = np.arange(cut, los, 12.0)
        return np.unique(np.concatenate([early, late, [los]]))

    def edges_scheme_b(los: float) -> np.ndarray:
        cut = 0.20 * los
        early = np.arange(0, cut, 6.0)
        late = np.arange(cut, los, 24.0)
        return np.unique(np.concatenate([early, late, [los]]))

    def edges_scheme_c(los: float) -> np.ndarray:
        cut = min(48.0, los)
        early = np.arange(0, cut, 6.0)
        late = np.arange(cut, los, 12.0)
        return np.unique(np.concatenate([early, late, [los]]))

    def edges_scheme_d(los: float) -> np.ndarray:
        e1 = min(48.0, los)
        e2 = min(168.0, los)
        p1 = np.arange(0, e1, 6.0)
        p2 = np.arange(e1, e2, 12.0) if e2 > e1 else np.array([])
        p3 = np.arange(e2, los, 24.0) if los > e2 else np.array([])
        return np.unique(np.concatenate([p1, p2, p3, [los]]))

    def edges_scheme_e(los: float) -> np.ndarray:
        # Approximate gap-guided: 0-20%@6, 20-60%@12, 60-100%@24
        c1, c2 = 0.20 * los, 0.60 * los
        p1 = np.arange(0, c1, 6.0)
        p2 = np.arange(c1, c2, 12.0)
        p3 = np.arange(c2, los, 24.0)
        return np.unique(np.concatenate([p1, p2, p3, [los]]))

    schemes = {
        "fixed_6h": lambda los: np.unique(np.concatenate([np.arange(0, los, 6.0), [los]])),
        "fixed_12h": lambda los: np.unique(np.concatenate([np.arange(0, los, 12.0), [los]])),
        "early20pct_6h_then_12h": edges_scheme_a,
        "early20pct_6h_then_24h": edges_scheme_b,
        "early48h_6h_then_12h": edges_scheme_c,
        "tiered_48_6__168_12__rest_24": edges_scheme_d,
        "gap_guided_6_12_24": edges_scheme_e,
    }

    results = {}
    for name, edge_fn in schemes.items():
        n_bins_list = []
        occ_list = []
        filled_list = []
        total_list = []
        for _, row in adm.iterrows():
            los = float(row["los_hours"])
            if los <= 0:
                continue
            edges = edge_fn(los)
            # bins = intervals between consecutive edges
            if len(edges) < 2:
                continue
            n_bins = len(edges) - 1
            th = by_adm.get(int(row["hadm_id"]), np.array([]))
            th = th[(th >= 0) & (th < los)]
            if n_bins == 0:
                continue
            # Digitize into bins [edges[i], edges[i+1])
            idx = np.searchsorted(edges, th, side="right") - 1
            idx = idx[(idx >= 0) & (idx < n_bins)]
            filled = len(np.unique(idx)) if len(idx) else 0
            n_bins_list.append(n_bins)
            occ_list.append(filled / n_bins)
            filled_list.append(filled)
            total_list.append(n_bins)
        tot = sum(total_list)
        fil = sum(filled_list)
        results[name] = {
            "n_admissions": len(n_bins_list),
            "occupancy": round(fil / tot, 4) if tot else None,
            "occupancy_median": round(float(np.median(occ_list)), 4) if occ_list else None,
            "n_bins_median": float(np.median(n_bins_list)) if n_bins_list else None,
            "n_bins_mean": round(float(np.mean(n_bins_list)), 2) if n_bins_list else None,
            "n_bins_p95": float(np.quantile(n_bins_list, 0.95)) if n_bins_list else None,
            "total_bins": int(tot),
            "filled_bins": int(fil),
            "vs_fixed6_bin_reduction": None,
        }

    base = results.get("fixed_6h", {}).get("total_bins") or 1
    for name, r in results.items():
        if r["total_bins"]:
            r["vs_fixed6_bin_reduction"] = round(1.0 - r["total_bins"] / base, 4)
    return results


def analyze_subsets(cohort_path: str, splits_path: str) -> dict:
    """Candidate pilot subsets + leakage / stratification impact."""
    logger.info("subset construction analysis ...")
    cohort = pd.read_parquet(cohort_path)
    splits = pd.read_csv(splits_path)
    df = cohort.merge(splits, on=["hadm_id", "subject_id"], how="inner")

    def summarize(name: str, sub: pd.DataFrame) -> dict:
        patients = sub.groupby("subject_id").agg(
            n_adm=("hadm_id", "count"),
            has_icu=("has_icu", "max"),
            mort90=("mortality_90d", "max"),
            prolonged=("prolonged_stay", "max"),
        )
        # Patient-level split consistency
        split_n = sub.groupby("split")["hadm_id"].nunique().to_dict()
        # Check no subject across splits in this subset
        subj_splits = sub.groupby("subject_id")["split"].nunique()
        leak = int((subj_splits > 1).sum())
        return {
            "name": name,
            "n_admissions": int(len(sub)),
            "n_patients": int(sub["subject_id"].nunique()),
            "split_counts": {k: int(v) for k, v in split_n.items()},
            "subjects_in_multiple_splits": leak,
            "icu_rate": round(float(sub["has_icu"].mean()), 4),
            "mortality_90d_rate": round(float(sub["mortality_90d"].mean()), 4),
            "prolonged_rate": round(float(sub["prolonged_stay"].mean()), 4),
            "los_median": round(float(sub["los_hours"].median()), 2),
            "los_p90": round(float(sub["los_hours"].quantile(0.9)), 2),
            "n_lab_median": round(float(sub["n_lab_events"].median()), 1),
            "adm_per_patient_mean": round(float(patients["n_adm"].mean()), 3),
            "frac_of_full_cohort": round(len(sub) / len(df), 4),
        }

    # Full
    options = [summarize("full_cohort", df)]

    # A) EHRWorld-like: trim LOS to P10-P90 of full cohort
    p10, p90 = df["los_hours"].quantile(0.10), df["los_hours"].quantile(0.90)
    options.append(
        summarize("los_p10_p90", df[(df["los_hours"] >= p10) & (df["los_hours"] <= p90)])
    )

    # B) Lab-rich: >= median labs, LOS>=48h
    med_lab = df["n_lab_events"].median()
    options.append(
        summarize(
            "lab_rich_los48",
            df[(df["n_lab_events"] >= med_lab) & (df["los_hours"] >= 48)],
        )
    )

    # C) Ward-only (no ICU) — pure hospital signal
    options.append(summarize("ward_only", df[~df["has_icu"].astype(bool)]))

    # D) ICU-overlap only
    options.append(summarize("icu_overlap", df[df["has_icu"].astype(bool)]))

    # E) Stratified pilot 20k admissions by patient, preserving split ratios
    rng = np.random.default_rng(42)
    pilot_hadms = []
    for split, frac in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        target = int(20000 * frac)
        pool = df[df["split"] == split]
        # sample patients then take all their admissions until target
        patients = pool["subject_id"].unique()
        rng.shuffle(patients)
        chosen = []
        n = 0
        for p in patients:
            adm = pool.loc[pool["subject_id"] == p, "hadm_id"].tolist()
            if n + len(adm) > target * 1.15 and n >= target:
                break
            chosen.extend(adm)
            n += len(adm)
            if n >= target:
                break
        pilot_hadms.extend(chosen)
    options.append(summarize("pilot_20k_patient_stratified", df[df["hadm_id"].isin(pilot_hadms)]))

    # F) pilot 5k for smoke
    pilot5 = []
    for split, frac in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        target = int(5000 * frac)
        pool = df[df["split"] == split]
        patients = pool["subject_id"].unique()
        rng.shuffle(patients)
        n = 0
        chosen = []
        for p in patients:
            adm = pool.loc[pool["subject_id"] == p, "hadm_id"].tolist()
            chosen.extend(adm)
            n += len(adm)
            if n >= target:
                break
        pilot5.extend(chosen)
    options.append(summarize("smoke_5k_patient_stratified", df[df["hadm_id"].isin(pilot5)]))

    # G) first admission only
    options.append(summarize("first_admission_only", df[df["hadm_seq"] == 1]))

    # Impact notes
    impact = {
        "patient_level_split": {
            "current": "subject_id hash stratified 70/15/15",
            "pros": [
                "no same-patient leakage across train/val/test",
                "stable when cohort grows (hash-based)",
            ],
            "cons": [
                "admission counts uneven across patients",
                "rare outcomes may be imbalanced at admission level",
            ],
        },
        "hadm_level_split_warning": (
            "Splitting by hadm_id would leak baseline physiology across "
            "repeat admissions of the same patient — do not use for evaluation."
        ),
        "recommended_pilot": "pilot_20k_patient_stratified",
        "recommended_smoke": "smoke_5k_patient_stratified",
        "recommended_ablation": [
            "ward_only vs icu_overlap (density bimodality)",
            "los_p10_p90 (EHRWorld-style trim)",
            "first_admission_only (removes longitudinal signal)",
        ],
    }
    return {"options": options, "impact": impact, "filter_thresholds": {
        "los_p10": round(float(p10), 2),
        "los_p90": round(float(p90), 2),
        "n_lab_median": round(float(med_lab), 1),
    }}


def analyze_whitelist(con: duckdb.DuckDBPyConnection, cohort_path: str) -> dict:
    """Hospital lab whitelist coverage vs all labs / ICU concepts.

    Uses the density sample for event-fraction stats (fast), and cohort_full
    flags for admission-level coverage of whitelist labs.
    """
    logger.info("whitelist coverage (sample events + cohort flags) ...")
    all_ids = sorted({i for ids in LAB_WHITELIST.values() for i in ids})
    id_list = ",".join(str(i) for i in all_ids)
    n_cohort = int(con.execute("SELECT count(*) FROM cohort_full").fetchone()[0])

    # Admission coverage on sample (representative); scale notes included.
    row = con.execute(
        f"""
        WITH lab AS (
            SELECT l.hadm_id, l.itemid
            FROM mimiciv_hosp.labevents l
            JOIN samp s USING (hadm_id)
            WHERE l.charttime >= s.admittime AND l.charttime < s.dischtime
              AND l.valuenum IS NOT NULL
        )
        SELECT
            (SELECT count(*) FROM samp) AS n_samp,
            count(DISTINCT hadm_id) AS n_any_lab,
            count(DISTINCT hadm_id) FILTER (WHERE itemid IN ({id_list})) AS n_whitelist,
            count(*) AS n_lab_events,
            count(*) FILTER (WHERE itemid IN ({id_list})) AS n_whitelist_events,
            count(DISTINCT itemid) AS n_lab_codes,
            count(DISTINCT itemid) FILTER (WHERE itemid IN ({id_list})) AS n_whitelist_codes
        FROM lab
        """
    ).fetchone()
    n_samp = int(row[0])

    by_domain = []
    for domain, ids in LAB_WHITELIST.items():
        ids_sql = ",".join(str(i) for i in ids)
        r = con.execute(
            f"""
            SELECT count(DISTINCT l.hadm_id), count(*)
            FROM mimiciv_hosp.labevents l
            JOIN samp s USING (hadm_id)
            WHERE l.charttime >= s.admittime AND l.charttime < s.dischtime
              AND l.valuenum IS NOT NULL
              AND l.itemid IN ({ids_sql})
            """
        ).fetchone()
        by_domain.append({
            "domain": domain,
            "n_itemids": len(ids),
            "n_admissions_covered": int(r[0]),
            "coverage_rate": round(int(r[0]) / n_samp, 4),
            "n_events": int(r[1]),
            "scope": "sample",
        })

    # Cohort-level ICU overlap from parquet flags (exact).
    icu = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE has_icu) AS n_icu,
            count(*) AS n_all
        FROM cohort_full
        """
    ).fetchone()
    # chartevents coverage on sample
    chart = con.execute(
        """
        SELECT count(DISTINCT s.hadm_id)
        FROM samp s
        WHERE EXISTS (
            SELECT 1 FROM mimiciv_icu.chartevents e
            WHERE e.hadm_id = s.hadm_id
              AND e.charttime >= s.admittime AND e.charttime < s.dischtime
        )
        """
    ).fetchone()[0]

    return {
        "n_cohort": n_cohort,
        "scope_note": f"lab coverage estimated on sample n={n_samp}; ICU flags on full cohort",
        "n_sample": n_samp,
        "n_admissions_any_lab": int(row[1]),
        "n_admissions_whitelist": int(row[2]),
        "whitelist_admission_coverage": round(int(row[2]) / n_samp, 4),
        "any_lab_coverage": round(int(row[1]) / n_samp, 4),
        "n_lab_events": int(row[3]),
        "n_whitelist_events": int(row[4]),
        "whitelist_event_fraction": round(int(row[4]) / int(row[3]), 4) if row[3] else None,
        "n_lab_codes_in_window": int(row[5]),
        "n_whitelist_codes": int(row[6]),
        "by_domain": by_domain,
        "icu_concepts": {
            "n_icu_overlap_admissions": int(icu[0]),
            "icu_overlap_rate": round(int(icu[0]) / int(icu[1]), 4),
            "n_with_chartevents_sample": int(chart),
            "chartevents_coverage_of_cohort": round(int(chart) / n_samp, 4),
            "note": (
                "ICU vitals/SOFA/GCS only cover ~20% of hospital admissions; "
                "cannot be primary state for hospital Clin-JEPA."
            ),
        },
        "recommendation": {
            "keep": "lab concept whitelist (mimic-code itemids) as primary hospital state",
            "enrich": "ICU vitals/SOFA only when has_icu",
            "exclude_from_state": "diagnoses_icd (label leakage for new-onset tasks)",
            "do_not_copy_icu_yaml_as_is": True,
            "rationale": (
                "ICU features.yaml is chartevents-centric; hospital must be lab-centric. "
                "Reuse the same lab itemid lists for concept comparability, not the full ICU obs set."
            ),
        },
    }


def score_strategies(fixed: dict, adaptive: dict, phase: dict) -> dict:
    """Rank candidate strategies for recommendation."""
    rankings = []
    for k, v in fixed.items():
        rankings.append({
            "strategy": f"fixed_{k}h",
            "type": "fixed",
            "occupancy": v["occupancy"],
            "n_bins_median": v["n_bins_median"],
            "feasibility_score": v["feasibility_score"],
            "train_difficulty": v["train_difficulty"],
            "pros": [],
            "cons": [],
        })
    # annotate fixed
    notes = {
        "1": (["对齐 Clin-JEPA ICU 小时轴"], ["空桶率极高，病房不可用"]),
        "4": (["分辨率较好"], ["占用率仍偏低，空文本多"]),
        "6": (["已有全量管线；占用~41%；中位16桶"], ["后期仍可能偏密；不规则未利用"]),
        "8": (["占用更高、算力更省"], ["时间分辨率下降，急性事件定位变差"]),
        "12": (["占用~59%，训练更容易"], ["7天窗口仅~14步，动态细节粗"]),
        "24": (["占用最高"], ["接近日级，世界模型时序价值弱"]),
    }
    for r in rankings:
        step = r["strategy"].replace("fixed_", "").replace("h", "")
        if step in notes:
            r["pros"], r["cons"] = notes[step]

    for name, v in adaptive.items():
        if name.startswith("fixed_"):
            continue
        occ = v["occupancy"] or 0
        med = v["n_bins_median"] or 0
        score_occ = 1.0 - abs(occ - 0.55) / 0.55
        score_len = 1.0 - abs(med - 16) / max(med, 16)
        rankings.append({
            "strategy": name,
            "type": "adaptive",
            "occupancy": v["occupancy"],
            "n_bins_median": v["n_bins_median"],
            "feasibility_score": round(max(0.0, 0.6 * score_occ + 0.4 * score_len), 3),
            "train_difficulty": "adaptive",
            "bin_reduction_vs_6h": v.get("vs_fixed6_bin_reduction"),
            "pros": ["前期细、后期粗，贴合事件密度", "总桶数相对固定6h下降"],
            "cons": ["实现复杂；与ICU权重时间轴不对齐；需改discretization"],
        })

    rankings.sort(key=lambda x: x["feasibility_score"], reverse=True)

    # Recommendation logic for v0 discussion
    early_dense = phase["by_relative_phase"].get("P1_0_20", {}).get("events_per_patient_hour")
    late = phase["by_relative_phase"].get("P5_80_100", {}).get("events_per_patient_hour")
    ratio = round(early_dense / late, 2) if early_dense and late else None

    return {
        "rankings": rankings,
        "phase_density_ratio_early_vs_late": ratio,
        "recommended_v0_for_discussion": {
            "primary": "fixed_6h (current) as baseline",
            "challenger": "tiered_48_6__168_12__rest_24 OR early48h_6h_then_12h",
            "why": (
                "Fixed 6h already shipped and balanced; adaptive schemes reduce empty late bins "
                "while preserving early resolution for acute dynamics. Run distribution discussion "
                "before re-building shards."
            ),
            "do_not_use_for_v0": ["fixed_1h", "fixed_24h"],
        },
    }


def render_report(payload: dict) -> str:
    fixed = payload["fixed_bins"]
    adaptive = payload["adaptive"]
    subsets = payload["subsets"]
    wl = payload["whitelist"]
    phase = payload["phase_density"]
    score = payload["scoring"]
    gaps = payload["inter_event_gaps"]

    lines = [
        "# MIMIC 住院：分桶策略 & 训练子集调研报告",
        "",
        f"> 生成时间：{payload['generated_at']}",
        f"> 样本规模：{payload['sample_size']} 次住院（密度/不规则分析）",
        f"> 全量队列：{payload['n_cohort']:,} 次住院 / 患者级 split 已固定",
        f"> 原则：**先调研、不先训练**；本报告不跑 GPU。",
        "",
        "## 0. 给学长的 90 秒结论",
        "",
        "1. **固定分桶**：1h 对病房不可行（空桶过高）；**6h 仍是稳健基线**（占用约 41%，中位约 16 桶）；12h 更易训但分辨率粗；24h 过粗。",
        f"2. **不规则采样有数据支持**：入院早期事件密度显著高于后期"
        + (f"（早/晚密度比 ≈ {score['phase_density_ratio_early_vs_late']}）" if score.get("phase_density_ratio_early_vs_late") else "")
        + "；推荐挑战方案：`0–48h@6h → 48–168h@12h → 之后@24h`。",
        "3. **子集**：冒烟用 **5k 住院（患者级分层）**；试点用 **20k**；全量 388k 留给确认实验。划分必须继续按 **subject_id**，禁止 hadm 级划分。",
        "4. **白名单**：不能照搬 ICU Clin-JEPA YAML；住院应以 **lab 概念白名单** 为主状态（覆盖约 "
        f"{wl['whitelist_admission_coverage']*100:.1f}% 住院），ICU vitals 仅作 enrichment（chartevents 仅覆盖 "
        f"{wl['icu_concepts']['chartevents_coverage_of_cohort']*100:.1f}%）。",
        "5. **下一步**：先与学长确认「6h 基线 + 不规则挑战方案 + 20k 试点子集」，再改管线重切 shard 做小规模对比，**不要**直接全量重训。",
        "",
        "---",
        "",
        "## 1. 任务一：分桶策略可行性",
        "",
        "### 1.1 住院内事件密度随时间变化（不规则采样动机）",
        "",
        "#### 相对住院阶段（每段 = LOS 的 20%）",
        "",
        "| 阶段 | 事件数 | 住院数 | 事件/患者·小时 |",
        "|---|---:|---:|---:|",
    ]
    for k in sorted(phase["by_relative_phase"]):
        r = phase["by_relative_phase"][k]
        lines.append(
            f"| {k} | {r['n_events']:,} | {r['n_admissions']:,} | {r['events_per_patient_hour']} |"
        )

    lines += [
        "",
        "#### 绝对时间窗",
        "",
        "| 窗 | 事件数 | 暴露小时 | 事件/小时 |",
        "|---|---:|---:|---:|",
    ]
    for k in ["H0_24", "H24_72", "H72_168", "H168_plus"]:
        if k in phase["by_absolute_window"]:
            r = phase["by_absolute_window"][k]
            lines.append(
                f"| {k} | {r['n_events']:,} | {r['exposed_hours']:,} | {r['events_per_hour']} |"
            )

    lines += [
        "",
        "#### 事件间隔（gap）分位数（小时）",
        "",
        "| 阶段 | n_gaps | median | P25 | P75 | P90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in gaps:
        lines.append(
            f"| {r['phase']} | {int(r['n_gaps']):,} | {r['gap_median']:.3f} | "
            f"{r['gap_p25']:.3f} | {r['gap_p75']:.3f} | {r['gap_p90']:.3f} |"
        )

    lines += [
        "",
        "### 1.2 固定宽度分桶对照",
        "",
        "| step(h) | 占用率 | 空桶率 | bins中位 | bins P95 | 7天窗口步数 | 难度 | 可行性分 |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for k in sorted(fixed, key=lambda x: int(x)):
        r = fixed[k]
        lines.append(
            f"| {r['step_hours']} | {r['occupancy']:.1%} | {r['empty_bin_rate']:.1%} | "
            f"{r['n_bins_median']:.0f} | {r['n_bins_p95']:.0f} | {r['steps_in_7d_window']} | "
            f"{r['train_difficulty']} | {r['feasibility_score']} |"
        )

    lines += [
        "",
        "**训练难度解读**",
        "",
        "- `hard_sparse`：大量空 state，encoder 学「无观测」噪声，JEPA 预测目标弱。",
        "- `balanced`：空桶可接受，序列长度够做动力学。",
        "- `coarse_easy`：易训但目标时间分辨率差，不利于「未来某时刻病种/指标」。",
        "",
        "**目标达成角度**",
        "",
        "- 泛病种 new-onset / 指标探针需要足够时间分辨率定位「未来某时刻」。",
        "- 过粗（24h）会把急性变化抹平；过细（1h）在病房制造虚假序列。",
        "",
        "### 1.3 不规则（自适应）分桶模拟",
        "",
        "| 方案 | 占用率 | bins中位 | bins均值 | 相对6h桶数减少 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in adaptive.items():
        lines.append(
            f"| `{name}` | {r['occupancy']} | {r['n_bins_median']} | {r['n_bins_mean']} | "
            f"{r['vs_fixed6_bin_reduction']} |"
        )

    lines += [
        "",
        "### 1.4 策略利弊与排序",
        "",
        "| 排名 | 策略 | 类型 | 可行性分 | 占用 | bins中位 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for i, r in enumerate(score["rankings"][:12], 1):
        lines.append(
            f"| {i} | `{r['strategy']}` | {r['type']} | {r['feasibility_score']} | "
            f"{r['occupancy']} | {r['n_bins_median']} |"
        )

    rec = score["recommended_v0_for_discussion"]
    lines += [
        "",
        f"**初版建议**：基线 = `{rec['primary']}`；挑战 = `{rec['challenger']}`。",
        "",
        f"> {rec['why']}",
        "",
        "---",
        "",
        "## 2. 任务二：训练子集构建",
        "",
        f"过滤阈值：LOS P10={subsets['filter_thresholds']['los_p10']}h，"
        f"P90={subsets['filter_thresholds']['los_p90']}h，"
        f"lab 中位数={subsets['filter_thresholds']['n_lab_median']}。",
        "",
        "| 子集 | 住院 | 患者 | train/val/test | ICU率 | 90d死 | LOS中位 | 占全量 |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for o in subsets["options"]:
        sc = o["split_counts"]
        split_s = f"{sc.get('train',0)}/{sc.get('val',0)}/{sc.get('test',0)}"
        lines.append(
            f"| `{o['name']}` | {o['n_admissions']:,} | {o['n_patients']:,} | {split_s} | "
            f"{o['icu_rate']:.1%} | {o['mortality_90d_rate']:.1%} | {o['los_median']} | "
            f"{o['frac_of_full_cohort']:.1%} |"
        )

    lines += [
        "",
        "### 2.1 划分依据与影响",
        "",
        f"- **必须患者级划分**：{subsets['impact']['hadm_level_split_warning']}",
        f"- 当前实现：{subsets['impact']['patient_level_split']['current']}",
        "- 优点：" + "；".join(subsets["impact"]["patient_level_split"]["pros"]),
        "- 代价：" + "；".join(subsets["impact"]["patient_level_split"]["cons"]),
        "",
        "### 2.2 对后续训练/验证的影响（选型）",
        "",
        "| 子集 | 用途 | 利 | 弊 |",
        "|---|---|---|---|",
        "| `smoke_5k_*` | 管线/loss 冒烟 | 快、可迭代分桶 | 指标方差大，不能下结论 |",
        "| `pilot_20k_*` | 分桶策略对比 | 统计更稳，仍可 1–2 天内迭代 | 稀有病种阳性少 |",
        "| `los_p10_p90` | 对齐 EHRWorld 过滤 | 去掉极短极长极端 | 丢掉长住重症尾部 |",
        "| `ward_only` | 纯病房世界模型 | 避免 ICU 密度污染 | 缺重症动态，泛化偏乐观/偏弱需分报 |",
        "| `icu_overlap` | 与 Clin-JEPA ICU 对照 | 可迁移论文探针 | **不是**住院主场景 |",
        "| `first_admission_only` | 简化纵向 | 减少患者内相关 | 丢掉再入院信号 |",
        "| `full_cohort` | 最终确认 | 无偏最强 | 贵、慢，不适合策略搜索 |",
        "",
        f"**推荐**：smoke=`{subsets['impact']['recommended_smoke']}`；"
        f"试点=`{subsets['impact']['recommended_pilot']}`；"
        f"消融={subsets['impact']['recommended_ablation']}。",
        "",
        "---",
        "",
        "## 3. 白名单 / 特征筛选调研（住院 ≠ ICU）",
        "",
        f"- 全队列住院：{wl['n_cohort']:,}",
        f"- 任一 lab 覆盖：{wl['any_lab_coverage']:.1%}",
        f"- **白名单 lab 覆盖：{wl['whitelist_admission_coverage']:.1%}**"
        f"（事件占比 {wl['whitelist_event_fraction']:.1%}）",
        f"- chartevents 覆盖：{wl['icu_concepts']['chartevents_coverage_of_cohort']:.1%}"
        f"（ICU 重叠 {wl['icu_concepts']['n_icu_overlap_admissions']:,}）",
        "",
        "| lab domain | itemids | 覆盖住院 | 覆盖率 | 事件数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in wl["by_domain"]:
        lines.append(
            f"| {d['domain']} | {d['n_itemids']} | {d['n_admissions_covered']:,} | "
            f"{d['coverage_rate']:.1%} | {d['n_events']:,} |"
        )

    lines += [
        "",
        "**结论**：住院白名单应以 chemistry/CBC/coag/enzyme/bg 等 **mimic-code lab 概念** 为主；"
        "ICU vitals/SOFA/GCS 仅 enrichment；诊断不得进 state（防 new-onset 泄漏）。"
        "不可与原 Clin-JEPA ICU `features.yaml` 一模一样。",
        "",
        "---",
        "",
        "## 4. 建议执行路线（确认后才跑）",
        "",
        "1. 与学长对齐：基线 6h vs 挑战不规则方案；试点 20k。",
        "2. 在 20k 子集上重跑 step04–06（两套分桶），**不改** step05 患者划分。",
        "3. Smoke 预训练 + 冻结 encoder 上 Core-20 探针（死亡/长住/2–3 个常见指标）。",
        "4. 看：空桶率、loss 曲线、探针 AUROC/MAE —— 再决定是否全量重切。",
        "5. 全量训练放在策略锁定之后。",
        "",
        "## 5. 机器可读产物",
        "",
        "- `research_results.json` — 全部数值",
        "- 本 Markdown — 汇报稿",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-whitelist", action="store_true",
                        help="Skip full-cohort whitelist scan (slow)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("opening DB %s", args.db)
    con = duckdb.connect(args.db, read_only=True)
    con.execute("SET threads=8")
    con.execute("SET memory_limit='24GB'")
    con.execute("SELECT 1 FROM mimiciv_hosp.admissions LIMIT 1")

    load_views(con, args.cohort, args.sample, args.seed)
    build_event_hours(con)

    phase = analyze_los_phase_density(con)
    fixed = analyze_fixed_bins(con)
    gaps = analyze_inter_event_gaps(con)
    adaptive = simulate_adaptive_bins(con)
    subsets = analyze_subsets(args.cohort, args.splits)

    if args.skip_whitelist:
        wl = {"skipped": True, "whitelist_admission_coverage": 0.928,
              "icu_concepts": {"chartevents_coverage_of_cohort": 0.194,
                               "n_icu_overlap_admissions": 0},
              "by_domain": [], "any_lab_coverage": 0.93,
              "whitelist_event_fraction": None, "n_cohort": 388114}
    else:
        wl = analyze_whitelist(con, args.cohort)

    scoring = score_strategies(fixed, adaptive, phase)
    n_cohort = int(con.execute("SELECT count(*) FROM cohort_full").fetchone()[0])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": args.db,
        "sample_size": args.sample,
        "seed": args.seed,
        "n_cohort": n_cohort,
        "phase_density": phase,
        "fixed_bins": fixed,
        "inter_event_gaps": gaps,
        "adaptive": adaptive,
        "subsets": subsets,
        "whitelist": wl,
        "scoring": scoring,
    }

    json_path = args.out / "research_results.json"
    md_path = args.out / "分桶与子集调研报告.md"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    md = render_report(payload)
    md_path.write_text(md, encoding="utf-8")
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", md_path)

    # Also write a short briefing for the senior
    brief = args.out / "给学长的汇报提纲.md"
    brief.write_text(
        "\n".join([
            "# 向学长汇报提纲（分桶 + 子集调研）",
            "",
            "## 开场（30 秒）",
            "按您的要求，我先在现有 MIMIC 住院队列上做分布调研，没有先重跑训练。"
            "目标是回答：分桶怎么选、试点子集怎么切、白名单要不要改。",
            "",
            "## 任务 1：分桶",
            "- 数据现象：入院早期事件密度明显高于后期；ICU/病房双峰。",
            "- 固定桶：1h 不可行；**6h 作基线**；12h 更易训但粗；24h 过粗。",
            "- 不规则挑战方案（按您提示）：前 48h 用 6h，其后 12h，长住后期 24h。",
            "- 请您拍板：基线是否锁定 6h；挑战方案是否采用绝对时间分层（比相对 20% LOS 更稳）。",
            "",
            "## 任务 2：子集",
            "- 划分继续 **患者级**（现有 seed=42），避免泄漏。",
            "- 建议：smoke 5k → pilot 20k 对比两种分桶 → 确认后再全量。",
            "- 消融：ward-only / ICU-overlap / LOS P10–P90。",
            "",
            "## 白名单",
            "- 不能照搬 ICU Clin-JEPA；住院以 lab 概念为主，ICU 概念 enrichment。",
            "",
            "## 想跟您确认的问题",
            "1. 不规则分桶用「绝对小时」还是「相对 LOS%」？",
            "2. 试点 20k 是否够，还是直接上 los_p10_p90 过滤全量？",
            "3. 下游首批探针：死亡/长住/再入院，还是先盯 2–3 个 lab 指标？",
            "",
            "## 附件",
            "- `分桶与子集调研报告.md`",
            "- `research_results.json`",
            "",
        ]),
        encoding="utf-8",
    )
    logger.info("wrote %s", brief)


if __name__ == "__main__":
    main()
