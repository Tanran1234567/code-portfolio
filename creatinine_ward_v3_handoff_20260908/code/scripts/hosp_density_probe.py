"""Hospital-scope event density probe.

Establishes the empirical basis for the discretization step size and cohort
filters used by the hospital (inpatient) Clin-JEPA pipeline. The ICU pipeline
bins at 1 hour because ICU charting is dense (~1113 chartevents/ICU-day). Ward
time is far sparser, so the bin width has to be chosen from measurement rather
than inherited.

Outputs a JSON + markdown summary under reports/hosp_pipeline/.

Usage:
    python scripts/hosp_density_probe.py
    python scripts/hosp_density_probe.py --sample-admissions 20000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB = "/data/general_disease_pred/mimic4.db"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "reports" / "hosp_pipeline"

# Candidate step sizes (hours) to evaluate occupancy for.
STEP_SIZES = [1, 2, 3, 4, 6, 8, 12, 24]


def _cohort_sql(min_age: int, min_los_hours: float) -> str:
    """Adult admissions with a usable length of stay."""
    return f"""
    WITH adm AS (
        SELECT
            a.subject_id,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.hospital_expire_flag,
            p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year) AS admission_age,
            date_diff('second', a.admittime, a.dischtime) / 3600.0 AS los_hours
        FROM mimiciv_hosp.admissions a
        JOIN mimiciv_hosp.patients p USING (subject_id)
        WHERE a.dischtime IS NOT NULL
          AND a.dischtime > a.admittime
    )
    SELECT * FROM adm
    WHERE admission_age >= {min_age}
      AND los_hours >= {min_los_hours}
    """


def probe_cohort(con: duckdb.DuckDBPyConnection, min_age: int, min_los: float) -> dict:
    """Cohort size and LOS distribution."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW cohort AS {_cohort_sql(min_age, min_los)}")
    row = con.execute(
        """
        SELECT
            count(*)                        AS n_admissions,
            count(DISTINCT subject_id)      AS n_patients,
            avg(los_hours)                  AS los_mean,
            median(los_hours)               AS los_median,
            quantile_cont(los_hours, 0.10)  AS los_p10,
            quantile_cont(los_hours, 0.90)  AS los_p90,
            quantile_cont(los_hours, 0.95)  AS los_p95,
            quantile_cont(los_hours, 0.99)  AS los_p99,
            max(los_hours)                  AS los_max,
            avg(hospital_expire_flag)       AS mortality_rate
        FROM cohort
        """
    ).fetchone()
    keys = [
        "n_admissions", "n_patients", "los_mean", "los_median",
        "los_p10", "los_p90", "los_p95", "los_p99", "los_max", "mortality_rate",
    ]
    return {k: (float(v) if v is not None else None) for k, v in zip(keys, row)}


def probe_icu_overlap(con: duckdb.DuckDBPyConnection) -> dict:
    """How many cohort admissions have any ICU time."""
    row = con.execute(
        """
        SELECT
            count(*) AS n_total,
            count(*) FILTER (WHERE has_icu) AS n_with_icu
        FROM (
            SELECT c.hadm_id,
                   EXISTS (SELECT 1 FROM mimiciv_icu.icustays i
                           WHERE i.hadm_id = c.hadm_id) AS has_icu
            FROM cohort c
        )
        """
    ).fetchone()
    n_total, n_icu = int(row[0]), int(row[1])
    return {
        "n_total": n_total,
        "n_with_icu": n_icu,
        "icu_fraction": round(n_icu / n_total, 4) if n_total else 0.0,
    }


def probe_source_counts(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-source event totals and admission coverage, restricted to the
    in-stay window [admittime, dischtime)."""
    sources = [
        # (label, domain, sql returning hadm_id, event_time)
        ("labevents", "observation", """
            SELECT l.hadm_id, l.charttime AS t
            FROM mimiciv_hosp.labevents l JOIN cohort c USING (hadm_id)
            WHERE l.charttime >= c.admittime AND l.charttime < c.dischtime
        """),
        ("microbiologyevents", "observation", """
            SELECT m.hadm_id, coalesce(m.charttime, m.chartdate) AS t
            FROM mimiciv_hosp.microbiologyevents m JOIN cohort c USING (hadm_id)
            WHERE coalesce(m.charttime, m.chartdate) >= c.admittime
              AND coalesce(m.charttime, m.chartdate) < c.dischtime
        """),
        ("chartevents", "observation", """
            SELECT e.hadm_id, e.charttime AS t
            FROM mimiciv_icu.chartevents e JOIN cohort c USING (hadm_id)
            WHERE e.charttime >= c.admittime AND e.charttime < c.dischtime
        """),
        ("prescriptions", "action", """
            SELECT r.hadm_id, r.starttime AS t
            FROM mimiciv_hosp.prescriptions r JOIN cohort c USING (hadm_id)
            WHERE r.starttime >= c.admittime AND r.starttime < c.dischtime
        """),
        ("emar", "action", """
            SELECT e.hadm_id, e.charttime AS t
            FROM mimiciv_hosp.emar e JOIN cohort c USING (hadm_id)
            WHERE e.charttime >= c.admittime AND e.charttime < c.dischtime
        """),
        ("procedures_icd", "action", """
            SELECT p.hadm_id, p.chartdate AS t
            FROM mimiciv_hosp.procedures_icd p JOIN cohort c USING (hadm_id)
        """),
        ("transfers", "observation", """
            SELECT t.hadm_id, t.intime AS t
            FROM mimiciv_hosp.transfers t JOIN cohort c USING (hadm_id)
            WHERE t.intime IS NOT NULL
        """),
    ]

    out = []
    for label, domain, sql in sources:
        logger.info("  counting %s ...", label)
        row = con.execute(
            f"""
            SELECT count(*) AS n_events,
                   count(DISTINCT hadm_id) AS n_admissions
            FROM ({sql})
            """
        ).fetchone()
        out.append({
            "source": label,
            "domain": domain,
            "n_events": int(row[0]),
            "n_admissions_covered": int(row[1]),
        })
    return out


def probe_occupancy(con: duckdb.DuckDBPyConnection, sample: int, seed: int) -> dict:
    """Bin occupancy at each candidate step size, on a random admission sample.

    Occupancy = fraction of bins within [0, LOS) that contain >=1 observation
    event. This is the number that decides the step size: a bin width where
    most bins are empty wastes encoder compute on "No observations recorded."
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE samp AS
        SELECT * FROM cohort USING SAMPLE {sample} ROWS (reservoir, {seed})
        """
    )
    n_samp = con.execute("SELECT count(*) FROM samp").fetchone()[0]
    logger.info("  occupancy sample: %d admissions", n_samp)

    # Union of observation-domain events for the sampled admissions.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE obs_ev AS
        SELECT hadm_id, date_diff('second', admittime, t) / 3600.0 AS th
        FROM (
            SELECT l.hadm_id, l.charttime AS t, s.admittime
            FROM mimiciv_hosp.labevents l JOIN samp s USING (hadm_id)
            WHERE l.charttime >= s.admittime AND l.charttime < s.dischtime
            UNION ALL
            SELECT m.hadm_id, coalesce(m.charttime, m.chartdate) AS t, s.admittime
            FROM mimiciv_hosp.microbiologyevents m JOIN samp s USING (hadm_id)
            WHERE coalesce(m.charttime, m.chartdate) >= s.admittime
              AND coalesce(m.charttime, m.chartdate) < s.dischtime
            UNION ALL
            SELECT e.hadm_id, e.charttime AS t, s.admittime
            FROM mimiciv_icu.chartevents e JOIN samp s USING (hadm_id)
            WHERE e.charttime >= s.admittime AND e.charttime < s.dischtime
        )
        """
    )

    results = {}
    for step in STEP_SIZES:
        logger.info("  occupancy @ step=%dh ...", step)
        row = con.execute(
            f"""
            WITH bins AS (
                SELECT hadm_id, floor(th / {step}) AS b, count(*) AS n
                FROM obs_ev
                WHERE th >= 0
                GROUP BY hadm_id, b
            ),
            tot AS (
                SELECT hadm_id,
                       greatest(1, ceil(los_hours / {step})) AS n_bins
                FROM samp
            ),
            per_adm AS (
                SELECT t.hadm_id, t.n_bins,
                       coalesce(count(b.b), 0) AS filled,
                       coalesce(sum(b.n), 0)   AS n_events
                FROM tot t LEFT JOIN bins b
                  ON b.hadm_id = t.hadm_id AND b.b < t.n_bins
                GROUP BY t.hadm_id, t.n_bins
            )
            SELECT
                sum(n_bins)                              AS total_bins,
                sum(filled)                              AS filled_bins,
                sum(filled) / sum(n_bins)                AS occupancy,
                avg(filled * 1.0 / n_bins)               AS occupancy_per_adm_mean,
                median(filled * 1.0 / n_bins)            AS occupancy_per_adm_median,
                sum(n_events) * 1.0 / sum(n_bins)        AS events_per_bin_mean,
                sum(n_events) * 1.0 / nullif(sum(filled), 0) AS events_per_filled_bin,
                median(n_bins)                           AS n_bins_median,
                quantile_cont(n_bins, 0.95)              AS n_bins_p95
            FROM per_adm
            """
        ).fetchone()
        results[str(step)] = {
            "step_hours": step,
            "total_bins": int(row[0]),
            "filled_bins": int(row[1]),
            "occupancy": round(float(row[2]), 4),
            "occupancy_per_adm_mean": round(float(row[3]), 4),
            "occupancy_per_adm_median": round(float(row[4]), 4),
            "events_per_bin_mean": round(float(row[5]), 3),
            "events_per_filled_bin": round(float(row[6]), 3),
            "n_bins_median": float(row[7]),
            "n_bins_p95": float(row[8]),
        }

    # Same, split by ICU presence — density is bimodal and the report must say so.
    strata = {}
    for name, pred in [("with_icu", "EXISTS"), ("no_icu", "NOT EXISTS")]:
        row = con.execute(
            f"""
            WITH sub AS (
                SELECT * FROM samp s
                WHERE {pred} (SELECT 1 FROM mimiciv_icu.icustays i WHERE i.hadm_id = s.hadm_id)
            ),
            bins AS (
                SELECT o.hadm_id, floor(o.th / 1) AS b, count(*) AS n
                FROM obs_ev o JOIN sub USING (hadm_id)
                WHERE o.th >= 0 GROUP BY o.hadm_id, b
            ),
            tot AS (SELECT hadm_id, greatest(1, ceil(los_hours)) AS n_bins FROM sub),
            per_adm AS (
                SELECT t.hadm_id, t.n_bins, coalesce(count(b.b), 0) AS filled,
                       coalesce(sum(b.n), 0) AS n_events
                FROM tot t LEFT JOIN bins b
                  ON b.hadm_id = t.hadm_id AND b.b < t.n_bins
                GROUP BY t.hadm_id, t.n_bins
            )
            SELECT count(*), sum(n_bins), sum(filled),
                   sum(filled) / sum(n_bins), sum(n_events) * 1.0 / sum(n_bins)
            FROM per_adm
            """
        ).fetchone()
        strata[name] = {
            "n_admissions": int(row[0]),
            "total_bins_1h": int(row[1]),
            "filled_bins_1h": int(row[2]),
            "occupancy_1h": round(float(row[3]), 4) if row[3] is not None else None,
            "events_per_bin_1h": round(float(row[4]), 3) if row[4] is not None else None,
        }

    return {"sample_size": int(n_samp), "by_step": results, "by_icu_stratum": strata}


def render_markdown(payload: dict) -> str:
    """Human-readable summary for the design report."""
    c = payload["cohort"]
    icu = payload["icu_overlap"]
    occ = payload["occupancy"]

    lines = [
        "# 住院场景事件密度探针结果",
        "",
        f"> 生成时间：{payload['generated_at']}",
        f"> 数据库：`{payload['db_path']}`",
        f"> 队列口径：年龄 ≥ {payload['params']['min_age']}，住院时长 ≥ {payload['params']['min_los_hours']}h",
        "",
        "## 1. 队列规模与住院时长",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 住院次数 | {c['n_admissions']:,.0f} |",
        f"| 患者数 | {c['n_patients']:,.0f} |",
        f"| LOS 均值 | {c['los_mean']:.1f} h |",
        f"| LOS 中位 | {c['los_median']:.1f} h |",
        f"| LOS P10 | {c['los_p10']:.1f} h |",
        f"| LOS P90 | {c['los_p90']:.1f} h |",
        f"| LOS P95 | {c['los_p95']:.1f} h |",
        f"| LOS P99 | {c['los_p99']:.1f} h |",
        f"| 院内死亡率 | {c['mortality_rate'] * 100:.2f}% |",
        "",
        "## 2. ICU 重叠",
        "",
        f"- 队列住院数：{icu['n_total']:,}",
        f"- 含 ICU 时段：{icu['n_with_icu']:,}（**{icu['icu_fraction'] * 100:.1f}%**）",
        "",
        "> 密度呈双峰：含 ICU 的住院有 chartevents 高频记录，纯病房住院只有检验/给药。",
        "",
        "## 3. 各来源事件量与覆盖",
        "",
        "| 来源 | domain | 窗内事件数 | 覆盖住院数 |",
        "|---|---|---:|---:|",
    ]
    for s in payload["source_counts"]:
        lines.append(
            f"| {s['source']} | {s['domain']} | {s['n_events']:,} | {s['n_admissions_covered']:,} |"
        )

    lines += [
        "",
        "## 4. 分桶占用率（决定 step_size 的核心证据）",
        "",
        f"抽样住院数：{occ['sample_size']:,}",
        "",
        "| step (h) | 总桶数 | 非空桶 | 占用率 | 每桶事件均值 | 非空桶事件均值 | 桶数中位 | 桶数 P95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(occ["by_step"], key=lambda k: int(k)):
        r = occ["by_step"][key]
        lines.append(
            f"| {r['step_hours']} | {r['total_bins']:,} | {r['filled_bins']:,} | "
            f"{r['occupancy'] * 100:.1f}% | {r['events_per_bin_mean']:.2f} | "
            f"{r['events_per_filled_bin']:.2f} | {r['n_bins_median']:.0f} | {r['n_bins_p95']:.0f} |"
        )

    st = occ["by_icu_stratum"]
    lines += [
        "",
        "### 4.1 按 ICU 分层（step = 1h）",
        "",
        "| 分层 | 住院数 | 总桶 | 非空桶 | 占用率 | 每桶事件均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in [("with_icu", "含 ICU"), ("no_icu", "纯病房")]:
        r = st[name]
        occ_s = f"{r['occupancy_1h'] * 100:.1f}%" if r["occupancy_1h"] is not None else "—"
        epb = f"{r['events_per_bin_1h']:.2f}" if r["events_per_bin_1h"] is not None else "—"
        lines.append(
            f"| {label} | {r['n_admissions']:,} | {r['total_bins_1h']:,} | "
            f"{r['filled_bins_1h']:,} | {occ_s} | {epb} |"
        )

    lines += ["", "---", "", "*由 `scripts/hosp_density_probe.py` 生成。*", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hospital-scope event density probe")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--min-age", type=int, default=18)
    parser.add_argument("--min-los-hours", type=float, default=24.0)
    parser.add_argument("--sample-admissions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.db, read_only=True)
    con.execute(f"SET threads={args.threads}")
    con.execute("SET memory_limit='24GB'")

    logger.info("Probing cohort ...")
    cohort = probe_cohort(con, args.min_age, args.min_los_hours)
    logger.info("  %d admissions / %d patients", cohort["n_admissions"], cohort["n_patients"])

    logger.info("Probing ICU overlap ...")
    icu_overlap = probe_icu_overlap(con)

    logger.info("Probing per-source counts ...")
    source_counts = probe_source_counts(con)

    logger.info("Probing bin occupancy ...")
    occupancy = probe_occupancy(con, args.sample_admissions, args.seed)

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "db_path": args.db,
        "params": {
            "min_age": args.min_age,
            "min_los_hours": args.min_los_hours,
            "sample_admissions": args.sample_admissions,
            "seed": args.seed,
        },
        "cohort": cohort,
        "icu_overlap": icu_overlap,
        "source_counts": source_counts,
        "occupancy": occupancy,
    }

    json_path = out_dir / "hosp_density_probe.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", json_path)

    md_path = out_dir / "hosp_density_probe.md"
    md_path.write_text(render_markdown(payload))
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
