"""Count Clin-JEPA candidate fields and retained ICU events.

The script has two modes:

1. Inventory only (does not require MIMIC):
   python scripts/count_clin_jepa_events.py --inventory-only

2. Actual event statistics from Step 04 output:
   python scripts/count_clin_jepa_events.py --data-root /path/to/CLIN_JEPA_DATA

Actual statistics are computed from ``discretized/*.pkl`` and reproduce the
Step 06 state/action filters. This makes ``final_rows`` the closest auditable
count of event rows that are serialized into Qwen input text.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clin_jepa.data.helpers.action_filter import (
    ACTION_SOURCE_TABLES_TO_DROP,
    ACTION_VARIABLES_TO_DROP,
    filter_act_events,
)
from clin_jepa.data.helpers.event_ordering import _has_value
from clin_jepa.data.helpers.text_templates import DISPLAY_NAMES


DEFAULT_FEATURES = PROJECT_ROOT / "configs/data/features.yaml"
DEFAULT_TRAJECTORIES = PROJECT_ROOT / "configs/data/trajectories.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/clin_jepa_event_stats"


@dataclass
class Count:
    extracted_rows: int = 0
    final_rows: int = 0
    stays_extracted: int = 0
    stays_final: int = 0
    stay_hours_extracted: int = 0
    stay_hours_final: int = 0


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected_all_columns(source_name: str) -> list[str]:
    """Return repository-expected columns for a ``value_cols: all`` source.

    Runtime behavior remains dynamic: the extraction code reads every
    non-ID/time column found in the materialized mimic-code CSV.
    """
    return [variable for source, variable in DISPLAY_NAMES if source == source_name]


def build_inventory(features: dict, trajectories: dict) -> pd.DataFrame:
    rows: list[dict] = []
    excluded_categories = set(trajectories["state_text"]["exclude_categories"])
    excluded_variables = set(trajectories["state_text"]["exclude_variables"])

    for source_name, config in features["observations"].items():
        configured = config["value_cols"]
        variables = (
            expected_all_columns(source_name)
            if configured == "all"
            else list(configured)
        )
        for variable in variables:
            excluded = (
                config["category"] in excluded_categories
                or variable in excluded_variables
            )
            rows.append(
                {
                    "domain": "observation",
                    "source_table": source_name,
                    "category": config["category"],
                    "variable_name": variable,
                    "selection_rule": "all (runtime dynamic)"
                    if configured == "all"
                    else "explicit value_cols",
                    "final_text_status": "excluded" if excluded else "retained",
                    "exclusion_reason": "Step06 state leakage/metadata exclusion"
                    if excluded
                    else "",
                }
            )

    # Diagnosis is extracted as an open ICD vocabulary, then excluded in Step06.
    rows.append(
        {
            "domain": "observation",
            "source_table": "diagnoses",
            "category": "diagnosis",
            "variable_name": "<ICD-9/10 open vocabulary>",
            "selection_rule": "all cohort admission diagnoses",
            "final_text_status": "excluded",
            "exclusion_reason": "entire diagnosis category excluded in Step06",
        }
    )

    for source_name, config in features["actions"].items():
        configured = config.get("value_cols")
        if config.get("is_raw_table"):
            variables = ["<open vocabulary from raw table>"]
            rule = "raw names/item labels"
        else:
            variables = (
                expected_all_columns(source_name)
                if configured == "all"
                else list(configured or [])
            )
            rule = (
                "all (runtime dynamic)"
                if configured == "all"
                else "explicit value_cols"
            )

        for variable in variables:
            source_dropped = source_name in ACTION_SOURCE_TABLES_TO_DROP
            variable_dropped = (source_name, variable) in ACTION_VARIABLES_TO_DROP
            excluded = source_dropped or variable_dropped
            if source_dropped:
                reason = "entire source dropped by Step06 action filter"
            elif variable_dropped:
                reason = "variable dropped by Step06 action filter"
            else:
                reason = ""
            rows.append(
                {
                    "domain": "action",
                    "source_table": source_name,
                    "category": config["category"],
                    "variable_name": variable,
                    "selection_rule": rule,
                    "final_text_status": "excluded" if excluded else "retained",
                    "exclusion_reason": reason,
                }
            )

    return pd.DataFrame(rows)


def meaningful_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    mask = [
        _has_value(numeric, text)
        for numeric, text in zip(df["numeric_value"], df["text_value"])
    ]
    return df.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)


def update_counts(
    counts: dict[Hashable, Count],
    extracted: pd.DataFrame,
    final: pd.DataFrame,
    key_columns: list[str],
) -> None:
    def grouped(df: pd.DataFrame) -> dict[Hashable, tuple[int, int]]:
        if len(df) == 0:
            return {}
        result: dict[Hashable, tuple[int, int]] = {}
        grouper = key_columns[0] if len(key_columns) == 1 else key_columns
        for key, group in df.groupby(grouper, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            result[key] = (len(group), int(group["step_idx"].nunique()))
        return result

    extracted_groups = grouped(extracted)
    final_groups = grouped(final)
    for key in extracted_groups.keys() | final_groups.keys():
        counter = counts[key]
        if key in extracted_groups:
            rows, hours = extracted_groups[key]
            counter.extracted_rows += rows
            counter.stays_extracted += 1
            counter.stay_hours_extracted += hours
        if key in final_groups:
            rows, hours = final_groups[key]
            counter.final_rows += rows
            counter.stays_final += 1
            counter.stay_hours_final += hours


def counts_to_frame(
    counts: dict[Hashable, Count],
    key_columns: list[str],
    total_stays: int,
    total_hours: int,
) -> pd.DataFrame:
    rows = []
    for key, counter in counts.items():
        row = dict(zip(key_columns, key))
        row.update(asdict(counter))
        row["removed_rows"] = counter.extracted_rows - counter.final_rows
        row["stay_coverage_pct"] = (
            100.0 * counter.stays_final / total_stays if total_stays else 0.0
        )
        row["hour_coverage_pct"] = (
            100.0 * counter.stay_hours_final / total_hours if total_hours else 0.0
        )
        row["events_per_covered_stay"] = (
            counter.final_rows / counter.stays_final
            if counter.stays_final
            else 0.0
        )
        row["events_per_icu_day"] = (
            24.0 * counter.final_rows / total_hours if total_hours else 0.0
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["final_rows", *key_columns], ascending=[False, *([True] * len(key_columns))]
        ).reset_index(drop=True)
    return frame


def scan_discretized(
    discretized_dir: Path,
    trajectories: dict,
    max_stays: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pickle_files = sorted(discretized_dir.glob("*.pkl"))
    if max_stays is not None:
        pickle_files = pickle_files[:max_stays]
    if not pickle_files:
        raise FileNotFoundError(f"No Step04 .pkl files found in {discretized_dir}")

    excluded_categories = set(trajectories["state_text"]["exclude_categories"])
    excluded_variables = set(trajectories["state_text"]["exclude_variables"])

    field_counts: dict[Hashable, Count] = defaultdict(Count)
    source_counts: dict[Hashable, Count] = defaultdict(Count)
    total_hours = 0
    processed = 0

    for index, pickle_path in enumerate(pickle_files, start=1):
        with pickle_path.open("rb") as handle:
            stay = pickle.load(handle)  # noqa: S301 - trusted pipeline output

        obs = stay["obs_events"].copy()
        actions = stay["act_events"].copy()
        total_hours += int(stay["n_steps"])

        final_obs = obs.loc[
            ~obs["category"].isin(excluded_categories)
            & ~obs["variable_name"].isin(excluded_variables)
        ].reset_index(drop=True)
        final_obs = meaningful_rows(final_obs)

        final_actions = meaningful_rows(filter_act_events(actions))

        update_counts(
            field_counts,
            obs.assign(domain="observation"),
            final_obs.assign(domain="observation"),
            ["domain", "source_table", "category", "variable_name"],
        )
        update_counts(
            field_counts,
            actions.assign(domain="action"),
            final_actions.assign(domain="action"),
            ["domain", "source_table", "category", "variable_name"],
        )
        update_counts(
            source_counts,
            obs.assign(domain="observation"),
            final_obs.assign(domain="observation"),
            ["domain", "source_table", "category"],
        )
        update_counts(
            source_counts,
            actions.assign(domain="action"),
            final_actions.assign(domain="action"),
            ["domain", "source_table", "category"],
        )

        processed += 1
        if index % 5000 == 0:
            print(f"Processed {index:,}/{len(pickle_files):,} stays")

    field_frame = counts_to_frame(
        field_counts,
        ["domain", "source_table", "category", "variable_name"],
        processed,
        total_hours,
    )
    source_frame = counts_to_frame(
        source_counts,
        ["domain", "source_table", "category"],
        processed,
        total_hours,
    )
    summary = {
        "stays_scanned": processed,
        "icu_hours_scanned": total_hours,
        "icu_days_scanned": total_hours / 24.0,
        "final_observation_rows": int(
            field_frame.loc[field_frame["domain"] == "observation", "final_rows"].sum()
        ),
        "final_action_rows": int(
            field_frame.loc[field_frame["domain"] == "action", "final_rows"].sum()
        ),
        "counting_unit": (
            "event rows after Step04 hourly expansion and Step06 filters; "
            "repeated measurements are counted separately"
        ),
    }
    return field_frame, source_frame, summary


def write_inventory_summary(inventory: pd.DataFrame, output_dir: Path) -> None:
    observations = inventory[
        (inventory["domain"] == "observation")
        & (inventory["source_table"] != "diagnoses")
    ]
    derived_actions = inventory[
        (inventory["domain"] == "action")
        & (inventory["selection_rule"] != "raw names/item labels")
    ]
    raw_actions = inventory[
        (inventory["domain"] == "action")
        & (inventory["selection_rule"] == "raw names/item labels")
    ]
    lines = [
        "# Clin-JEPA 字段清单（无需 MIMIC 数据）",
        "",
        "该文件只统计代码配置中的候选字段，不是实际事件行数。",
        "",
        "## 配置字段数量",
        "",
        f"- observation 候选字段对：{len(observations)}",
        f"- observation 最终保留字段对："
        f"{(observations['final_text_status'] == 'retained').sum()}",
        "- diagnosis：开放 ICD 词表，但在 Step06 整类排除",
        f"- 派生 action 候选字段对：{len(derived_actions)}",
        f"- 派生 action 最终保留字段对："
        f"{(derived_actions['final_text_status'] == 'retained').sum()}",
        f"- raw action 来源：{len(raw_actions)} 个；其中最终保留 "
        f"{(raw_actions['final_text_status'] == 'retained').sum()} 个来源",
        "- raw action 最终唯一名称数：必须扫描实际数据",
        "- 另有 52 个精确动作噪声名称组合在 Step06 删除",
    ]
    lines.extend(
        [
            "",
            "## 重要口径",
            "",
            "- observation 的 `value_cols: all` 会随实际 mimic-code CSV 列变化。",
            "- raw action 是开放名称，无法仅由代码确定唯一词表大小。",
            "- GCS、尿量等实际行数和 stay 覆盖率必须扫描处理后的数据。",
        ]
    )
    (output_dir / "字段清单说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_actual_summary(
    field_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    focus_sources = {
        "gcs",
        "urine_output",
        "urine_output_rate",
        "kdigo_stages",
        "vitalsign",
        "sofa",
    }
    focus = field_frame[
        (field_frame["domain"] == "observation")
        & field_frame["source_table"].isin(focus_sources)
    ]
    lines = [
        "# Clin-JEPA ICU 实际事件数量统计",
        "",
        f"- 扫描 ICU stays：{summary['stays_scanned']:,}",
        f"- 扫描 ICU 小时：{summary['icu_hours_scanned']:,}",
        f"- 最终 observation 事件行：{summary['final_observation_rows']:,}",
        f"- 最终 action 事件行：{summary['final_action_rows']:,}",
        "",
        "事件行按实际记录计数；同小时同变量重复测量分别计数。",
        "",
        "## GCS、尿量、KDIGO 等重点字段",
        "",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"- `{row.source_table}.{row.variable_name}`："
            f"{row.final_rows:,} 行，覆盖 {row.stays_final:,} stays "
            f"({row.stay_coverage_pct:.2f}%)，"
            f"覆盖 {row.stay_hours_final:,} stay-hours"
        )

    lines.extend(["", "## 按来源汇总", ""])
    for row in source_frame.itertuples(index=False):
        lines.append(
            f"- {row.domain} / `{row.source_table}`："
            f"{row.final_rows:,} 行，覆盖 {row.stays_final:,} stays "
            f"({row.stay_coverage_pct:.2f}%)"
        )
    (output_dir / "实际事件统计摘要.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count Clin-JEPA candidate fields and actual retained events"
    )
    parser.add_argument("--data-root", type=Path, help="CLIN_JEPA_DATA directory")
    parser.add_argument(
        "--discretized-dir",
        type=Path,
        help="Direct Step04 discretized directory; overrides --data-root",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-stays", type=int, help="Optional smoke-test limit")
    parser.add_argument("--features-config", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--trajectories-config", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_yaml(args.features_config)
    trajectories = load_yaml(args.trajectories_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(features, trajectories)
    inventory.to_csv(args.output_dir / "clin_jepa_field_inventory.csv", index=False)
    write_inventory_summary(inventory, args.output_dir)
    print(f"Wrote field inventory to {args.output_dir}")

    if args.inventory_only:
        return

    if args.discretized_dir:
        discretized_dir = args.discretized_dir
    elif args.data_root:
        candidates = [
            args.data_root / "discretized",
            args.data_root / "04_discretized",
        ]
        discretized_dir = next((path for path in candidates if path.exists()), candidates[0])
    else:
        raise SystemExit(
            "Actual counts require --data-root or --discretized-dir. "
            "Use --inventory-only when MIMIC-derived outputs are unavailable."
        )

    field_frame, source_frame, summary = scan_discretized(
        discretized_dir, trajectories, args.max_stays
    )
    field_frame.to_csv(args.output_dir / "clin_jepa_event_counts_by_field.csv", index=False)
    source_frame.to_csv(args.output_dir / "clin_jepa_event_counts_by_source.csv", index=False)
    with (args.output_dir / "clin_jepa_event_count_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_actual_summary(field_frame, source_frame, summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
