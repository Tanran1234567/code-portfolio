"""Select and freeze a hospital checkpoint using full validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from clin_jepa.utils import load_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(_sha256_file(child).encode())
    return digest.hexdigest()


def _score(summary: dict, horizon: str, stratum: str, metric: str) -> float:
    if metric == "balanced_measured_mase":
        icu = summary["metrics"][horizon]["icu_measured"].get("mase")
        ward = summary["metrics"][horizon]["ward_measured"].get("mase")
        if icu is None or ward is None:
            raise ValueError(
                f"Balanced 24h MASE requires non-empty measured ICU and WARD "
                f"strata at horizon={horizon}; got icu={icu}, ward={ward}"
            )
        return (float(icu) + float(ward)) / 2.0
    try:
        value = summary["metrics"][horizon][stratum][metric]
    except KeyError as exc:
        raise KeyError(
            f"Selection metric metrics.{horizon}.{stratum}.{metric} is absent"
        ) from exc
    if value is None:
        raise ValueError(
            f"Selection metric is null for horizon={horizon}, stratum={stratum}"
        )
    return float(value)


def select_checkpoint(config: dict, config_path: Path) -> dict:
    selection = config["selection"]
    verify_report = Path(config["evaluation"]["data_verify_report"])
    if not verify_report.exists():
        raise FileNotFoundError(
            f"Patient/time/schema verification report is required: {verify_report}"
        )
    horizon = str(selection.get("horizon_hours", 24))
    stratum = selection.get("stratum", "all")
    metric = selection.get("metric", "model_mean_l1")
    candidates = selection.get("candidates") or list(config["plans"])
    scored: list[dict] = []

    for plan in candidates:
        plan_cfg = config["plans"][plan]
        summary_path = Path(plan_cfg["output_dir"]) / "val" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing full-validation result for {plan}: {summary_path}"
            )
        with summary_path.open() as f:
            summary = json.load(f)
        if summary.get("split") != "val" or not summary.get("is_full_split"):
            raise ValueError(
                f"{summary_path} is not a full validation evaluation; "
                "remove --max_windows and rerun it"
            )
        gate = summary.get("encoder_collapse_gate", {})
        if not gate.get("passed"):
            raise ValueError(
                f"{summary_path} failed encoder collapse gate: {gate}"
            )
        scored.append({
            "plan": plan,
            "score": _score(summary, horizon, stratum, metric),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256_file(summary_path),
        })

    scored.sort(key=lambda item: (item["score"], item["plan"]))
    selected_plan = scored[0]["plan"]
    selected_cfg = config["plans"][selected_plan]
    predictor = Path(selected_cfg["predictor_checkpoint"]).resolve()
    encoder = Path(selected_cfg["encoder_checkpoint"]).resolve()
    manifest = {
        "schema_version": "medin_hosp_checkpoint_selection_v1",
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_plan": selected_plan,
        "selection_split": "val",
        "selection_metric": {
            "horizon_hours": horizon,
            "stratum": stratum,
            "metric": metric,
            "definition": (
                "equal-weight mean of measured ICU and measured WARD MASE"
                if metric == "balanced_measured_mase" else None
            ),
            "direction": "minimize",
        },
        "candidate_scores": scored,
        "selected_artifacts": {
            "predictor_checkpoint": str(predictor),
            "predictor_sha256": _sha256_path(predictor),
            "encoder_checkpoint": str(encoder),
            "encoder_sha256": _sha256_path(encoder),
        },
        "evaluation_config": str(config_path.resolve()),
        "evaluation_config_sha256": _sha256_file(config_path),
        "data_verify_report": str(verify_report.resolve()),
        "data_verify_report_sha256": _sha256_file(verify_report),
        "test_has_been_read": False,
    }
    output = Path(config["evaluation"]["selection_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose best/latest/final on full hospital validation"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    manifest = select_checkpoint(load_config(config_path), config_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
