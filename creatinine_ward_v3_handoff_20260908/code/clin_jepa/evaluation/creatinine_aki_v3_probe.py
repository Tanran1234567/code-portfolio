"""Downstream AKI classifiers for Creatinine Ward v3.

Features follow the v3 analysis plan:

* ``z_hist = mean(z_state[t-2:t]) || mean(z_action[t-2:t]) || z_static``
* ``z_full48 = z_hist || mean(predicted z[t+1], predicted z[t+2])``
* ``z_full72 = z_hist || mean(predicted z[t+1], predicted z[t+2], predicted z[t+3])``

The encoder and JEPA predictor are consumed as frozen inference artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clin_jepa.evaluation.rollout import load_predictor
from clin_jepa.utils import load_config


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def patient_equal_weights(subjects: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(subjects, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1.0 / count_map[int(subject)] for subject in subjects])


def load_embedding_split(root: Path, split: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted((root / split).glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        states = shard["z_state_embeddings"].float()
        actions = shard["z_action_embeddings"].float()
        statics = shard["z_statics"].float()
        for window in shard["windows"]:
            labels = {
                name: value.long()
                for name, value in window["labels"].items()
            }
            records.append({
                "subject_id": int(window["subject_id"]),
                "hadm_id": int(window["hadm_id"]),
                "state": states[window["emb_indices_state"].long()],
                "action": actions[window["emb_indices_action"].long()],
                "static": statics[int(window["static_idx"])],
                "time_hours": window["time_hours"].float(),
                "duration_hours": window["duration_hours"].float(),
                "labels": labels,
            })
    if not records:
        raise FileNotFoundError(f"no embedding shards in {root / split}")
    return records


def _rollout_future_pool(
    predictor,
    state: torch.Tensor,
    action: torch.Tensor,
    static: torch.Tensor,
    time_hours: torch.Tensor,
    duration_hours: torch.Tensor,
    anchor_idx: int,
    horizon: int,
    action_alignment: str,
    device: torch.device,
) -> np.ndarray:
    context = anchor_idx + 1
    z_sequence = state[:context].unsqueeze(0).to(device)
    static_batch = static.unsqueeze(0).to(device)
    time_sequence = time_hours[:context].unsqueeze(0).to(device)
    duration_sequence = duration_hours[:context].unsqueeze(0).to(device)
    if action_alignment == "target_interval":
        action_slice = slice(1, context + 1)
    elif action_alignment == "current_interval":
        action_slice = slice(0, context)
    else:
        raise ValueError(f"invalid action_alignment={action_alignment!r}")
    action_sequence = action[action_slice].unsqueeze(0).to(device)
    action_time_sequence = time_hours[action_slice].unsqueeze(0).to(device)
    action_duration_sequence = duration_hours[action_slice].unsqueeze(0).to(device)
    lengths = torch.full((1,), context, dtype=torch.long, device=device)
    predictions = []
    current_prediction = None
    for step in range(horizon):
        current_prediction = predictor(
            z_sequence,
            static_batch,
            action_sequence,
            lengths,
            return_last_only=True,
            time_hours=time_sequence,
            duration_hours=duration_sequence,
            action_time_hours=action_time_sequence,
            action_duration_hours=action_duration_sequence,
        )
        predictions.append(current_prediction.squeeze(0).detach().cpu())
        if step + 1 < horizon:
            z_sequence = torch.cat(
                [z_sequence, current_prediction.unsqueeze(1)], dim=1
            )
            next_action_index = (
                context + step + 1
                if action_alignment == "target_interval"
                else context + step
            )
            next_action = action[next_action_index].view(1, 1, -1).to(device)
            action_sequence = torch.cat([action_sequence, next_action], dim=1)
            next_time = time_hours[context + step].view(1, 1).to(device)
            time_sequence = torch.cat([time_sequence, next_time], dim=1)
            next_duration = duration_hours[context + step].view(1, 1).to(device)
            duration_sequence = torch.cat([duration_sequence, next_duration], dim=1)
            next_action_time = time_hours[next_action_index].view(1, 1).to(device)
            action_time_sequence = torch.cat(
                [action_time_sequence, next_action_time], dim=1
            )
            next_action_duration = duration_hours[next_action_index].view(1, 1).to(device)
            action_duration_sequence = torch.cat(
                [action_duration_sequence, next_action_duration], dim=1
            )
            lengths = lengths + 1
    return torch.stack(predictions).mean(dim=0).numpy()


@torch.no_grad()
def build_examples(
    records: list[dict],
    label_name: str,
    horizon_steps: int,
    feature_kind: str,
    predictor,
    action_alignment: str,
    device: torch.device,
    min_anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    xs = []
    ys = []
    subjects = []
    rows = []
    needs_future = feature_kind.startswith("full")
    for record in records:
        length = int(len(record["labels"][label_name]))
        max_anchor = length - horizon_steps - 1 if needs_future else length - 1
        for anchor_idx in range(min_anchor_idx, max_anchor + 1):
            y = int(record["labels"][label_name][anchor_idx])
            state_hist = record["state"][anchor_idx - 2:anchor_idx + 1].mean(dim=0).numpy()
            action_hist = record["action"][anchor_idx - 2:anchor_idx + 1].mean(dim=0).numpy()
            base = [state_hist, action_hist, record["static"].numpy()]
            if feature_kind == "history":
                feature = np.concatenate(base)
            elif feature_kind in {"full48", "full72"}:
                z_future = _rollout_future_pool(
                    predictor,
                    record["state"],
                    record["action"],
                    record["static"],
                    record["time_hours"],
                    record["duration_hours"],
                    anchor_idx,
                    horizon_steps,
                    action_alignment,
                    device,
                )
                feature = np.concatenate(base + [z_future])
            else:
                raise ValueError(f"unknown feature_kind={feature_kind!r}")
            xs.append(feature.astype(np.float32, copy=False))
            ys.append(y)
            subjects.append(record["subject_id"])
            cur_arr = record["labels"].get("label_scr_aki_current_rclass")
            current_rclass = 3 if cur_arr is None else int(cur_arr[anchor_idx])
            rows.append({
                "subject_id": record["subject_id"],
                "hadm_id": record["hadm_id"],
                "anchor_idx": anchor_idx,
                "anchor_time_hours": float(record["time_hours"][anchor_idx]),
                "raw_label": y,
                "current_rclass": current_rclass,
            })
    if not xs:
        raise ValueError(f"no examples for label={label_name} feature={feature_kind}")
    return (
        np.stack(xs),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
        pd.DataFrame(rows),
    )


def _prepare_target(
    y: np.ndarray,
    subjects: np.ndarray,
    rows: pd.DataFrame,
    target_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, dict]:
    include = target_cfg.get("include_classes")
    exclude = set(int(v) for v in target_cfg.get("exclude_classes", []))
    require_current = target_cfg.get("require_current_classes")
    mask = np.ones(len(y), dtype=bool)
    current = None
    if "current_rclass" in rows.columns:
        current = rows["current_rclass"].to_numpy(dtype=np.int64)
    if require_current is not None:
        if current is None:
            raise ValueError("require_current_classes needs current_rclass on rows")
        allowed = set(int(v) for v in require_current)
        mask &= np.asarray([int(v) in allowed for v in current], dtype=bool)
    if include is not None:
        include_set = set(int(v) for v in include)
        mask &= np.asarray([int(v) in include_set for v in y], dtype=bool)
    if exclude:
        mask &= np.asarray([int(v) not in exclude for v in y], dtype=bool)
    y2 = y[mask].copy()
    current_m = current[mask] if current is not None else None
    task_type = str(target_cfg.get("task_type", "binary"))
    if task_type == "worsening":
        if current_m is None:
            raise ValueError("worsening task needs current_rclass")
        y2 = np.asarray(
            [1 if int(fut) > int(cur) else 0 for fut, cur in zip(y2, current_m)],
            dtype=np.int64,
        )
        class_names = ["NO_WORSEN", "WORSEN"]
    elif task_type == "binary":
        positive = set(int(v) for v in target_cfg["positive_classes"])
        y2 = np.asarray([1 if int(v) in positive else 0 for v in y2], dtype=np.int64)
        class_names = list(target_cfg.get("binary_class_names") or ["NO_AKI", "AKI"])
    else:
        remap = {int(v): i for i, v in enumerate(sorted(set(y2.tolist())))}
        y2 = np.asarray([remap[int(v)] for v in y2], dtype=np.int64)
        configured = list(target_cfg.get("class_names", []))
        class_names = [
            configured[raw] if raw < len(configured) else str(raw)
            for raw in sorted(remap)
        ]
    meta = {
        "n_excluded": int((~mask).sum()),
        "class_names": class_names,
        "require_current_classes": require_current,
        "raw_class_counts_after_filter": {
            str(int(k)): int(v) for k, v in zip(*np.unique(y[mask], return_counts=True))
        },
        "current_class_counts_after_filter": (
            {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(current_m, return_counts=True))
            }
            if current_m is not None and len(current_m)
            else {}
        ),
    }
    return y2, subjects[mask], rows.loc[mask].reset_index(drop=True), mask, meta


def _metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    class_names: list[str],
) -> dict:
    pred = probabilities.argmax(axis=1)
    result = {
        "n": int(len(y)),
        "n_patients": int(len(np.unique(subjects))),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(len(class_names)))
        ).tolist(),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(class_names))), zero_division=0
    )
    result["per_class"] = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    if len(class_names) == 2 and len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probabilities[:, 1]))
        result["auprc"] = float(average_precision_score(y, probabilities[:, 1]))
    return result


def _bootstrap(
    y: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    class_names: list[str],
    reps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    patients = np.unique(subjects)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        draw = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == p) for p in draw])
        if len(np.unique(y[indices])) < min(2, len(class_names)):
            continue
        metrics = _metrics(y[indices], probabilities[indices], subjects[indices], class_names)
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc"):
            if key in metrics:
                samples[key].append(metrics[key])
    return {
        f"{key}_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in samples.items()
        if values
    }


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float],
    max_iter: int,
) -> tuple[object, float]:
    weights = patient_equal_weights(train_subjects)
    best = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value),
                max_iter=int(max_iter),
                solver="lbfgs",
                class_weight="balanced",
            ),
        )
        model.fit(x_train, y_train, logisticregression__sample_weight=weights)
        score = f1_score(y_val, model.predict(x_val), average="macro")
        if best is None or score > best[0]:
            best = (float(score), float(c_value), model)
    assert best is not None
    return best[2], best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = _expand(yaml.safe_load(handle))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    paths["output"].mkdir(parents=True, exist_ok=True)

    records = {
        split: load_embedding_split(paths["embeddings"], split)
        for split in ("train", "val", "test")
    }
    pretrain_config = load_config(paths["pretrain_config"])
    action_alignment = str(
        pretrain_config["predictor"].get("action_alignment", "target_interval")
    )
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    predictor = load_predictor(
        paths["predictor"],
        device,
        model_config=pretrain_config["predictor"],
    )

    results = {
        "schema_version": "creatinine_aki_v3_probe",
        "feature_definitions": {
            "history": "mean(state[t-2:t]) || mean(action[t-2:t]) || static",
            "full48": "history || mean(pred_z[t+1], pred_z[t+2])",
            "full72": "history || mean(pred_z[t+1], pred_z[t+2], pred_z[t+3])",
        },
        "action_alignment": action_alignment,
        "targets": {},
    }
    c_values = [float(v) for v in config["probe"]["c_values"]]
    max_iter = int(config["probe"].get("max_iter", 3000))
    reps = int(config["metrics"].get("bootstrap_replicates", 200))
    seed = int(config["metrics"].get("bootstrap_seed", 42))
    min_anchor_idx = int(config["anchor"].get("min_anchor_idx", 2))

    example_cache = {}

    for target_cfg in config["targets"]:
        target_name = str(target_cfg["name"])
        label_name = str(target_cfg["label_name"])
        horizon_steps = int(target_cfg["horizon_steps"])
        feature_kinds = list(target_cfg["feature_kinds"])
        target_out = paths["output"] / target_name
        target_out.mkdir(parents=True, exist_ok=True)
        results["targets"][target_name] = {}
        for feature_kind in feature_kinds:
            cache_key = (label_name, horizon_steps, feature_kind)
            if cache_key not in example_cache:
                example_cache[cache_key] = {
                    split: build_examples(
                        records[split],
                        label_name,
                        horizon_steps,
                        feature_kind,
                        predictor,
                        action_alignment,
                        device,
                        min_anchor_idx,
                    )
                    for split in ("train", "val", "test")
                }
            built = example_cache[cache_key]
            prepared = {}
            metas = {}
            for split, (x, y, subjects, rows) in built.items():
                y2, subjects2, rows2, mask, meta = _prepare_target(
                    y, subjects, rows, target_cfg
                )
                prepared[split] = (x[mask], y2, subjects2, rows2)
                metas[split] = meta
            x_train, y_train, train_subjects, _ = prepared["train"]
            x_val, y_val, _val_subjects, _ = prepared["val"]
            x_test, y_test, test_subjects, test_rows = prepared["test"]
            class_names = metas["train"]["class_names"]
            model, selected_c = fit_probe(
                x_train, y_train, train_subjects, x_val, y_val, c_values, max_iter
            )
            probabilities = model.predict_proba(x_test)
            metrics = _metrics(y_test, probabilities, test_subjects, class_names)
            metrics.update(_bootstrap(
                y_test, probabilities, test_subjects, class_names,
                reps, seed + horizon_steps,
            ))
            subset_metrics = {}
            for subset_name, subset_cfg in (target_cfg.get("report_current_subsets") or {}).items():
                allowed = set(int(v) for v in subset_cfg["current"])
                subset_mask = test_rows["current_rclass"].isin(allowed).to_numpy()
                if int(subset_mask.sum()) < 20 or len(np.unique(y_test[subset_mask])) < 2:
                    subset_metrics[subset_name] = {
                        "n": int(subset_mask.sum()),
                        "skipped": True,
                    }
                    continue
                sub = _metrics(
                    y_test[subset_mask],
                    probabilities[subset_mask],
                    test_subjects[subset_mask],
                    class_names,
                )
                sub.update(_bootstrap(
                    y_test[subset_mask],
                    probabilities[subset_mask],
                    test_subjects[subset_mask],
                    class_names,
                    reps,
                    seed + horizon_steps + 17,
                ))
                subset_metrics[subset_name] = sub
            feature_out = target_out / feature_kind
            feature_out.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, feature_out / "linear_probe.joblib")
            pred_frame = test_rows.assign(
                y_true=y_test,
                y_pred=probabilities.argmax(axis=1),
            )
            if probabilities.shape[1] == 2:
                pred_frame["y_score"] = probabilities[:, 1]
            pred_frame.to_csv(feature_out / "test_predictions.csv", index=False)
            results["targets"][target_name][feature_kind] = {
                "selected_C": selected_c,
                "feature_dim": int(x_train.shape[1]),
                "class_names": class_names,
                "filter_meta": metas,
                "test": metrics,
                "test_current_subsets": subset_metrics,
            }
    (paths["output"] / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("CREATININE_AKI_V3_PROBE_OK", paths["output"] / "results.json")


if __name__ == "__main__":
    main()
