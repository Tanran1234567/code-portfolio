"""Current-state and 24/48h creatinine probes for the ICU-free ward pilot."""

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
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

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


def load_embedding_split(root: Path, split: str, label_name: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted((root / split).glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        states = shard["z_state_embeddings"].float()
        actions = shard["z_action_embeddings"].float()
        statics = shard["z_statics"].float()
        for window in shard["windows"]:
            labels = window["labels"][label_name].long()
            if not bool(((labels >= 0) & (labels <= 2)).all()):
                raise ValueError(f"invalid {label_name} in {path}")
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
        raise FileNotFoundError(f"no embedding windows in {root / split}")
    return records


def flatten_nodes(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.cat([record["state"] for record in records]).numpy()
    y = torch.cat([record["labels"] for record in records]).numpy()
    subjects = np.concatenate([
        np.full(len(record["labels"]), record["subject_id"], dtype=np.int64)
        for record in records
    ])
    return x, y, subjects


def patient_equal_weights(subjects: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(subjects, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1.0 / count_map[int(subject)] for subject in subjects])


def metric_dict(
    y: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    weights: np.ndarray | None = None,
) -> dict:
    prediction = probabilities.argmax(axis=1)
    result = {
        "n": int(len(y)),
        "n_patients": None,
        "accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", sample_weight=weights)
        ),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2], sample_weight=weights)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "class_prevalence": {
            name: float(np.mean(y == index))
            for index, name in enumerate(class_names)
        },
        "predicted_class_fraction": {
            name: float(np.mean(prediction == index))
            for index, name in enumerate(class_names)
        },
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1, 2], zero_division=0
    )
    result["per_class"] = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    binary = label_binarize(y, classes=[0, 1, 2])
    try:
        result["macro_ovr_auroc"] = float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        )
        result["macro_ovr_auprc"] = float(
            average_precision_score(binary, probabilities, average="macro")
        )
    except ValueError:
        result["macro_ovr_auroc"] = None
        result["macro_ovr_auprc"] = None
    measured_y = (y != 2).astype(int)
    measured_p = 1.0 - probabilities[:, 2]
    result["measured_vs_not"] = {
        "auroc": float(roc_auc_score(measured_y, measured_p)),
        "auprc": float(average_precision_score(measured_y, measured_p)),
        "f1": float(f1_score(measured_y, measured_p >= 0.5)),
    }
    measured_mask = y != 2
    if bool(measured_mask.any()) and len(np.unique(y[measured_mask])) == 2:
        high_y = (y[measured_mask] == 0).astype(int)
        denom = probabilities[measured_mask, 0] + probabilities[measured_mask, 1]
        high_p = probabilities[measured_mask, 0] / np.maximum(denom, 1e-8)
        result["high_vs_not_high_given_measured"] = {
            "auroc": float(roc_auc_score(high_y, high_p)),
            "auprc": float(average_precision_score(high_y, high_p)),
            "f1": float(f1_score(high_y, high_p >= 0.5)),
        }
    return result


def patient_bootstrap(
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
        indices = np.concatenate([
            np.flatnonzero(subjects == patient) for patient in draw
        ])
        if len(np.unique(y[indices])) < 3:
            continue
        metrics = metric_dict(y[indices], probabilities[indices], class_names)
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            samples[key].append(metrics[key])
    return {
        f"{key}_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in samples.items() if values
    }


def fit_probe(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    c_values: list[float],
    max_iter: int,
) -> tuple[object, float]:
    x_train, y_train, train_subjects = train
    x_val, y_val, _ = val
    train_weights = patient_equal_weights(train_subjects)
    best: tuple[float, float, object] | None = None
    for c_value in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value), max_iter=max_iter, solver="lbfgs"
            ),
        )
        model.fit(
            x_train, y_train,
            logisticregression__sample_weight=train_weights,
        )
        score = f1_score(y_val, model.predict(x_val), average="macro")
        if best is None or score > best[0]:
            best = (float(score), float(c_value), model)
    assert best is not None
    return best[2], best[1]


@torch.no_grad()
def rollout_records(
    model,
    records: list[dict],
    horizon: int,
    device: torch.device,
    action_alignment: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted/actual/current vectors and future labels by anchor."""
    grouped: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    for record in records:
        length = len(record["labels"])
        for context in range(1, length - horizon + 1):
            grouped[context].append((record, context))

    predicted: list[torch.Tensor] = []
    actual: list[torch.Tensor] = []
    current: list[torch.Tensor] = []
    labels: list[int] = []
    subjects: list[int] = []
    for context, anchors in sorted(grouped.items()):
        for offset in range(0, len(anchors), batch_size):
            batch = anchors[offset:offset + batch_size]
            z_sequence = torch.stack([
                record["state"][:context] for record, _ in batch
            ]).to(device)
            if action_alignment == "target_interval":
                action_slice = slice(1, context + 1)
            elif action_alignment == "current_interval":
                action_slice = slice(0, context)
            else:
                raise ValueError(f"invalid action_alignment={action_alignment!r}")
            action_sequence = torch.stack([
                record["action"][action_slice] for record, _ in batch
            ]).to(device)
            static = torch.stack([
                record["static"] for record, _ in batch
            ]).to(device)
            time_sequence = torch.stack([
                record["time_hours"][:context] for record, _ in batch
            ]).to(device)
            duration_sequence = torch.stack([
                record["duration_hours"][:context] for record, _ in batch
            ]).to(device)
            action_time_sequence = torch.stack([
                record["time_hours"][action_slice] for record, _ in batch
            ]).to(device)
            action_duration_sequence = torch.stack([
                record["duration_hours"][action_slice] for record, _ in batch
            ]).to(device)
            lengths = torch.full(
                (len(batch),), context, dtype=torch.long, device=device
            )
            prediction = None
            for step in range(horizon):
                prediction = model(
                    z_sequence, static, action_sequence, lengths,
                    return_last_only=True,
                    time_hours=time_sequence,
                    duration_hours=duration_sequence,
                    action_time_hours=action_time_sequence,
                    action_duration_hours=action_duration_sequence,
                )
                if step + 1 < horizon:
                    z_sequence = torch.cat(
                        [z_sequence, prediction.unsqueeze(1)], dim=1
                    )
                    next_action_index = (
                        context + step + 1
                        if action_alignment == "target_interval"
                        else context + step
                    )
                    next_action = torch.stack([
                        record["action"][next_action_index]
                        for record, _ in batch
                    ]).to(device)
                    action_sequence = torch.cat(
                        [action_sequence, next_action.unsqueeze(1)], dim=1
                    )
                    next_time = torch.stack([
                        record["time_hours"][context + step]
                        for record, _ in batch
                    ]).to(device)
                    time_sequence = torch.cat(
                        [time_sequence, next_time.unsqueeze(1)], dim=1
                    )
                    next_duration = torch.stack([
                        record["duration_hours"][context + step]
                        for record, _ in batch
                    ]).to(device)
                    duration_sequence = torch.cat(
                        [duration_sequence, next_duration.unsqueeze(1)], dim=1
                    )
                    next_action_time = torch.stack([
                        record["time_hours"][next_action_index]
                        for record, _ in batch
                    ]).to(device)
                    action_time_sequence = torch.cat(
                        [action_time_sequence, next_action_time.unsqueeze(1)], dim=1
                    )
                    next_action_duration = torch.stack([
                        record["duration_hours"][next_action_index]
                        for record, _ in batch
                    ]).to(device)
                    action_duration_sequence = torch.cat(
                        [action_duration_sequence, next_action_duration.unsqueeze(1)],
                        dim=1,
                    )
                    lengths = lengths + 1
            assert prediction is not None
            predicted.append(prediction.cpu())
            actual.append(torch.stack([
                record["state"][context + horizon - 1]
                for record, _ in batch
            ]))
            current.append(torch.stack([
                record["state"][context - 1] for record, _ in batch
            ]))
            labels.extend([
                int(record["labels"][context + horizon - 1])
                for record, _ in batch
            ])
            subjects.extend([record["subject_id"] for record, _ in batch])
    return (
        torch.cat(predicted).numpy(),
        torch.cat(actual).numpy(),
        torch.cat(current).numpy(),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = _expand(yaml.safe_load(handle))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    if args.embeddings:
        paths["embeddings"] = Path(args.embeddings)
    if args.output:
        paths["output"] = Path(args.output)
    class_names = list(config["label"]["class_names"])
    label_name = str(config["label"]["name"])
    records = {
        split: load_embedding_split(paths["embeddings"], split, label_name)
        for split in ("train", "val", "test")
    }
    flattened = {split: flatten_nodes(value) for split, value in records.items()}
    probe, selected_c = fit_probe(
        flattened["train"], flattened["val"],
        [float(value) for value in config["probe"]["c_values"]],
        int(config["probe"].get("max_iter", 3000)),
    )
    x_test, y_test, test_subjects = flattened["test"]
    current_probabilities = probe.predict_proba(x_test)
    current_metrics = metric_dict(
        y_test, current_probabilities, class_names,
        patient_equal_weights(test_subjects),
    )
    current_metrics["n_patients"] = int(len(np.unique(test_subjects)))
    current_metrics.update(patient_bootstrap(
        y_test, current_probabilities, test_subjects, class_names,
        int(config["metrics"].get("bootstrap_replicates", 500)),
        int(config["metrics"].get("bootstrap_seed", 42)),
    ))

    rollout_rows: list[dict] = []
    if not args.skip_rollout:
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        predictor_config = load_config(paths["pretrain_config"])
        action_alignment = str(
            predictor_config["predictor"].get(
                "action_alignment", "current_interval"
            )
        )
        predictor = load_predictor(
            paths["predictor"], device,
            model_config=predictor_config["predictor"],
        )
        for horizon in config["rollout"].get("horizons_steps", [1, 2]):
            pred, actual, current, future_y, future_subjects = rollout_records(
                predictor, records["test"], int(horizon), device,
                action_alignment=action_alignment,
            )
            candidates = {
                "actual_future_encoder_ceiling": probe.predict_proba(actual),
                "predictor_native_rollout": probe.predict_proba(pred),
                "persistence": probe.predict_proba(current),
            }
            for name, probabilities in candidates.items():
                row = metric_dict(future_y, probabilities, class_names)
                row.update({
                    "horizon_steps": int(horizon),
                    "horizon_hours": int(horizon) * 24,
                    "model": name,
                    "n_patients": int(len(np.unique(future_subjects))),
                })
                row.update(patient_bootstrap(
                    future_y, probabilities, future_subjects, class_names,
                    int(config["metrics"].get("bootstrap_replicates", 500)),
                    int(config["metrics"].get("bootstrap_seed", 42)) + int(horizon),
                ))
                if name == "predictor_native_rollout":
                    row["latent_l1_mean"] = float(np.abs(pred - actual).sum(axis=1).mean())
                    cosine = np.sum(pred * actual, axis=1) / (
                        np.linalg.norm(pred, axis=1) * np.linalg.norm(actual, axis=1)
                        + 1e-8
                    )
                    row["latent_cosine_distance_mean"] = float(np.mean(1.0 - cosine))
                    row["predicted_norm_mean"] = float(
                        np.linalg.norm(pred, axis=1).mean()
                    )
                    row["actual_norm_mean"] = float(
                        np.linalg.norm(actual, axis=1).mean()
                    )
                    row["predicted_mean_feature_std"] = float(
                        np.std(pred, axis=0).mean()
                    )
                    row["actual_mean_feature_std"] = float(
                        np.std(actual, axis=0).mean()
                    )
                elif name == "persistence":
                    row["latent_l1_mean"] = float(
                        np.abs(current - actual).sum(axis=1).mean()
                    )
                    cosine = np.sum(current * actual, axis=1) / (
                        np.linalg.norm(current, axis=1)
                        * np.linalg.norm(actual, axis=1) + 1e-8
                    )
                    row["latent_cosine_distance_mean"] = float(
                        np.mean(1.0 - cosine)
                    )
                rollout_rows.append(row)

    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, output / "linear_probe.joblib")
    result = {
        "schema_version": "creatinine_probe_v1",
        "selected_C": selected_c,
        "class_names": class_names,
        "current_test": current_metrics,
        "rollout": rollout_rows,
        "action_alignment": (
            action_alignment if not args.skip_rollout else None
        ),
        "future_action_semantics": (
            "target-interval recorded-action conditional rollout"
            if not args.skip_rollout and action_alignment == "target_interval"
            else "current-interval action conditional rollout"
        ),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rollout_rows
    ]).to_csv(output / "rollout_metrics.csv", index=False)
    print("CREATININE_PROBE_OK", output / "results.json")


if __name__ == "__main__":
    main()
