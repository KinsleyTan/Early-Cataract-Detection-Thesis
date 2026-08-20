"""Subject-level 5-fold x 3-repeat CV for frozen ROI-224 EfficientNetB0.

Only the existing training and validation workbooks form the development pool.
The fixed test split is never loaded, predicted, or used for any decision.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from data import Sample, build_dataset, load_metadata, select_samples
from metrics import calculate_metrics, plot_confusion, plot_training_history
from model import build_model, compile_model, parameter_counts
from roi import roi_box_for_dimensions
from utils import PROJECT_ROOT, load_config, project_path, sha256_file


REPEAT_SEEDS = (2026, 2027, 2028)
N_FOLDS = 5
BASE_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"
ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "repeated_cv_roi224"
SUMMARY = ROOT / "summary"
EXPECTED_INTERPRETER = Path.home() / ".venvs" / "cat-screen-tf217-gpu" / "bin" / "python"
EXPECTED_TF = "2.17.1"
EXPECTED_GPU = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_ROI = (928, 424, 3105, 2601)
METRICS = ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc")
AGGREGATE_METRICS = METRICS + ("fn",)

# These controls must exist before TensorFlow is imported by the gate.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def environment_gate() -> dict[str, Any]:
    import tensorflow as tf

    if Path(sys.executable) != EXPECTED_INTERPRETER:
        raise RuntimeError(f"Wrong interpreter: {sys.executable}; expected {EXPECTED_INTERPRETER}")
    if tf.__version__ != EXPECTED_TF:
        raise RuntimeError(f"Wrong TensorFlow: {tf.__version__}; expected {EXPECTED_TF}")
    if not tf.test.is_built_with_cuda():
        raise RuntimeError("TensorFlow CUDA build is false")
    kernel = platform.release()
    if "microsoft-standard-WSL2" not in kernel:
        raise RuntimeError(f"WSL2 not confirmed: {kernel}")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus or gpus[0].name != "/physical_device:GPU:0":
        raise RuntimeError(f"GPU:0 unavailable: {gpus}")
    details = tf.config.experimental.get_device_details(gpus[0])
    if details.get("device_name") != EXPECTED_GPU:
        raise RuntimeError(f"Unexpected GPU: {details}")
    with tf.device("/GPU:0"):
        result = tf.linalg.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    checksum = float(tf.reduce_sum(result).numpy())
    if "GPU:0" not in result.device.upper() or checksum != 4096.0:
        raise RuntimeError(f"GPU operation not confirmed: {result.device}, {checksum}")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    os_release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    build = tf.sysconfig.get_build_info()
    return {
        "wsl2_confirmed": True,
        "ubuntu": os_release.get("PRETTY_NAME"),
        "kernel": kernel,
        "interpreter": sys.executable,
        "tensorflow_version": tf.__version__,
        "cuda_build": True,
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu": details.get("device_name"),
        "physical_device": gpus[0].name,
        "operation_device": result.device,
        "operation_checksum": checksum,
        "nvidia_smi": smi,
    }


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def cv_config(seed: int) -> dict[str, Any]:
    """Return the frozen config with the test split removed from visibility."""
    cfg = copy.deepcopy(load_config(BASE_CONFIG))
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["name"] = f"repeated_subject_cv_roi224_seed_{seed}"
    cfg["fixed_splits"] = {
        key: value for key, value in cfg["fixed_splits"].items() if key in ("train", "validation")
    }
    if set(cfg["fixed_splits"]) != {"train", "validation"}:
        raise RuntimeError("Development configuration must expose train and validation only")
    return cfg


def development_samples() -> list[Sample]:
    cfg = cv_config(REPEAT_SEEDS[0])
    samples: list[Sample] = []
    for source in ("train", "validation"):
        samples.extend(select_samples(cfg, load_metadata(cfg, source)))
    return samples


def protected_manifest() -> dict[str, str]:
    """Hash prior outputs plus development workbooks; exclude this new root."""
    items: dict[str, str] = {}
    outputs = PROJECT_ROOT / "outputs"
    if outputs.exists():
        for path in sorted(outputs.rglob("*")):
            if not path.is_file() or ROOT in path.parents:
                continue
            items[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    cfg = cv_config(REPEAT_SEEDS[0])
    dataset = project_path(cfg["paths"]["dataset_root"])
    for split, spec in cfg["fixed_splits"].items():
        path = dataset / spec["workbook"]
        items[f"../Fixed Dataset/Clean/{spec['workbook']}"] = sha256_file(path)
    return items


def verify_integrity(*, save: bool) -> dict[str, Any]:
    before_path = SUMMARY / "protected_artifact_hashes_before.json"
    if not before_path.is_file():
        raise RuntimeError("Missing protected-artifact manifest")
    before = read_json(before_path)
    after = protected_manifest()
    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    }
    result = {
        "pass": not changed,
        "protected_file_count_before": len(before),
        "protected_file_count_after": len(after),
        "changed_or_missing": changed,
    }
    if save:
        write_json(SUMMARY / "protected_artifact_integrity_after.json", result)
    if changed:
        raise RuntimeError(f"Protected artifacts changed: {list(changed)}")
    return result


def subject_map(samples: list[Sample]) -> dict[str, list[Sample]]:
    result: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        result[sample.subject_id].append(sample)
    return dict(result)


def construct_folds(samples: list[Sample]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_subject = subject_map(samples)
    subject_ids = np.asarray(sorted(by_subject), dtype=object)
    subject_labels: list[int] = []
    for subject_id in subject_ids:
        labels = {sample.label for sample in by_subject[str(subject_id)]}
        if len(labels) != 1:
            raise RuntimeError(f"Mixed-label subject {subject_id}: {labels}")
        subject_labels.append(next(iter(labels)))
    labels_array = np.asarray(subject_labels, dtype=int)
    rows: list[dict[str, Any]] = []
    audit_repeats: dict[str, Any] = {}
    for seed in REPEAT_SEEDS:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        repeat_audit: dict[str, Any] = {}
        seen_validation: Counter[str] = Counter()
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(subject_ids, labels_array), start=1
        ):
            train_subjects = set(subject_ids[train_index].tolist())
            validation_subjects = set(subject_ids[validation_index].tolist())
            overlap = sorted(train_subjects & validation_subjects)
            if overlap:
                raise RuntimeError(f"Subject leakage repeat={seed} fold={fold}: {overlap}")
            for subject_id in validation_subjects:
                seen_validation[str(subject_id)] += 1
            train_samples = [sample for sid in sorted(train_subjects) for sample in by_subject[str(sid)]]
            validation_samples = [sample for sid in sorted(validation_subjects) for sample in by_subject[str(sid)]]
            for subject_id in subject_ids:
                role = "validation" if subject_id in validation_subjects else "train"
                subject_samples = by_subject[str(subject_id)]
                rows.append(
                    {
                        "repeat_seed": seed,
                        "fold": fold,
                        "subject_id": subject_id,
                        "role": role,
                        "label": subject_samples[0].label,
                        "class_name": "Mild Cataract" if subject_samples[0].label else "Normal",
                        "image_count": len(subject_samples),
                        "filenames": " | ".join(sample.filename for sample in subject_samples),
                    }
                )
            repeat_audit[str(fold)] = {
                "subject_overlap": overlap,
                "train_subjects": len(train_subjects),
                "validation_subjects": len(validation_subjects),
                "train_subject_class_counts": dict(Counter(labels_array[train_index].tolist())),
                "validation_subject_class_counts": dict(Counter(labels_array[validation_index].tolist())),
                "train_images": len(train_samples),
                "validation_images": len(validation_samples),
                "train_image_class_counts": dict(Counter(s.label for s in train_samples)),
                "validation_image_class_counts": dict(Counter(s.label for s in validation_samples)),
            }
        bad_coverage = {sid: count for sid, count in seen_validation.items() if count != 1}
        if len(seen_validation) != len(subject_ids) or bad_coverage:
            raise RuntimeError(f"Invalid held-out coverage for repeat {seed}: {bad_coverage}")
        audit_repeats[str(seed)] = repeat_audit
    return rows, {
        "development_sources": ["train.xlsx", "val.xlsx"],
        "fixed_test_loaded": False,
        "images": len(samples),
        "subjects": len(by_subject),
        "image_class_counts": dict(Counter(sample.label for sample in samples)),
        "subject_class_counts": dict(Counter(subject_labels)),
        "mixed_label_subjects": [],
        "images_per_subject": dict(Counter(len(value) for value in by_subject.values())),
        "repeat_seeds": list(REPEAT_SEEDS),
        "folds_per_repeat": N_FOLDS,
        "total_runs": len(REPEAT_SEEDS) * N_FOLDS,
        "all_subject_overlap_checks_passed": True,
        "every_subject_held_out_once_per_repeat": True,
        "repeats": audit_repeats,
    }


def audit_protocol(samples: list[Sample]) -> dict[str, Any]:
    cfg = cv_config(REPEAT_SEEDS[0])
    if cfg["model"] != load_config(BASE_CONFIG)["model"]:
        raise RuntimeError("Model protocol differs from frozen ROI baseline")
    required = {
        "architecture": "EfficientNetB0",
        "weights": "imagenet",
        "include_top": False,
        "backbone_trainable": False,
        "dense_units": 64,
        "dropout_rate": 0.30,
        "l2_regularization": 1e-4,
    }
    for key, expected in required.items():
        if cfg["model"].get(key) != expected:
            raise RuntimeError(f"Model control mismatch {key}: {cfg['model'].get(key)}")
    training_expected = {
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "loss": "binary_crossentropy",
        "epochs": 30,
        "monitor": "val_loss",
        "threshold": 0.5,
        "class_weights": None,
        "early_stopping_patience": 6,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.2,
        "min_learning_rate": 1e-6,
        "fine_tuning": False,
    }
    for key, expected in training_expected.items():
        if cfg["training"].get(key) != expected:
            raise RuntimeError(f"Training control mismatch {key}: {cfg['training'].get(key)}")
    if cfg["data"]["image_size"] != [224, 224] or cfg["data"]["batch_size"] != 8:
        raise RuntimeError("Input size or batch size changed")
    expected_hashes: dict[str, str] = {}
    actual_hashes: dict[str, str] = {}
    dataset = project_path(cfg["paths"]["dataset_root"])
    for split, spec in cfg["fixed_splits"].items():
        expected_hashes[split] = str(spec["sha256"]).upper()
        actual_hashes[split] = sha256_file(dataset / spec["workbook"])
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"Development workbook hash mismatch: {actual_hashes}")
    sizes: Counter[str] = Counter()
    boxes: Counter[str] = Counter()
    for sample in samples:
        with Image.open(sample.image_path) as image:
            size = image.size
            box = roi_box_for_dimensions(*size, cfg["roi"]).as_tuple()
        sizes[str(size)] += 1
        boxes[str(box)] += 1
        if box != EXPECTED_ROI:
            raise RuntimeError(f"ROI mismatch for {sample.filename}: {box}")
    return {
        "controls_passed": True,
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "development_workbook_hashes": actual_hashes,
        "fixed_test_workbook_loaded": False,
        "image_dimensions": dict(sizes),
        "roi_boxes": dict(boxes),
        "model": required,
        "training": training_expected,
        "augmentation": cfg["augmentation"],
        "data": cfg["data"],
    }


def prepare() -> None:
    if ROOT.exists():
        raise RuntimeError(f"Refusing to reuse existing experiment directory: {ROOT}")
    environment = environment_gate()
    samples = development_samples()
    assignment_rows, split_audit = construct_folds(samples)
    protocol = audit_protocol(samples)
    SUMMARY.mkdir(parents=True)
    write_json(SUMMARY / "protected_artifact_hashes_before.json", protected_manifest())
    write_json(SUMMARY / "environment.json", environment)
    write_json(SUMMARY / "subject_level_split_audit.json", split_audit)
    write_json(SUMMARY / "protocol_audit.json", protocol)
    write_csv(SUMMARY / "all_fold_subject_assignments.csv", assignment_rows)
    print(json.dumps({"status": "READY", "environment": environment, "split_audit": split_audit}, indent=2))


def assignments_for(seed: int, fold: int) -> list[dict[str, str]]:
    rows = read_csv(SUMMARY / "all_fold_subject_assignments.csv")
    selected = [row for row in rows if int(row["repeat_seed"]) == seed and int(row["fold"]) == fold]
    if not selected:
        raise RuntimeError(f"Missing assignments for repeat={seed}, fold={fold}")
    return selected


def samples_for_assignments(
    all_samples: list[Sample], assignments: list[dict[str, str]], role: str
) -> list[Sample]:
    subjects = {row["subject_id"] for row in assignments if row["role"] == role}
    return [sample for sample in all_samples if sample.subject_id in subjects]


def fold_config(seed: int, fold: int, assignments: list[dict[str, str]]) -> dict[str, Any]:
    cfg = cv_config(seed)
    cfg.pop("_config_path", None)
    cfg["experiment"]["repeat_seed"] = seed
    cfg["experiment"]["fold"] = fold
    cfg["experiment"]["total_folds"] = N_FOLDS
    cfg["paths"] = {
        "dataset_root": "../Fixed Dataset/Clean",
        "output_dir": str((ROOT / f"repeat_{seed}" / f"fold_{fold}").relative_to(PROJECT_ROOT)),
        "keras_home": ".keras",
    }
    cfg["development_policy"] = {
        "sources": ["train.xlsx", "val.xlsx"],
        "fixed_test_excluded": True,
        "split_unit": "subject_id",
        "splitter": "StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)",
        "positive_class": "Mild Cataract",
        "selection": "minimum validation loss only",
        "roi_pixels": list(EXPECTED_ROI),
        "train_subjects": sorted(row["subject_id"] for row in assignments if row["role"] == "train"),
        "validation_subjects": sorted(row["subject_id"] for row in assignments if row["role"] == "validation"),
    }
    return cfg


def callbacks_for(output: Path):
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            output / "best_checkpoint.keras", monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.2, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(output / "training_history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def prediction_rows(
    samples: list[Sample], probabilities: np.ndarray, predicted: np.ndarray, seed: int, fold: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, probability, label in zip(samples, probabilities, predicted, strict=True):
        rows.append(
            {
                "filename": sample.filename,
                "subject_id": sample.subject_id,
                "true_label": sample.label,
                "mild_probability": f"{float(probability):.8f}",
                "predicted_label": int(label),
                "fold": fold,
                "repeat_seed": seed,
            }
        )
    return rows


def train_fold(seed: int, fold: int, all_samples: list[Sample]) -> None:
    import tensorflow as tf

    output = ROOT / f"repeat_{seed}" / f"fold_{fold}"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite fold directory: {output}")
    output.mkdir(parents=True)
    assignments = assignments_for(seed, fold)
    train_subjects = {row["subject_id"] for row in assignments if row["role"] == "train"}
    validation_subjects = {row["subject_id"] for row in assignments if row["role"] == "validation"}
    overlap = sorted(train_subjects & validation_subjects)
    if overlap:
        raise RuntimeError(f"Subject leakage immediately before training: {overlap}")
    train_samples = samples_for_assignments(all_samples, assignments, "train")
    validation_samples = samples_for_assignments(all_samples, assignments, "validation")
    if {sample.subject_id for sample in train_samples} & {sample.subject_id for sample in validation_samples}:
        raise RuntimeError("Sample-derived subject leakage detected")
    cfg = cv_config(seed)
    write_json(output / "config_used.json", fold_config(seed, fold, assignments))
    write_csv(output / "subject_assignments.csv", assignments)
    set_determinism(seed)
    train_data = build_dataset(train_samples, cfg, training=True)
    validation_data = build_dataset(validation_samples, cfg, training=False)
    model, backbone = build_model(cfg)
    compile_model(model, cfg)
    if backbone.trainable or any(layer.trainable for layer in backbone.layers):
        raise RuntimeError("Backbone is not fully frozen")
    trainable, non_trainable = parameter_counts(model)
    history_object = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=30,
        callbacks=callbacks_for(output),
        class_weight=None,
        shuffle=False,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    write_json(output / "training_history.json", history)
    plot_training_history(history, output)
    best_index = int(np.argmin(history["val_loss"]))
    overfit = bool(
        best_index < len(history["loss"]) - 1
        and history["val_loss"][-1] > history["val_loss"][best_index] * 1.05
        and history["loss"][-1] < history["loss"][best_index]
    )
    del model
    tf.keras.backend.clear_session()
    selected = tf.keras.models.load_model(output / "best_checkpoint.keras", compile=False)
    probabilities = selected.predict(validation_data, verbose=0).reshape(-1)
    true = np.asarray([sample.label for sample in validation_samples], dtype=int)
    calculated = calculate_metrics(true, probabilities, 0.5)
    predicted = np.asarray(calculated.pop("predicted_labels"), dtype=int)
    metrics = {
        **calculated,
        "repeat_seed": seed,
        "fold": fold,
        "positive_class": "Mild Cataract",
        "best_epoch": best_index + 1,
        "minimum_validation_loss": history["val_loss"][best_index],
        "fixed_test_accessed": False,
    }
    write_json(output / "validation_metrics.json", metrics)
    write_csv(
        output / "validation_predictions.csv",
        prediction_rows(validation_samples, probabilities, predicted, seed, fold),
    )
    plot_confusion(
        metrics,
        output / "validation_confusion_matrix.png",
        class_names=("Normal", "Mild Cataract"),
        title=f"Held-Out Validation: Repeat {seed}, Fold {fold}",
    )
    write_json(
        output / "training_summary.json",
        {
            "repeat_seed": seed,
            "fold": fold,
            "fixed_test_accessed": False,
            "selection": "minimum validation loss only",
            "train_subjects": len(train_subjects),
            "validation_subjects": len(validation_subjects),
            "subject_overlap": overlap,
            "train_images": len(train_samples),
            "validation_images": len(validation_samples),
            "train_image_class_counts": dict(Counter(sample.label for sample in train_samples)),
            "validation_image_class_counts": dict(Counter(sample.label for sample in validation_samples)),
            "epochs_requested": 30,
            "epochs_run": len(history["loss"]),
            "best_epoch": best_index + 1,
            "minimum_validation_loss": history["val_loss"][best_index],
            "overfitting_observed": overfit,
            "backbone_fully_frozen": True,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "validation_metrics": metrics,
        },
    )
    del selected
    tf.keras.backend.clear_session()


def run_all() -> None:
    environment_gate()
    verify_integrity(save=False)
    all_samples = development_samples()
    for seed in REPEAT_SEEDS:
        for fold in range(1, N_FOLDS + 1):
            print(f"\n===== REPEAT {seed}, FOLD {fold}/{N_FOLDS} =====", flush=True)
            train_fold(seed, fold, all_samples)
            verify_integrity(save=False)
        write_json(
            ROOT / f"repeat_{seed}" / "repeat_completed.json",
            {"repeat_seed": seed, "folds_completed": N_FOLDS, "fixed_test_accessed": False},
        )


def collect_fold_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in REPEAT_SEEDS:
        for fold in range(1, N_FOLDS + 1):
            metrics = read_json(ROOT / f"repeat_{seed}" / f"fold_{fold}" / "validation_metrics.json")
            training = read_json(ROOT / f"repeat_{seed}" / f"fold_{fold}" / "training_summary.json")
            rows.append(
                {
                    "repeat_seed": seed,
                    "fold": fold,
                    "validation_images": training["validation_images"],
                    "validation_subjects": training["validation_subjects"],
                    "best_epoch": metrics["best_epoch"],
                    "minimum_validation_loss": metrics["minimum_validation_loss"],
                    **{key: metrics[key] for key in METRICS},
                    **{key: metrics[key] for key in ("tn", "fp", "fn", "tp")},
                    "overfitting_observed": training["overfitting_observed"],
                }
            )
    return rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in AGGREGATE_METRICS:
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "descriptive_percentile_2_5": float(np.percentile(values, 2.5)),
            "descriptive_percentile_97_5": float(np.percentile(values, 97.5)),
        }
    return result


def per_repeat_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in REPEAT_SEEDS:
        selected = [row for row in rows if row["repeat_seed"] == seed]
        result.append(
            {
                "repeat_seed": seed,
                **{key: float(np.mean([row[key] for row in selected])) for key in AGGREGATE_METRICS},
            }
        )
    return result


def make_oof_predictions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_oof: list[dict[str, Any]] = []
    repeat_summary: list[dict[str, Any]] = []
    development = development_samples()
    expected = {(sample.filename, sample.subject_id) for sample in development}
    for seed in REPEAT_SEEDS:
        rows: list[dict[str, Any]] = []
        for fold in range(1, N_FOLDS + 1):
            rows.extend(read_csv(ROOT / f"repeat_{seed}" / f"fold_{fold}" / "validation_predictions.csv"))
        keys = [(row["filename"], row["subject_id"]) for row in rows]
        counts = Counter(keys)
        duplicates = [key for key, count in counts.items() if count != 1]
        if set(keys) != expected or duplicates or len(rows) != len(development):
            raise RuntimeError(f"Invalid OOF coverage for repeat {seed}: duplicates={duplicates}")
        rows.sort(key=lambda row: (int(row["fold"]), row["subject_id"], row["filename"]))
        write_csv(SUMMARY / f"oof_predictions_repeat_{seed}.csv", rows)
        true = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
        probability = np.asarray([float(row["mild_probability"]) for row in rows], dtype=float)
        metrics = calculate_metrics(true, probability, 0.5)
        metrics.pop("predicted_labels")
        repeat_summary.append({"repeat_seed": seed, **metrics})
        all_oof.extend(rows)
    write_csv(SUMMARY / "oof_metrics_by_repeat.csv", repeat_summary)
    return all_oof, repeat_summary


def error_analysis(all_oof: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_oof:
        by_subject[row["subject_id"]].append(row)
        by_file[(row["subject_id"], row["filename"])].append(row)
    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(by_subject.items()):
        labels = {int(row["true_label"]) for row in rows}
        if len(labels) != 1:
            raise RuntimeError(f"Mixed OOF labels for subject {subject_id}")
        errors = [row for row in rows if int(row["predicted_label"]) != int(row["true_label"])]
        repeat_errors = {
            int(row["repeat_seed"]) for row in errors
        }
        subject_rows.append(
            {
                "subject_id": subject_id,
                "true_label": next(iter(labels)),
                "class_name": "Mild Cataract" if next(iter(labels)) else "Normal",
                "unique_images": len({row["filename"] for row in rows}),
                "total_oof_predictions": len(rows),
                "errors": len(errors),
                "error_rate": len(errors) / len(rows),
                "repeats_with_any_error": len(repeat_errors),
                "misclassified_in_all_3_repeats": len(repeat_errors) == 3,
                "false_negative_predictions": sum(
                    int(row["true_label"]) == 1 and int(row["predicted_label"]) == 0 for row in rows
                ),
                "false_positive_predictions": sum(
                    int(row["true_label"]) == 0 and int(row["predicted_label"]) == 1 for row in rows
                ),
                "mean_mild_probability": float(np.mean([float(row["mild_probability"]) for row in rows])),
                "filenames": " | ".join(sorted({row["filename"] for row in rows})),
            }
        )
    mild_rows: list[dict[str, Any]] = []
    for (subject_id, filename), rows in sorted(by_file.items()):
        if int(rows[0]["true_label"]) != 1:
            continue
        false_negatives = sum(int(row["predicted_label"]) == 0 for row in rows)
        mild_rows.append(
            {
                "subject_id": subject_id,
                "filename": filename,
                "oof_predictions": len(rows),
                "false_negative_count": false_negatives,
                "false_negative_in_all_3_repeats": false_negatives == 3,
                "mean_mild_probability": float(np.mean([float(row["mild_probability"]) for row in rows])),
                "minimum_mild_probability": float(np.min([float(row["mild_probability"]) for row in rows])),
                "maximum_mild_probability": float(np.max([float(row["mild_probability"]) for row in rows])),
            }
        )
    subject_rows.sort(key=lambda row: (-row["errors"], row["subject_id"]))
    mild_rows.sort(key=lambda row: (-row["false_negative_count"], row["subject_id"], row["filename"]))
    write_csv(SUMMARY / "repeated_subject_error_analysis.csv", subject_rows)
    write_csv(SUMMARY / "repeated_mild_false_negative_analysis.csv", mild_rows)
    return subject_rows, mild_rows


def make_figures(rows: list[dict[str, Any]]) -> None:
    labels = {
        "roc_auc": "ROC-AUC",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1": "F1",
    }
    colors = {2026: "#1f77b4", 2027: "#ff7f0e", 2028: "#2ca02c"}
    for key, label in labels.items():
        values = np.asarray([row[key] for row in rows], dtype=float)
        fig, axis = plt.subplots(figsize=(5.5, 4.5))
        axis.boxplot(values, widths=0.35, showmeans=True)
        jitter = np.linspace(-0.08, 0.08, len(rows))
        axis.scatter(1 + jitter, values, c=[colors[row["repeat_seed"]] for row in rows], s=34, alpha=0.85)
        axis.set(title=f"{label} Across 15 Held-Out Folds", ylabel=label, xticks=[], ylim=(0, 1))
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(SUMMARY / f"{key}_distribution.png", dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5.2))
    x = np.arange(1, len(rows) + 1)
    for key, label, marker in (
        ("roc_auc", "ROC-AUC", "o"),
        ("sensitivity", "Sensitivity", "s"),
        ("specificity", "Specificity", "^"),
        ("f1", "F1", "D"),
    ):
        axis.plot(x, [row[key] for row in rows], marker=marker, linewidth=1.5, label=label)
    tick_labels = [f"{row['repeat_seed']}\nF{row['fold']}" for row in rows]
    axis.set(
        title="Held-Out Metrics by Repeat and Fold",
        xlabel="Repeat seed and fold",
        ylabel="Metric value",
        xticks=x,
        xticklabels=tick_labels,
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(SUMMARY / "per_fold_metrics.png", dpi=180)
    plt.close(fig)


def choose_verdict(aggregate: dict[str, Any]) -> str:
    key_metrics = ("sensitivity", "specificity", "roc_auc", "f1")
    values = [aggregate[key] for key in key_metrics]
    if any(not math.isfinite(item["standard_deviation"]) for item in values):
        return "REPEATED CV IS INCONCLUSIVE"
    if any(item["standard_deviation"] >= 0.15 or item["maximum"] - item["minimum"] >= 0.40 for item in values):
        return "ROI-224 PERFORMANCE IS HIGHLY VARIABLE"
    if all(item["standard_deviation"] <= 0.075 and item["maximum"] - item["minimum"] <= 0.25 for item in values):
        return "ROI-224 PERFORMANCE IS STABLE"
    return "ROI-224 PERFORMANCE IS MODERATELY VARIABLE"


def report_text(
    fold_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    repeat_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    subject_errors: list[dict[str, Any]],
    mild_errors: list[dict[str, Any]],
    verdict: str,
) -> str:
    env = read_json(SUMMARY / "environment.json")
    split = read_json(SUMMARY / "subject_level_split_audit.json")
    auc_above = sum(row["roc_auc"] > 0.5 for row in fold_rows)
    auc_equal = sum(abs(row["roc_auc"] - 0.5) <= 1e-12 for row in fold_rows)
    all_repeat_subject_errors = sum(row["misclassified_in_all_3_repeats"] for row in subject_errors)
    all_repeat_mild_fn = sum(row["false_negative_in_all_3_repeats"] for row in mild_errors)
    top_mild = [row for row in mild_errors if row["false_negative_count"] > 0][:10]
    lines = [
        "SUBJECT-LEVEL REPEATED STRATIFIED CV: FROZEN ROI-224 EFFICIENTNETB0",
        "=" * 88,
        "",
        "A. ENVIRONMENT",
        "-" * 88,
        f"WSL2 Ubuntu confirmed: yes ({env['ubuntu']}; kernel {env['kernel']})",
        f"Interpreter: {env['interpreter']}",
        f"TensorFlow: {env['tensorflow_version']}; CUDA build: {env['cuda_build']}",
        f"GPU: {env['gpu']} ({env['physical_device']})",
        f"TensorFlow operation device: {env['operation_device']}",
        "",
        "B. SUBJECT-LEVEL SPLIT AUDIT",
        "-" * 88,
        f"Development pool: {split['images']} images from {split['subjects']} subjects.",
        f"Image counts: Normal={split['image_class_counts']['0']}, Mild Cataract={split['image_class_counts']['1']}.",
        f"Subject counts: Normal={split['subject_class_counts']['0']}, Mild Cataract={split['subject_class_counts']['1']}.",
        "No subject had mixed class labels.",
        "All images from each subject remained together. Every train/validation overlap check passed.",
        "Every subject was held out exactly once per repeat.",
        "Only train.xlsx and val.xlsx formed the development pool.",
        "The current fixed test set was deliberately excluded and was never loaded, predicted,",
        "used for fold construction, training, validation, early stopping, or model selection.",
        "",
        "C. FOLD CONSTRUCTION AND FIXED MODEL PROTOCOL",
        "-" * 88,
        "StratifiedKFold was applied to 104 subject records with 5 folds, shuffle=True,",
        "and repeat seeds 2026, 2027, 2028, producing 15 predetermined held-out runs.",
        "Fixed ROI=(928,424,3105,2601); 224x224 RGB; ImageNet EfficientNetB0 fully frozen;",
        "GlobalAveragePooling; Dense 64 with L2=1e-4; Dropout=0.30; sigmoid output;",
        "Adam LR=1e-3; binary crossentropy; batch size 8; threshold=0.5; unchanged augmentation.",
        "Maximum epochs=30; EarlyStopping patience=6; ReduceLROnPlateau patience=3,",
        "factor=0.2, minimum LR=1e-6. Checkpoints were selected by validation loss only.",
        "No fine-tuning, class weighting, focal loss, threshold tuning, or architecture search was used.",
        "",
        "D. PER-FOLD HELD-OUT METRICS",
        "-" * 88,
        "Repeat Fold N  BestEpoch MinValLoss Acc Prec Sens Spec F1 AUC TN FP FN TP Overfit",
    ]
    for row in fold_rows:
        lines.append(
            f"{row['repeat_seed']} {row['fold']:>4} {row['validation_images']:>2} {row['best_epoch']:>10} "
            f"{row['minimum_validation_loss']:.5f} {row['accuracy']:.3f} {row['precision']:.3f} "
            f"{row['sensitivity']:.3f} {row['specificity']:.3f} {row['f1']:.3f} "
            f"{row['roc_auc']:.3f} {row['tn']} {row['fp']} {row['fn']} {row['tp']} {row['overfitting_observed']}"
        )
    lines.extend([
        "",
        "E. AGGREGATE ACROSS 15 FOLDS",
        "-" * 88,
        "Metric       Mean+/-SD      Median   Min     Max     Descriptive 2.5th-97.5th percentile",
    ])
    for key in AGGREGATE_METRICS:
        item = aggregate[key]
        lines.append(
            f"{key:<12} {item['mean']:.4f}+/-{item['standard_deviation']:.4f}  "
            f"{item['median']:.4f}  {item['minimum']:.4f}  {item['maximum']:.4f}  "
            f"[{item['descriptive_percentile_2_5']:.4f}, {item['descriptive_percentile_97_5']:.4f}]"
        )
    lines.extend([
        "The percentile range is descriptive, not a confidence interval; folds and repeats are dependent.",
        "",
        "F. PER-REPEAT MEAN FOLD METRICS",
        "-" * 88,
        "Repeat Accuracy Precision Sensitivity Specificity F1 ROC-AUC FN",
    ])
    for row in repeat_rows:
        lines.append(
            f"{row['repeat_seed']} {row['accuracy']:.4f} {row['precision']:.4f} "
            f"{row['sensitivity']:.4f} {row['specificity']:.4f} {row['f1']:.4f} "
            f"{row['roc_auc']:.4f} {row['fn']:.2f}"
        )
    lines.extend([
        "",
        "G. OUT-OF-FOLD RESULTS",
        "-" * 88,
        "Each repeat contains exactly one held-out prediction for every one of the 111 development images.",
        "Every prediction came from a model that trained on neither that image nor any image from its subject.",
        "Repeat Accuracy Precision Sensitivity Specificity F1 ROC-AUC TN FP FN TP",
    ])
    for row in oof_rows:
        lines.append(
            f"{row['repeat_seed']} {row['accuracy']:.4f} {row['precision']:.4f} "
            f"{row['sensitivity']:.4f} {row['specificity']:.4f} {row['f1']:.4f} "
            f"{row['roc_auc']:.4f} {row['tn']} {row['fp']} {row['fn']} {row['tp']}"
        )
    lines.extend([
        "",
        "H. REPEATED-SUBJECT ERROR ANALYSIS",
        "-" * 88,
        f"Subjects with at least one error in all three repeats: {all_repeat_subject_errors}/{len(subject_errors)}.",
        f"Mild image cases classified false-negative in all three repeats: {all_repeat_mild_fn}/{len(mild_errors)}.",
        "Most recurrent Mild false-negative cases (up to 10):",
    ])
    if top_mild:
        for row in top_mild:
            lines.append(
                f"- subject={row['subject_id']}, file={row['filename']}, FN repeats="
                f"{row['false_negative_count']}/3, mean p(Mild)={row['mean_mild_probability']:.4f}"
            )
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "I. STABILITY INTERPRETATION",
        "-" * 88,
        f"Sensitivity mean/SD/range: {aggregate['sensitivity']['mean']:.4f} / "
        f"{aggregate['sensitivity']['standard_deviation']:.4f} / "
        f"{aggregate['sensitivity']['minimum']:.4f}-{aggregate['sensitivity']['maximum']:.4f}.",
        f"Specificity mean/SD/range: {aggregate['specificity']['mean']:.4f} / "
        f"{aggregate['specificity']['standard_deviation']:.4f} / "
        f"{aggregate['specificity']['minimum']:.4f}-{aggregate['specificity']['maximum']:.4f}.",
        f"ROC-AUC mean/SD/range: {aggregate['roc_auc']['mean']:.4f} / "
        f"{aggregate['roc_auc']['standard_deviation']:.4f} / "
        f"{aggregate['roc_auc']['minimum']:.4f}-{aggregate['roc_auc']['maximum']:.4f}.",
        f"ROC-AUC was above 0.5 in {auc_above}/15 folds, exactly 0.5 in {auc_equal}/15, "
        f"and below 0.5 in {15-auc_above-auc_equal}/15.",
        "Near-chance versus consistently-above-chance behavior is interpreted descriptively; no",
        "independence assumption or formal significance claim is made for the repeated folds.",
        "Concentration of repeated errors and Mild false negatives is detailed in the two summary CSVs.",
        "",
        "J. LIMITATIONS",
        "-" * 88,
        "The dataset is small (111 development images from 104 subjects).",
        "Normal subjects are limited (47), constraining held-out specificity estimates.",
        "The ROI is a fixed heuristic crop rather than learned anatomical localization.",
        "Repeated cross-validation is still internal validation and does not establish external validity.",
        "The folds within and across repeats are statistically dependent.",
        "The current fixed test set was deliberately excluded from this entire experiment.",
        "",
        "FINAL VERDICT",
        "-" * 88,
        verdict,
        "",
        "ONE NEXT RESEARCH STEP (NOT IMPLEMENTED)",
        "-" * 88,
        "Evaluate the frozen ROI-224 pipeline once on a newly collected external subject-level cohort with adequate Normal representation.",
    ])
    return "\n".join(lines) + "\n"


def summarize() -> None:
    environment_gate()
    verify_integrity(save=False)
    fold_rows = collect_fold_metrics()
    aggregate = aggregate_metrics(fold_rows)
    repeat_rows = per_repeat_metrics(fold_rows)
    all_oof, oof_rows = make_oof_predictions()
    subject_errors, mild_errors = error_analysis(all_oof)
    verdict = choose_verdict(aggregate)
    write_csv(SUMMARY / "per_fold_metrics.csv", fold_rows)
    write_csv(SUMMARY / "per_repeat_mean_metrics.csv", repeat_rows)
    write_json(SUMMARY / "aggregate_statistics.json", aggregate)
    write_json(SUMMARY / "final_verdict.json", {"verdict": verdict})
    make_figures(fold_rows)
    report = report_text(
        fold_rows, aggregate, repeat_rows, oof_rows, subject_errors, mild_errors, verdict
    )
    (SUMMARY / "repeated_cv_report.txt").write_text(report, encoding="utf-8")
    integrity = verify_integrity(save=True)
    print(report)
    print(json.dumps({"integrity": integrity}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "integrity"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run_all()
    elif args.stage == "summarize":
        summarize()
    else:
        print(json.dumps(verify_integrity(save=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
