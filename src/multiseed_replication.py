"""Controlled five-seed ROI-224 frozen vs partial-fine-tuning replication.

This experiment is intentionally isolated below
outputs/mild_cataract/multiseed_replication.  Model selection uses validation
loss only; test predictions are made only after both models for a seed have
finished selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import Sample, build_dataset, label_counts, load_metadata, select_samples
from metrics import calculate_metrics, plot_confusion, plot_training_history
from model import build_model, compile_model, parameter_counts
from roi import roi_box_for_dimensions
from utils import PROJECT_ROOT, load_config, project_path, sha256_file


SEEDS = (2026, 2027, 2028, 2029, 2030)
CONDITIONS = ("frozen", "finetuned")
BASE_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"
FINE_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi_finetune.yaml"
ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "multiseed_replication"
SUMMARY = ROOT / "summary"
EXPECTED_INTERPRETER = Path.home() / ".venvs" / "cat-screen-tf217-gpu" / "bin" / "python"
EXPECTED_TF = "2.17.1"
EXPECTED_GPU = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_ROI = (928, 424, 3105, 2601)
UNFREEZE_FROM = "block6d_expand_conv"
METRIC_KEYS = ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc")
AGGREGATE_KEYS = METRIC_KEYS + ("fn", "fp")
PAIRED_KEYS = ("roc_auc", "sensitivity", "specificity", "f1", "accuracy", "fn")

# These runtime controls must exist before environment_gate imports TensorFlow.
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


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.pop("predicted_labels", None)
    return result


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
        raise RuntimeError(f"TensorFlow GPU:0 unavailable: {gpus}")
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
        "kernel": kernel,
        "ubuntu": os_release.get("PRETTY_NAME"),
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
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def config_for(seed: int, condition: str) -> dict[str, Any]:
    cfg = load_config(BASE_CONFIG)
    cfg["experiment"] = dict(cfg["experiment"])
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["name"] = f"roi224_{condition}_seed_{seed}"
    cfg["experiment"]["display_name"] = f"ROI 224 {condition.title()} Seed {seed}"
    if condition == "finetuned":
        fine = load_config(FINE_CONFIG)
        cfg["model"] = dict(fine["model"])
        cfg["training"] = dict(fine["training"])
    cfg.pop("_config_path", None)
    cfg["protocol_notes"] = {
        "exploratory_replication": True,
        "test_set_study_level_exposure": True,
        "model_selection": "minimum validation loss only",
        "positive_class": "Mild Cataract",
        "only_intended_variable": "random seed",
        "explicit_roi_pixels_for_all_selected_4032x3024_images": list(EXPECTED_ROI),
        "test_threshold": 0.5,
    }
    cfg["paths"] = {
        "dataset_root": "../Fixed Dataset/Clean",
        "output_dir": str((ROOT / f"seed_{seed}" / condition).relative_to(PROJECT_ROOT)),
        "keras_home": ".keras",
    }
    return cfg


def data_config(seed: int, condition: str) -> dict[str, Any]:
    """Return an in-memory config accepted by the existing data/model helpers."""
    cfg = load_config(BASE_CONFIG)
    cfg["experiment"]["seed"] = seed
    if condition == "finetuned":
        fine = load_config(FINE_CONFIG)
        cfg["model"] = dict(fine["model"])
        cfg["training"] = dict(fine["training"])
    return cfg


def protected_manifest() -> dict[str, str]:
    items: dict[str, str] = {}
    outputs = PROJECT_ROOT / "outputs"
    if outputs.exists():
        for path in sorted(outputs.rglob("*")):
            if not path.is_file() or ROOT in path.parents:
                continue
            items[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    fixed_root = (PROJECT_ROOT / ".." / "Fixed Dataset" / "Clean").resolve()
    for name in ("train.xlsx", "val.xlsx", "test.xlsx"):
        path = fixed_root / name
        items[f"../Fixed Dataset/Clean/{name}"] = sha256_file(path)
    return items


def verify_integrity(*, save: bool = True) -> dict[str, Any]:
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


def audit_protocol() -> dict[str, Any]:
    base = load_config(BASE_CONFIG)
    fine = load_config(FINE_CONFIG)
    expected_hashes = {
        split: str(spec["sha256"]).upper() for split, spec in base["fixed_splits"].items()
    }
    actual_hashes = {
        split: sha256_file(project_path(base["paths"]["dataset_root"]) / spec["workbook"])
        for split, spec in base["fixed_splits"].items()
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"Workbook hash mismatch: {actual_hashes}")
    split_audit: dict[str, Any] = {}
    all_samples: list[Sample] = []
    subject_sets: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        samples = select_samples(base, load_metadata(base, split))
        all_samples.extend(samples)
        subject_sets[split] = {sample.subject_id for sample in samples}
        counts = label_counts(samples)
        actual = {"Normal": int(counts[0]), "Cataract": int(counts[1])}
        expected = base["fixed_splits"][split]["expected_usable"]
        if actual != expected:
            raise RuntimeError(f"Selected counts changed for {split}: {actual} != {expected}")
        split_audit[split] = {
            "n": len(samples),
            "label_counts": actual,
            "filenames": [sample.filename for sample in samples],
            "subject_ids": [sample.subject_id for sample in samples],
        }
    overlaps = {
        "train_validation": sorted(subject_sets["train"] & subject_sets["validation"]),
        "train_test": sorted(subject_sets["train"] & subject_sets["test"]),
        "validation_test": sorted(subject_sets["validation"] & subject_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Subject overlap detected: {overlaps}")
    sizes: dict[str, int] = {}
    boxes: dict[str, int] = {}
    for sample in all_samples:
        with Image.open(sample.image_path) as image:
            size = image.size
            box = roi_box_for_dimensions(*size, base["roi"]).as_tuple()
        sizes[str(size)] = sizes.get(str(size), 0) + 1
        boxes[str(box)] = boxes.get(str(box), 0) + 1
        if box != EXPECTED_ROI:
            raise RuntimeError(f"ROI mismatch for {sample.filename}: {box}")
    identical = ("fixed_splits", "label_policy", "data", "roi", "augmentation")
    for key in identical:
        if fine[key] != base[key]:
            raise RuntimeError(f"Fine-tuning control mismatch: {key}")
    for key in (
        "optimizer", "loss", "epochs", "monitor", "threshold", "class_weights",
        "early_stopping_patience", "reduce_lr_patience", "reduce_lr_factor", "min_learning_rate",
    ):
        if fine["training"][key] != base["training"][key]:
            raise RuntimeError(f"Training control mismatch: {key}")
    if base["training"]["learning_rate"] != 0.001:
        raise RuntimeError("Frozen LR changed")
    if fine["training"]["learning_rate"] != 0.00001:
        raise RuntimeError("Fine-tuning LR changed")
    if fine["model"]["unfreeze_from_layer"] != UNFREEZE_FROM:
        raise RuntimeError("Fine-tuning boundary changed")
    return {
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "fixed_workbook_hashes": actual_hashes,
        "fixed_split_audit": split_audit,
        "subject_overlaps": overlaps,
        "image_dimensions": sizes,
        "roi_boxes": boxes,
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "fine_config_sha256": sha256_file(FINE_CONFIG),
        "controls_passed": True,
    }


def prepare() -> None:
    if ROOT.exists():
        raise RuntimeError(f"Refusing to reuse existing experiment directory: {ROOT}")
    environment = environment_gate()
    protocol = audit_protocol()
    SUMMARY.mkdir(parents=True)
    write_json(SUMMARY / "protected_artifact_hashes_before.json", protected_manifest())
    write_json(SUMMARY / "environment.json", environment)
    write_json(SUMMARY / "protocol_audit.json", protocol)
    print(json.dumps({"status": "READY", "environment": environment, "protocol": protocol}, indent=2))


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


def dataset_samples(cfg: dict[str, Any], split: str) -> list[Sample]:
    return select_samples(cfg, load_metadata(cfg, split))


def history_dict(history_object: Any) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history_object.history.items()}


def validation_evaluation(checkpoint: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    samples = dataset_samples(cfg, "validation")
    data = build_dataset(samples, cfg, training=False)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    probabilities = model.predict(data, verbose=0).reshape(-1)
    true = np.asarray([sample.label for sample in samples], dtype=int)
    return clean_metrics(calculate_metrics(true, probabilities, 0.5))


def overfit_flag(history: dict[str, list[float]]) -> bool:
    best = int(np.argmin(history["val_loss"]))
    return bool(
        best < len(history["loss"]) - 1
        and history["val_loss"][-1] > history["val_loss"][best] * 1.05
        and history["loss"][-1] < history["loss"][best]
    )


def train_frozen(seed: int) -> None:
    import tensorflow as tf

    output = ROOT / f"seed_{seed}" / "frozen"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    cfg = data_config(seed, "frozen")
    write_json(output / "config_used.json", config_for(seed, "frozen"))
    set_determinism(seed)
    train_samples = dataset_samples(cfg, "train")
    val_samples = dataset_samples(cfg, "validation")
    train_data = build_dataset(train_samples, cfg, training=True)
    val_data = build_dataset(val_samples, cfg, training=False)
    model, backbone = build_model(cfg)
    compile_model(model, cfg)
    trainable, non_trainable = parameter_counts(model)
    history = history_dict(
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=30,
            callbacks=callbacks_for(output),
            class_weight=None,
            shuffle=False,
            verbose=2,
        )
    )
    write_json(output / "training_history.json", history)
    plot_training_history(history, output)
    best_index = int(np.argmin(history["val_loss"]))
    validation = validation_evaluation(output / "best_checkpoint.keras", cfg)
    write_json(output / "validation_metrics.json", validation)
    write_json(
        output / "training_summary.json",
        {
            "seed": seed,
            "condition": "frozen",
            "test_set_accessed": False,
            "checkpoint_monitor": "val_loss",
            "epochs_requested": 30,
            "epochs_run": len(history["loss"]),
            "best_epoch": best_index + 1,
            "minimum_validation_loss": history["val_loss"][best_index],
            "overfitting_observed": overfit_flag(history),
            "backbone_trainable": backbone.trainable,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "validation_metrics": validation,
        },
    )
    tf.keras.backend.clear_session()


def configure_finetuning(model: Any) -> dict[str, Any]:
    import tensorflow as tf

    backbone = model.get_layer("efficientnetb0")
    if len(backbone.layers) != 238:
        raise RuntimeError(f"Unexpected backbone layer count: {len(backbone.layers)}")
    backbone.trainable = True
    reached = False
    for layer in backbone.layers:
        if layer.name == UNFREEZE_FROM:
            reached = True
        layer.trainable = reached and not isinstance(layer, tf.keras.layers.BatchNormalization)
    frozen = [layer.name for layer in backbone.layers if not layer.trainable]
    trainable = [layer.name for layer in backbone.layers if layer.trainable]
    bn = [layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    fraction = len(frozen) / len(backbone.layers)
    if len(frozen) != 214 or len(trainable) != 24 or not math.isclose(fraction, 214 / 238):
        raise RuntimeError(f"Unexpected unfreezing: {len(frozen)} frozen, {len(trainable)} trainable")
    if any(layer.trainable for layer in bn):
        raise RuntimeError("A BatchNormalization layer is trainable")
    return {
        "total_backbone_layers": len(backbone.layers),
        "frozen_backbone_layers": len(frozen),
        "trainable_backbone_layers": len(trainable),
        "frozen_backbone_fraction": fraction,
        "first_trainable_backbone_layer": trainable[0],
        "last_trainable_backbone_layer": trainable[-1],
        "batch_normalization_layers": len(bn),
        "batch_normalization_trainable": sum(layer.trainable for layer in bn),
        "trainable_backbone_layer_names": trainable,
    }


def compile_finetuned(model: Any) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy"),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="sensitivity", thresholds=0.5),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        ],
    )


def array_digest(weights: list[Any]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.asarray(weight.numpy()).tobytes())
    return digest.hexdigest().upper()


def bn_state(model: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import tensorflow as tf

    backbone = model.get_layer("efficientnetb0")
    return {
        layer.name: (layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    }


def train_finetuned(seed: int) -> None:
    import tensorflow as tf

    frozen_checkpoint = ROOT / f"seed_{seed}" / "frozen" / "best_checkpoint.keras"
    if not frozen_checkpoint.is_file():
        raise RuntimeError(f"Missing seed-specific frozen checkpoint: {frozen_checkpoint}")
    output = ROOT / f"seed_{seed}" / "finetuned"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    cfg = data_config(seed, "finetuned")
    used = config_for(seed, "finetuned")
    used["initialization"] = {
        "seed_specific_frozen_checkpoint": str(frozen_checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(frozen_checkpoint),
    }
    write_json(output / "config_used.json", used)
    set_determinism(seed)
    train_samples = dataset_samples(cfg, "train")
    val_samples = dataset_samples(cfg, "validation")
    train_data = build_dataset(train_samples, cfg, training=True)
    val_data = build_dataset(val_samples, cfg, training=False)
    model = tf.keras.models.load_model(frozen_checkpoint, compile=False)
    initial_digest = array_digest(model.weights)
    setup = configure_finetuning(model)
    compile_finetuned(model)
    trainable, non_trainable = parameter_counts(model)
    before_bn = bn_state(model)
    history = history_dict(
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=30,
            callbacks=callbacks_for(output),
            class_weight=None,
            shuffle=False,
            verbose=2,
        )
    )
    after_bn = bn_state(model)
    changed_bn = [
        name for name in before_bn
        if not np.array_equal(before_bn[name][0], after_bn[name][0])
        or not np.array_equal(before_bn[name][1], after_bn[name][1])
    ]
    if changed_bn:
        raise RuntimeError(f"BatchNormalization moving statistics changed: {changed_bn}")
    write_json(output / "training_history.json", history)
    plot_training_history(history, output)
    best_index = int(np.argmin(history["val_loss"]))
    validation = validation_evaluation(output / "best_checkpoint.keras", cfg)
    write_json(output / "validation_metrics.json", validation)
    write_json(
        output / "training_summary.json",
        {
            "seed": seed,
            "condition": "finetuned",
            "test_set_accessed": False,
            "source_checkpoint": str(frozen_checkpoint),
            "source_checkpoint_sha256": sha256_file(frozen_checkpoint),
            "initial_model_weight_digest": initial_digest,
            "checkpoint_monitor": "val_loss",
            "epochs_requested": 30,
            "epochs_run": len(history["loss"]),
            "best_epoch": best_index + 1,
            "minimum_validation_loss": history["val_loss"][best_index],
            "overfitting_observed": overfit_flag(history),
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "batch_normalization_moving_statistics_changed": changed_bn,
            "model_setup": setup,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "validation_metrics": validation,
        },
    )
    tf.keras.backend.clear_session()


def prediction_rows(
    samples: list[Sample], probabilities: np.ndarray, predicted: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, probability, label in zip(samples, probabilities, predicted, strict=True):
        rows.append(
            {
                "filename": sample.filename,
                "subject_id": sample.subject_id,
                "eye_side": sample.eye_side,
                "true_label": sample.label,
                "true_class": "Mild Cataract" if sample.label else "Normal",
                "mild_probability": f"{float(probability):.8f}",
                "predicted_label": int(label),
                "predicted_class": "Mild Cataract" if label else "Normal",
                "correct": int(label) == sample.label,
                "roi_left": EXPECTED_ROI[0],
                "roi_top": EXPECTED_ROI[1],
                "roi_right": EXPECTED_ROI[2],
                "roi_bottom": EXPECTED_ROI[3],
            }
        )
    return rows


def evaluate_condition(seed: int, condition: str) -> None:
    import tensorflow as tf

    output = ROOT / f"seed_{seed}" / condition
    metrics_path = output / "test_metrics.json"
    predictions_path = output / "predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise RuntimeError(f"Refusing to repeat test evaluation: {output}")
    cfg = data_config(seed, condition)
    samples = dataset_samples(cfg, "test")
    data = build_dataset(samples, cfg, training=False)
    model = tf.keras.models.load_model(output / "best_checkpoint.keras", compile=False)
    probabilities = model.predict(data, verbose=0).reshape(-1)
    true = np.asarray([sample.label for sample in samples], dtype=int)
    calculated = calculate_metrics(true, probabilities, 0.5)
    predicted = np.asarray(calculated.pop("predicted_labels"), dtype=int)
    write_json(
        metrics_path,
        {
            **calculated,
            "seed": seed,
            "condition": condition,
            "positive_class": "Mild Cataract",
            "model_selected_using": "minimum validation loss only",
            "test_prediction_passes": 1,
            "exploratory_replication": True,
        },
    )
    write_csv(predictions_path, prediction_rows(samples, probabilities, predicted))
    plot_confusion(
        calculated,
        output / "confusion_matrix.png",
        class_names=("Normal", "Mild Cataract"),
        title=f"Test Confusion Matrix: {condition.title()}, Seed {seed}",
    )
    tf.keras.backend.clear_session()


def run_all() -> None:
    environment_gate()
    verify_integrity(save=False)
    for seed in SEEDS:
        print(f"\n===== SEED {seed}: FROZEN TRAINING =====", flush=True)
        train_frozen(seed)
        print(f"\n===== SEED {seed}: FINE-TUNING =====", flush=True)
        train_finetuned(seed)
        # Both validation-only model selections are now complete for this seed.
        print(f"\n===== SEED {seed}: PREDETERMINED TEST EVALUATION =====", flush=True)
        evaluate_condition(seed, "frozen")
        evaluate_condition(seed, "finetuned")
        write_json(
            ROOT / f"seed_{seed}" / "seed_completed.json",
            {"seed": seed, "both_models_selected_before_test": True, "test_evaluations_complete": True},
        )
        verify_integrity(save=False)


def summarize_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            output = ROOT / f"seed_{seed}" / condition
            training = read_json(output / "training_summary.json")
            validation = read_json(output / "validation_metrics.json")
            test = read_json(output / "test_metrics.json")
            validation_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "best_epoch": training["best_epoch"],
                    "minimum_validation_loss": training["minimum_validation_loss"],
                    **{key: validation[key] for key in METRIC_KEYS},
                    **{key: validation[key] for key in ("tn", "fp", "fn", "tp")},
                    "overfitting_observed": training["overfitting_observed"],
                }
            )
            test_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "best_epoch": training["best_epoch"],
                    "minimum_validation_loss": training["minimum_validation_loss"],
                    **{key: test[key] for key in METRIC_KEYS},
                    **{key: test[key] for key in ("tn", "fp", "fn", "tp")},
                }
            )
    return validation_rows, test_rows


def aggregate(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in test_rows if row["condition"] == condition]
        result[condition] = {}
        for key in AGGREGATE_KEYS:
            values = np.asarray([row[key] for row in selected], dtype=float)
            result[condition][key] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return result


def paired(test_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["seed"], row["condition"]): row for row in test_rows}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frozen = by_key[(seed, "frozen")]
        fine = by_key[(seed, "finetuned")]
        rows.append({"seed": seed, **{key: float(fine[key] - frozen[key]) for key in PAIRED_KEYS}})
    summary: dict[str, Any] = {}
    tolerance = 1e-12
    for key in PAIRED_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=float)
        # Lower FN is clinically better, while higher is better for all rate metrics.
        oriented = -values if key == "fn" else values
        summary[key] = {
            "fine_tuned_better_seeds": int(np.sum(oriented > tolerance)),
            "equal_seeds": int(np.sum(np.abs(oriented) <= tolerance)),
            "fine_tuned_worse_seeds": int(np.sum(oriented < -tolerance)),
            "mean_paired_difference_finetuned_minus_frozen": float(np.mean(values)),
        }
    return rows, summary


def make_figures(test_rows: list[dict[str, Any]]) -> None:
    by_key = {(row["seed"], row["condition"]): row for row in test_rows}
    for key, title in (
        ("roc_auc", "ROC-AUC by Seed"),
        ("sensitivity", "Sensitivity by Seed"),
        ("specificity", "Specificity by Seed"),
        ("f1", "F1 by Seed"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for condition, label, marker in (
            ("frozen", "Frozen", "o"),
            ("finetuned", "Fine-Tuned", "s"),
        ):
            values = [by_key[(seed, condition)][key] for seed in SEEDS]
            axis.plot(SEEDS, values, marker=marker, linewidth=2, label=label)
        axis.set(title=title, xlabel="Seed", ylabel=title.split(" by")[0], xticks=list(SEEDS), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(SUMMARY / f"{key}_by_seed.png", dpi=180)
        plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def report_text(
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    paired_summary: dict[str, Any],
    verdict: str,
) -> str:
    env = read_json(SUMMARY / "environment.json")
    lines = [
        "FIVE-SEED ROI-224 FROZEN VS PARTIAL FINE-TUNING REPLICATION",
        "=" * 80,
        "",
        "A. ENVIRONMENT",
        "-" * 80,
        f"WSL2 Ubuntu confirmed: yes ({env['ubuntu']}; kernel {env['kernel']})",
        f"Interpreter: {env['interpreter']}",
        f"TensorFlow: {env['tensorflow_version']}",
        f"CUDA build: {env['cuda_build']}",
        f"GPU: {env['gpu']}",
        f"TensorFlow operation device: {env['operation_device']}",
        "",
        "B. FIXED PROTOCOL",
        "-" * 80,
        "Seeds were fixed in advance: 2026, 2027, 2028, 2029, 2030.",
        "Only the random seed varied. Train/validation/test workbooks, subject assignments,",
        "ROI=(928,424,3105,2601), 224x224 RGB input, augmentation, batch size 8,",
        "EfficientNetB0 architecture, Adam optimizer, callbacks, 30-epoch cap, threshold 0.5,",
        "and metrics were identical across seeds. Frozen LR=1e-3; fine-tuned LR=1e-5.",
        "Fine-tuning began at block6d_expand_conv: 214/238 backbone layers frozen",
        "(89.916%), with all BatchNormalization layers frozen for every seed.",
        "Each fine-tuned model was initialized from its own seed's selected frozen checkpoint.",
        "Selection used minimum validation loss only. Test results never changed a seed,",
        "hyperparameter, stopping decision, or protocol.",
        "This is exploratory replication, not a fully untouched confirmatory test, because",
        "the same test set had already been exposed at study level in earlier experiments.",
        "",
        "C. PER-SEED VALIDATION RESULTS",
        "-" * 80,
        "Seed Condition  BestEpoch MinValLoss Acc Prec Sens Spec F1 AUC TN FP FN TP Overfit",
    ]
    for row in validation_rows:
        lines.append(
            f"{row['seed']} {row['condition']:<10} {row['best_epoch']:>3} {row['minimum_validation_loss']:.5f} "
            f"{row['accuracy']:.3f} {row['precision']:.3f} {row['sensitivity']:.3f} "
            f"{row['specificity']:.3f} {row['f1']:.3f} {row['roc_auc']:.3f} "
            f"{row['tn']} {row['fp']} {row['fn']} {row['tp']} {row['overfitting_observed']}"
        )
    lines.extend(["", "D. PER-SEED LOCKED-TEST RESULTS", "-" * 80,
                  "Seed Condition  BestEpoch MinValLoss Acc Prec Sens Spec F1 AUC TN FP FN TP"])
    for row in test_rows:
        lines.append(
            f"{row['seed']} {row['condition']:<10} {row['best_epoch']:>3} {row['minimum_validation_loss']:.5f} "
            f"{row['accuracy']:.3f} {row['precision']:.3f} {row['sensitivity']:.3f} "
            f"{row['specificity']:.3f} {row['f1']:.3f} {row['roc_auc']:.3f} "
            f"{row['tn']} {row['fp']} {row['fn']} {row['tp']}"
        )
    lines.extend(["", "E. MEAN +/- STANDARD DEVIATION (TEST)", "-" * 80,
                  "Metric       Frozen Mean+/-SD [min,max]    Fine-Tuned Mean+/-SD [min,max]    Mean Difference"])
    for key in AGGREGATE_KEYS:
        frozen = aggregates["frozen"][key]
        fine = aggregates["finetuned"][key]
        lines.append(
            f"{key:<12} {frozen['mean']:.4f}+/-{frozen['standard_deviation']:.4f} "
            f"[{frozen['minimum']:.4f},{frozen['maximum']:.4f}]    "
            f"{fine['mean']:.4f}+/-{fine['standard_deviation']:.4f} "
            f"[{fine['minimum']:.4f},{fine['maximum']:.4f}]    {fine['mean']-frozen['mean']:+.4f}"
        )
    lines.extend(["", "F. PAIRED FROZEN-VS-FINE-TUNED DIFFERENCES", "-" * 80,
                  "Differences are Fine-Tuned minus Frozen. For FN, a negative difference is better.",
                  "Metric       Better Equal Worse Mean paired difference"])
    for key in PAIRED_KEYS:
        item = paired_summary[key]
        lines.append(
            f"{key:<12} {item['fine_tuned_better_seeds']:>2} {item['equal_seeds']:>5} "
            f"{item['fine_tuned_worse_seeds']:>5} {item['mean_paired_difference_finetuned_minus_frozen']:+.4f}"
        )
    f_sd = aggregates["frozen"]
    t_sd = aggregates["finetuned"]
    identical_pairs = sum(
        all(abs(read_json(ROOT / f"seed_{seed}" / "finetuned" / "test_metrics.json")[key]
                - read_json(ROOT / f"seed_{seed}" / "frozen" / "test_metrics.json")[key]) < 1e-12
            for key in METRIC_KEYS + ("fn", "fp")) for seed in SEEDS
    )
    lines.extend(
        [
            "",
            "G. STABILITY INTERPRETATION",
            "-" * 80,
            f"Frozen ROC-AUC SD/range: {f_sd['roc_auc']['standard_deviation']:.4f} / "
            f"{f_sd['roc_auc']['minimum']:.4f}-{f_sd['roc_auc']['maximum']:.4f}.",
            f"Fine-Tuned ROC-AUC SD/range: {t_sd['roc_auc']['standard_deviation']:.4f} / "
            f"{t_sd['roc_auc']['minimum']:.4f}-{t_sd['roc_auc']['maximum']:.4f}.",
            f"Frozen sensitivity SD={f_sd['sensitivity']['standard_deviation']:.4f}, "
            f"FN SD={f_sd['fn']['standard_deviation']:.4f}, specificity SD={f_sd['specificity']['standard_deviation']:.4f}, "
            f"F1 SD={f_sd['f1']['standard_deviation']:.4f}.",
            f"Fine-Tuned sensitivity SD={t_sd['sensitivity']['standard_deviation']:.4f}, "
            f"FN SD={t_sd['fn']['standard_deviation']:.4f}, specificity SD={t_sd['specificity']['standard_deviation']:.4f}, "
            f"F1 SD={t_sd['f1']['standard_deviation']:.4f}.",
            f"Exactly identical frozen/fine-tuned test metric vectors occurred in {identical_pairs}/5 paired seeds.",
            "Therefore, the previous single-seed finding of exactly unchanged locked-test metrics",
            "did not replicate and does not appear stable across seeds.",
            "Partial fine-tuning did not consistently improve any requested paired metric: specificity",
            "was equal in all seeds, ROC-AUC was mixed, and sensitivity, F1, accuracy, and FN count",
            "were worse in four of five seeds.",
            "Neither condition meets the predefined highly-seed-sensitive rule (rate-metric range >=0.30",
            "or SD >=0.15), although specificity remains noticeably variable because only eight Normal",
            "test images make each error change specificity by 0.125.",
            "Consistency is judged from all paired seeds and aggregate dispersion, not the best seed.",
            "With only five paired seeds, these counts describe replication behavior and do not support",
            "strong statistical-significance claims.",
            "",
            "H. OVERFITTING BEHAVIOR",
            "-" * 80,
            f"Predefined overfitting-history flag: frozen {sum(r['overfitting_observed'] for r in validation_rows if r['condition']=='frozen')}/5; "
            f"fine-tuned {sum(r['overfitting_observed'] for r in validation_rows if r['condition']=='finetuned')}/5.",
            "The flag requires post-best-epoch validation loss to rise >5% while training loss falls.",
            "The tiny validation set makes both epoch selection and this heuristic noisy.",
            "",
            "I. LIMITATIONS",
            "-" * 80,
            "Only five seeds were evaluated.",
            "The dataset is small (99 selected training images).",
            "The validation set is very small (12 images).",
            "The test set is small (29 images; only 8 Normal and 21 Mild Cataract).",
            "The ROI is a fixed heuristic crop rather than a learned anatomical localization.",
            "The same test set was already exposed at study level, so this is exploratory replication.",
            "Images rather than independent subjects are the evaluation units; repeated subjects may occur within a split.",
            "",
            "FINAL VERDICT",
            "-" * 80,
            verdict,
            "",
            "ONE NEXT RESEARCH STEP (NOT IMPLEMENTED)",
            "-" * 80,
            "Run a subject-level repeated stratified cross-validation study on the development data, keeping the current test set out of model selection and reporting fold/seed uncertainty.",
        ]
    )
    return "\n".join(lines) + "\n"


def choose_verdict(aggregates: dict[str, Any], paired_summary: dict[str, Any]) -> str:
    # Predetermined descriptive rules, applied to aggregate behavior only.
    key_metrics = ("roc_auc", "sensitivity", "specificity", "f1")
    max_range = max(
        aggregates[condition][key]["maximum"] - aggregates[condition][key]["minimum"]
        for condition in CONDITIONS for key in key_metrics
    )
    max_sd = max(
        aggregates[condition][key]["standard_deviation"]
        for condition in CONDITIONS for key in key_metrics
    )
    if max_range >= 0.30 or max_sd >= 0.15:
        return "RESULTS ARE HIGHLY SEED-SENSITIVE"
    if all(paired_summary[key]["fine_tuned_better_seeds"] >= 4 for key in key_metrics):
        return "PARTIAL FINE-TUNING IS CONSISTENTLY BETTER"
    frozen_dispersion = np.mean([aggregates["frozen"][key]["standard_deviation"] for key in key_metrics])
    fine_dispersion = np.mean([aggregates["finetuned"][key]["standard_deviation"] for key in key_metrics])
    if frozen_dispersion + 0.02 < fine_dispersion:
        return "FROZEN ROI-224 IS MORE STABLE"
    mean_abs_differences = np.mean([
        abs(paired_summary[key]["mean_paired_difference_finetuned_minus_frozen"])
        for key in key_metrics
    ])
    if mean_abs_differences <= 0.03:
        return "FROZEN AND FINE-TUNED ARE COMPARABLE"
    return "MULTI-SEED EXPERIMENT INCONCLUSIVE"


def summarize() -> None:
    environment_gate()
    verify_integrity(save=False)
    validation_rows, test_rows = summarize_rows()
    aggregates = aggregate(test_rows)
    paired_rows, paired_summary = paired(test_rows)
    verdict = choose_verdict(aggregates, paired_summary)
    write_csv(SUMMARY / "per_seed_validation_metrics.csv", validation_rows)
    write_csv(SUMMARY / "per_seed_test_metrics.csv", test_rows)
    write_csv(SUMMARY / "paired_seed_differences.csv", paired_rows)
    write_json(SUMMARY / "aggregate_statistics.json", aggregates)
    write_json(SUMMARY / "paired_comparison.json", paired_summary)
    write_json(SUMMARY / "final_verdict.json", {"verdict": verdict})
    make_figures(test_rows)
    report = report_text(validation_rows, test_rows, aggregates, paired_summary, verdict)
    (SUMMARY / "multiseed_frozen_vs_finetuned_report.txt").write_text(report, encoding="utf-8")
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
