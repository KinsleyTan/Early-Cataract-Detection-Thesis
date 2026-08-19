"""Controlled ROI-224 partial fine-tuning, evaluation, and paired analysis.

The stages are deliberately separate so the locked test split is unavailable to
the training stage. Existing experiment artifacts are hashed and treated as
read-only; every new artifact is written below outputs/mild_cataract/roi_finetune.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import Sample, build_dataset, label_counts, load_metadata, select_samples
from metrics import calculate_metrics, plot_confusion, plot_roc, plot_training_history
from model import parameter_counts
from roi import crop_pil
from utils import (
    PROJECT_ROOT,
    config_sha256,
    dataset_root,
    load_config,
    project_path,
    read_json,
    set_global_determinism,
    sha256_file,
    write_json,
)


CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi_finetune.yaml"
FROZEN_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"
NEW_ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_finetune"
STAGE_START = "block6d_expand_conv"
EXPECTED_GPU = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_INTERPRETER = Path.home() / ".venvs" / "cat-screen-tf217-gpu" / "bin" / "python"
OUTCOME_NAMES = {
    (1, 1): "TP",
    (1, 0): "FN",
    (0, 0): "TN",
    (0, 1): "FP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "train", "evaluate", "analyze", "integrity")
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    return parser.parse_args()


def dirs(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: project_path(config["paths"][key])
        for key in (
            "checkpoints_dir",
            "figures_dir",
            "predictions_dir",
            "reports_dir",
            "gradcam_dir",
            "logs_dir",
        )
    }


def create_output_tree(config: dict[str, Any]) -> dict[str, Path]:
    result = dirs(config)
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def protected_manifest() -> dict[str, str]:
    """Hash all prior outputs and the three fixed workbooks, excluding NEW_ROOT."""
    items: dict[str, str] = {}
    outputs = PROJECT_ROOT / "outputs"
    if outputs.exists():
        for path in sorted(outputs.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved == NEW_ROOT.resolve() or NEW_ROOT.resolve() in resolved.parents:
                continue
            items[str(resolved.relative_to(PROJECT_ROOT))] = sha256_file(resolved)
    fixed_root = (PROJECT_ROOT / ".." / "Fixed Dataset" / "Clean").resolve()
    for filename in ("train.xlsx", "val.xlsx", "test.xlsx"):
        path = fixed_root / filename
        items[f"../Fixed Dataset/Clean/{filename}"] = sha256_file(path)
    return items


def verify_integrity(config: dict[str, Any], *, save: bool) -> dict[str, Any]:
    reports = dirs(config)["reports_dir"]
    before_path = reports / "protected_artifact_hashes_before.json"
    if not before_path.is_file():
        raise RuntimeError("Missing protected-artifact manifest; run prepare first")
    before = read_json(before_path)
    current = protected_manifest()
    changed = {
        name: {"before": before.get(name), "after": current.get(name)}
        for name in sorted(set(before) | set(current))
        if before.get(name) != current.get(name)
    }
    result = {
        "pass": not changed,
        "protected_file_count_before": len(before),
        "protected_file_count_now": len(current),
        "changed_or_missing": changed,
    }
    if save:
        write_json(reports / "protected_artifact_integrity_after.json", result)
    if changed:
        raise RuntimeError(f"Protected artifacts changed: {list(changed)}")
    return result


def assert_controlled_config(config: dict[str, Any]) -> None:
    frozen = load_config(FROZEN_CONFIG)
    identical_top_level = ("fixed_splits", "label_policy", "data", "roi", "augmentation")
    for key in identical_top_level:
        if config[key] != frozen[key]:
            raise RuntimeError(f"Controlled field differs from frozen ROI config: {key}")
    for key in (
        "architecture",
        "weights",
        "include_top",
        "dense_units",
        "dropout_rate",
        "l2_regularization",
    ):
        if config["model"][key] != frozen["model"][key]:
            raise RuntimeError(f"Controlled model field differs: {key}")
    for key in (
        "optimizer",
        "loss",
        "epochs",
        "monitor",
        "threshold",
        "class_weights",
        "early_stopping_patience",
        "reduce_lr_patience",
        "reduce_lr_factor",
        "min_learning_rate",
    ):
        if config["training"][key] != frozen["training"][key]:
            raise RuntimeError(f"Controlled training field differs: {key}")
    if float(config["training"]["learning_rate"]) != 1e-5:
        raise RuntimeError("Fine-tuning learning rate must be exactly 1e-5")
    if config["model"]["unfreeze_from_layer"] != STAGE_START:
        raise RuntimeError(f"Unfreezing must begin at {STAGE_START}")
    if config["model"]["batch_normalization_trainable"] is not False:
        raise RuntimeError("BatchNormalization must remain frozen")


def assert_fixed_workbooks(config: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    root = dataset_root(config)
    for split, split_cfg in config["fixed_splits"].items():
        path = root / split_cfg["workbook"]
        actual = sha256_file(path)
        expected = str(split_cfg["sha256"]).upper()
        if actual != expected:
            raise RuntimeError(f"Fixed {split} workbook hash mismatch: {actual} != {expected}")
        hashes[split] = actual
    return hashes


def sample_audit(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for split in ("train", "validation", "test"):
        rows = load_metadata(config, split)
        samples = select_samples(config, rows)
        counts = label_counts(samples)
        expected = config["fixed_splits"][split]["expected_usable"]
        actual = {"Normal": int(counts[0]), "Cataract": int(counts[1])}
        if actual != expected:
            raise RuntimeError(f"{split} usable counts changed: {actual} != {expected}")
        result[split] = {
            "workbook_rows": len(rows),
            "usable_samples": len(samples),
            "label_counts": actual,
            "subject_ids": [sample.subject_id for sample in samples],
            "filenames": [sample.filename for sample in samples],
        }
    return result


def gpu_environment() -> dict[str, Any]:
    import tensorflow as tf

    if Path(sys.executable) != EXPECTED_INTERPRETER:
        raise RuntimeError(
            f"Wrong interpreter: {sys.executable}; expected {EXPECTED_INTERPRETER}"
        )
    if not tf.test.is_built_with_cuda():
        raise RuntimeError("TensorFlow CUDA build is false")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus or gpus[0].name != "/physical_device:GPU:0":
        raise RuntimeError(f"TensorFlow GPU:0 unavailable: {gpus}")
    details = tf.config.experimental.get_device_details(gpus[0])
    if details.get("device_name") != EXPECTED_GPU:
        raise RuntimeError(f"Unexpected GPU: {details}")
    with tf.device("/GPU:0"):
        gpu_result = tf.linalg.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    checksum = float(tf.reduce_sum(gpu_result).numpy())
    if "GPU:0" not in gpu_result.device.upper() or checksum != 4096.0:
        raise RuntimeError("TensorFlow GPU operation was not confirmed")
    build = tf.sysconfig.get_build_info()
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    return {
        "interpreter": sys.executable,
        "virtual_env_expected": str(EXPECTED_INTERPRETER.parent.parent),
        "tensorflow_version": tf.__version__,
        "cuda_build": tf.test.is_built_with_cuda(),
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu_device": gpus[0].name,
        "gpu_details": details,
        "gpu_operation_device": gpu_result.device,
        "gpu_operation_checksum": checksum,
        "nvidia_smi": nvidia_smi,
        "kernel": platform.release(),
        "ubuntu": os_release.get("PRETTY_NAME"),
        "wsl2_confirmed": "microsoft-standard-WSL2" in platform.release(),
    }


def weight_digest(weights: list[Any]) -> str:
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(np.asarray(weight.numpy()).tobytes())
    return digest.hexdigest().upper()


def load_and_configure(config: dict[str, Any]):
    import tensorflow as tf

    checkpoint = project_path(config["paths"]["frozen_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 1):
        raise RuntimeError(f"Frozen checkpoint topology is incompatible: {model.input_shape}")
    expected_top = [
        "rgb_image_0_255",
        "training_augmentation",
        "efficientnetb0",
        "global_average_pooling",
        "classification_dense",
        "classification_dropout",
        "cataract_probability",
    ]
    if [layer.name for layer in model.layers] != expected_top:
        raise RuntimeError("Frozen checkpoint classification topology has changed")
    backbone = model.get_layer("efficientnetb0")
    if len(backbone.layers) != 238:
        raise RuntimeError(f"Unexpected EfficientNetB0 layer count: {len(backbone.layers)}")
    if STAGE_START not in {layer.name for layer in backbone.layers}:
        raise RuntimeError(f"Fine-tuning boundary not found: {STAGE_START}")

    head_layers = [
        model.get_layer("classification_dense"), model.get_layer("cataract_probability")
    ]
    head_before = weight_digest([w for layer in head_layers for w in layer.weights])
    backbone.trainable = True
    reached_boundary = False
    for layer in backbone.layers:
        if layer.name == STAGE_START:
            reached_boundary = True
        layer.trainable = reached_boundary and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    head_after = weight_digest([w for layer in head_layers for w in layer.weights])
    if head_before != head_after:
        raise RuntimeError("Classification-head weights changed during selective unfreezing")
    if any(layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)):
        raise RuntimeError("A BatchNormalization layer became trainable")
    return model, backbone, head_before


def model_setup(model, backbone, head_digest: str) -> dict[str, Any]:
    import tensorflow as tf

    frozen_layers = [layer.name for layer in backbone.layers if not layer.trainable]
    trainable_layers = [layer.name for layer in backbone.layers if layer.trainable]
    weighted_trainable = [
        layer.name for layer in backbone.layers if layer.trainable and layer.weights
    ]
    fraction = len(frozen_layers) / len(backbone.layers)
    if not 0.80 <= fraction <= 0.90:
        raise RuntimeError(f"Frozen backbone fraction {fraction:.4f} is outside [0.80, 0.90]")
    trainable_params, non_trainable_params = parameter_counts(model)
    bn_layers = [
        layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    # A training-mode forward pass must not update any BN moving statistic.
    before = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy()) for layer in bn_layers]
    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=True)
    after = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy()) for layer in bn_layers]
    bn_unchanged = all(
        np.array_equal(mean_before, mean_after) and np.array_equal(var_before, var_after)
        for (mean_before, var_before), (mean_after, var_after) in zip(before, after, strict=True)
    )
    if not bn_unchanged:
        raise RuntimeError("BatchNormalization moving statistics changed in training mode")

    with tf.GradientTape() as tape:
        output = model(tf.ones((1, 224, 224, 3), dtype=tf.float32), training=False)
    gradients = tape.gradient(output, model.trainable_weights)
    if not gradients or not all(gradient is not None for gradient in gradients):
        raise RuntimeError("Gradient path to selectively trainable weights is incomplete")
    return {
        "total_backbone_layers": len(backbone.layers),
        "frozen_backbone_layers": len(frozen_layers),
        "trainable_backbone_layers": len(trainable_layers),
        "frozen_backbone_fraction": fraction,
        "first_trainable_backbone_layer": trainable_layers[0],
        "last_trainable_backbone_layer": trainable_layers[-1],
        "first_trainable_weighted_backbone_layer": weighted_trainable[0],
        "last_trainable_weighted_backbone_layer": weighted_trainable[-1],
        "trainable_backbone_layer_names": trainable_layers,
        "trainable_weighted_backbone_layer_names": weighted_trainable,
        "batch_normalization_layer_count": len(bn_layers),
        "batch_normalization_trainable_count": sum(layer.trainable for layer in bn_layers),
        "batch_normalization_moving_statistics_unchanged_in_training_mode_probe": bn_unchanged,
        "batch_normalization_implementation": (
            "All EfficientNetB0 BatchNormalization layers have trainable=False, and the "
            "serialized backbone is called with training=False, forcing inference behavior "
            "and preventing moving-mean/moving-variance updates."
        ),
        "classification_head_weight_sha256": head_digest,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
    }


def compile_finetune(model, config: dict[str, Any]) -> None:
    import tensorflow as tf

    train_cfg = config["training"]
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
    if train_cfg["optimizer"] != "Adam" or train_cfg["loss"] != "binary_crossentropy":
        raise RuntimeError("Optimizer/loss control violation")


def prepare(config: dict[str, Any]) -> None:
    if NEW_ROOT.exists():
        raise RuntimeError(f"Refusing to reuse existing experiment directory: {NEW_ROOT}")
    assert_controlled_config(config)
    workbook_hashes = assert_fixed_workbooks(config)
    samples = sample_audit(config)
    environment = gpu_environment()
    model, backbone, head_digest = load_and_configure(config)
    setup = model_setup(model, backbone, head_digest)
    output = create_output_tree(config)
    write_json(output["reports_dir"] / "protected_artifact_hashes_before.json", protected_manifest())
    frozen_checkpoint = project_path(config["paths"]["frozen_checkpoint"])
    payload = {
        "status": "READY FOR CONTROLLED PARTIAL FINE-TUNING",
        "config_sha256": config_sha256(config),
        "frozen_config": str(FROZEN_CONFIG),
        "frozen_checkpoint": str(frozen_checkpoint),
        "frozen_checkpoint_sha256": sha256_file(frozen_checkpoint),
        "checkpoint_reuse_compatible": True,
        "classification_head_retained": True,
        "environment": environment,
        "fixed_workbook_hashes": workbook_hashes,
        "fixed_split_audit": samples,
        "model_setup": setup,
        "controls": {
            "input_size": config["data"]["image_size"],
            "batch_size": config["data"]["batch_size"],
            "seed": config["experiment"]["seed"],
            "roi": config["roi"],
            "augmentation": config["augmentation"],
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "loss": "binary_crossentropy",
            "threshold": 0.5,
            "class_weights": None,
            "epochs_maximum": 30,
            "early_stopping_patience": 6,
            "reduce_lr_patience": 3,
            "reduce_lr_factor": 0.2,
            "minimum_learning_rate": 1e-6,
        },
    }
    write_json(output["reports_dir"] / "fine_tuning_setup.json", payload)
    layer_lines = "\n".join(f"- {name}" for name in setup["trainable_backbone_layer_names"])
    text = f"""ROI 224 PARTIAL FINE-TUNING SETUP
===================================
Status: {payload['status']}
Interpreter: {environment['interpreter']}
TensorFlow: {environment['tensorflow_version']}
CUDA build: {environment['cuda_build']}
GPU: {environment['gpu_details']['device_name']} ({environment['gpu_device']})
GPU operation device: {environment['gpu_operation_device']}
Frozen checkpoint reused: {frozen_checkpoint}
Classification head retained: yes (digest {head_digest})

BACKBONE SELECTION
------------------
Total backbone layers: {setup['total_backbone_layers']}
Frozen backbone layers: {setup['frozen_backbone_layers']} ({100*setup['frozen_backbone_fraction']:.1f}%)
Trainable backbone layers: {setup['trainable_backbone_layers']}
First trainable backbone layer: {setup['first_trainable_backbone_layer']}
Last trainable backbone layer: {setup['last_trainable_backbone_layer']}
First/last weighted trainable layers: {setup['first_trainable_weighted_backbone_layer']} / {setup['last_trainable_weighted_backbone_layer']}
Total parameters: {setup['total_parameters']:,}
Trainable parameters: {setup['trainable_parameters']:,}
Non-trainable parameters: {setup['non_trainable_parameters']:,}

BATCHNORMALIZATION
------------------
{setup['batch_normalization_implementation']}
BN layers trainable: {setup['batch_normalization_trainable_count']} of {setup['batch_normalization_layer_count']}
Moving-statistics probe unchanged: {setup['batch_normalization_moving_statistics_unchanged_in_training_mode_probe']}

EXACT TRAINABLE BACKBONE LAYERS
-------------------------------
{layer_lines}
"""
    (output["reports_dir"] / "fine_tuning_setup.txt").write_text(text, encoding="utf-8")
    print(text)


def bn_state(backbone) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import tensorflow as tf

    return {
        layer.name: (layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    }


def train(config: dict[str, Any]) -> None:
    output = dirs(config)
    setup_path = output["reports_dir"] / "fine_tuning_setup.json"
    if not setup_path.is_file():
        raise RuntimeError("Preparation report missing")
    if config_sha256(config) != read_json(setup_path)["config_sha256"]:
        raise RuntimeError("Fine-tuning config changed after preparation")
    verify_integrity(config, save=False)
    assert_fixed_workbooks(config)
    environment = gpu_environment()
    if not environment["wsl2_confirmed"]:
        raise RuntimeError("WSL2 not confirmed")

    import tensorflow as tf

    train_samples = select_samples(config, load_metadata(config, "train"))
    validation_samples = select_samples(config, load_metadata(config, "validation"))
    train_data = build_dataset(train_samples, config, training=True)
    validation_data = build_dataset(validation_samples, config, training=False)
    model, backbone, head_digest = load_and_configure(config)
    setup = model_setup(model, backbone, head_digest)
    compile_finetune(model, config)
    checkpoint = output["checkpoints_dir"] / "best_partial_finetuned_efficientnetb0.keras"
    history_csv = output["logs_dir"] / "training_history.csv"
    for path in (checkpoint, history_csv, output["reports_dir"] / "training_summary.json"):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing fine-tuning artifact: {path}")
    before_bn = bn_state(backbone)
    train_cfg = config["training"]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_csv),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history_object = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=30,
        callbacks=callbacks,
        class_weight=None,
        shuffle=False,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    after_bn = bn_state(backbone)
    changed_bn = [
        name
        for name in before_bn
        if not (
            np.array_equal(before_bn[name][0], after_bn[name][0])
            and np.array_equal(before_bn[name][1], after_bn[name][1])
        )
    ]
    if changed_bn:
        raise RuntimeError(f"BatchNormalization moving statistics changed: {changed_bn}")
    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1
    epochs_run = len(history["loss"])
    final_learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    overfit = bool(
        best_index < epochs_run - 1
        and history["val_loss"][-1] > history["val_loss"][best_index] * 1.05
        and history["loss"][-1] < history["loss"][best_index]
    )
    plot_training_history(history, output["figures_dir"])
    write_json(output["reports_dir"] / "training_history.json", history)

    best_model = tf.keras.models.load_model(checkpoint, compile=False)
    val_probabilities = best_model.predict(validation_data, verbose=0).reshape(-1)
    val_true = np.array([sample.label for sample in validation_samples], dtype=int)
    validation_metrics = calculate_metrics(val_true, val_probabilities, 0.5)
    validation_metrics.pop("predicted_labels")
    write_json(output["reports_dir"] / "validation_metrics.json", validation_metrics)
    summary = {
        "training_data_only": ["train.xlsx", "val.xlsx"],
        "test_set_accessed": False,
        "source_checkpoint": str(project_path(config["paths"]["frozen_checkpoint"])),
        "source_checkpoint_sha256": sha256_file(project_path(config["paths"]["frozen_checkpoint"])),
        "classification_head_weight_sha256_at_start": head_digest,
        "optimizer": "Adam",
        "initial_learning_rate": 1e-5,
        "final_learning_rate": final_learning_rate,
        "loss": "binary_crossentropy",
        "batch_size": 8,
        "seed": 2026,
        "epochs_requested": 30,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "minimum_validation_loss": float(history["val_loss"][best_index]),
        "best_logged_validation_accuracy": float(history["val_accuracy"][best_index]),
        "early_stopping_patience": 6,
        "reduce_lr_patience": 3,
        "restore_best_weights": True,
        "checkpoint_monitor": "val_loss",
        "overfitting_observed": overfit,
        "batch_normalization_moving_statistics_changed": changed_bn,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_label_counts": dict(label_counts(train_samples)),
        "validation_label_counts": dict(label_counts(validation_samples)),
        "checkpoint": str(checkpoint),
        "environment": environment,
        "model_setup": setup,
        "validation_metrics_at_selected_checkpoint": validation_metrics,
    }
    write_json(output["reports_dir"] / "training_summary.json", summary)
    write_json(
        output["logs_dir"] / "training_completed.json",
        {"completed": True, "best_epoch": best_epoch, "epochs_run": epochs_run},
    )
    print(json.dumps(summary, indent=2, default=str))


def prediction_rows(
    samples: list[Sample], probabilities: np.ndarray, predicted: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for sample, probability, prediction in zip(samples, probabilities, predicted, strict=True):
        rows.append(
            {
                "filename": sample.filename,
                "subject_id": sample.subject_id,
                "true_label": sample.label,
                "predicted_label": int(prediction),
                "mild_probability": f"{float(probability):.8f}",
                "correct_incorrect": "correct" if int(prediction) == sample.label else "incorrect",
                "cataract_type": sample.cataract_type,
                "illumination_type": sample.illumination_type,
                "image_quality": sample.image_quality,
                "reflection_metadata": sample.reflection,
                "eye_side": sample.eye_side,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config: dict[str, Any]) -> None:
    output = dirs(config)
    training_path = output["reports_dir"] / "training_summary.json"
    if not training_path.is_file():
        raise RuntimeError("Training summary missing")
    forbidden_existing = (
        output["predictions_dir"] / "test_predictions.csv",
        output["reports_dir"] / "locked_test_metrics.json",
        output["logs_dir"] / "locked_test_evaluation_completed.json",
    )
    if any(path.exists() for path in forbidden_existing):
        raise RuntimeError("Locked test evaluation artifact already exists; refusing to repeat")
    verify_integrity(config, save=False)
    assert_fixed_workbooks(config)
    gpu_environment()

    import tensorflow as tf

    checkpoint = output["checkpoints_dir"] / "best_partial_finetuned_efficientnetb0.keras"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    test_samples = select_samples(config, load_metadata(config, "test"))
    test_data = build_dataset(test_samples, config, training=False)
    # This is the one locked-test prediction pass used for performance evaluation.
    probabilities = model.predict(test_data, verbose=0).reshape(-1)
    true = np.array([sample.label for sample in test_samples], dtype=int)
    metrics = calculate_metrics(true, probabilities, 0.5)
    predicted = np.array(metrics.pop("predicted_labels"), dtype=int)
    rows = prediction_rows(test_samples, probabilities, predicted)
    write_csv(output["predictions_dir"] / "test_predictions.csv", rows)
    validation = read_json(output["reports_dir"] / "validation_metrics.json")
    payload = {
        "threshold": 0.5,
        "model_selected_using": "minimum validation loss only",
        "test_evaluation_passes": 1,
        "validation": validation,
        "test": metrics,
    }
    write_json(output["reports_dir"] / "locked_test_metrics.json", payload)
    plot_roc(
        true,
        probabilities,
        metrics["roc_auc"],
        output["figures_dir"] / "test_roc_curve.png",
        title="Locked Test ROC: ROI 224 Partial Fine-Tuning",
    )
    plot_confusion(
        metrics,
        output["figures_dir"] / "test_confusion_matrix.png",
        class_names=("Normal", "Mild Cataract"),
        title="Locked Test: ROI 224 Partial Fine-Tuning",
    )
    write_json(
        output["logs_dir"] / "locked_test_evaluation_completed.json",
        {"completed": True, "prediction_passes": 1, "threshold": 0.5},
    )
    print(json.dumps(payload, indent=2))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def transition_name(true: int, frozen: int, fine: int) -> str:
    if true == 1 and frozen == 0 and fine == 1:
        return "FN→TP"
    if true == 1 and frozen == 1 and fine == 0:
        return "TP→FN"
    if true == 0 and frozen == 1 and fine == 0:
        return "FP→TN"
    if true == 0 and frozen == 0 and fine == 1:
        return "TN→FP"
    return "unchanged correct" if fine == true else "unchanged incorrect"


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def transition_contact_sheet(
    rows: list[dict[str, Any]], samples: dict[str, Sample], config: dict[str, Any], path: Path, title: str
) -> None:
    columns, panel_width, panel_height = 3, 350, 310
    count_rows = max(1, math.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * panel_width, 55 + count_rows * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=17)
    font = ImageFont.load_default(size=13)
    draw.text((12, 16), title, fill="black", font=title_font)
    if not rows:
        draw.text((12, 70), "No cases in this transition category.", fill="black", font=font)
    for index, row in enumerate(rows):
        x, y = (index % columns) * panel_width + 8, 55 + (index // columns) * panel_height
        sample = samples[row["filename"]]
        with Image.open(sample.image_path) as image:
            roi_image, _ = crop_pil(image.convert("RGB"), config["roi"])
        sheet.paste(fit_image(roi_image, (330, 220)), (x, y))
        lines = [
            row["filename"],
            row["transition"],
            f"True={row['true_label']} Frozen={row['frozen_predicted_label']} Fine={row['finetuned_predicted_label']}",
            f"P(Mild): {row['frozen_mild_probability']:.4f} → {row['finetuned_mild_probability']:.4f}",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x, y + 226 + 18 * line_index), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def overlap_coefficient(normal: np.ndarray, mild: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 21)
    normal_hist, _ = np.histogram(normal, bins=bins, density=True)
    mild_hist, _ = np.histogram(mild, bins=bins, density=True)
    return float(np.minimum(normal_hist, mild_hist).sum() * (bins[1] - bins[0]))


def distribution_stats(probabilities: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    normal, mild = probabilities[true == 0], probabilities[true == 1]
    return {
        "normal_mean": float(normal.mean()),
        "normal_median": float(np.median(normal)),
        "normal_standard_deviation": float(normal.std(ddof=1)),
        "mild_mean": float(mild.mean()),
        "mild_median": float(np.median(mild)),
        "mild_standard_deviation": float(mild.std(ddof=1)),
        "mean_class_separation_mild_minus_normal": float(mild.mean() - normal.mean()),
        "histogram_overlap_coefficient": overlap_coefficient(normal, mild),
        "mean_absolute_margin_from_0_5": float(np.mean(np.abs(probabilities - 0.5))),
        "extreme_prediction_fraction_p_le_0_1_or_ge_0_9": float(
            np.mean((probabilities <= 0.1) | (probabilities >= 0.9))
        ),
    }


def probability_figure(
    frozen: np.ndarray, fine: np.ndarray, true: np.ndarray, path: Path
) -> None:
    bins = np.linspace(0.0, 1.0, 16)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, label, title in zip(axes, (0, 1), ("Normal", "Mild Cataract"), strict=True):
        axis.hist(frozen[true == label], bins=bins, alpha=0.55, label="Frozen ROI 224", color="#4C78A8")
        axis.hist(fine[true == label], bins=bins, alpha=0.55, label="Fine-tuned ROI 224", color="#F58518")
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Predicted Mild probability", ylabel="Image count")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Frozen vs Fine-Tuned ROI 224 Probability Distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(config: dict[str, Any]) -> None:
    output = dirs(config)
    metrics_path = output["reports_dir"] / "locked_test_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("Locked test evaluation must finish before paired analysis")
    frozen_prediction_path = project_path(config["paths"]["frozen_predictions"])
    fine_prediction_path = output["predictions_dir"] / "test_predictions.csv"
    frozen_rows = {row["filename"]: row for row in read_csv(frozen_prediction_path)}
    fine_rows = {row["filename"]: row for row in read_csv(fine_prediction_path)}
    samples_list = select_samples(config, load_metadata(config, "test"))
    samples = {sample.filename: sample for sample in samples_list}
    if set(frozen_rows) != set(fine_rows) or set(fine_rows) != set(samples):
        raise RuntimeError("Frozen/fine-tuned predictions do not match the fixed test split")
    transitions = []
    for sample in samples_list:
        frozen = frozen_rows[sample.filename]
        fine = fine_rows[sample.filename]
        frozen_label, fine_label = int(frozen["predicted_label"]), int(fine["predicted_label"])
        transitions.append(
            {
                "filename": sample.filename,
                "subject_id": sample.subject_id,
                "true_label": sample.label,
                "frozen_predicted_label": frozen_label,
                "finetuned_predicted_label": fine_label,
                "frozen_mild_probability": float(frozen["mild_probability"]),
                "finetuned_mild_probability": float(fine["mild_probability"]),
                "probability_change": float(fine["mild_probability"]) - float(frozen["mild_probability"]),
                "transition": transition_name(sample.label, frozen_label, fine_label),
                "cataract_type": sample.cataract_type,
                "illumination_type": sample.illumination_type,
                "image_quality": sample.image_quality,
                "reflection_metadata": sample.reflection,
            }
        )
    transition_path = output["predictions_dir"] / "frozen_vs_finetuned_case_transitions.csv"
    write_csv(transition_path, transitions)
    transition_counts = Counter(row["transition"] for row in transitions)
    for category, stem in (("FN→TP", "fn_to_tp"), ("TP→FN", "tp_to_fn"), ("FP→TN", "fp_to_tn"), ("TN→FP", "tn_to_fp")):
        subset = [row for row in transitions if row["transition"] == category]
        transition_contact_sheet(
            subset,
            samples,
            config,
            output["figures_dir"] / "case_transitions" / f"{stem}_contact_sheet.png",
            f"Frozen ROI 224 vs Fine-Tuned ROI 224: {category} ({len(subset)})",
        )

    frozen_metrics = read_json(project_path(config["paths"]["frozen_metrics"]))["test"]
    fine_metrics = read_json(metrics_path)["test"]
    comparison = []
    for name in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"):
        change = float(fine_metrics[name]) - float(frozen_metrics[name])
        comparison.append(
            {
                "metric": name,
                "frozen_roi_224": float(frozen_metrics[name]),
                "finetuned_roi_224": float(fine_metrics[name]),
                "absolute_difference": change,
                "percentage_point_change": 100.0 * change,
            }
        )
    for name in ("tn", "fp", "fn", "tp"):
        comparison.append(
            {
                "metric": name,
                "frozen_roi_224": int(frozen_metrics[name]),
                "finetuned_roi_224": int(fine_metrics[name]),
                "absolute_difference": int(fine_metrics[name]) - int(frozen_metrics[name]),
                "percentage_point_change": None,
            }
        )
    write_json(output["reports_dir"] / "frozen_vs_finetuned_metrics.json", comparison)
    write_csv(output["reports_dir"] / "frozen_vs_finetuned_metrics.csv", comparison)
    write_json(output["reports_dir"] / "case_transition_summary.json", dict(transition_counts))

    true = np.array([sample.label for sample in samples_list], dtype=int)
    frozen_prob = np.array([float(frozen_rows[s.filename]["mild_probability"]) for s in samples_list])
    fine_prob = np.array([float(fine_rows[s.filename]["mild_probability"]) for s in samples_list])
    frozen_stats = distribution_stats(frozen_prob, true)
    fine_stats = distribution_stats(fine_prob, true)
    separation_change = (
        fine_stats["mean_class_separation_mild_minus_normal"]
        - frozen_stats["mean_class_separation_mild_minus_normal"]
    )
    overlap_change = (
        fine_stats["histogram_overlap_coefficient"]
        - frozen_stats["histogram_overlap_coefficient"]
    )
    margin_change = (
        fine_stats["mean_absolute_margin_from_0_5"]
        - frozen_stats["mean_absolute_margin_from_0_5"]
    )
    normal_shift = float(np.mean(fine_prob[true == 0] - frozen_prob[true == 0]))
    mild_shift = float(np.mean(fine_prob[true == 1] - frozen_prob[true == 1]))
    probability_analysis = {
        "frozen": frozen_stats,
        "finetuned": fine_stats,
        "mean_class_separation_change": separation_change,
        "histogram_overlap_change": overlap_change,
        "mean_absolute_margin_change": margin_change,
        "normal_mean_paired_probability_shift": normal_shift,
        "mild_mean_paired_probability_shift": mild_shift,
        "increases_class_separation": separation_change > 0,
        "merely_shifts_probabilities": abs(separation_change) < 0.01 and normal_shift * mild_shift > 0,
        "creates_more_extreme_predictions": margin_change > 0,
        "increases_class_overlap": overlap_change > 0,
        "interpretation": (
            f"Mean class separation changed by {separation_change:+.4f}; histogram overlap "
            f"changed by {overlap_change:+.4f}; mean distance from 0.5 changed by "
            f"{margin_change:+.4f}. Normal and Mild paired mean shifts were "
            f"{normal_shift:+.4f} and {mild_shift:+.4f}, respectively."
        ),
    }
    write_json(output["reports_dir"] / "probability_analysis.json", probability_analysis)
    probability_figure(
        frozen_prob,
        fine_prob,
        true,
        output["figures_dir"] / "frozen_vs_finetuned_probability_distributions.png",
    )
    print(json.dumps({"comparison": comparison, "transitions": dict(transition_counts), "probability": probability_analysis}, indent=2))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_global_determinism(config)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "train":
        train(config)
    elif args.stage == "evaluate":
        evaluate(config)
    elif args.stage == "analyze":
        analyze(config)
    elif args.stage == "integrity":
        print(json.dumps(verify_integrity(config, save=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
