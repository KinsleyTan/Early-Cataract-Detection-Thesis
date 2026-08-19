"""Train the frozen EfficientNetB0 head using only official train/validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import build_dataset, label_counts, load_metadata, select_samples
from metrics import plot_training_history
from model import build_model, compile_model, parameter_counts
from utils import (
    DEFAULT_CONFIG,
    load_config,
    output_path,
    require_preflight,
    set_global_determinism,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_global_determinism(config)
    require_preflight(config)

    import keras
    import tensorflow as tf

    if config["training"].get("fine_tuning", False):
        raise ValueError("Fine-tuning is intentionally disabled for this first baseline")

    train_samples = select_samples(config, load_metadata(config, "train"))
    validation_samples = select_samples(config, load_metadata(config, "validation"))
    train_data = build_dataset(train_samples, config, training=True)
    validation_data = build_dataset(validation_samples, config, training=False)

    model, backbone = build_model(config)
    compile_model(model, config)
    trainable_params, non_trainable_params = parameter_counts(model)

    checkpoints_dir = output_path(config, "checkpoints_dir")
    figures_dir = output_path(config, "figures_dir")
    reports_dir = output_path(config, "reports_dir")
    checkpoint = checkpoints_dir / "best_frozen_efficientnetb0.keras"
    history_csv = reports_dir / "training_history.csv"
    train_cfg = config["training"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor=train_cfg["monitor"],
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=train_cfg["monitor"],
            mode="min",
            patience=int(train_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=train_cfg["monitor"],
            mode="min",
            factor=float(train_cfg["reduce_lr_factor"]),
            patience=int(train_cfg["reduce_lr_patience"]),
            min_lr=float(train_cfg["min_learning_rate"]),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_csv),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history_object = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=int(train_cfg["epochs"]),
        callbacks=callbacks,
        class_weight=train_cfg.get("class_weights"),
        # Shuffling is already seeded and performed inside the training dataset.
        shuffle=False,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    plot_training_history(history, figures_dir)
    write_json(reports_dir / "training_history.json", history)

    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1
    epochs_run = len(history["loss"])
    overfitting_observed = bool(
        best_index < epochs_run - 1
        and history["val_loss"][-1] > history["val_loss"][best_index] * 1.05
        and history["loss"][-1] < history["loss"][best_index]
    )
    roi_config = config.get("roi", {})
    image_height, image_width = [int(value) for value in config["data"]["image_size"]]
    if roi_config.get("enabled", False):
        preprocessing = (
            "RGB JPEG -> deterministic fixed center-square ROI "
            f"(center=({roi_config['center_x_fraction']}, {roi_config['center_y_fraction']}), "
            f"side={roi_config['side_fraction_of_short_edge']} of short edge) -> "
            f"bilinear resize to {image_height}x{image_width} -> float32 [0,255]. "
            "No external normalization "
            "or preprocess_input; EfficientNetB0's internal Rescaling(1/255) performs "
            "the only normalization."
        )
    else:
        preprocessing = (
            f"RGB JPEG -> bilinear resize to {image_height}x{image_width} -> "
            "float32 [0,255]. "
            "No external normalization or preprocess_input; EfficientNetB0's internal "
            "Rescaling(1/255) performs the only normalization."
        )
    summary = {
        "architecture": config["model"]["architecture"],
        "weights": config["model"]["weights"],
        "backbone_trainable": backbone.trainable,
        "fine_tuning_used": False,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_label_counts": dict(label_counts(train_samples)),
        "validation_label_counts": dict(label_counts(validation_samples)),
        "epochs_requested": int(train_cfg["epochs"]),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_loss": history["val_loss"][best_index],
        "best_validation_accuracy": history["val_accuracy"][best_index],
        "overfitting_observed": overfitting_observed,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "checkpoint": str(checkpoint),
        "checkpoint_monitor": train_cfg["monitor"],
        "input_representation": "pupil/lens ROI" if roi_config.get("enabled", False) else "full image",
        "roi": roi_config if roi_config.get("enabled", False) else None,
        "preprocessing": preprocessing,
    }
    write_json(reports_dir / "training_summary.json", summary)
    print(f"Training complete. Best epoch: {best_epoch}/{epochs_run}")
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
