"""Validation reporting and one-time locked test evaluation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data import Sample, build_dataset, load_metadata, select_samples
from metrics import calculate_metrics, plot_confusion, plot_roc
from utils import (
    DEFAULT_CONFIG,
    load_config,
    output_path,
    read_json,
    require_preflight,
    set_global_determinism,
    write_json,
)


def percent(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{100.0 * value:.1f}%"


def metric_line(name: str, value: float) -> str:
    return f"{name}: {value:.4f} ({percent(value)})"


def export_predictions(
    samples: list[Sample],
    probabilities: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    class_names: tuple[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for sample, probability, prediction in zip(samples, probabilities, predicted, strict=True):
        rows.append(
            {
                "filename": sample.filename,
                "subject_id": sample.subject_id,
                "eye_side": sample.eye_side,
                "true_label": sample.label,
                "true_class_name": class_names[sample.label],
                "predicted_probability_cataract": f"{float(probability):.8f}",
                "predicted_label": int(prediction),
                "predicted_class_name": class_names[int(prediction)],
                "correct": int(prediction) == sample.label,
                "outcome": "correct" if int(prediction) == sample.label else "incorrect",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def choose_verdict(metrics: dict[str, Any]) -> tuple[str, str]:
    balanced_accuracy = (metrics["sensitivity"] + metrics["specificity"]) / 2.0
    if metrics["roc_auc"] < 0.60 or balanced_accuracy < 0.60:
        return (
            "BASELINE UNRELIABLE",
            "Discrimination or balanced class performance is too weak for a trustworthy baseline.",
        )
    if (
        metrics["accuracy"] >= 0.80
        and metrics["roc_auc"] >= 0.80
        and metrics["sensitivity"] >= 0.75
        and metrics["specificity"] >= 0.75
    ):
        return (
            "BASELINE SUCCESSFUL",
            "The locked test metrics are consistently strong, while the small sample sizes still limit precision.",
        )
    return (
        "BASELINE WORKS WITH WARNINGS",
        "The pipeline is functional, but class performance is uneven or only moderate on the small locked test set.",
    )


def build_report(
    config: dict[str, Any],
    training: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    errors: list[tuple[Sample, float, int]],
) -> str:
    train_expected = config["fixed_splits"]["train"]["expected_usable"]
    val_expected = config["fixed_splits"]["validation"]["expected_usable"]
    test_expected = config["fixed_splits"]["test"]["expected_usable"]
    augmentation = config["augmentation"]
    train_cfg = config["training"]
    task_name = config["experiment"].get("display_name", "Normal vs Cataract")
    negative_class = config["label_policy"].get("negative_class", "Normal")
    positive_class = config["label_policy"].get("positive_class", "Cataract")
    grades = config["label_policy"].get("cataract_grades")
    label_policy_text = (
        f"Class 0={negative_class}; class 1={positive_class}; Severe Cataract and Other excluded."
        if grades is not None
        else f"Class 0={negative_class}; class 1={positive_class}; Other excluded."
    )
    verdict, verdict_reason = choose_verdict(test_metrics)
    lines = [
        f"EFFICIENTNETB0 {task_name.upper()} BASELINE",
        "=" * 80,
        "",
        "DATASET",
        "-" * 80,
        f"Train usable: {sum(train_expected.values())} ({negative_class}={train_expected['Normal']}, {positive_class}={train_expected['Cataract']})",
        f"Validation usable: {sum(val_expected.values())} ({negative_class}={val_expected['Normal']}, {positive_class}={val_expected['Cataract']})",
        f"Test usable: {sum(test_expected.values())} ({negative_class}={test_expected['Normal']}, {positive_class}={test_expected['Cataract']})",
        "Official train.xlsx, val.xlsx, and test.xlsx assignments were kept fixed.",
        label_policy_text,
        "No random split was created.",
        "The locked test set was not used during fitting or model selection.",
        "",
        "CONFIGURATION",
        "-" * 80,
        f"Architecture: {training['architecture']} (include_top=False)",
        f"Pretrained weights: {training['weights']}",
        f"Image size: {config['data']['image_size'][0]}x{config['data']['image_size'][1]}x3 RGB",
        f"Preprocessing: {training['preprocessing']}",
        f"Optimizer: {train_cfg['optimizer']}",
        f"Initial learning rate: {train_cfg['learning_rate']}",
        f"Loss: {train_cfg['loss']}",
        f"Batch size: {config['data']['batch_size']}",
        f"Decision threshold: {train_cfg['threshold']}",
        f"Random seed: {config['experiment']['seed']} (Python, NumPy, TensorFlow)",
        f"Epochs requested: {training['epochs_requested']}",
        f"Epochs run: {training['epochs_run']}",
        f"Best epoch: {training['best_epoch']} (selected by minimum {training['checkpoint_monitor']})",
        (
            "Training-only augmentation: "
            f"rotation factor +/-{augmentation['rotation_factor']}, zoom +/-{augmentation['zoom_factor']}, "
            f"brightness +/-{augmentation['brightness_factor']}, contrast +/-{augmentation['contrast_factor']}, "
            f"horizontal flip={augmentation['horizontal_flip']}."
        ),
        "Horizontal flipping was omitted to preserve laterality/acquisition geometry conservatively.",
        "",
        "MODEL",
        "-" * 80,
        f"Trainable parameters: {training['trainable_parameters']:,}",
        f"Non-trainable parameters: {training['non_trainable_parameters']:,}",
        "Backbone: frozen EfficientNetB0; BatchNormalization used inference behavior.",
        f"Fine-tuning used: {training['fine_tuning_used']}",
        "",
        "VALIDATION RESULT",
        "-" * 80,
        metric_line("Accuracy", validation_metrics["accuracy"]),
        metric_line("Precision", validation_metrics["precision"]),
        metric_line("Sensitivity", validation_metrics["sensitivity"]),
        metric_line("Specificity", validation_metrics["specificity"]),
        metric_line("F1", validation_metrics["f1"]),
        f"ROC-AUC: {validation_metrics['roc_auc']:.4f}",
        (
            f"TN={validation_metrics['tn']}, FP={validation_metrics['fp']}, "
            f"FN={validation_metrics['fn']}, TP={validation_metrics['tp']}"
        ),
        "",
        "LOCKED TEST RESULT",
        "-" * 80,
        metric_line("Accuracy", test_metrics["accuracy"]),
        metric_line("Precision", test_metrics["precision"]),
        metric_line(f"Sensitivity / {positive_class} recall", test_metrics["sensitivity"]),
        metric_line("Specificity", test_metrics["specificity"]),
        metric_line("F1", test_metrics["f1"]),
        f"ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}",
        (
            f"Normal raw outcomes: {test_metrics['tn']}/{test_metrics['tn'] + test_metrics['fp']} correct, "
            f"{test_metrics['fp']}/{test_metrics['tn'] + test_metrics['fp']} false positive."
        ),
        (
            f"{positive_class} raw outcomes: {test_metrics['tp']}/{test_metrics['tp'] + test_metrics['fn']} detected, "
            f"{test_metrics['fn']}/{test_metrics['tp'] + test_metrics['fn']} missed."
        ),
        "",
        "ERRORS",
        "-" * 80,
    ]
    if not errors:
        lines.append("No misclassified locked-test images.")
    else:
        for sample, probability, predicted in errors:
            lines.append(
                f"{sample.filename}: true={(positive_class if sample.label == 1 else negative_class)} ({sample.label}), "
                f"predicted={(positive_class if predicted == 1 else negative_class)} ({predicted}), "
                f"p({positive_class})={probability:.6f}, subject_id={sample.subject_id}, "
                f"illumination={sample.illumination_type}, grade={sample.cataract_grade}"
            )
    lines.extend(
        [
            "",
            "TRAINING BEHAVIOR",
            "-" * 80,
            f"Best validation-loss epoch: {training['best_epoch']} of {training['epochs_run']} run.",
            f"Overfitting signal observed by the predefined history heuristic: {training['overfitting_observed']}.",
            "No fine-tuning or hyperparameter search was performed.",
            "",
            "LIMITATIONS",
            "-" * 80,
            f"The dataset is small ({sum(train_expected.values())} training samples).",
            f"The validation set has only {sum(val_expected.values())} usable images, so model-selection estimates are unstable.",
            f"The locked test set has only {sum(test_expected.values())} usable images.",
            f"The test set is class-imbalanced ({test_expected['Normal']} {negative_class} vs {test_expected['Cataract']} {positive_class}).",
            f"This is the controlled {task_name} baseline.",
            "Percentages can change substantially with one image; raw counts should be emphasized.",
            "",
            "BASELINE VERDICT",
            "-" * 80,
            verdict,
            verdict_reason,
            "",
            "NEXT-STEP RECOMMENDATION",
            "-" * 80,
            (
                "Treat this as the initial thesis-task baseline and diagnose its fixed-threshold "
                "false-positive/false-negative pattern before proposing later improvements. "
                "Do not tune this completed run from locked-test outcomes."
                if grades is not None
                else
                "The software pipeline can be reconfigured for a narrower label policy only after "
                "a fresh class-count and split-adequacy review. Do not reuse locked-test outcomes to tune it."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Explicitly allow overwriting an existing locked-test evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_global_determinism(config)
    require_preflight(config)

    import tensorflow as tf

    reports_dir = output_path(config, "reports_dir")
    figures_dir = output_path(config, "figures_dir")
    predictions_dir = output_path(config, "predictions_dir")
    metrics_path = reports_dir / "baseline_metrics.json"
    if metrics_path.exists() and not args.allow_repeat:
        raise RuntimeError(
            "Locked test evaluation already exists. Refusing to repeat without --allow-repeat."
        )

    checkpoint = output_path(config, "checkpoints_dir") / "best_frozen_efficientnetb0.keras"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    training_summary = read_json(reports_dir / "training_summary.json")
    model = tf.keras.models.load_model(checkpoint, compile=False)

    validation_samples = select_samples(config, load_metadata(config, "validation"))
    test_samples = select_samples(config, load_metadata(config, "test"))
    validation_data = build_dataset(validation_samples, config, training=False)
    test_data = build_dataset(test_samples, config, training=False)

    validation_probabilities = model.predict(validation_data, verbose=0).reshape(-1)
    # This is the single final locked-test inference pass.
    test_probabilities = model.predict(test_data, verbose=0).reshape(-1)
    threshold = float(config["training"]["threshold"])
    validation_true = np.array([sample.label for sample in validation_samples], dtype=int)
    test_true = np.array([sample.label for sample in test_samples], dtype=int)
    validation_metrics = calculate_metrics(validation_true, validation_probabilities, threshold)
    test_metrics = calculate_metrics(test_true, test_probabilities, threshold)
    test_predicted = np.array(test_metrics.pop("predicted_labels"), dtype=int)
    validation_metrics.pop("predicted_labels")
    negative_class = config["label_policy"].get("negative_class", "Normal")
    positive_class = config["label_policy"].get("positive_class", "Cataract")
    class_names = (negative_class, positive_class)
    task_name = config["experiment"].get("display_name", "Normal vs Cataract")

    export_predictions(
        test_samples,
        test_probabilities,
        test_predicted,
        predictions_dir / "test_predictions.csv",
        class_names,
    )
    plot_roc(
        test_true,
        test_probabilities,
        test_metrics["roc_auc"],
        figures_dir / "test_roc_curve.png",
        title=f"Locked Test ROC: {task_name}",
    )
    plot_confusion(
        test_metrics,
        figures_dir / "test_confusion_matrix.png",
        class_names=class_names,
        title=f"Locked Test Confusion Matrix: {task_name}",
    )

    errors = [
        (sample, float(probability), int(predicted))
        for sample, probability, predicted in zip(
            test_samples, test_probabilities, test_predicted, strict=True
        )
        if int(predicted) != sample.label
    ]
    verdict, verdict_reason = choose_verdict(test_metrics)
    metrics_output = {
        "task": task_name,
        "positive_class": positive_class,
        "threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_class_counts": dict(Counter(int(value) for value in test_true)),
        "misclassified_test_images": len(errors),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "test_evaluation_passes_in_this_run": 1,
    }
    write_json(metrics_path, metrics_output)
    report = build_report(
        config, training_summary, validation_metrics, test_metrics, errors
    )
    (reports_dir / "baseline_results.txt").write_text(report, encoding="utf-8")
    print(f"Locked test evaluation complete: {verdict}")
    print(f"Report: {reports_dir / 'baseline_results.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
