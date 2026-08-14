"""Compare the completed all-cataract and Mild Cataract baseline runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from utils import PROJECT_ROOT, read_json


METRICS = (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("F1", "f1"),
    ("ROC-AUC", "roc_auc"),
)


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_comparison(
    broad: dict[str, Any],
    mild: dict[str, Any],
    broad_training: dict[str, Any],
    mild_training: dict[str, Any],
    task_audit: dict[str, Any],
    broad_predictions: list[dict[str, str]],
    mild_predictions: list[dict[str, str]],
) -> str:
    broad_test = broad["test"]
    mild_test = mild["test"]
    mild_false_negatives = [
        row
        for row in mild_predictions
        if row["true_label"] == "1" and row["predicted_label"] == "0"
    ]
    confident_mild_misses = [
        row
        for row in mild_false_negatives
        if float(row["predicted_probability_cataract"]) < 0.40
    ]
    broad_wrong = sum(row["correct"].casefold() == "false" for row in broad_predictions)
    mild_wrong = sum(row["correct"].casefold() == "false" for row in mild_predictions)
    protocol_match = bool(task_audit["protocol_matches_reference"])
    mild_balanced_accuracy = (mild_test["sensitivity"] + mild_test["specificity"]) / 2.0

    if mild_test["roc_auc"] < 0.60 or mild_balanced_accuracy < 0.60:
        viability = (
            "The Normal-vs-Mild task is viable as a thesis research problem and this run is a valid "
            "weak reference baseline, but the current frozen EfficientNetB0 result is not a reliable "
            "detector. It should motivate a later, separately justified improvement rather than be "
            "presented as adequate performance."
        )
    else:
        viability = (
            "The task is viable as a thesis baseline, subject to the severe uncertainty from the "
            "small validation and locked-test sets."
        )

    lines = [
        "CONTROLLED BASELINE COMPARISON",
        "Normal vs All Cataract compared with Normal vs Mild Cataract",
        "=" * 88,
        "",
        "EXPERIMENT CONTROL",
        "-" * 88,
        f"Training protocol identical: {protocol_match}",
        "Both runs used EfficientNetB0, ImageNet weights, frozen backbone, 224x224 RGB,",
        "the same 64-unit head, dropout 0.30, Adam 0.001, batch size 8, augmentation,",
        "seed 2026, callbacks, validation-loss selection, and fixed threshold 0.5.",
        "No fine-tuning, test-based selection, model comparison, or threshold tuning was used.",
        "",
        "DATASET COMPARISON",
        "-" * 88,
        "task                         train                 validation            locked test",
        "Normal vs All Cataract       45 N / 65 C = 110     5 N / 10 C = 15      8 N / 27 C = 35",
        "Normal vs Mild Cataract      45 N / 54 M = 99      5 N / 7 M = 12       8 N / 21 M = 29",
        "Mild-task overlap: 0 subjects in every split pair; exact duplicates=0; near duplicates=0.",
        "",
        "LOCKED TEST METRICS",
        "-" * 88,
        "metric             All Cataract      Mild Cataract      Mild minus All",
    ]
    for label, key in METRICS:
        broad_value = float(broad_test[key])
        mild_value = float(mild_test[key])
        if key == "roc_auc":
            lines.append(
                f"{label:<18} {broad_value:>12.3f} {mild_value:>18.3f} {mild_value - broad_value:>19.3f}"
            )
        else:
            lines.append(
                f"{label:<18} {percentage(broad_value):>12} {percentage(mild_value):>18} "
                f"{100.0 * (mild_value - broad_value):>+18.1f} pp"
            )
    lines.extend(
        [
            "",
            "CONFUSION COUNTS",
            "-" * 88,
            (
                f"All Cataract:  TN={broad_test['tn']}, FP={broad_test['fp']}, "
                f"FN={broad_test['fn']}, TP={broad_test['tp']} ({broad_wrong} errors / {broad_test['n']})"
            ),
            (
                f"Mild Cataract: TN={mild_test['tn']}, FP={mild_test['fp']}, "
                f"FN={mild_test['fn']}, TP={mild_test['tp']} ({mild_wrong} errors / {mild_test['n']})"
            ),
            "",
            "TRAINING BEHAVIOR",
            "-" * 88,
            (
                f"All Cataract: best epoch {broad_training['best_epoch']} of "
                f"{broad_training['epochs_run']}; overfitting={broad_training['overfitting_observed']}."
            ),
            (
                f"Mild Cataract: best epoch {mild_training['best_epoch']} of "
                f"{mild_training['epochs_run']}; overfitting={mild_training['overfitting_observed']}."
            ),
            "",
            "MAIN FAILURE PATTERN",
            "-" * 88,
            (
                f"The Mild model missed {len(mild_false_negatives)} of 21 Mild Cataract images "
                f"(false-negative rate {100.0 * mild_test['fn'] / (mild_test['fn'] + mild_test['tp']):.1f}%)."
            ),
            (
                f"{len(confident_mild_misses)} of those {len(mild_false_negatives)} misses had "
                "p(Mild Cataract) below 0.40, so the failure is not only a few threshold-borderline cases."
            ),
            (
                f"ROC-AUC fell from {broad_test['roc_auc']:.3f} to {mild_test['roc_auc']:.3f}, "
                "indicating near-chance ranking after severe cases were removed."
            ),
            "The central failure is under-detection of subtle mild disease as Normal, not excessive",
            "false alarms alone. Locked-test threshold tuning is neither justified nor permitted.",
            "",
            "THESIS BASELINE VIABILITY",
            "-" * 88,
            viability,
            "The 12-image validation set and 29-image test set make all estimates highly uncertain.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "baseline_comparison.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    broad_reports = PROJECT_ROOT / "outputs" / "reports"
    mild_root = PROJECT_ROOT / "outputs" / "mild_cataract"
    report = build_comparison(
        read_json(broad_reports / "baseline_metrics.json"),
        read_json(mild_root / "reports" / "baseline_metrics.json"),
        read_json(broad_reports / "training_summary.json"),
        read_json(mild_root / "reports" / "training_summary.json"),
        read_json(mild_root / "reports" / "task_audit.json"),
        load_predictions(PROJECT_ROOT / "outputs" / "predictions" / "test_predictions.csv"),
        load_predictions(mild_root / "predictions" / "test_predictions.csv"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Comparison report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

