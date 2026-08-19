"""Recompute and verify the final ROI resolution experiment artifacts."""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from metrics import calculate_metrics
from utils import PROJECT_ROOT, read_json, write_json


ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(first: float, second: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance)


def main() -> int:
    failures: list[str] = []
    predictions = csv_rows(ROOT / "predictions" / "test_predictions.csv")
    if len(predictions) != 29 or len({row["filename"] for row in predictions}) != 29:
        failures.append("Test predictions must contain 29 unique images.")
    y_true = np.array([int(row["true_label"]) for row in predictions], dtype=int)
    probabilities = np.array([float(row["mild_probability"]) for row in predictions])
    recomputed = calculate_metrics(y_true, probabilities, threshold=0.5)
    recomputed.pop("predicted_labels")
    saved = read_json(ROOT / "reports" / "baseline_metrics.json")
    for key in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"):
        if not close(recomputed[key], saved["test"][key]):
            failures.append(f"Metric mismatch for {key}: {recomputed[key]} vs {saved['test'][key]}")
    for key in ("tn", "fp", "fn", "tp", "n"):
        if int(recomputed[key]) != int(saved["test"][key]):
            failures.append(f"Count mismatch for {key}: {recomputed[key]} vs {saved['test'][key]}")
    if saved.get("test_evaluation_passes_in_this_run") != 1:
        failures.append("Locked test evaluation pass count is not exactly one.")

    transitions = csv_rows(ROOT / "predictions" / "224_vs_320_case_transitions.csv")
    transition_counts = dict(Counter(row["transition_category"] for row in transitions))
    if len(transitions) != 29:
        failures.append("Case transition CSV must contain 29 rows.")
    if transition_counts != read_json(ROOT / "reports" / "case_transition_summary.json"):
        failures.append("Case transition counts disagree with the recorded summary.")

    metric_rows = csv_rows(ROOT / "reports" / "roi_224_vs_320_metrics.csv")
    if len(metric_rows) != 10:
        failures.append("Metric comparison CSV must contain ten metric/count rows.")
    metric_json = {row["metric"]: row for row in read_json(ROOT / "reports" / "metric_comparison_rows.json")}
    for row in metric_rows:
        source = metric_json[row["metric"]]
        for key in ("roi_224", "roi_320", "absolute_change"):
            if not close(float(row[key]), float(source[key])):
                failures.append(f"Comparison mismatch for {row['metric']} {key}.")

    gradcam_rows = read_json(ROOT / "gradcam" / "roi320_gradcam_analysis_rows.json")
    gradcam_summary = read_json(ROOT / "gradcam" / "roi320_gradcam_summary.json")
    if len(gradcam_rows) != 29:
        failures.append("Grad-CAM rows must contain all 29 locked test images.")
    gradcam_counts = dict(Counter(row["prediction_outcome"] for row in gradcam_rows))
    if gradcam_counts != gradcam_summary["outcomes"]:
        failures.append("Grad-CAM outcome counts disagree with the locked predictions.")
    directory_names = {"TP": "true_positive", "FN": "false_negative", "TN": "true_negative", "FP": "false_positive"}
    overlay_counts = {
        outcome: len(list((ROOT / "gradcam" / directory).glob("*.png")))
        for outcome, directory in directory_names.items()
    }
    if overlay_counts != gradcam_summary["outcomes"]:
        failures.append(f"Grad-CAM overlay counts are incomplete: {overlay_counts}")

    training = read_json(ROOT / "reports" / "training_summary.json")
    if training["devices"] != ["/physical_device:CPU:0"]:
        failures.append(f"Unexpected training device record: {training['devices']}")
    if training["trainable_parameters"] != 82049 or training["non_trainable_parameters"] != 4049571:
        failures.append("ROI-320 model parameter count differs from ROI-224.")
    if "320x320" not in training["preprocessing"]:
        failures.append("Training preprocessing summary does not record 320x320 input.")

    required = [
        ROOT / "checkpoints" / "best_frozen_efficientnetb0.keras",
        ROOT / "reports" / "roi_224_vs_320_report.txt",
        ROOT / "figures" / "case_transitions" / "all_changed_cases_contact_sheet.png",
        ROOT / "gradcam" / "false_negative_roi320_gradcam_contact_sheet.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        failures.append(f"Missing required artifacts: {missing}")

    result = {
        "pass": not failures,
        "recomputed_locked_test_metrics": recomputed,
        "prediction_rows": len(predictions),
        "transition_rows": len(transitions),
        "transition_counts": transition_counts,
        "gradcam_rows": len(gradcam_rows),
        "gradcam_overlay_counts": overlay_counts,
        "failures": failures,
    }
    write_json(ROOT / "reports" / "final_verification.json", result)
    print(result)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
