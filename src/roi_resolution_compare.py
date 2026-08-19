"""Compare locked ROI-224 and ROI-320 predictions, metrics, and training behavior."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data import load_metadata, select_samples
from roi import crop_pil
from utils import PROJECT_ROOT, load_config, read_json, write_json


CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi_320.yaml"
ROOT_224 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
ROOT_320 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320"


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["filename"]: row for row in rows}


def prediction_label(row: dict[str, str]) -> int:
    return int(row["predicted_label"])


def outcome(true_label: int, predicted_label: int) -> str:
    if true_label == 1:
        return "TP" if predicted_label == 1 else "FN"
    return "FP" if predicted_label == 1 else "TN"


def transition_name(before: str, after: str) -> str:
    labels = {
        ("FN", "TP"): "FN -> TP improvement",
        ("TP", "FN"): "TP -> FN regression",
        ("FP", "TN"): "FP -> TN improvement",
        ("TN", "FP"): "TN -> FP regression",
    }
    return labels.get((before, after), f"{before} -> {after} unchanged")


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def contact_sheet(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    image_size = (260, 195)
    columns = 3
    panel_width = image_size[0] * 2 + 30
    panel_height = image_size[1] + 110
    display_rows = max(1, math.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * panel_width, 62 + display_rows * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=17)
    font = ImageFont.load_default(size=13)
    draw.text((14, 18), title, fill="black", font=title_font)
    if not rows:
        draw.text((14, 72), "No cases in this transition category.", fill="black", font=font)
    for index, row in enumerate(rows):
        x = (index % columns) * panel_width + 8
        y = 62 + (index // columns) * panel_height
        sheet.paste(fit(row["original"], image_size), (x, y))
        sheet.paste(fit(row["roi"], image_size), (x + image_size[0] + 5, y))
        lines = [
            row["filename"],
            row["transition_category"],
            f"True: {row['true_class']}",
            f"P224={row['roi_224_mild_probability']:.6f} | P320={row['roi_320_mild_probability']:.6f}",
            "Panels: original | identical fixed ROI",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x, y + image_size[1] + 5 + 18 * line_index), line, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def training_snapshot(root: Path, label: str) -> dict[str, Any]:
    summary = read_json(root / "reports" / "training_summary.json")
    history = read_json(root / "reports" / "training_history.json")
    best = int(summary["best_epoch"]) - 1
    final = len(history["loss"]) - 1
    return {
        "experiment": label,
        "best_epoch": best + 1,
        "epochs_run": len(history["loss"]),
        "best_epoch_metrics": {
            "training_accuracy": history["accuracy"][best],
            "validation_accuracy": history["val_accuracy"][best],
            "training_loss": history["loss"][best],
            "validation_loss": history["val_loss"][best],
            "accuracy_divergence_train_minus_validation": history["accuracy"][best] - history["val_accuracy"][best],
            "loss_divergence_validation_minus_train": history["val_loss"][best] - history["loss"][best],
        },
        "final_epoch_metrics": {
            "training_accuracy": history["accuracy"][final],
            "validation_accuracy": history["val_accuracy"][final],
            "training_loss": history["loss"][final],
            "validation_loss": history["val_loss"][final],
            "accuracy_divergence_train_minus_validation": history["accuracy"][final] - history["val_accuracy"][final],
            "loss_divergence_validation_minus_train": history["val_loss"][final] - history["loss"][final],
        },
        "overfitting_observed": bool(summary["overfitting_observed"]),
    }


def main() -> None:
    config = load_config(CONFIG)
    samples = select_samples(config, load_metadata(config, "test"))
    by_name = {sample.filename: sample for sample in samples}
    rows_224 = read_csv(ROOT_224 / "predictions" / "test_predictions.csv")
    rows_320 = read_csv(ROOT_320 / "predictions" / "test_predictions.csv")
    expected = set(by_name)
    if set(rows_224) != expected or set(rows_320) != expected:
        raise RuntimeError("Prediction files do not match the same fixed Mild test split")

    transitions: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for sample in samples:
        label_224 = prediction_label(rows_224[sample.filename])
        label_320 = prediction_label(rows_320[sample.filename])
        outcome_224 = outcome(sample.label, label_224)
        outcome_320 = outcome(sample.label, label_320)
        category = transition_name(outcome_224, outcome_320)
        row = {
            "filename": sample.filename,
            "subject_id": sample.subject_id,
            "true_label": sample.label,
            "true_class": "Mild Cataract" if sample.label else "Normal",
            "roi_224_predicted_label": label_224,
            "roi_224_predicted_class": "Mild Cataract" if label_224 else "Normal",
            "roi_224_mild_probability": float(rows_224[sample.filename]["mild_probability"]),
            "roi_224_outcome": outcome_224,
            "roi_320_predicted_label": label_320,
            "roi_320_predicted_class": "Mild Cataract" if label_320 else "Normal",
            "roi_320_mild_probability": float(rows_320[sample.filename]["mild_probability"]),
            "roi_320_outcome": outcome_320,
            "transition_category": category,
            "prediction_changed": label_224 != label_320,
            "illumination_type": sample.illumination_type,
            "cataract_type": sample.cataract_type,
            "image_quality": sample.image_quality,
        }
        transitions.append(row)
        if label_224 != label_320:
            with Image.open(sample.image_path) as image:
                original = image.convert("RGB").copy()
            roi_image, _ = crop_pil(original, config["roi"])
            visual_rows.append({**row, "original": original, "roi": roi_image})

    metrics_224 = read_json(ROOT_224 / "reports" / "baseline_metrics.json")["test"]
    metrics_320 = read_json(ROOT_320 / "reports" / "baseline_metrics.json")["test"]
    comparison = []
    for metric in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"):
        before = float(metrics_224[metric])
        after = float(metrics_320[metric])
        comparison.append(
            {
                "metric": metric,
                "roi_224": before,
                "roi_320": after,
                "absolute_change": after - before,
                "percentage_point_change": 100.0 * (after - before),
                "priority": {"sensitivity": 1, "roc_auc": 2}.get(metric),
            }
        )
    for metric in ("fn", "fp", "tn", "tp"):
        before = int(metrics_224[metric])
        after = int(metrics_320[metric])
        comparison.append(
            {
                "metric": metric,
                "roi_224": before,
                "roi_320": after,
                "absolute_change": after - before,
                "percentage_point_change": None,
                "priority": 3 if metric == "fn" else None,
            }
        )

    reports = ROOT_320 / "reports"
    figures = ROOT_320 / "figures" / "case_transitions"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_json(reports / "metric_comparison_rows.json", comparison)
    write_json(reports / "case_transition_rows.json", transitions)
    transition_counts = dict(Counter(row["transition_category"] for row in transitions))
    write_json(reports / "case_transition_summary.json", transition_counts)
    training = {
        "roi_224": training_snapshot(ROOT_224, "ROI 224x224"),
        "roi_320": training_snapshot(ROOT_320, "ROI 320x320"),
    }
    write_json(reports / "training_comparison.json", training)

    important = {
        "fn_to_tp": "FN -> TP improvement",
        "tp_to_fn": "TP -> FN regression",
        "fp_to_tn": "FP -> TN improvement",
        "tn_to_fp": "TN -> FP regression",
    }
    for stem, category in important.items():
        subset = [row for row in visual_rows if row["transition_category"] == category]
        contact_sheet(subset, figures / f"{stem}_contact_sheet.png", f"ROI 224 vs 320: {category} ({len(subset)} cases)")
    contact_sheet(
        visual_rows,
        figures / "all_changed_cases_contact_sheet.png",
        f"ROI 224 vs 320: all changed predictions ({len(visual_rows)} cases)",
    )
    print(json.dumps({"metrics": comparison, "transitions": transition_counts, "training": training}, indent=2))


if __name__ == "__main__":
    main()

