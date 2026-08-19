"""Paired full-image versus ROI comparison after the single locked ROI test run."""

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


FULL_METRICS = PROJECT_ROOT / "outputs" / "mild_cataract" / "reports" / "baseline_metrics.json"
FULL_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "mild_cataract" / "predictions" / "test_predictions.csv"
)
ROI_ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
ROI_METRICS = ROI_ROOT / "reports" / "baseline_metrics.json"
ROI_PREDICTIONS = ROI_ROOT / "predictions" / "test_predictions.csv"
CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction_label(row: dict[str, str]) -> int:
    return int(row["predicted_label"])


def transition_category(true_label: int, full_label: int, roi_label: int) -> str:
    if true_label == 1:
        if full_label == 0 and roi_label == 1:
            return "FN -> TP improvement"
        if full_label == 1 and roi_label == 0:
            return "TP -> FN regression"
        return "stable TP" if full_label == 1 else "stable FN"
    if full_label == 1 and roi_label == 0:
        return "FP -> TN improvement"
    if full_label == 0 and roi_label == 1:
        return "TN -> FP regression"
    return "stable TN" if full_label == 0 else "stable FP"


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def contact_sheet(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    image_size = (270, 205)
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
            f"Full p={row['full_image_mild_probability']:.6f} | ROI p={row['roi_mild_probability']:.6f}",
            "Panels: original | fixed ROI",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x, y + image_size[1] + 5 + 18 * line_index), line, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    config = load_config(CONFIG)
    samples = select_samples(config, load_metadata(config, "test"))
    samples_by_name = {sample.filename: sample for sample in samples}
    full_rows = {row["filename"]: row for row in read_csv(FULL_PREDICTIONS)}
    roi_rows = {row["filename"]: row for row in read_csv(ROI_PREDICTIONS)}
    expected = set(samples_by_name)
    if set(full_rows) != expected or set(roi_rows) != expected:
        raise RuntimeError("Prediction files do not match the fixed Mild test split")

    transitions = []
    visual_rows = []
    for sample in samples:
        full = full_rows[sample.filename]
        roi = roi_rows[sample.filename]
        full_label = prediction_label(full)
        roi_label = prediction_label(roi)
        category = transition_category(sample.label, full_label, roi_label)
        row = {
            "filename": sample.filename,
            "subject_id": sample.subject_id,
            "true_class": "Mild Cataract" if sample.label == 1 else "Normal",
            "full_image_predicted_class": "Mild Cataract" if full_label == 1 else "Normal",
            "roi_predicted_class": "Mild Cataract" if roi_label == 1 else "Normal",
            "full_image_mild_probability": float(full["predicted_probability_cataract"]),
            "roi_mild_probability": float(roi["mild_probability"]),
            "transition_category": category,
            "illumination_type": sample.illumination_type,
            "cataract_type": sample.cataract_type,
            "image_quality": sample.image_quality,
        }
        transitions.append(row)
        if category in {
            "FN -> TP improvement",
            "TP -> FN regression",
            "FP -> TN improvement",
            "TN -> FP regression",
        }:
            with Image.open(sample.image_path) as image:
                original = image.convert("RGB").copy()
            roi_image, _ = crop_pil(original, config["roi"])
            visual_rows.append({**row, "original": original, "roi": roi_image})

    full_metrics = read_json(FULL_METRICS)["test"]
    roi_metrics = read_json(ROI_METRICS)["test"]
    metric_names = ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc")
    comparison = []
    for metric in metric_names:
        full_value = float(full_metrics[metric])
        roi_value = float(roi_metrics[metric])
        comparison.append(
            {
                "metric": metric,
                "full_image": full_value,
                "roi": roi_value,
                "absolute_change": roi_value - full_value,
                "percentage_point_change": 100.0 * (roi_value - full_value),
            }
        )
    for metric in ("fn", "fp"):
        comparison.append(
            {
                "metric": metric,
                "full_image": int(full_metrics[metric]),
                "roi": int(roi_metrics[metric]),
                "absolute_change": int(roi_metrics[metric]) - int(full_metrics[metric]),
                "percentage_point_change": None,
            }
        )

    reports_dir = ROI_ROOT / "reports"
    figures_dir = ROI_ROOT / "figures" / "case_transitions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "case_transition_rows.json", transitions)
    write_json(reports_dir / "metric_comparison_rows.json", comparison)
    transition_counts = Counter(row["transition_category"] for row in transitions)
    write_json(reports_dir / "case_transition_summary.json", dict(transition_counts))

    named_categories = {
        "fn_to_tp": "FN -> TP improvement",
        "tp_to_fn": "TP -> FN regression",
        "fp_to_tn": "FP -> TN improvement",
        "tn_to_fp": "TN -> FP regression",
    }
    for filename, category in named_categories.items():
        subset = [row for row in visual_rows if row["transition_category"] == category]
        contact_sheet(
            subset,
            figures_dir / f"{filename}_contact_sheet.png",
            f"Full image vs ROI: {category} ({len(subset)} cases)",
        )
    print(json.dumps({"metrics": comparison, "transitions": dict(transition_counts)}, indent=2))


if __name__ == "__main__":
    main()
