"""Generate inference-only Grad-CAM for ROI-320 and compare with ROI-224."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import load_metadata, select_samples
from explainability import (
    FOCUS_CATEGORIES,
    GRADCAM_LAYER,
    attention_summary,
    build_gradcam_components,
    cataract_gradcam,
    create_contact_sheet,
    load_image_for_model,
    make_overlay,
    outcome_for,
    safe_stem,
)
from roi import crop_pil
from utils import PROJECT_ROOT, load_config, read_json, require_preflight, set_global_determinism


CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi_320.yaml"
ROOT_224 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
ROOT_320 = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320"
CHECKPOINT = ROOT_320 / "checkpoints" / "best_frozen_efficientnetb0.keras"
PREDICTIONS = ROOT_320 / "predictions" / "test_predictions.csv"
OUTPUT_DIR = ROOT_320 / "gradcam"
OUTCOME_DIRS = {"TP": "true_positive", "FN": "false_negative", "TN": "true_negative", "FP": "false_positive"}


def read_predictions() -> dict[str, dict[str, str]]:
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    nested = defaultdict(Counter)
    for row in rows:
        nested[row["prediction_outcome"]][row["qualitative_gradcam_focus_category"]] += 1
    return {
        outcome: {category: nested[outcome][category] for category in FOCUS_CATEGORIES}
        for outcome in OUTCOME_DIRS
    }


def overall_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["qualitative_gradcam_focus_category"] for row in rows)
    return {category: counts[category] for category in FOCUS_CATEGORIES}


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def create_case_figure(roi_image: Image.Image, overlay: Image.Image, row: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(roi_image)
    axes[0].set_title("Identical fixed ROI crop")
    axes[1].imshow(overlay)
    axes[1].set_title(f"ROI-320 Grad-CAM: {GRADCAM_LAYER}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{row['filename']}\nTrue: {row['true_class']} | Predicted: {row['predicted_class']} | "
        f"P(Mild): {row['cataract_probability']:.6f} | {row['prediction_outcome']}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, facecolor="white")
    plt.close(fig)


def main() -> None:
    config = load_config(CONFIG)
    set_global_determinism(config)
    require_preflight(config)
    import tensorflow as tf

    samples = select_samples(config, load_metadata(config, "test"))
    saved = read_predictions()
    if set(saved) != {sample.filename for sample in samples}:
        raise RuntimeError("ROI-320 prediction file does not match the fixed test split")
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    model.trainable = False
    components = build_gradcam_components(model)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    threshold = float(config["training"]["threshold"])
    for directory in OUTCOME_DIRS.values():
        (OUTPUT_DIR / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    probability_differences = []
    for index, sample in enumerate(samples, start=1):
        batch = load_image_for_model(Path(sample.image_path), image_size, config["roi"])
        heatmap, probability = cataract_gradcam(model, components, batch)
        saved_probability = float(saved[sample.filename]["mild_probability"])
        difference = abs(probability - saved_probability)
        probability_differences.append(difference)
        if difference > 1e-5:
            raise RuntimeError(f"Probability drift for {sample.filename}: {difference}")
        predicted = int(probability >= threshold)
        if predicted != int(saved[sample.filename]["predicted_label"]):
            raise RuntimeError(f"Prediction drift for {sample.filename}")
        prediction_outcome = outcome_for(sample.label, predicted)
        with Image.open(sample.image_path) as image:
            original = image.convert("RGB").copy()
        roi_image, box = crop_pil(original, config["roi"])
        overlay = make_overlay(roi_image, heatmap)
        focus = attention_summary(heatmap, batch.numpy()[0])
        row = {
            "filename": sample.filename,
            "subject_id": sample.subject_id,
            "true_class": "Mild Cataract" if sample.label else "Normal",
            "predicted_class": "Mild Cataract" if predicted else "Normal",
            "cataract_probability": probability,
            "prediction_outcome": prediction_outcome,
            "cataract_type": sample.cataract_type,
            "illumination_type": sample.illumination_type,
            "image_quality": sample.image_quality,
            "reflection_metadata": sample.reflection,
            "roi_coordinates": list(box.as_tuple()),
            "qualitative_gradcam_focus_category": focus.focus_category,
            "central_attention_mass": focus.central_attention_mass,
            "border_attention_mass": focus.border_attention_mass,
            "bright_attention_mass": focus.bright_attention_mass,
            "normalized_entropy": focus.normalized_entropy,
        }
        rows.append(row)
        cases.append({**row, "original": roi_image, "overlay": overlay})
        create_case_figure(
            roi_image,
            overlay,
            row,
            OUTPUT_DIR / OUTCOME_DIRS[prediction_outcome] / f"{safe_stem(sample.filename)}__roi320_gradcam.png",
        )
        print(f"[{index:02d}/{len(samples):02d}] {prediction_outcome} {sample.filename}: p={probability:.6f}, focus={focus.focus_category}")

    for prediction_outcome, directory in OUTCOME_DIRS.items():
        subset = [case for case in cases if case["prediction_outcome"] == prediction_outcome]
        create_contact_sheet(
            subset,
            OUTPUT_DIR / f"{directory}_roi320_gradcam_contact_sheet.png",
            f"ROI-320 {prediction_outcome}: fixed ROI | Grad-CAM ({len(subset)} cases)",
            columns=3,
        )

    summary_224 = read_json(ROOT_224 / "gradcam" / "roi_gradcam_summary.json")
    counts_320 = overall_counts(rows)
    central_320 = mean_metric(rows, "central_attention_mass")
    border_320 = mean_metric(rows, "border_attention_mass")
    summary = {
        "gradcam_layer": GRADCAM_LAYER,
        "roi_224_layer_shape": summary_224["layer_shape"],
        "roi_320_layer_shape": list(components["conv_layer"].output.shape[1:]),
        "outcomes": dict(Counter(row["prediction_outcome"] for row in rows)),
        "category_counts_by_outcome": category_counts(rows),
        "roi_224_overall_category_counts": summary_224["roi_overall_category_counts"],
        "roi_320_overall_category_counts": counts_320,
        "roi_224_mean_central_attention_mass": summary_224["roi_mean_central_attention_mass"],
        "roi_320_mean_central_attention_mass": central_320,
        "central_attention_change_320_minus_224": central_320 - summary_224["roi_mean_central_attention_mass"],
        "roi_224_mean_border_attention_mass": summary_224["roi_mean_border_attention_mass"],
        "roi_320_mean_border_attention_mass": border_320,
        "border_attention_change_320_minus_224": border_320 - summary_224["roi_mean_border_attention_mass"],
        "maximum_probability_difference_from_locked_roi_320_evaluation": max(probability_differences),
        "interpretation_limit": (
            "Qualitative focus categories and center/border/brightness proxies are exploratory. "
            "The feature grid changes from 7x7 to 10x10 and Grad-CAM does not prove clinical causality."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "roi320_gradcam_analysis_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "roi320_gradcam_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""ROI 224x224 VS ROI 320x320 GRAD-CAM COMPARISON
=================================================

IMPLEMENTATION
--------------
Grad-CAM was generated after the locked test evaluation from the saved ROI-320
checkpoint. The positive P(Mild Cataract) output was differentiated with respect
to EfficientNetB0 `top_conv`, the final spatial Conv2D feature layer. The feature
grid changed from {summary['roi_224_layer_shape']} at 224 to {summary['roi_320_layer_shape']} at 320.

ATTENTION SUMMARY
-----------------
ROI-224 categories: {summary['roi_224_overall_category_counts']}
ROI-320 categories: {summary['roi_320_overall_category_counts']}
Mean central attention: 224={summary['roi_224_mean_central_attention_mass']:.3f}, 320={central_320:.3f}, change={summary['central_attention_change_320_minus_224']:+.3f}.
Mean border attention: 224={summary['roi_224_mean_border_attention_mass']:.3f}, 320={border_320:.3f}, change={summary['border_attention_change_320_minus_224']:+.3f}.

LIMITATION
----------
Grad-CAM is a coarse post-hoc interpretability tool. These categories are
exploratory and do not establish anatomical localization or clinical causality.
The different feature-grid sizes also limit pixelwise comparison.
"""
    (OUTPUT_DIR / "roi224_vs_roi320_gradcam_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
