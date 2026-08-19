"""Post-evaluation Grad-CAM analysis for the locked Mild Cataract ROI model."""

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


CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"
ROI_ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment"
CHECKPOINT = ROI_ROOT / "checkpoints" / "best_frozen_efficientnetb0.keras"
PREDICTIONS = ROI_ROOT / "predictions" / "test_predictions.csv"
OUTPUT_DIR = ROI_ROOT / "gradcam"
FULL_GRADCAM_ROWS = (
    PROJECT_ROOT
    / "outputs"
    / "mild_cataract"
    / "gradcam"
    / "gradcam_analysis_rows.json"
)
OUTCOME_DIRS = {
    "TP": "true_positive",
    "FN": "false_negative",
    "TN": "true_negative",
    "FP": "false_positive",
}


def read_predictions() -> dict[str, dict[str, str]]:
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["filename"]: row for row in rows}


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


def create_case_figure(
    roi_image: Image.Image,
    overlay: Image.Image,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(roi_image)
    axes[0].set_title("Fixed ROI crop")
    axes[1].imshow(overlay)
    axes[1].set_title(f"ROI Grad-CAM: {GRADCAM_LAYER}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{row['filename']}\nTrue: {row['true_class']} | Predicted: {row['predicted_class']} | "
        f"P(Mild): {row['cataract_probability']:.6f} | {row['prediction_outcome']}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, facecolor="white")
    plt.close(fig)


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    config = load_config(CONFIG)
    set_global_determinism(config)
    require_preflight(config)
    import tensorflow as tf

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not FULL_GRADCAM_ROWS.is_file():
        raise FileNotFoundError(FULL_GRADCAM_ROWS)
    samples = select_samples(config, load_metadata(config, "test"))
    saved = read_predictions()
    if set(saved) != {sample.filename for sample in samples}:
        raise RuntimeError("ROI prediction file does not match the fixed test split")
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    model.trainable = False
    components = build_gradcam_components(model)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    threshold = float(config["training"]["threshold"])
    for directory in OUTCOME_DIRS.values():
        (OUTPUT_DIR / directory).mkdir(parents=True, exist_ok=True)

    rows = []
    cases = []
    differences = []
    for index, sample in enumerate(samples, start=1):
        batch = load_image_for_model(Path(sample.image_path), image_size, config["roi"])
        heatmap, probability = cataract_gradcam(model, components, batch)
        saved_probability = float(saved[sample.filename]["mild_probability"])
        difference = abs(probability - saved_probability)
        differences.append(difference)
        if difference > 1e-5:
            raise RuntimeError(f"Probability drift for {sample.filename}: {difference}")
        predicted = int(probability >= threshold)
        if predicted != int(saved[sample.filename]["predicted_label"]):
            raise RuntimeError(f"Prediction drift for {sample.filename}")
        outcome = outcome_for(sample.label, predicted)
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
            "prediction_outcome": outcome,
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
            OUTPUT_DIR / OUTCOME_DIRS[outcome] / f"{safe_stem(sample.filename)}__roi_gradcam.png",
        )
        print(
            f"[{index:02d}/{len(samples):02d}] {outcome} {sample.filename}: "
            f"p={probability:.6f}, focus={focus.focus_category}"
        )

    for outcome, directory in OUTCOME_DIRS.items():
        subset = [case for case in cases if case["prediction_outcome"] == outcome]
        if subset:
            create_contact_sheet(
                subset,
                OUTPUT_DIR / f"{directory}_roi_gradcam_contact_sheet.png",
                f"ROI model {outcome}: fixed ROI crop | ROI Grad-CAM ({len(subset)} cases)",
                columns=3,
            )

    full_rows = read_json(FULL_GRADCAM_ROWS)
    full_overall = overall_counts(full_rows)
    roi_overall = overall_counts(rows)
    full_central = mean_metric(full_rows, "central_attention_mass")
    roi_central = mean_metric(rows, "central_attention_mass")
    full_border = mean_metric(full_rows, "border_attention_mass")
    roi_border = mean_metric(rows, "border_attention_mass")
    summary = {
        "gradcam_layer": GRADCAM_LAYER,
        "layer_shape": list(components["conv_layer"].output.shape[1:]),
        "outcomes": dict(Counter(row["prediction_outcome"] for row in rows)),
        "category_counts_by_outcome": category_counts(rows),
        "full_image_overall_category_counts": full_overall,
        "roi_overall_category_counts": roi_overall,
        "full_image_mean_central_attention_mass": full_central,
        "roi_mean_central_attention_mass": roi_central,
        "central_attention_change": roi_central - full_central,
        "full_image_mean_border_attention_mass": full_border,
        "roi_mean_border_attention_mass": roi_border,
        "border_attention_change": roi_border - full_border,
        "maximum_probability_difference_from_locked_roi_evaluation": max(differences),
        "interpretation_limit": (
            "The same exploratory center/border/brightness proxies were applied to different "
            "input frames. ROI centrality can increase mechanically after cropping and is not "
            "proof of clinical causality or anatomical segmentation."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "roi_gradcam_analysis_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "roi_gradcam_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""ROI MODEL GRAD-CAM COMPARISON
=================================

IMPLEMENTATION
--------------
Inference-only Grad-CAM was generated after the locked ROI evaluation from the
saved ROI checkpoint. The positive P(Mild Cataract) output was differentiated with
respect to EfficientNetB0 `top_conv` ({summary['layer_shape'][0]}x{summary['layer_shape'][1]}x{summary['layer_shape'][2]}), the final Conv2D
spatial feature map. No model selection or tuning used these maps.

OUTCOMES
--------
{summary['outcomes']}

FULL-IMAGE VS ROI ATTENTION
---------------------------
Full-image categories: {full_overall}
ROI categories: {roi_overall}
Mean central-proxy attention mass: full={full_central:.3f}, ROI={roi_central:.3f}, change={roi_central-full_central:+.3f}.
Mean border attention mass: full={full_border:.3f}, ROI={roi_border:.3f}, change={roi_border-full_border:+.3f}.

INTERPRETATION
--------------
The ROI maps show whether attention becomes more central within the cropped input
and whether edge/background dominance decreases. This is descriptive only. The ROI
itself mechanically removes outer anatomy/background and redefines the coordinate
frame, so increased centrality cannot by itself prove stronger lens-based reasoning.
Reflection and slit-illumination structure can remain inside the ROI and may still
serve as acquisition shortcuts.

LIMITATION
----------
Grad-CAM is a coarse post-hoc tool and does not prove clinical causality. The final
feature map is only 7x7, the test set is small, and neither experiment has expert
pupil/lens segmentation ground truth.
"""
    (OUTPUT_DIR / "roi_gradcam_comparison_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
