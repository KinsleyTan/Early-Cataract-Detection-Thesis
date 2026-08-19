"""Post-evaluation Grad-CAM for the controlled ROI-224 fine-tuned model."""

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
from utils import PROJECT_ROOT, load_config, project_path, read_json, set_global_determinism, write_json


CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi_finetune.yaml"
ROOT = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_finetune"
CHECKPOINT = ROOT / "checkpoints" / "best_partial_finetuned_efficientnetb0.keras"
PREDICTIONS = ROOT / "predictions" / "test_predictions.csv"
OUTPUT = ROOT / "gradcam"
OUTCOME_DIRS = {
    "TP": "true_positive",
    "FN": "false_negative",
    "TN": "true_negative",
    "FP": "false_positive",
}


def read_predictions() -> dict[str, dict[str, str]]:
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def counts_by_outcome(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
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


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def case_figure(
    roi_image: Image.Image, overlay: Image.Image, row: dict[str, Any], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(roi_image)
    axes[0].set_title("Fixed ROI crop")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Fine-tuned Grad-CAM: {GRADCAM_LAYER}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{row['filename']}\nTrue: {row['true_class']} | Predicted: {row['predicted_class']} | "
        f"P(Mild): {row['mild_probability']:.6f} | {row['prediction_outcome']}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def main() -> int:
    config = load_config(CONFIG)
    set_global_determinism(config)
    import tensorflow as tf

    if (OUTPUT / "finetuned_gradcam_summary.json").exists():
        raise RuntimeError("Fine-tuned Grad-CAM already exists; refusing to overwrite")
    if not CHECKPOINT.is_file() or not PREDICTIONS.is_file():
        raise RuntimeError("Checkpoint and locked predictions are required before Grad-CAM")
    frozen_summary_path = project_path(config["paths"]["frozen_gradcam_summary"])
    if not frozen_summary_path.is_file():
        raise FileNotFoundError(frozen_summary_path)
    samples = select_samples(config, load_metadata(config, "test"))
    saved = read_predictions()
    if set(saved) != {sample.filename for sample in samples}:
        raise RuntimeError("Saved predictions do not match the fixed test split")

    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    model.trainable = False
    components = build_gradcam_components(model)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    for directory in OUTCOME_DIRS.values():
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    probability_differences = []
    for index, sample in enumerate(samples, start=1):
        batch = load_image_for_model(Path(sample.image_path), image_size, config["roi"])
        heatmap, probability = cataract_gradcam(model, components, batch)
        saved_probability = float(saved[sample.filename]["mild_probability"])
        difference = abs(probability - saved_probability)
        probability_differences.append(difference)
        # GPU convolution kernels can differ slightly between the locked batched
        # prediction pass and Grad-CAM's single-image decomposed pass. Keep a
        # strict 0.002 guard and independently require the thresholded label to
        # match the locked CSV for every case.
        if difference > 2e-3:
            raise RuntimeError(f"Probability drift for {sample.filename}: {difference}")
        predicted = int(probability >= 0.5)
        if predicted != int(saved[sample.filename]["predicted_label"]):
            raise RuntimeError(f"Prediction drift for {sample.filename}")
        outcome = outcome_for(sample.label, predicted)
        with Image.open(sample.image_path) as image:
            roi_image, box = crop_pil(image.convert("RGB"), config["roi"])
        overlay = make_overlay(roi_image, heatmap)
        focus = attention_summary(heatmap, batch.numpy()[0])
        row = {
            "filename": sample.filename,
            "subject_id": sample.subject_id,
            "true_class": "Mild Cataract" if sample.label else "Normal",
            "predicted_class": "Mild Cataract" if predicted else "Normal",
            "mild_probability": probability,
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
            "bright_area_fraction": focus.bright_area_fraction,
            "normalized_entropy": focus.normalized_entropy,
            "peak_in_central_proxy": focus.peak_in_central_proxy,
            "peak_in_border": focus.peak_in_border,
        }
        rows.append(row)
        cases.append({**row, "original": roi_image, "overlay": overlay})
        case_figure(
            roi_image,
            overlay,
            row,
            OUTPUT / OUTCOME_DIRS[outcome] / f"{safe_stem(sample.filename)}__finetuned_gradcam.png",
        )
        print(
            f"[{index:02d}/{len(samples):02d}] {outcome} {sample.filename}: "
            f"p={probability:.6f}, focus={focus.focus_category}",
            flush=True,
        )

    for outcome, directory in OUTCOME_DIRS.items():
        subset = [case for case in cases if case["prediction_outcome"] == outcome]
        if subset:
            create_contact_sheet(
                subset,
                OUTPUT / f"{directory}_finetuned_gradcam_contact_sheet.png",
                f"Fine-tuned ROI model {outcome}: ROI | Grad-CAM ({len(subset)} cases)",
                columns=3,
            )

    frozen = read_json(frozen_summary_path)
    fine_counts = overall_counts(rows)
    frozen_counts = frozen["roi_overall_category_counts"]
    fine_central = mean(rows, "central_attention_mass")
    fine_border = mean(rows, "border_attention_mass")
    fine_bright = mean(rows, "bright_attention_mass")
    fine_entropy = mean(rows, "normalized_entropy")
    summary = {
        "methodology": "Same top_conv Grad-CAM and fixed heuristic attention_summary as the frozen ROI experiment",
        "used_for_model_selection": False,
        "gradcam_layer": GRADCAM_LAYER,
        "layer_shape": list(components["conv_layer"].output.shape[1:]),
        "outcomes": dict(Counter(row["prediction_outcome"] for row in rows)),
        "category_counts_by_outcome": counts_by_outcome(rows),
        "frozen_roi_category_counts": frozen_counts,
        "finetuned_roi_category_counts": fine_counts,
        "category_count_changes": {
            category: int(fine_counts[category]) - int(frozen_counts[category])
            for category in FOCUS_CATEGORIES
        },
        "frozen_mean_central_attention_mass": frozen["roi_mean_central_attention_mass"],
        "finetuned_mean_central_attention_mass": fine_central,
        "central_attention_change": fine_central - float(frozen["roi_mean_central_attention_mass"]),
        "frozen_mean_border_attention_mass": frozen["roi_mean_border_attention_mass"],
        "finetuned_mean_border_attention_mass": fine_border,
        "border_attention_change": fine_border - float(frozen["roi_mean_border_attention_mass"]),
        "finetuned_mean_bright_attention_mass": fine_bright,
        "finetuned_mean_normalized_entropy": fine_entropy,
        "maximum_probability_difference_from_locked_predictions": max(probability_differences),
        "interpretation_limit": (
            "The center, border, and brightness measures are fixed exploratory proxies, not "
            "anatomical segmentations. Grad-CAM is coarse and cannot establish clinical causality."
        ),
    }
    write_json(OUTPUT / "finetuned_gradcam_analysis_rows.json", rows)
    write_json(OUTPUT / "finetuned_gradcam_summary.json", summary)
    report = f"""FROZEN VS FINE-TUNED ROI 224 GRAD-CAM
========================================
Method: {summary['methodology']}.
Layer: {GRADCAM_LAYER}, shape {summary['layer_shape']}.
Grad-CAM was generated only after the locked test evaluation and was not used for selection.

Frozen categories: {frozen_counts}
Fine-tuned categories: {fine_counts}
Category changes: {summary['category_count_changes']}
Mean central-proxy mass: {summary['frozen_mean_central_attention_mass']:.4f} -> {fine_central:.4f} ({summary['central_attention_change']:+.4f})
Mean border mass: {summary['frozen_mean_border_attention_mass']:.4f} -> {fine_border:.4f} ({summary['border_attention_change']:+.4f})
Fine-tuned mean bright-attention mass: {fine_bright:.4f}
Fine-tuned mean normalized entropy: {fine_entropy:.4f}

Interpretation: counts distinguish lens/pupil dominant, partially lens-related,
reflection/illumination dominant, eyelid/background dominant, and diffuse/unclear
patterns using the same fixed heuristics as the prior ROI analysis. These maps are
descriptive only and do not prove causal or clinically valid localization.
"""
    (OUTPUT / "frozen_vs_finetuned_gradcam_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
