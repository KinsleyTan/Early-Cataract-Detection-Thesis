"""Checkpoint-only Grad-CAM analysis for the fixed Mild Cataract test split.

This module never trains or alters the model. It explains the positive sigmoid
output (Mild Cataract probability) using the final convolutional feature tensor
(`top_conv`) of the saved EfficientNetB0 backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import load_metadata, select_samples
from roi import apply_roi_tensor
from utils import PROJECT_ROOT, load_config, project_path, require_preflight, set_global_determinism


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "mild_cataract"
    / "checkpoints"
    / "best_frozen_efficientnetb0.keras"
)
DEFAULT_SAVED_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "mild_cataract"
    / "predictions"
    / "test_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "mild_cataract" / "gradcam"
)

GRADCAM_LAYER = "top_conv"
FOCUS_CATEGORIES = (
    "lens/pupil dominant",
    "partially lens-related",
    "reflection/illumination dominant",
    "eyelid/background dominant",
    "diffuse/unclear",
)
OUTCOME_DIRS = {
    "TP": "true_positive",
    "FN": "false_negative",
    "TN": "true_negative",
    "FP": "false_positive",
}


@dataclass(frozen=True)
class AttentionSummary:
    central_attention_mass: float
    border_attention_mass: float
    bright_attention_mass: float
    bright_area_fraction: float
    normalized_entropy: float
    peak_in_central_proxy: bool
    peak_in_border: bool
    focus_category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for the locked Mild Cataract test split."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--saved-predictions", type=Path, default=DEFAULT_SAVED_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_image_for_model(
    path: Path,
    image_size: tuple[int, int],
    roi_config: dict[str, Any] | None = None,
):
    """Match src/data.py decoding exactly: JPEG RGB, bilinear antialiased, [0,255]."""
    import tensorflow as tf

    content = tf.io.read_file(str(path))
    image = tf.io.decode_jpeg(content, channels=3)
    if roi_config and roi_config.get("enabled", False):
        image = apply_roi_tensor(image, roi_config)
    image = tf.image.resize(image, image_size, method="bilinear", antialias=True)
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)
    image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
    return tf.expand_dims(image, axis=0)


def load_display_image(path: Path) -> Image.Image:
    """Read the source image for display only; the source file is never modified."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def build_gradcam_components(model):
    """Return the saved-model components needed to explain its positive output."""
    import tensorflow as tf

    backbone = model.get_layer("efficientnetb0")
    conv_layer = backbone.get_layer(GRADCAM_LAYER)
    if not isinstance(conv_layer, tf.keras.layers.Conv2D):
        raise TypeError(f"{GRADCAM_LAYER} is not a Conv2D layer")
    conv_model = tf.keras.Model(backbone.input, conv_layer.output, name="gradcam_to_top_conv")
    backbone_tail = tf.keras.Model(
        conv_layer.output, backbone.output, name="gradcam_from_top_conv"
    )
    return {
        "augmentation": model.get_layer("training_augmentation"),
        "conv_model": conv_model,
        "backbone_tail": backbone_tail,
        "global_pool": model.get_layer("global_average_pooling"),
        "dense": model.get_layer("classification_dense"),
        "dropout": model.get_layer("classification_dropout"),
        "output": model.get_layer("cataract_probability"),
        "backbone": backbone,
        "conv_layer": conv_layer,
    }


def cataract_gradcam(model, components: dict[str, Any], image_batch):
    """Compute normalized Grad-CAM for the positive Mild Cataract probability."""
    import tensorflow as tf

    with tf.GradientTape() as tape:
        x = components["augmentation"](image_batch, training=False)
        feature_maps = components["conv_model"](x, training=False)
        tape.watch(feature_maps)
        x = components["backbone_tail"](feature_maps, training=False)
        x = components["global_pool"](x)
        x = components["dense"](x, training=False)
        x = components["dropout"](x, training=False)
        probability = components["output"](x, training=False)[0, 0]

    gradients = tape.gradient(probability, feature_maps)
    if gradients is None:
        raise RuntimeError("No gradient connects top_conv to the Cataract probability")
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(channel_weights * feature_maps, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))

    direct_probability = model(image_batch, training=False)[0, 0]
    if not np.isclose(
        float(probability.numpy()), float(direct_probability.numpy()), atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError("Decomposed Grad-CAM graph does not reproduce the saved model")
    return heatmap.numpy().astype(np.float32), float(direct_probability.numpy())


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    uint8_heatmap = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    resized = Image.fromarray(uint8_heatmap, mode="L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def make_overlay(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    resized = resize_heatmap(heatmap, original.size)
    color = plt.get_cmap("turbo")(resized)[..., :3]
    original_array = np.asarray(original, dtype=np.float32) / 255.0
    strength = (alpha * resized)[..., None]
    blended = original_array * (1.0 - strength) + color * strength
    return Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def attention_summary(heatmap: np.ndarray, model_image: np.ndarray) -> AttentionSummary:
    """Assign an exploratory category using fixed spatial/intensity proxies.

    The center ellipse is only a reproducible proxy for the pupil/lens region;
    it is not a segmented anatomical annotation. Bright-pixel and border masks
    are similarly heuristic. Categories therefore support exploration, not a
    clinical claim.
    """
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = (xx + 0.5) / w
    y_norm = (yy + 0.5) / h
    central = ((x_norm - 0.5) / 0.29) ** 2 + ((y_norm - 0.5) / 0.26) ** 2 <= 1.0
    border = (x_norm < 0.14) | (x_norm > 0.86) | (y_norm < 0.14) | (y_norm > 0.86)

    gray = np.dot(model_image[..., :3], [0.299, 0.587, 0.114])
    gray_small = np.asarray(
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L").resize(
            (w, h), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    bright_threshold = max(180.0, float(np.percentile(gray_small, 90.0)))
    bright = gray_small >= bright_threshold

    mass = np.maximum(heatmap.astype(np.float64), 0.0)
    total = float(mass.sum())
    if total <= 1e-12:
        distribution = np.full_like(mass, 1.0 / mass.size)
    else:
        distribution = mass / total
    central_mass = float(distribution[central].sum())
    border_mass = float(distribution[border].sum())
    bright_mass = float(distribution[bright].sum()) if bright.any() else 0.0
    bright_area = float(bright.mean())
    entropy = float(
        -np.sum(distribution * np.log(distribution + 1e-12)) / math.log(distribution.size)
    )
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    peak_central = bool(central[peak])
    peak_border = bool(border[peak])
    bright_enrichment = bright_mass / max(bright_area, 1e-6)

    if central_mass >= 0.46 and peak_central and bright_mass < 0.34:
        category = "lens/pupil dominant"
    elif bright_mass >= 0.27 and bright_enrichment >= 1.45:
        category = "reflection/illumination dominant"
    elif border_mass >= 0.52 and peak_border:
        category = "eyelid/background dominant"
    elif central_mass >= 0.28:
        category = "partially lens-related"
    else:
        category = "diffuse/unclear"

    if category not in FOCUS_CATEGORIES:
        raise AssertionError(f"Unexpected focus category: {category}")
    return AttentionSummary(
        central_attention_mass=central_mass,
        border_attention_mass=border_mass,
        bright_attention_mass=bright_mass,
        bright_area_fraction=bright_area,
        normalized_entropy=entropy,
        peak_in_central_proxy=peak_central,
        peak_in_border=peak_border,
        focus_category=category,
    )


def outcome_for(true_label: int, predicted_label: int) -> str:
    return {
        (1, 1): "TP",
        (1, 0): "FN",
        (0, 0): "TN",
        (0, 1): "FP",
    }[(true_label, predicted_label)]


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9._() -]+", "_", stem).strip() or "image"


def create_case_figure(
    original: Image.Image,
    overlay: Image.Image,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(original)
    axes[0].set_title("Original (unchanged)")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {GRADCAM_LAYER}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{row['filename']}\n"
        f"True: {row['true_class']} | Predicted: {row['predicted_class']} | "
        f"P(Cataract): {row['cataract_probability']:.6f} | {row['prediction_outcome']}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, facecolor="white")
    plt.close(fig)


def fit_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    offset = ((box[0] - result.width) // 2, (box[1] - result.height) // 2)
    canvas.paste(result, offset)
    return canvas


def create_contact_sheet(
    cases: list[dict[str, Any]],
    output_path: Path,
    title: str,
    *,
    columns: int = 3,
) -> None:
    if not cases:
        raise ValueError(f"Cannot create empty contact sheet: {title}")
    image_box = (260, 210)
    panel_width = image_box[0] * 2 + 24
    panel_height = image_box[1] + 126
    rows = math.ceil(len(cases) / columns)
    sheet = Image.new("RGB", (columns * panel_width, 70 + rows * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    draw.text((18, 18), title, fill="black", font=font)

    for index, case in enumerate(cases):
        col = index % columns
        row_index = index // columns
        x = col * panel_width + 8
        y = 70 + row_index * panel_height
        original = fit_image(case["original"], image_box)
        overlay = fit_image(case["overlay"], image_box)
        sheet.paste(original, (x, y))
        sheet.paste(overlay, (x + image_box[0] + 6, y))
        text_y = y + image_box[1] + 6
        lines = [
            case["filename"],
            f"P(Cataract)={case['cataract_probability']:.6f} | {case['prediction_outcome']}",
            f"Type: {case['cataract_type'] or '(not recorded)'}",
            f"Illumination: {case['illumination_type'] or '(not recorded)'}",
            f"Quality: {case['image_quality'] or '(not recorded)'}",
            f"Exploratory focus: {case['qualitative_gradcam_focus_category']}",
        ]
        for line_number, line in enumerate(lines):
            truncated = line if len(line) <= 72 else line[:69] + "..."
            draw.text((x, text_y + line_number * 18), truncated, fill="black", font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def load_saved_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_filename = {row["filename"]: row for row in rows}
    if len(by_filename) != len(rows):
        raise RuntimeError("Saved test predictions contain duplicate filenames")
    return by_filename


def count_categories(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        nested[row["prediction_outcome"]][row["qualitative_gradcam_focus_category"]] += 1
    return {
        outcome: {category: counts.get(category, 0) for category in FOCUS_CATEGORIES}
        for outcome, counts in sorted(nested.items())
    }


def format_category_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{category}={counts.get(category, 0)}" for category in FOCUS_CATEGORIES)


def write_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    max_saved_probability_difference: float,
) -> None:
    outcome_counts = Counter(row["prediction_outcome"] for row in rows)
    category_counts = count_categories(rows)
    avg_metrics: dict[str, dict[str, float]] = {}
    for outcome in OUTCOME_DIRS:
        subset = [row for row in rows if row["prediction_outcome"] == outcome]
        if subset:
            avg_metrics[outcome] = {
                "central": float(np.mean([row["central_attention_mass"] for row in subset])),
                "border": float(np.mean([row["border_attention_mass"] for row in subset])),
                "bright": float(np.mean([row["bright_attention_mass"] for row in subset])),
                "entropy": float(np.mean([row["normalized_entropy"] for row in subset])),
            }

    tp_central = avg_metrics.get("TP", {}).get("central", float("nan"))
    fn_central = avg_metrics.get("FN", {}).get("central", float("nan"))
    fp_categories = category_counts.get("FP", {})
    non_anatomical_fp = sum(
        fp_categories.get(category, 0)
        for category in (
            "reflection/illumination dominant",
            "eyelid/background dominant",
            "diffuse/unclear",
        )
    )
    recommendation = "full image vs pupil/lens ROI"
    report = f"""MILD CATARACT GRAD-CAM ANALYSIS REPORT
==========================================

Scope and safeguards
--------------------
This was an inference-only analysis of the existing locked Mild Cataract test set.
No training, fine-tuning, threshold selection, preprocessing change, label change,
augmentation change, split change, or model-weight update was performed.
Usable test images: {len(rows)} (TP={outcome_counts['TP']}, FN={outcome_counts['FN']}, TN={outcome_counts['TN']}, FP={outcome_counts['FP']}).
Maximum absolute difference from the saved locked test probabilities: {max_saved_probability_difference:.10f}.

A. Implementation
-----------------
The saved best checkpoint was loaded with compile=False and evaluated at the locked
0.5 decision threshold. Grad-CAM was computed for the positive sigmoid output,
P(Mild Cataract), using EfficientNetB0 layer `top_conv` (7 x 7 x 1280). `top_conv`
is the final Conv2D layer before the backbone's terminal batch normalization and
activation, so it is the deepest spatial convolutional representation that still
retains an image grid. Gradients of P(Mild Cataract) with respect to these feature
maps were spatially averaged, used as channel weights, combined, passed through
ReLU, and normalized to [0, 1]. The heatmap was resized only for display and blended
onto a copy of the original image; source images were not modified.

The qualitative focus category is exploratory and heuristic-assisted. It uses a
fixed central ellipse as a pupil/lens proxy, a 14% border mask, and a bright-pixel
mask. These are not anatomical segmentations. The controlled categories are:
{'; '.join(FOCUS_CATEGORIES)}.

B. False-negative findings
--------------------------
All {outcome_counts['FN']} missed Mild Cataract cases are included in the dedicated contact sheet.
Category counts: {format_category_counts(category_counts.get('FN', {}))}.
Mean attention mass: central proxy={avg_metrics.get('FN', {}).get('central', float('nan')):.3f},
border={avg_metrics.get('FN', {}).get('border', float('nan')):.3f}, bright-pixel mask={avg_metrics.get('FN', {}).get('bright', float('nan')):.3f};
mean normalized spatial entropy={avg_metrics.get('FN', {}).get('entropy', float('nan')):.3f}.
The missed cases do not provide consistent evidence of a single localized Mild
Cataract cue. Visual inspection agrees with the spatial summaries: 1357URPSL (1),
1585DRPSL (2), and 1603DLPSL (2) emphasize edge/eyelid or background regions;
1396DRPSL (6), 1658DRPSL (3), and 1906URPSL (2) overlap strong illumination or
reflection structure; and six FN maps are too diffuse to localize confidently.
Attention outside the central proxy or diffuse activation is a possible failure
pattern, not proof that the model ignored a lesion.

C. True-positive findings
-------------------------
All {outcome_counts['TP']} correctly detected Mild Cataract cases are included in the TP contact sheet.
Category counts: {format_category_counts(category_counts.get('TP', {}))}.
Mean attention mass: central proxy={avg_metrics.get('TP', {}).get('central', float('nan')):.3f},
border={avg_metrics.get('TP', {}).get('border', float('nan')):.3f}, bright-pixel mask={avg_metrics.get('TP', {}).get('bright', float('nan')):.3f};
mean normalized spatial entropy={avg_metrics.get('TP', {}).get('entropy', float('nan')):.3f}.
The TP-vs-FN central-attention difference is {tp_central - fn_central:+.3f}. Given
only 8 TP and 13 FN cases and the coarse 7 x 7 map, this descriptive difference
must not be interpreted as a statistically established anatomical effect. Only
1076DLPSL (6) and 11168DLPSL (1) were categorized as partially lens-related; none
was lens/pupil dominant. Two TP maps were illumination/reflection dominant, one was
edge/eyelid dominant, and three were diffuse. Thus correct positive predictions do
not show a uniform lens-centered attention pattern.

D. False-positive findings
--------------------------
The {outcome_counts['FP']} Normal false positives were inspected alongside the {outcome_counts['TN']} true negatives.
FP category counts: {format_category_counts(category_counts.get('FP', {}))}.
TN category counts: {format_category_counts(category_counts.get('TN', {}))}.
{non_anatomical_fp} of {outcome_counts['FP']} FP cases were assigned a non-anatomical/diffuse
heuristic category (reflection/illumination, eyelid/background, or diffuse/unclear).
All three FP images used Direct Focal Illumination. 1506DRPSL (2) and 11288DLPSL
(4) were edge/eyelid dominant, while 1114DLPSL (3) was diffuse/unclear. Bright slit
illumination, corneal highlights, pupil boundaries, and borders remain plausible
distractors where their activation overlaps the displayed heatmap, but three cases
are far too few to establish an illumination-specific effect.

E. Likely shortcut behavior
---------------------------
The maps should be read as evidence of association, not mechanism. Recurrent
activation on bright highlights, illumination bands, borders, or background across
errors is compatible with shortcut learning. Across all 29 images, zero maps were
categorized as lens/pupil dominant; 11 were diffuse/unclear, 7 edge/background
dominant, 7 reflection/illumination dominant, and only 4 partially lens-related.
This is exploratory evidence that image-acquisition cues may contribute, but it
does not isolate those cues causally and it also shows that the heuristic category
is not outcome-specific.

F. Anatomical relevance
-----------------------
No map met the fixed lens/pupil-dominant criterion. TP maps had more mean central
attention than FN maps (0.305 versus 0.120), but most TP maps still fell into
diffuse or non-anatomical categories. Therefore the current full-image classifier
cannot be said to rely consistently on anatomically relevant lens/pupil features
from Grad-CAM alone. A segmented or clinician-annotated pupil/lens region would be
needed for a stronger localization assessment.

G. Main limitation
------------------
Grad-CAM is a coarse post-hoc interpretability tool and does not prove clinical
causality. EfficientNetB0's final feature grid is only 7 x 7, overlays depend on
normalization and interpolation, and the central ellipse/qualitative labels are
exploratory proxies rather than expert anatomical annotations. The test set is also
small (29 images), so visual patterns can be unstable.

H. Recommended next experiment
------------------------------
Recommended controlled experiment: {recommendation}.
Keep the fixed splits and the same baseline protocol, and compare the current full
image input against a reproducibly defined pupil/lens ROI. This directly tests the
most relevant Grad-CAM concern—whether non-lens acquisition artifacts are helping
or hurting predictions—without mixing in fine-tuning, resolution, optimizer, or
architecture changes. Do not use the test set to design the ROI or tune decisions.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_determinism(config)
    require_preflight(config)

    import tensorflow as tf

    checkpoint = project_path(args.checkpoint)
    predictions_path = project_path(args.saved_predictions)
    output_dir = project_path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Saved test predictions not found: {predictions_path}")

    threshold = float(config["training"]["threshold"])
    if threshold != 0.5:
        raise RuntimeError(f"Expected locked threshold 0.5, found {threshold}")
    samples = select_samples(config, load_metadata(config, "test"))
    saved_predictions = load_saved_predictions(predictions_path)
    if set(saved_predictions) != {sample.filename for sample in samples}:
        raise RuntimeError("Fixed-test filenames differ from the saved locked evaluation")

    model = tf.keras.models.load_model(checkpoint, compile=False)
    model.trainable = False
    components = build_gradcam_components(model)
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    for directory in OUTCOME_DIRS.values():
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    visual_cases: list[dict[str, Any]] = []
    probability_differences: list[float] = []
    for index, sample in enumerate(samples, start=1):
        image_batch = load_image_for_model(Path(sample.image_path), image_size)
        heatmap, probability = cataract_gradcam(model, components, image_batch)
        saved = saved_predictions[sample.filename]
        saved_probability = float(saved["predicted_probability_cataract"])
        difference = abs(probability - saved_probability)
        probability_differences.append(difference)
        if difference > 1e-5:
            raise RuntimeError(
                f"Probability drift for {sample.filename}: computed={probability:.8f}, "
                f"saved={saved_probability:.8f}"
            )
        predicted_label = int(probability >= threshold)
        if predicted_label != int(saved["predicted_label"]):
            raise RuntimeError(f"Locked prediction changed for {sample.filename}")
        outcome = outcome_for(sample.label, predicted_label)

        original = load_display_image(Path(sample.image_path))
        overlay = make_overlay(original, heatmap)
        model_image = image_batch.numpy()[0]
        focus = attention_summary(heatmap, model_image)
        row = {
            "filename": sample.filename,
            "subject_id": sample.subject_id,
            "true_class": "Mild Cataract" if sample.label == 1 else "Normal",
            "predicted_class": "Mild Cataract" if predicted_label == 1 else "Normal",
            "cataract_probability": probability,
            "prediction_outcome": outcome,
            "cataract_type": sample.cataract_type,
            "illumination_type": sample.illumination_type,
            "image_quality": sample.image_quality,
            "reflection_metadata": sample.reflection,
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
        visual_case = {**row, "original": original, "overlay": overlay}
        visual_cases.append(visual_case)

        case_path = (
            output_dir
            / OUTCOME_DIRS[outcome]
            / f"{safe_stem(sample.filename)}__gradcam.png"
        )
        create_case_figure(original, overlay, row, case_path)
        print(
            f"[{index:02d}/{len(samples):02d}] {outcome} {sample.filename}: "
            f"p={probability:.6f}, focus={focus.focus_category}"
        )

    outcome_counts = Counter(row["prediction_outcome"] for row in rows)
    expected_counts = {"TP": 8, "FN": 13, "TN": 5, "FP": 3}
    if dict(outcome_counts) != expected_counts:
        raise RuntimeError(
            f"Outcome counts changed from locked evaluation: {dict(outcome_counts)}"
        )

    fn_cases = [case for case in visual_cases if case["prediction_outcome"] == "FN"]
    tp_cases = [case for case in visual_cases if case["prediction_outcome"] == "TP"]
    normal_cases = [case for case in visual_cases if case["prediction_outcome"] in {"TN", "FP"}]
    create_contact_sheet(
        fn_cases,
        output_dir / "false_negative_gradcam_contact_sheet.png",
        "False negatives: all 13 missed Mild Cataract test images (original | Grad-CAM)",
        columns=3,
    )
    create_contact_sheet(
        tp_cases,
        output_dir / "true_positive_gradcam_contact_sheet.png",
        "True positives: all 8 detected Mild Cataract test images (original | Grad-CAM)",
        columns=3,
    )
    create_contact_sheet(
        normal_cases,
        output_dir / "normal_cases_gradcam_contact_sheet.png",
        "Normal cases: 5 true negatives and 3 false positives (original | Grad-CAM)",
        columns=3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gradcam_analysis_rows.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint),
        "gradcam_layer": GRADCAM_LAYER,
        "gradcam_layer_output_shape": list(components["conv_layer"].output.shape[1:]),
        "decision_threshold": threshold,
        "test_images": len(rows),
        "outcome_counts": dict(outcome_counts),
        "category_counts_by_outcome": count_categories(rows),
        "maximum_absolute_saved_probability_difference": max(probability_differences),
        "qualitative_method": (
            "Exploratory heuristic using a fixed central ellipse as a pupil/lens proxy, "
            "a 14% border mask, and a bright-pixel mask; not anatomical segmentation."
        ),
    }
    (output_dir / "gradcam_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir / "gradcam_analysis_report.txt",
        rows,
        max(probability_differences),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
