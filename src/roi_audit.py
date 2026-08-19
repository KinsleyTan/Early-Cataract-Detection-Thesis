"""Create and record the pre-training visual audit for the fixed ROI rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data import load_metadata, select_samples
from roi import crop_pil, roi_box_for_dimensions, validate_roi_config
from utils import PROJECT_ROOT, load_config, project_path, write_json


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--review-status", choices=("pending", "pass", "fail"), default="pending"
    )
    parser.add_argument(
        "--failure",
        action="append",
        default=[],
        help="Filename or concise note for a visually identified ROI failure.",
    )
    return parser.parse_args()


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def boxed_image(image: Image.Image, roi_config: dict[str, Any]) -> Image.Image:
    result = image.copy()
    box = roi_box_for_dimensions(result.width, result.height, roi_config)
    draw = ImageDraw.Draw(result)
    width = max(8, result.width // 300)
    draw.rectangle(box.as_tuple(), outline="#FF2D2D", width=width)
    return result


def make_contact_sheet(
    samples,
    roi_config: dict[str, Any],
    output_path: Path,
    title: str,
    columns: int = 3,
) -> None:
    image_size = (220, 165)
    panel_width = image_size[0] * 3 + 24
    panel_height = image_size[1] + 92
    rows = math.ceil(len(samples) / columns)
    sheet = Image.new("RGB", (columns * panel_width, 60 + rows * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=17)
    font = ImageFont.load_default(size=13)
    draw.text((12, 16), title, fill="black", font=title_font)
    for index, sample in enumerate(samples):
        x = (index % columns) * panel_width + 6
        y = 60 + (index // columns) * panel_height
        original = load_rgb(sample.image_path)
        boxed = boxed_image(original, roi_config)
        cropped, box = crop_pil(original, roi_config)
        for offset, image in enumerate((original, boxed, cropped)):
            sheet.paste(fit(image, image_size), (x + offset * (image_size[0] + 4), y))
        label = "Mild Cataract" if sample.label == 1 else "Normal"
        lines = [
            sample.filename,
            f"{label} | {sample.illumination_type}",
            f"ROI px: ({box.left},{box.top})-({box.right},{box.bottom}) | source={original.width}x{original.height}",
            "Panels: original | bounding box | cropped ROI",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x, y + image_size[1] + 5 + line_index * 18), line, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def deterministic_representative_sample(samples) -> list:
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample.label, sample.illumination_type)].append(sample)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda sample: sample.filename.casefold())
        indexes = sorted({0, len(group) // 2, len(group) - 1})
        selected.extend(group[index] for index in indexes)
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def protected_manifest() -> dict[str, str]:
    protected = [PROJECT_ROOT / "outputs" / "reports", PROJECT_ROOT / "outputs" / "mild_cataract"]
    manifest = {}
    roi_root = (PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment").resolve()
    for root in protected:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if roi_root in resolved.parents:
                continue
            manifest[str(resolved.relative_to(PROJECT_ROOT))] = sha256(resolved)
    return manifest


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    roi_config = config.get("roi", {})
    if not roi_config.get("enabled", False):
        raise RuntimeError("ROI must be enabled for this audit")
    validate_roi_config(roi_config)
    train_samples = select_samples(config, load_metadata(config, "train"))
    representatives = deterministic_representative_sample(train_samples)
    audit_dir = PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_experiment" / "roi_audit"
    reports_dir = project_path(config["paths"]["reports_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    make_contact_sheet(
        representatives,
        roi_config,
        audit_dir / "roi_audit_representative_train.png",
        "Train-only representative ROI audit: original | fixed box | ROI",
        columns=3,
    )
    grouped = defaultdict(list)
    for sample in train_samples:
        grouped[(sample.label, sample.illumination_type)].append(sample)
    group_files = []
    for (label, illumination), samples in sorted(grouped.items()):
        label_name = "mild" if label == 1 else "normal"
        safe_illumination = illumination.casefold().replace(" ", "_")
        filename = f"all_train_{label_name}_{safe_illumination}.png"
        make_contact_sheet(
            sorted(samples, key=lambda sample: sample.filename.casefold()),
            roi_config,
            audit_dir / filename,
            f"All train: {label_name.title()} | {illumination}",
            columns=3,
        )
        group_files.append(filename)

    first_image = load_rgb(train_samples[0].image_path)
    box = roi_box_for_dimensions(first_image.width, first_image.height, roi_config)
    result = {
        "review_status": args.review_status,
        "training_allowed": args.review_status == "pass",
        "selection_scope": "ROI parameters and visual audit used training images only.",
        "method": roi_config["method"],
        "parameters": roi_config,
        "source_dimensions": [first_image.width, first_image.height],
        "roi_coordinates_for_dataset_dimensions": list(box.as_tuple()),
        "roi_dimensions": [box.width, box.height],
        "train_images_reviewable": len(train_samples),
        "representative_images": len(representatives),
        "representative_group_count": len(grouped),
        "group_contact_sheets": group_files,
        "visually_identified_failures": args.failure,
        "failure_count": len(args.failure),
    }
    write_json(reports_dir / "roi_audit.json", result)
    write_json(reports_dir / "protected_artifact_hashes_before.json", protected_manifest())
    report = f"""PUPIL/LENS ROI PRE-TRAINING AUDIT
=====================================

Review status: {args.review_status.upper()}
Training allowed by audit: {result['training_allowed']}

METHOD
------
Deterministic label-independent fixed center-square crop. The crop center is
({roi_config['center_x_fraction']:.2f}, {roi_config['center_y_fraction']:.2f}) of the source image and its side is
{roi_config['side_fraction_of_short_edge']:.2f} of the source short edge. For the uniform {first_image.width}x{first_image.height}
dataset images, this is pixel box ({box.left}, {box.top})-({box.right}, {box.bottom}),
producing a {box.width}x{box.height} square before bilinear resize to 224x224.

SELECTION AND LEAKAGE CONTROL
-----------------------------
ROI parameters were selected and audited from the official training split only.
No validation or test labels/images were used to choose or tune ROI coordinates.
The rule does not inspect diagnosis, label, illumination metadata, or pixels.

AUDIT COVERAGE
--------------
Representative contact sheet: 18 train images, covering Normal and Mild Cataract
for Diffuse, Direct Focal, and Retro Illumination (three deterministic examples per group).
Six group sheets expose all {len(train_samples)} usable training images for visual review.
Each panel shows original, bounding box, and cropped ROI.

VISUALLY IDENTIFIED FAILURES
----------------------------
{chr(10).join(args.failure) if args.failure else 'None recorded at this review stage.'}

GATE
----
{'PASS: The fixed ROI is approved for the controlled training run.' if result['training_allowed'] else 'STOP: Training remains blocked until this visual audit is explicitly passed.'}
"""
    (reports_dir / "roi_audit_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if args.review_status != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
