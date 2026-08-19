"""Audit the controlled ROI 224-to-320 resolution change before training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from data import load_metadata, select_samples
from roi import roi_box_for_dimensions
from task_audit import run_audit
from utils import PROJECT_ROOT, load_config, output_path, read_json, write_json


CONFIG_224 = PROJECT_ROOT / "configs" / "mild_cataract_roi.yaml"
CONFIG_320 = PROJECT_ROOT / "configs" / "mild_cataract_roi_320.yaml"
ROI_AUDIT_224 = (
    PROJECT_ROOT
    / "outputs"
    / "mild_cataract"
    / "roi_experiment"
    / "reports"
    / "roi_audit.json"
)


def differences(first: Any, second: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(first, dict) and isinstance(second, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in first:
                rows.append({"path": path, "roi_224": None, "roi_320": second[key]})
            elif key not in second:
                rows.append({"path": path, "roi_224": first[key], "roi_320": None})
            else:
                rows.extend(differences(first[key], second[key], path))
        return rows
    if first != second:
        return [{"path": prefix, "roi_224": first, "roi_320": second}]
    return []


def main() -> int:
    config_224 = load_config(CONFIG_224)
    config_320 = load_config(CONFIG_320)
    normalized_reference = copy.deepcopy(config_224)
    normalized_reference["data"]["image_size"] = copy.deepcopy(
        config_320["data"]["image_size"]
    )
    audit = run_audit(config_320, normalized_reference)

    controlled_sections = (
        "fixed_splits",
        "label_policy",
        "roi",
        "augmentation",
        "model",
        "training",
    )
    controlled_differences = []
    for section in controlled_sections:
        controlled_differences.extend(
            differences(config_224[section], config_320[section], section)
        )
    data_224 = {key: value for key, value in config_224["data"].items() if key != "image_size"}
    data_320 = {key: value for key, value in config_320["data"].items() if key != "image_size"}
    controlled_differences.extend(differences(data_224, data_320, "data"))

    samples = select_samples(config_320, load_metadata(config_320, "train"))
    from PIL import Image

    with Image.open(samples[0].image_path) as image:
        source_dimensions = [image.width, image.height]
        roi_box = roi_box_for_dimensions(image.width, image.height, config_320["roi"])
    expected_box = [928, 424, 3105, 2601]
    actual_box = list(roi_box.as_tuple())
    roi_coordinates_pass = actual_box == expected_box
    resolution_pass = (
        config_224["data"]["image_size"] == [224, 224]
        and config_320["data"]["image_size"] == [320, 320]
    )
    previous_roi_audit = read_json(ROI_AUDIT_224)
    roi_audit_reuse_pass = previous_roi_audit.get("training_allowed") is True

    failures = list(audit["failures"])
    if controlled_differences:
        failures.append(f"Unexpected controlled-protocol differences: {controlled_differences}")
    if not resolution_pass:
        failures.append("Expected resolution change 224x224 to 320x320 was not configured.")
    if not roi_coordinates_pass:
        failures.append(f"ROI coordinates changed: expected {expected_box}, got {actual_box}.")
    if not roi_audit_reuse_pass:
        failures.append("The completed train-only ROI visual audit was not passed.")

    overall_pass = not failures
    result = {
        "overall_pass": overall_pass,
        "ready_phrase": config_320["audit"]["ready_phrase"] if overall_pass else "",
        "only_intended_variable_changed": not controlled_differences and resolution_pass,
        "resolution": {"roi_224": [224, 224], "roi_320": [320, 320]},
        "controlled_protocol_differences": controlled_differences,
        "source_dimensions": source_dimensions,
        "roi_coordinates_expected": expected_box,
        "roi_coordinates_actual": actual_box,
        "roi_coordinates_unchanged": roi_coordinates_pass,
        "reused_train_only_roi_visual_audit": str(ROI_AUDIT_224),
        "reused_roi_visual_audit_pass": roi_audit_reuse_pass,
        "split_audit": audit,
        "failures": failures,
    }
    reports = output_path(config_320, "reports_dir")
    write_json(reports / "resolution_audit.json", result)
    reused = dict(previous_roi_audit)
    reused["reused_without_retuning"] = True
    reused["source_audit"] = str(ROI_AUDIT_224)
    reused["resolution_change_does_not_change_crop"] = True
    write_json(reports / "roi_audit.json", reused)

    counts = audit["counts"]
    overlaps = audit["subject_overlap"]
    lines = [
        "ROI 224x224 VS ROI 320x320 PRE-TRAINING AUDIT",
        "=" * 80,
        f"Verdict: {'PASS' if overall_pass else 'FAIL'}",
    ]
    if overall_pass:
        lines.append(config_320["audit"]["ready_phrase"])
    lines.extend(
        [
            "",
            "CONTROLLED VARIABLE",
            "-" * 80,
            "Only intended scientific variable: input resize 224x224 -> 320x320 RGB.",
            f"Unexpected controlled-protocol differences: {controlled_differences or 'none'}",
            f"ROI crop unchanged: {actual_box} from source {source_dimensions}.",
            "The already approved train-only ROI visual audit was reused; ROI was not retuned.",
            "EfficientNetB0, ImageNet weights, frozen backbone, head, optimizer, learning rate, "
            "batch size, seed, augmentation, callbacks, val-loss selection, and threshold remain fixed.",
            "",
            "FIXED SPLITS AND USABLE COUNTS",
            "-" * 80,
        ]
    )
    for split in ("train", "validation", "test"):
        value = counts[split]
        lines.append(
            f"{split}: {value['usable_total']} usable "
            f"(Normal={value['normal']}, Mild Cataract={value['mild_cataract']}), "
            f"unique subjects={value['unique_subjects']}"
        )
    lines.extend(
        [
            "",
            "LEAKAGE AND DUPLICATES",
            "-" * 80,
            f"Train vs validation subject overlap: {len(overlaps['train_vs_validation'])}",
            f"Train vs test subject overlap: {len(overlaps['train_vs_test'])}",
            f"Validation vs test subject overlap: {len(overlaps['validation_vs_test'])}",
            f"Duplicate filenames across splits: {audit['duplicate_filename_groups_across_splits']}",
            f"Exact SHA-256 duplicate groups across splits: {audit['exact_duplicate_groups_across_splits']}",
            f"Near-duplicate pHash pairs across splits: {len(audit['near_duplicate_pairs_across_splits'])} "
            f"(64-bit pHash Hamming distance <= {audit['phash_threshold']})",
            "",
            "WARNINGS",
            "-" * 80,
            *(audit["warnings"] or ["none"]),
            "",
            "FAILURES",
            "-" * 80,
            *(failures or ["none"]),
        ]
    )
    (reports / "resolution_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

