"""Target-specific integrity audit for a configured fixed-split label policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit import hamming_distance, perceptual_hash
from data import MetadataRow, Sample, load_metadata, select_samples
from utils import (
    DEFAULT_CONFIG,
    config_sha256,
    dataset_root,
    load_config,
    output_path,
    project_path,
    sha256_file,
    write_json,
)


PROTOCOL_SECTIONS = ("data", "augmentation", "model", "training")


def protocol_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config["experiment"]["seed"]),
        **{section: config[section] for section in PROTOCOL_SECTIONS},
    }


def excluded_counts(rows: list[MetadataRow]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        diagnosis = row.diagnosis.casefold()
        grade = row.cataract_grade.casefold()
        if diagnosis == "normal":
            counts["Normal"] += 1
        elif diagnosis == "cataract" and grade == "mild":
            counts["Mild Cataract"] += 1
        elif diagnosis == "cataract" and grade == "severe":
            counts["Excluded Severe Cataract"] += 1
        else:
            counts["Excluded Other"] += 1
    return dict(counts)


def sample_description(sample: Sample) -> str:
    return (
        f"{sample.split}:{sample.filename} "
        f"(id={sample.subject_id}, diagnosis={sample.diagnosis}, grade={sample.cataract_grade})"
    )


def run_audit(
    config: dict[str, Any], reference_config: dict[str, Any]
) -> dict[str, Any]:
    root = dataset_root(config)
    split_rows: dict[str, list[MetadataRow]] = {}
    split_samples: dict[str, list[Sample]] = {}
    failures: list[str] = []
    warnings: list[str] = []
    workbook_hashes: dict[str, str] = {}

    protocol_match = protocol_snapshot(config) == protocol_snapshot(reference_config)
    if not protocol_match:
        failures.append("Configured training protocol differs from the completed reference baseline.")

    for split, split_cfg in config["fixed_splits"].items():
        workbook = root / split_cfg["workbook"]
        actual_hash = sha256_file(workbook)
        workbook_hashes[split] = actual_hash
        if actual_hash != split_cfg["sha256"]:
            failures.append(f"{split} workbook SHA-256 does not match the fixed audited split.")

        rows = load_metadata(config, split)
        samples = select_samples(config, rows)
        split_rows[split] = rows
        split_samples[split] = samples
        if len(rows) != int(split_cfg["expected_total_rows"]):
            failures.append(
                f"{split} total rows changed: expected {split_cfg['expected_total_rows']}, got {len(rows)}."
            )

        actual = Counter(sample.diagnosis for sample in samples)
        expected = {
            name: int(count) for name, count in split_cfg["expected_usable"].items()
        }
        if dict(actual) != expected:
            failures.append(
                f"{split} usable counts differ: expected {expected}, got {dict(actual)}."
            )
        minimum = int(config["audit"]["minimum_per_class"][split])
        label_counts = Counter(sample.label for sample in samples)
        for label in (0, 1):
            if label_counts[label] < minimum:
                failures.append(
                    f"{split} label {label} has {label_counts[label]} samples; minimum is {minimum}."
                )

    if len(split_samples["validation"]) < 20:
        warnings.append(
            f"Validation remains very small ({len(split_samples['validation'])} usable images)."
        )
    if len(split_samples["test"]) < 50:
        warnings.append(f"Locked test remains small ({len(split_samples['test'])} usable images).")

    bad_positive = [
        sample_description(sample)
        for samples in split_samples.values()
        for sample in samples
        if sample.label == 1
        and not (
            sample.diagnosis.casefold() == "cataract"
            and sample.cataract_grade.casefold() == "mild"
        )
    ]
    if bad_positive:
        failures.append("Label 1 contains non-mild cases: " + " | ".join(bad_positive))
    bad_negative = [
        sample_description(sample)
        for samples in split_samples.values()
        for sample in samples
        if sample.label == 0 and sample.diagnosis.casefold() != "normal"
    ]
    if bad_negative:
        failures.append("Label 0 contains non-Normal cases: " + " | ".join(bad_negative))

    subject_sets = {
        split: {sample.subject_id for sample in samples if sample.subject_id}
        for split, samples in split_samples.items()
    }
    subject_overlap: dict[str, list[str]] = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{first}_vs_{second}"
        overlap = sorted(subject_sets[first] & subject_sets[second])
        subject_overlap[key] = overlap
        if overlap:
            failures.append(f"Critical subject overlap in {key}: {overlap}")

    all_samples = [sample for samples in split_samples.values() for sample in samples]
    filename_groups: dict[str, list[Sample]] = defaultdict(list)
    hash_groups: dict[str, list[Sample]] = defaultdict(list)
    phash_by_path: dict[str, int] = {}
    sha_by_path: dict[str, str] = {}
    image_errors: list[str] = []
    for sample in all_samples:
        filename_groups[sample.filename.casefold()].append(sample)
        if sample.image_path not in sha_by_path:
            try:
                sha_by_path[sample.image_path] = sha256_file(Path(sample.image_path))
                phash_by_path[sample.image_path] = perceptual_hash(Path(sample.image_path))
            except (OSError, ValueError) as exc:
                image_errors.append(f"{sample.image_path}: {type(exc).__name__}: {exc}")
        if sample.image_path in sha_by_path:
            hash_groups[sha_by_path[sample.image_path]].append(sample)
    if image_errors:
        failures.append("Image hashing errors: " + " | ".join(image_errors))

    duplicate_filename_groups = [
        group
        for group in filename_groups.values()
        if len({sample.split for sample in group}) > 1
    ]
    if duplicate_filename_groups:
        failures.append(
            "Duplicate filenames across splits: "
            + " || ".join(" | ".join(map(sample_description, group)) for group in duplicate_filename_groups)
        )

    exact_duplicate_groups = [
        group
        for group in hash_groups.values()
        if len(group) > 1 and len({sample.split for sample in group}) > 1
    ]
    if exact_duplicate_groups:
        failures.append(
            "SHA-256 duplicates across splits: "
            + " || ".join(" | ".join(map(sample_description, group)) for group in exact_duplicate_groups)
        )

    threshold = int(config["audit"]["phash_threshold"])
    near_duplicate_pairs: list[dict[str, Any]] = []
    for index, first in enumerate(all_samples):
        for second in all_samples[index + 1 :]:
            if first.split == second.split:
                continue
            if first.image_path not in phash_by_path or second.image_path not in phash_by_path:
                continue
            if sha_by_path[first.image_path] == sha_by_path[second.image_path]:
                continue
            distance = hamming_distance(
                phash_by_path[first.image_path], phash_by_path[second.image_path]
            )
            if distance <= threshold:
                near_duplicate_pairs.append(
                    {
                        "distance": distance,
                        "first": sample_description(first),
                        "second": sample_description(second),
                    }
                )
    if near_duplicate_pairs:
        failures.append(
            f"Found {len(near_duplicate_pairs)} pHash cross-split pair(s) at distance <= {threshold}."
        )

    counts = {}
    for split in config["fixed_splits"]:
        labels = Counter(sample.label for sample in split_samples[split])
        counts[split] = {
            "source_rows": len(split_rows[split]),
            "normal": labels[0],
            "mild_cataract": labels[1],
            "usable_total": len(split_samples[split]),
            "unique_subjects": len(subject_sets[split]),
            **excluded_counts(split_rows[split]),
        }

    return {
        "overall_pass": not failures,
        "verdict": "MILD TASK AUDIT PASSED WITH WARNINGS" if not failures and warnings else (
            "MILD TASK AUDIT PASSED" if not failures else "MILD TASK AUDIT FAILED"
        ),
        "ready_phrase": config["audit"]["ready_phrase"] if not failures else "",
        "config_sha256": config_sha256(config),
        "reference_config_sha256": config_sha256(reference_config),
        "dataset_root": str(root),
        "protocol_matches_reference": protocol_match,
        "workbook_sha256": workbook_hashes,
        "counts": counts,
        "subject_overlap": subject_overlap,
        "duplicate_filename_groups_across_splits": len(duplicate_filename_groups),
        "exact_duplicate_groups_across_splits": len(exact_duplicate_groups),
        "near_duplicate_pairs_across_splits": near_duplicate_pairs,
        "phash_threshold": threshold,
        "warnings": warnings,
        "failures": failures,
    }


def write_text_report(config: dict[str, Any], result: dict[str, Any], path: Path) -> None:
    lines = [
        "NORMAL VS MILD CATARACT TARGET AUDIT",
        "=" * 80,
        f"Verdict: {result['verdict']}",
    ]
    if result["overall_pass"]:
        lines.append(result["ready_phrase"])
    lines.extend(
        [
            "Official train.xlsx, val.xlsx, and test.xlsx assignments remain fixed.",
            "Label 0: diagnosis=Normal.",
            "Label 1: diagnosis=Cataract AND cataract_grade=mild.",
            "Severe Cataract and Other are excluded without modifying source rows.",
            f"Protocol matches completed Normal-vs-All-Cataract baseline: {result['protocol_matches_reference']}",
            "",
            "FILTERED SPLIT COUNTS",
            "-" * 80,
            "split       source  normal  mild  usable  unique_subjects  excluded_severe  excluded_other",
        ]
    )
    for split, counts in result["counts"].items():
        lines.append(
            f"{split:<11} {counts['source_rows']:>6} {counts['normal']:>7} "
            f"{counts['mild_cataract']:>5} {counts['usable_total']:>7} "
            f"{counts['unique_subjects']:>16} {counts.get('Excluded Severe Cataract', 0):>16} "
            f"{counts.get('Excluded Other', 0):>15}"
        )
    lines.extend(["", "LEAKAGE AND DUPLICATES", "-" * 80])
    for pair, overlap in result["subject_overlap"].items():
        lines.append(f"Subject overlap {pair}: {len(overlap)} ({overlap or 'none'})")
    lines.append(
        f"Duplicate filenames across splits: {result['duplicate_filename_groups_across_splits']}"
    )
    lines.append(
        f"SHA-256 duplicate groups across splits: {result['exact_duplicate_groups_across_splits']}"
    )
    lines.append(
        f"pHash near-duplicate pairs across splits: {len(result['near_duplicate_pairs_across_splits'])} "
        f"(64-bit pHash Hamming distance <= {result['phash_threshold']})"
    )
    lines.extend(["", "WARNINGS", "-" * 80])
    lines.extend(result["warnings"] or ["none"])
    lines.extend(["", "FAILURES", "-" * 80])
    lines.extend(result["failures"] or ["none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    reference = load_config(args.reference_config)
    result = run_audit(config, reference)
    reports = output_path(config, "reports_dir")
    json_path = reports / "task_audit.json"
    text_path = project_path(config["paths"]["audit_report"])
    text_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_path, result)
    write_text_report(config, result, text_path)
    print(f"Target audit: {result['verdict']}")
    print(f"Report: {text_path}")
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

