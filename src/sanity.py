"""Hard pre-training gates for fixed-split integrity and label semantics."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any

from data import load_metadata, select_samples
from utils import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    config_sha256,
    dataset_root,
    load_config,
    output_path,
    sha256_file,
    write_json,
)


def result(passed: bool, details: Any) -> dict[str, Any]:
    return {"pass": bool(passed), "details": details}


def training_code_is_test_isolated() -> tuple[bool, list[str]]:
    """Confirm train.py has no literal test split or test workbook reference."""
    train_path = PROJECT_ROOT / "src" / "train.py"
    tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    literals = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    )
    prohibited = [value for value in literals if value.casefold() in {"test", "test.xlsx"}]
    return not prohibited, prohibited


def run_checks(config: dict[str, Any]) -> dict[str, Any]:
    root = dataset_root(config)
    checks: dict[str, dict[str, Any]] = {}

    audit_path = output_path(config, "reports_dir") / "dataset_audit.txt"
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else ""
    checks["audit_ready"] = result(
        "READY FOR BASELINE TRAINING" in audit_text,
        str(audit_path),
    )

    all_rows = {}
    all_samples = {}
    workbook_hashes = {}
    for split, split_cfg in config["fixed_splits"].items():
        workbook = root / split_cfg["workbook"]
        actual_hash = sha256_file(workbook)
        workbook_hashes[split] = actual_hash
        checks[f"{split}_workbook_unchanged"] = result(
            actual_hash == split_cfg["sha256"],
            {"expected": split_cfg["sha256"], "actual": actual_hash},
        )

        rows = load_metadata(config, split)
        samples = select_samples(config, rows)
        all_rows[split] = rows
        all_samples[split] = samples
        expected_total = int(split_cfg["expected_total_rows"])
        checks[f"{split}_total_rows"] = result(
            len(rows) == expected_total,
            {"expected": expected_total, "actual": len(rows)},
        )
        actual_usable = Counter(sample.diagnosis for sample in samples)
        expected_usable = {
            name: int(count) for name, count in split_cfg["expected_usable"].items()
        }
        checks[f"{split}_usable_counts"] = result(
            dict(actual_usable) == expected_usable,
            {"expected": expected_usable, "actual": dict(actual_usable)},
        )

    subject_ids = {
        split: {row.subject_id for row in rows if row.subject_id}
        for split, rows in all_rows.items()
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(subject_ids[first] & subject_ids[second])
        checks[f"subject_overlap_{first}_{second}"] = result(not overlap, overlap)

    selected_diagnoses = {
        sample.diagnosis for samples in all_samples.values() for sample in samples
    }
    checks["other_excluded"] = result(
        "Other" not in selected_diagnoses,
        sorted(selected_diagnoses),
    )
    bad_positive_labels = [
        sample.to_dict()
        for samples in all_samples.values()
        for sample in samples
        if sample.label == 1 and sample.diagnosis != "Cataract"
    ]
    checks["label_1_is_cataract"] = result(not bad_positive_labels, bad_positive_labels)
    unexpected_labels = sorted(
        {
            sample.label
            for samples in all_samples.values()
            for sample in samples
            if sample.label not in {0, 1}
        }
    )
    checks["binary_labels_only"] = result(not unexpected_labels, unexpected_labels)

    isolated, prohibited = training_code_is_test_isolated()
    checks["training_code_test_isolation"] = result(
        isolated,
        {"prohibited_literals": prohibited, "training_splits": ["train", "validation"]},
    )
    checks["fixed_split_policy"] = result(
        set(config["fixed_splits"]) == {"train", "validation", "test"},
        "No random split operation is used; official workbooks are loaded directly.",
    )

    overall_pass = all(item["pass"] for item in checks.values())
    return {
        "overall_pass": overall_pass,
        "config_sha256": config_sha256(config),
        "dataset_root": str(root),
        "workbook_sha256": workbook_hashes,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = run_checks(config)
    output = output_path(config, "reports_dir") / "preflight_checks.json"
    write_json(output, report)
    print(f"Preflight: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"Report: {output}")
    if not report["overall_pass"]:
        for name, check in report["checks"].items():
            if not check["pass"]:
                print(f"FAILED {name}: {check['details']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

