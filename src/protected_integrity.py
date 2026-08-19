"""Record and verify that completed experiment artifacts remain byte-identical."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import PROJECT_ROOT, read_json, sha256_file, write_json


NEW_ROOT = (PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320").resolve()
REPORTS = NEW_ROOT / "reports"


def manifest() -> dict[str, str]:
    roots = [PROJECT_ROOT / "outputs", PROJECT_ROOT / "configs"]
    items: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved == NEW_ROOT or NEW_ROOT in resolved.parents:
                continue
            items[str(resolved.relative_to(PROJECT_ROOT))] = sha256_file(resolved)
    dataset_root = (PROJECT_ROOT / ".." / "Fixed Dataset" / "Clean").resolve()
    for filename in ("train.xlsx", "val.xlsx", "test.xlsx"):
        path = dataset_root / filename
        items[f"../Fixed Dataset/Clean/{filename}"] = sha256_file(path)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    before_path = REPORTS / "protected_artifact_hashes_before.json"
    current = manifest()
    if args.mode == "before":
        write_json(before_path, current)
        print(f"Recorded {len(current)} protected hashes.")
        return 0
    before = read_json(before_path)
    changed = {
        name: {"before": before.get(name), "after": current.get(name)}
        for name in sorted(set(before) | set(current))
        if before.get(name) != current.get(name)
    }
    result = {
        "pass": not changed,
        "protected_file_count_before": len(before),
        "protected_file_count_after": len(current),
        "changed_or_missing": changed,
    }
    write_json(REPORTS / "protected_artifact_integrity_after.json", result)
    print(f"Protected artifact integrity: {'PASS' if result['pass'] else 'FAIL'}")
    if changed:
        for name in changed:
            print(name)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

