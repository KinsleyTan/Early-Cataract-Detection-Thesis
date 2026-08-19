"""Verify EfficientNetB0 model compatibility and parameter invariance at 320x320."""

from __future__ import annotations

import json

from model import build_model, parameter_counts
from utils import PROJECT_ROOT, configure_environment, load_config, read_json, write_json


def main() -> int:
    config = load_config(PROJECT_ROOT / "configs" / "mild_cataract_roi_320.yaml")
    configure_environment(config)
    model, backbone = build_model(config)
    trainable, non_trainable = parameter_counts(model)
    reference = read_json(
        PROJECT_ROOT
        / "outputs"
        / "mild_cataract"
        / "roi_experiment"
        / "reports"
        / "training_summary.json"
    )
    expected = {
        "trainable_parameters": int(reference["trainable_parameters"]),
        "non_trainable_parameters": int(reference["non_trainable_parameters"]),
    }
    actual = {
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
    }
    result = {
        "pass": (
            list(model.input_shape[1:]) == [320, 320, 3]
            and backbone.trainable is False
            and actual == expected
        ),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "backbone": backbone.name,
        "backbone_trainable": backbone.trainable,
        "weights_policy": config["model"]["weights"],
        "include_top": config["model"]["include_top"],
        "classification_head": "GlobalAveragePooling2D -> Dense(64, ReLU, L2=0.0001) -> Dropout(0.30) -> sigmoid",
        "expected_parameter_counts_from_roi_224": expected,
        "actual_parameter_counts_roi_320": actual,
        "parameter_counts_unchanged": actual == expected,
    }
    output = (
        PROJECT_ROOT
        / "outputs"
        / "mild_cataract"
        / "roi_resolution_320"
        / "reports"
        / "model_compatibility.json"
    )
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
