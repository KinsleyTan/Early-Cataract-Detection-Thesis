"""Configuration, paths, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def dataset_root(config: dict[str, Any]) -> Path:
    return project_path(config["paths"]["dataset_root"])


def output_path(config: dict[str, Any], key: str) -> Path:
    path = project_path(config["paths"][key])
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_environment(config: dict[str, Any]) -> None:
    """Set deterministic/runtime variables before TensorFlow is imported."""
    seed = int(config["experiment"]["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["KERAS_HOME"] = str(project_path(config["paths"]["keras_home"]))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def set_global_determinism(config: dict[str, Any]) -> None:
    """Seed Python, NumPy, and TensorFlow and request deterministic kernels."""
    configure_environment(config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def config_sha256(config: dict[str, Any]) -> str:
    return sha256_file(Path(config["_config_path"]))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_preflight(config: dict[str, Any]) -> dict[str, Any]:
    report = output_path(config, "reports_dir") / "preflight_checks.json"
    if not report.is_file():
        raise RuntimeError("Missing preflight_checks.json. Run src/sanity.py before training.")
    result = read_json(report)
    if not result.get("overall_pass"):
        raise RuntimeError("Preflight checks did not pass; training is blocked.")
    if result.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Configuration changed after preflight; rerun src/sanity.py.")
    return result

