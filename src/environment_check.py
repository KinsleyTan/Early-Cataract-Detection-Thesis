"""Record TensorFlow and NVIDIA device visibility before the ROI-320 run."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from utils import PROJECT_ROOT


OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "mild_cataract" / "roi_resolution_320" / "reports"
)


def command_output(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "output": None, "returncode": None}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "output": (completed.stdout or completed.stderr).strip(),
    }


def main() -> int:
    import keras
    import tensorflow as tf

    system = platform.system()
    release = platform.release()
    version = platform.version()
    wsl_env = bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))
    wsl_kernel = bool(re.search(r"microsoft|wsl", release, flags=re.IGNORECASE))
    under_wsl2 = system == "Linux" and (wsl_env or wsl_kernel)
    native_windows = system == "Windows" and not under_wsl2

    all_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices("GPU")
    nvidia_smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_smi_full = command_output(["nvidia-smi"])

    gpu_name = None
    driver_version = None
    if nvidia_smi["available"] and nvidia_smi["returncode"] == 0 and nvidia_smi["output"]:
        first_line = str(nvidia_smi["output"]).splitlines()[0]
        fields = [field.strip() for field in first_line.split(",")]
        if len(fields) >= 2:
            gpu_name, driver_version = fields[:2]

    cuda_version = None
    full_output = str(nvidia_smi_full.get("output") or "")
    match = re.search(r"CUDA Version:\s*([0-9.]+)", full_output)
    if match:
        cuda_version = match.group(1)

    result = {
        "operating_system": {
            "system": system,
            "release": release,
            "version": version,
            "platform": platform.platform(),
        },
        "python_version": sys.version,
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "tensorflow_physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in all_devices
        ],
        "tensorflow_gpu_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in gpu_devices
        ],
        "tensorflow_built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "tensorflow_gpu_available": bool(gpu_devices),
        "native_windows": native_windows,
        "wsl2": under_wsl2,
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_full": nvidia_smi_full,
        "nvidia_gpu_name": gpu_name,
        "nvidia_driver_version": driver_version,
        "nvidia_reported_cuda_version": cuda_version,
    }
    if gpu_devices:
        result["training_device_conclusion"] = (
            "TensorFlow detects at least one GPU; the ROI-320 run may use GPU acceleration."
        )
    elif native_windows and tuple(int(part) for part in tf.__version__.split(".")[:2]) >= (2, 11):
        result["training_device_conclusion"] = (
            "TensorFlow cannot use native-Windows CUDA in TensorFlow 2.11 or newer. "
            "This isolated run will execute on CPU unless launched from a separate WSL2 GPU environment."
        )
    else:
        result["training_device_conclusion"] = (
            "TensorFlow reports no GPU device; the isolated run will execute on CPU."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "environment_check.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    text_lines = [
        "PRE-TRAINING ENVIRONMENT CHECK",
        "==============================",
        f"Operating system: {result['operating_system']['platform']}",
        f"Execution mode: {'WSL2' if under_wsl2 else 'native Windows' if native_windows else system}",
        f"Python: {sys.version.splitlines()[0]}",
        f"Python executable: {sys.executable}",
        f"TensorFlow: {tf.__version__}",
        f"Keras: {keras.__version__}",
        f"tf.config.list_physical_devices(): {all_devices}",
        f"tf.config.list_physical_devices('GPU'): {gpu_devices}",
        f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}",
        f"nvidia-smi GPU: {gpu_name or 'not detected'}",
        f"NVIDIA driver: {driver_version or 'not available'}",
        f"nvidia-smi CUDA compatibility: {cuda_version or 'not available'}",
        "",
        str(result["training_device_conclusion"]),
        "",
        "FULL NVIDIA-SMI",
        "----------------",
        full_output or "nvidia-smi unavailable",
    ]
    (OUTPUT_DIR / "environment_check.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
