"""校验已脱敏的本地开发环境基线快照。"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "metadata/environment/baseline-2026-07-16.yaml"
VALID_STATUSES = {"available", "installed", "not_installed", "unavailable"}
REQUIRED_PACKAGES = {
    "pyyaml",
    "pytest",
    "torch",
    "torchvision",
    "torchaudio",
    "onnx",
    "onnxruntime",
    "tensorrt",
    "opencv_python",
    "transformers",
    "numpy",
    "pillow",
    "carla",
}
SENSITIVE_KEYS = {"hostname", "username", "serial_number", "uuid", "user_directory"}


def _walk(value: object):
    """递归遍历嵌套映射和列表，逐项产出键和值。"""

    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _validate_component(name: str, value: object, errors: list[str]) -> None:
    """校验带 ``status`` 和 ``version`` 的软件或工具条目。"""

    if not isinstance(value, dict):
        errors.append(f"{name} must be a mapping")
        return
    status = value.get("status")
    version = value.get("version")
    if status not in VALID_STATUSES:
        errors.append(f"{name}.status must be one of {sorted(VALID_STATUSES)}")
    if status == "installed" and not version:
        errors.append(f"{name}.version is required when installed")
    if status == "not_installed" and version is not None:
        errors.append(f"{name}.version must be null when not installed")


def validate() -> list[str]:
    """检查系统、硬件、框架版本和隐私字段并返回全部错误。"""

    errors: list[str] = []
    try:
        baseline = yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read environment baseline: {exc}"]

    if not isinstance(baseline, dict):
        return ["environment baseline root must be a mapping"]
    if baseline.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if baseline.get("environment_role") != "local_development":
        errors.append("environment_role must be local_development")
    if not baseline.get("captured_at"):
        errors.append("captured_at is required")

    privacy = baseline.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("sanitized") is not True:
        errors.append("privacy.sanitized must be true")

    system = baseline.get("system")
    if not isinstance(system, dict) or not isinstance(system.get("os"), dict):
        errors.append("system.os must be a mapping")
    elif not all(system["os"].get(key) for key in ("name", "version", "build", "architecture")):
        errors.append("system.os must include name, version, build, and architecture")

    hardware = baseline.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("hardware must be a mapping")
    else:
        cpu = hardware.get("cpu")
        if not isinstance(cpu, dict) or not all(
            cpu.get(key) for key in ("model", "physical_cores", "logical_processors")
        ):
            errors.append("hardware.cpu baseline is incomplete")
        storage = hardware.get("storage")
        if not isinstance(storage, dict) or not all(
            isinstance(storage.get(key), (int, float)) and storage[key] > 0
            for key in ("total_gib", "free_gib")
        ):
            errors.append("hardware.storage total_gib and free_gib must be positive")
        gpus = hardware.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            errors.append("hardware.gpus must contain at least one GPU")
        elif not any(gpu.get("vendor") == "NVIDIA" for gpu in gpus if isinstance(gpu, dict)):
            errors.append("hardware.gpus must include the measured NVIDIA GPU")

    cuda = baseline.get("cuda")
    if not isinstance(cuda, dict):
        errors.append("cuda must be a mapping")
    else:
        toolkit = cuda.get("toolkit")
        _validate_component("cuda.toolkit", toolkit, errors)
        framework_runtime = cuda.get("framework_runtime")
        _validate_component("cuda.framework_runtime", framework_runtime, errors)
        if isinstance(framework_runtime, dict) and framework_runtime.get("cuda_available") is not True:
            errors.append("cuda.framework_runtime must record an available CUDA device")
        compatibility = cuda.get("driver_compatibility")
        if not isinstance(compatibility, dict) or not compatibility.get("maximum_cuda_version"):
            errors.append("cuda.driver_compatibility.maximum_cuda_version is required")

    python_environment = baseline.get("python_environment")
    if not isinstance(python_environment, dict) or not all(
        python_environment.get(key) for key in ("implementation", "version", "architecture")
    ):
        errors.append("python_environment baseline is incomplete")

    packages = baseline.get("packages")
    if not isinstance(packages, dict):
        errors.append("packages must be a mapping")
    else:
        missing = REQUIRED_PACKAGES - packages.keys()
        if missing:
            errors.append(f"packages missing required entries: {sorted(missing)}")
        for name, value in packages.items():
            _validate_component(f"packages.{name}", value, errors)
        torch_entry = packages.get("torch")
        if isinstance(torch_entry, dict):
            if not str(torch_entry.get("version", "")).endswith("+cu130"):
                errors.append("packages.torch must record the installed CUDA 13.0 build")
            if torch_entry.get("cuda_available") is not True:
                errors.append("packages.torch.cuda_available must be true")

    tools = baseline.get("tools")
    if not isinstance(tools, dict):
        errors.append("tools must be a mapping")
    else:
        for name, value in tools.items():
            _validate_component(f"tools.{name}", value, errors)

    # 最后递归扫描整个快照，避免嵌套位置意外写入本机敏感信息。
    for key, value in _walk(baseline):
        if key.lower() in SENSITIVE_KEYS and value not in (None, ""):
            errors.append(f"sensitive field must not contain a value: {key}")
        if isinstance(value, str) and "c:\\users\\" in value.lower():
            errors.append(f"absolute user directory must not be stored: {key}")

    return errors


def main() -> int:
    """打印环境基线验收结论并返回标准退出码。"""

    errors = validate()
    if errors:
        print("Environment baseline validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Environment baseline validation passed.")
    print("Checked system, hardware, CUDA, Python, packages, tools, and privacy fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
