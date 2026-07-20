"""下载并完整校验任务 5 固定版本的 SegFormer 权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/models/segformer_cityscapes.yaml"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def _sha256(path: Path) -> str:
    """分块计算文件 SHA-256，适用于较大的模型权重。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """下载文件到临时路径，成功后原子替换，并最多重试三次。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "driver-vision-risk/0.1"})
    last_error: OSError | None = None
    # ``.part`` 文件不会被误认为完整权重；失败时立即清理。
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(destination)
            return
        except OSError as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)
    raise OSError(f"download failed after three attempts: {url}") from last_error


def _validate_model_directory(config: dict[str, object], directory: Path) -> None:
    """核对模型架构、道路类别、权重大小和固定 SHA-256。"""

    model_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    architectures = model_config.get("architectures", [])
    expected_architecture = config["model"]["architecture"]
    if expected_architecture not in architectures:
        raise RuntimeError(f"unexpected model architecture: {architectures}")
    if str(model_config.get("id2label", {}).get("0")) != "road":
        raise RuntimeError("model class 0 must be road")

    weights = directory / config["model"]["weights_file"]
    expected_size = int(config["model"]["weights_size_bytes"])
    if weights.stat().st_size != expected_size:
        raise RuntimeError(
            f"weight size mismatch: expected {expected_size}, got {weights.stat().st_size}"
        )
    expected_sha256 = str(config["model"]["weights_sha256"])
    actual_sha256 = _sha256(weights)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"weight SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")


def main() -> int:
    """解析下载参数，补齐缺失文件并执行最终完整性校验。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--weights-url",
        help="Optional temporary official signed URL when the canonical weight URL is unreachable.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    directory = ROOT / model["local_directory"]
    repository = model["repository"]
    revision = model["revision"]
    # 已存在文件不重复下载，但循环结束后仍会统一验证关键内容。
    for filename in REQUIRED_FILES:
        destination = directory / filename
        if destination.is_file():
            print(f"Exists: {destination.relative_to(ROOT)}")
            continue
        if filename == model["weights_file"] and args.weights_url:
            url = args.weights_url
        else:
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true"
        print(f"Downloading {filename} from pinned official revision...")
        _download(str(url), destination)

    _validate_model_directory(config, directory)
    print(f"SegFormer checkpoint verification passed: {directory.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        # 对终端返回非零状态，防止自动流程继续使用不完整或错误的权重。
        print(f"Segmentation model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
