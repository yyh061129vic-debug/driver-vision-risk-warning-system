"""下载并校验 team 3 sub/pidnet 使用的固定 PIDNet-S ONNX 资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "pidnet_s_cityscapes.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "team3sub-pidnet/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def validate_model_directory(config: dict, directory: Path) -> None:
    model = config["model"]
    required = {
        model["weights_file"]: (int(model["weights_size_bytes"]), str(model["weights_sha256"])),
        model["external_data_file"]: (
            int(model["external_data_size_bytes"]),
            str(model["external_data_sha256"]),
        ),
        model["metadata_file"]: (None, str(model["metadata_sha256"])),
        model["labels_file"]: (None, str(model["labels_sha256"])),
    }
    for filename, (expected_size, expected_sha) in required.items():
        path = directory / str(filename)
        if not path.is_file():
            raise RuntimeError(f"missing PIDNet asset file: {path}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(f"unexpected size for {filename}")
        if sha256(path) != expected_sha:
            raise RuntimeError(f"unexpected SHA256 for {filename}")

    metadata = json.loads((directory / str(model["metadata_file"])).read_text(encoding="utf-8"))
    model_files = metadata.get("model_files", {}).get(str(model["weights_file"]), {})
    input_shape = tuple(model_files.get("inputs", {}).get("image", {}).get("shape", []))
    output_shape = tuple(model_files.get("outputs", {}).get("mask", {}).get("shape", []))
    if input_shape != (1, 3, 1024, 2048):
        raise RuntimeError(f"unexpected PIDNet input shape: {input_shape}")
    if output_shape != (1, 19, 128, 256):
        raise RuntimeError(f"unexpected PIDNet output shape: {output_shape}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-url", help="Temporary override for the fixed PIDNet asset URL.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    directory = ROOT / model["local_directory"]
    directory.mkdir(parents=True, exist_ok=True)

    required_paths = [
        directory / str(model["weights_file"]),
        directory / str(model["external_data_file"]),
        directory / str(model["metadata_file"]),
        directory / str(model["labels_file"]),
    ]
    if all(path.is_file() for path in required_paths):
        validate_model_directory(config, directory)
        print(f"PIDNet checkpoint verification passed: {directory}")
        return 0

    asset_url = args.asset_url or str(model["asset_url"])
    with tempfile.TemporaryDirectory(prefix="pidnet-download-") as temporary_directory:
        archive_path = Path(temporary_directory) / "pidnet-onnx-float.zip"
        print("Downloading pinned PIDNet-S ONNX asset...")
        download(asset_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            root_prefix = "pidnet-onnx-float/"
            for filename in (
                str(model["weights_file"]),
                str(model["external_data_file"]),
                str(model["metadata_file"]),
                str(model["labels_file"]),
            ):
                member = root_prefix + filename
                target = directory / filename
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

    validate_model_directory(config, directory)
    print(f"PIDNet checkpoint verification passed: {directory}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"PIDNet model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
