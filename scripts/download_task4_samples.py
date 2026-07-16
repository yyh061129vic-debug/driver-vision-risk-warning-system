"""仅获取任务 4 可视化清单所需的最小原始数据。"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/task4_samples.yaml"


class HTTPRangeReader(io.RawIOBase):
    """支持 seek 的 HTTP 分段读取器，用有限缓存访问远端 ZIP。"""

    def __init__(self, url: str, cache_size: int = 8 * 1024 * 1024) -> None:
        """通过 HEAD 请求确认远端长度及字节范围能力。"""

        self.url = url
        self.cache_size = cache_size
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            ranges = response.headers.get("Accept-Ranges", "")
        if length is None:
            raise RuntimeError("remote ZIP does not report Content-Length")
        if ranges.lower() != "bytes":
            raise RuntimeError("remote ZIP does not advertise byte-range support")
        self.length = int(length)

    def readable(self) -> bool:
        """声明该流支持读取，供 ``zipfile`` 查询能力。"""

        return True

    def seekable(self) -> bool:
        """声明该流支持随机定位，远端 ZIP 中央目录依赖此能力。"""

        return True

    def tell(self) -> int:
        """返回当前逻辑读取位置。"""

        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """按文件开头、当前位置或文件结尾更新逻辑读取位置。"""

        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        """优先从预读缓存返回数据，未命中时发起 HTTP Range 请求。"""

        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        cache_end = self.cache_start + len(self.cache)
        # 缓存未覆盖本次请求时，至少预读 ``cache_size`` 以减少网络往返。
        if not (self.cache_start <= self.position and self.position + size <= cache_end):
            fetch_size = max(size, self.cache_size)
            end = min(self.length - 1, self.position + fetch_size - 1)
            request = urllib.request.Request(
                self.url,
                headers={"Range": f"bytes={self.position}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"server ignored byte range: HTTP {response.status}")
                self.cache_start = self.position
                self.cache = response.read()
            cache_end = self.cache_start + len(self.cache)
        start = self.position - self.cache_start
        available = min(size, cache_end - self.position)
        data = self.cache[start : start + available]
        self.position += len(data)
        return data


def _hash_file(path: Path, algorithm: str) -> str:
    """使用指定摘要算法分块校验本地归档文件。"""

    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(root: Path, metadata: dict[str, object]) -> None:
    """依据配置核对归档是否存在、大小正确且摘要一致。"""

    path = root / str(metadata["path"])
    if not path.is_file():
        raise FileNotFoundError(f"required archive not found: {path.relative_to(root)}")
    expected_size = int(metadata["size_bytes"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {path.name}")
    for algorithm in ("md5", "sha256"):
        expected = metadata.get(algorithm)
        if expected and _hash_file(path, algorithm) != expected:
            raise RuntimeError(f"{algorithm} mismatch for {path.name}")


def _lost_and_found_paths(sample: dict[str, object]) -> tuple[str, Path]:
    """构造 Lost and Found 样例在 ZIP 中的后缀和本地目标路径。"""

    split = str(sample["split"])
    sequence = str(sample["sequence_id"])
    frame = str(sample["frame_id"])
    stem = f"{sequence}_{frame}"
    archive_suffix = f"leftImg8bit/{split}/{sequence}/{stem}_leftImg8bit.png"
    local_path = ROOT / "data_raw/lost-and-found/leftImg8bit_samples" / archive_suffix
    return archive_suffix, local_path


def _extract_selected_lost_and_found_images(config: dict[str, object], url: str) -> int:
    """通过远端 ZIP 随机读取，只解压配置中选定的图像。"""

    dataset = next(item for item in config["datasets"] if item["id"] == "lost-and-found")
    requested = [_lost_and_found_paths(sample) for sample in dataset["samples"]]
    if all(target.is_file() for _, target in requested):
        print("Lost and Found selected images already exist; skipping remote extraction.")
        return len(requested)

    reader = HTTPRangeReader(url)
    expected_size = int(config["archives"]["lost_and_found_images"]["size_bytes"])
    if reader.length != expected_size:
        raise RuntimeError(
            f"leftImg8bit.zip size mismatch: expected {expected_size}, got {reader.length}"
        )

    extracted = 0
    # 不下载完整大归档，只根据中央目录定位选定成员并逐个提取。
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        for suffix, target in requested:
            matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"expected one archive entry for {suffix}, got {len(matches)}")
            if target.is_file():
                extracted += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(matches[0]) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
            extracted += 1
            print(f"Extracted {target.relative_to(ROOT)}")
    return extracted


def main() -> int:
    """校验任务 4 本地归档，并按需提取 Lost and Found 样例。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--lost-and-found-image-archive-url",
        help="Optional signed official archive URL when the canonical URL is unreachable.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    archives = config["archives"]
    _verify_archive(ROOT, archives["lost_and_found_annotations"])
    _verify_archive(ROOT, archives["road_obstacle_21"])
    image_url = args.lost_and_found_image_archive_url or archives["lost_and_found_images"][
        "official_url"
    ]
    count = _extract_selected_lost_and_found_images(config, str(image_url))
    print(f"Task-4 raw input verification passed; {count} Lost and Found images are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        # 统一错误出口，使下载失败不会被误报为样例已就绪。
        print(f"Task-4 sample acquisition failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
