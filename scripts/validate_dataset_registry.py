"""校验候选数据集登记的完整性、来源和启用安全门。"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "metadata/dataset_registry.yaml"
DATASETS_CONFIG_PATH = ROOT / "configs/data/datasets.yaml"
EXPECTED_IDS = {
    "cityscapes",
    "bdd100k",
    "mapillary-vistas",
    "lost-and-found",
    "fishyscapes",
    "road-anomaly-21",
    "segment-me-if-you-can",
}
EXPECTED_STATUS = {
    "lost-and-found": "enabled_local_noncommercial_validation",
    "segment-me-if-you-can": "enabled_local_evaluation",
}
REQUIRED_DATASET_FIELDS = {
    "id",
    "display_name",
    "version_scope",
    "status",
    "tasks",
    "scale",
    "labels",
    "license",
    "download",
    "recommended_use",
    "limitations",
    "sources",
}
REQUIRED_LICENSE_FIELDS = {
    "name",
    "review_status",
    "commercial_use",
    "redistribution",
    "obligations",
    "note",
}
REQUIRED_DOWNLOAD_FIELDS = {
    "official_url",
    "method",
    "availability",
    "expected_minimum",
}


def _is_https_url(value: object) -> bool:
    """判断字段是否为包含主机名的 HTTPS URL。"""

    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate() -> list[str]:
    """执行数据集登记校验，并以字符串列表返回全部错误。"""

    errors: list[str] = []
    try:
        registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot read registry: {exc}"]

    if not isinstance(registry, dict):
        return ["registry root must be a mapping"]
    if registry.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if not registry.get("reviewed_on"):
        errors.append("reviewed_on is required")

    datasets = registry.get("datasets")
    if not isinstance(datasets, list):
        return errors + ["datasets must be a list"]

    ids = [item.get("id") for item in datasets if isinstance(item, dict)]
    if len(ids) != len(set(ids)):
        errors.append("dataset ids must be unique")
    if set(ids) != EXPECTED_IDS:
        errors.append(f"dataset ids mismatch: expected {sorted(EXPECTED_IDS)}, got {sorted(ids)}")

    # 每个候选项都必须具备规模、标签、许可、下载方式和限制信息。
    for item in datasets:
        if not isinstance(item, dict):
            errors.append("each dataset entry must be a mapping")
            continue
        dataset_id = item.get("id", "<missing>")
        missing = REQUIRED_DATASET_FIELDS - item.keys()
        if missing:
            errors.append(f"{dataset_id}: missing fields {sorted(missing)}")
        expected_status = EXPECTED_STATUS.get(dataset_id, "researched_not_enabled")
        if item.get("status") != expected_status:
            errors.append(f"{dataset_id}: status must be {expected_status}")
        if dataset_id in EXPECTED_STATUS:
            enablement = item.get("local_enablement")
            if not isinstance(enablement, dict):
                errors.append(f"{dataset_id}: enabled dataset requires local_enablement metadata")
            elif not all(
                enablement.get(key)
                for key in ("enabled_on", "scope", "config", "license_snapshot")
            ):
                errors.append(f"{dataset_id}: local_enablement metadata is incomplete")
        if not item.get("tasks"):
            errors.append(f"{dataset_id}: tasks must not be empty")
        if not item.get("limitations"):
            errors.append(f"{dataset_id}: limitations must not be empty")

        license_info = item.get("license")
        if not isinstance(license_info, dict):
            errors.append(f"{dataset_id}: license must be a mapping")
        else:
            missing_license = REQUIRED_LICENSE_FIELDS - license_info.keys()
            if missing_license:
                errors.append(f"{dataset_id}: missing license fields {sorted(missing_license)}")
            if not license_info.get("obligations"):
                errors.append(f"{dataset_id}: license obligations must not be empty")

        download = item.get("download")
        if not isinstance(download, dict):
            errors.append(f"{dataset_id}: download must be a mapping")
        else:
            missing_download = REQUIRED_DOWNLOAD_FIELDS - download.keys()
            if missing_download:
                errors.append(f"{dataset_id}: missing download fields {sorted(missing_download)}")
            if not _is_https_url(download.get("official_url")):
                errors.append(f"{dataset_id}: official download URL must use HTTPS")

        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{dataset_id}: at least one official source is required")
        else:
            for source in sources:
                if not isinstance(source, dict) or not _is_https_url(source.get("url")):
                    errors.append(f"{dataset_id}: every source must contain an HTTPS URL")

    # 登记表和运行配置交叉检查，防止“已启用”状态只修改一处。
    try:
        dataset_config = yaml.safe_load(DATASETS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot read dataset configuration: {exc}")
    else:
        enabled = dataset_config.get("enabled") if isinstance(dataset_config, dict) else None
        if not isinstance(enabled, list):
            errors.append("configs/data/datasets.yaml enabled must be a list")
        else:
            enabled_ids = {item.get("id") for item in enabled if isinstance(item, dict)}
            if enabled_ids != set(EXPECTED_STATUS):
                errors.append(
                    f"enabled dataset ids mismatch: expected {sorted(EXPECTED_STATUS)}, got {sorted(enabled_ids)}"
                )
            for item in enabled:
                if not isinstance(item, dict):
                    errors.append("each enabled dataset entry must be a mapping")
                    continue
                dataset_id = item.get("id", "<missing>")
                if not item.get("scope"):
                    errors.append(f"{dataset_id}: enabled scope is required")
                for key in ("sample_config", "license_snapshot"):
                    value = item.get(key)
                    if not value or not (ROOT / value).is_file():
                        errors.append(f"{dataset_id}: enabled {key} must reference an existing file")

    return errors


def main() -> int:
    """输出人类可读校验结果，并用退出码表示是否通过。"""

    errors = validate()
    if errors:
        print("Dataset registry validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Dataset registry validation passed for {len(EXPECTED_IDS)} candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
