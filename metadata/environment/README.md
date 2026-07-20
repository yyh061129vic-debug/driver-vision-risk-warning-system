# 环境基线元数据

`baseline-2026-07-16.yaml` 是当前本地开发机的脱敏环境快照，与 `docs/environment-baseline.md` 对应。它只描述采集时的事实，不代表项目最终部署环境，也不用于替代 `configs/system.yaml` 中尚待确认的目标硬件决策。

刷新基线时应新建带日期的 YAML 文件、同步更新报告，并保留旧快照用于实验复现。禁止写入主机名、用户名、用户目录绝对路径、序列号、UUID、访问令牌或其他设备标识。

完成更新后运行：

```bash
python scripts/validate_environment_baseline.py
```
