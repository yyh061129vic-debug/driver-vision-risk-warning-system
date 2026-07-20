# 本地开发环境基线

## 1. 基线范围

- 最近刷新：2026-07-16 13:35:31（UTC+08:00）
- 环境角色：当前本地开发机，不代表最终部署或验收目标机
- 采集方式：PowerShell、WMI/CIM、`nvidia-smi` 和项目 `.venv`
- 隐私处理：不保存主机名、用户名、设备序列号、UUID 或用户目录绝对路径
- 机器可读快照：[`metadata/environment/baseline-2026-07-16.yaml`](../metadata/environment/baseline-2026-07-16.yaml)

## 2. 基线结论

本机具备 NVIDIA GeForce RTX 4070 Laptop GPU，NVIDIA 驱动版本为 581.42，显存为 8188 MiB，计算能力为 8.9。驱动报告的最高 CUDA 兼容版本为 13.0，但本机没有安装 CUDA Toolkit，因此不存在可用的 `nvcc` 编译器或 `CUDA_PATH`。

项目虚拟环境使用 CPython 3.12.13。为任务 5 已安装 CUDA 13.0 构建的 PyTorch 2.13.0、TorchVision 0.28.0、Transformers 5.14.0 和 OpenCV 5.0.0.93。`torch.cuda.is_available()` 已实测为 true，PyTorch 可识别并使用 NVIDIA GeForce RTX 4070 Laptop GPU，SegFormer GPU 图像推理已成功完成。ONNX Runtime、TensorRT 和 CARLA Python 包仍未安装。

C 盘总容量为 284.85 GiB，刷新时仅剩 17.67 GiB。该空间不适合直接下载完整候选数据集、CARLA 资源和多份模型权重；执行后续数据下载前应先确认外部存储目录与容量预算。

## 3. 系统与硬件

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Microsoft Windows 11 家庭版 中文版 |
| 系统版本 / 构建 | 10.0.26200 / 26200 |
| 架构 | 64-bit / AMD64 |
| CPU | Intel Core Ultra 9 185H |
| CPU 核心 | 16 个物理核心，22 个逻辑处理器 |
| 物理内存 | 31.42 GiB |
| C 盘容量 | 284.85 GiB 总计，17.67 GiB 可用 |
| 时区 | China Standard Time（UTC+08:00） |

## 4. GPU、驱动与 CUDA

| 项目 | 实测值 | 状态说明 |
| --- | --- | --- |
| 独立 GPU | NVIDIA GeForce RTX 4070 Laptop GPU | 可识别 |
| NVIDIA 驱动 | 581.42 | `nvidia-smi` 可用 |
| 独立显存 | 8188 MiB | `nvidia-smi` 实测 |
| CUDA 计算能力 | 8.9 | `nvidia-smi` 实测 |
| 驱动 CUDA 兼容上限 | 13.0 | 仅表示驱动兼容能力，不等于已安装 Toolkit |
| CUDA Toolkit / `nvcc` | 未安装 | `nvcc` 与 `CUDA_PATH` 均不可用 |
| PyTorch CUDA 运行时 | 13.0 | 由 PyTorch 轮子携带，可使用 RTX 4070 |
| 集成 GPU | Intel Arc Graphics | WMI 驱动 32.0.101.8132 |

## 5. Python 与框架

| 组件 | 版本 | 状态 |
| --- | --- | --- |
| CPython | 3.12.13 | 已安装于项目 `.venv` |
| pip | 25.0.1 | 已安装 |
| PyYAML | 6.0.3 | 已安装 |
| pytest | 8.4.2 | 已安装 |
| NumPy | 2.5.1 | 已安装，用于掩码计算 |
| Pillow | 12.3.0 | 已安装，用于图像可视化 |
| PyTorch | 2.13.0+cu130 | 已安装，CUDA 13.0 构建，GPU 可用 |
| TorchVision | 0.28.0+cu130 | 已安装，CUDA 13.0 构建 |
| Transformers | 5.14.0 | 已安装 |
| TorchAudio | — | 未安装 |
| ONNX / ONNX Runtime | — | 未安装 |
| TensorRT | — | 未安装 |
| OpenCV | 5.0.0.93 | 已安装，headless 构建 |
| CARLA Python 包 | — | 未安装 |

补充工具版本：Git 2.54.0.windows.1 已安装；Docker、Conda 和 CMake 在当前命令行环境中不可用。

## 6. 后续环境准入条件

在开始可行驶区域分割 Demo 或 CARLA 联调前，应先完成以下事项：

1. 确认最终目标 GPU、Python 版本和模型框架兼容矩阵；当前 RTX 4070 与 CUDA PyTorch 仅作为本地 GPU 基线。
2. 明确是否需要本地编译 CUDA 算子；只有确有需要时才安装与框架匹配的 CUDA Toolkit。
3. 单独确认 CARLA 服务端版本、Python 客户端版本和 Unreal 资源需求，保持三者一致。
4. 将数据集、CARLA、权重和输出目录迁移到容量充足的磁盘，再更新 `configs/paths.yaml`。
5. GPU 或 CARLA 环境安装完成后重新采集本快照，并运行 `python scripts/validate_environment_baseline.py`。

本报告满足启动清单中“记录 GPU、驱动、CUDA、Python、框架版本”的环境基线要求；缺失组件按“未安装”记录，不将驱动兼容版本误记为本地 CUDA Toolkit 版本。
