import os
import time
import torch

from config.config import Config
from models.segformer import build_segformer


WEIGHT_PATH = Config.BEST_MODEL


def main():

    cfg = Config

    print("=" * 50)
    print("SegFormer-B0 Baseline Benchmark")
    print("=" * 50)

    # =============================
    # 1. 加载模型
    # =============================

    print("\nLoading model...")

    # 本地权重存在时使用训练模型；否则使用预训练权重兜底（仅测速）
    model_dir = (
        cfg.MODEL_DIR
        if os.path.isdir(cfg.MODEL_DIR)
        else cfg.SEGFORMER_PRETRAINED
    )

    model = build_segformer(
        model_dir=model_dir,
        num_labels=2,
        device=cfg.DEVICE,
    )

    if os.path.exists(WEIGHT_PATH):
        checkpoint = torch.load(
            WEIGHT_PATH,
            map_location=cfg.DEVICE,
        )

        model.load_state_dict(checkpoint)
        print("Loaded trained checkpoint.")
    else:
        print(
            "Trained checkpoint missing:",
            WEIGHT_PATH,
            "- using pretrained weights (timing only).",
        )

    model.eval()

    print("Model loaded.")

    # =============================
    # 2. 参数量
    # =============================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nParameter Count")
    print("-" * 50)

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Total parameters: "
        f"{total_params / 1e6:.2f} M"
    )

    # =============================
    # 3. GPU 显存
    # =============================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # =============================
    # 4. 创建测试输入
    # =============================

    x = torch.randn(
        1,
        3,
        cfg.IMAGE_SIZE[0],
        cfg.IMAGE_SIZE[1],
        device=cfg.DEVICE
    )

    # =============================
    # 5. Warm-up
    # =============================

    print("\nWarm-up...")

    with torch.no_grad():

        for _ in range(20):

            _ = model(
                pixel_values=x
            )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    # =============================
    # 6. 正式测速
    # =============================

    print("Benchmarking...")

    iterations = 100

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):

            _ = model(
                pixel_values=x
            )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start

    avg_time = total_time / iterations

    fps = 1.0 / avg_time

    # =============================
    # 7. 输出结果
    # =============================

    print("\n" + "=" * 50)
    print("Benchmark Result")
    print("=" * 50)

    print(
        f"Average inference time: "
        f"{avg_time * 1000:.2f} ms"
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    if torch.cuda.is_available():

        peak_memory = torch.cuda.max_memory_allocated()

        print(
            f"Peak GPU memory: "
            f"{peak_memory / 1024**2:.2f} MB"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()