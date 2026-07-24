import torch
print(f"显卡名称: {torch.cuda.get_device_name(0)}")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")