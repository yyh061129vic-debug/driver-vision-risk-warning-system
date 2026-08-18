import os
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, path):
    parent = os.path.dirname(path)

    if parent:
        ensure_dir(parent)

    torch.save(
        model.state_dict(),
        path,
    )


def load_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    return model


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable
