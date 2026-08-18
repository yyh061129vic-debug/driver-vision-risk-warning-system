import torch


def binary_iou(pred, target):
    pred = pred == 1
    target = target == 1

    intersection = (
        pred & target
    ).sum().item()

    union = (
        pred | target
    ).sum().item()

    if union == 0:
        return 0.0

    return intersection / union


def binary_dice(pred, target):
    pred = pred == 1
    target = target == 1

    intersection = (
        pred & target
    ).sum().item()

    denominator = (
        pred.sum().item()
        + target.sum().item()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


def pixel_accuracy(pred, target):
    return (
        (pred == target).sum().item()
        / target.numel()
    )
