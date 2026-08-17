from transformers import SegformerForSemanticSegmentation


def build_segformer(model_dir, num_labels=2, device=None):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    if device is not None:
        model.to(device)

    return model
