"""Backbone builder using timm pretrained weights."""
import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int = 100, drop_rate: float = 0.1,
                pretrained: bool = True):
    """Create a pretrained backbone with a fresh `num_classes` head."""
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def set_backbone_trainable(model, trainable: bool):
    """Freeze/unfreeze everything except the classifier head."""
    head = model.get_classifier()
    head_params = set(head.parameters())
    for p in model.parameters():
        if p not in head_params:
            p.requires_grad_(trainable)
    for p in head_params:
        p.requires_grad_(True)
