"""Loading CLIP and freezing towers."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

DEFAULT_MODEL = "ViT-B-32-quickgelu"  # OpenAI weights were trained with QuickGELU
DEFAULT_PRETRAINED = "openai"
MAX_LOGIT_SCALE = math.log(100.0)


def load_clip(
    name: str = DEFAULT_MODEL,
    pretrained: str | None = DEFAULT_PRETRAINED,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Callable, Callable, Callable]:
    """Return ``(model, train_transform, eval_transform, tokenizer)``."""
    import open_clip  # imported lazily so the rest of the package works without it

    model, train_tf, eval_tf = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(name)
    return model, train_tf, eval_tf, tokenizer


def load_checkpoint(path: str, device: torch.device | str = "cpu"):
    """Rebuild a model saved by :func:`scorecliff.engine.train`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own checkpoints (config dict)
    cfg = ckpt["config"]
    model, train_tf, eval_tf, tokenizer = load_clip(cfg["model_name"], pretrained=None, device="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), train_tf, eval_tf, tokenizer, cfg


def text_tower_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Every parameter of the text tower, including embeddings and projection."""
    if hasattr(model, "text"):  # open_clip CustomTextCLIP (e.g. HF text towers)
        return list(model.text.parameters())
    params: list[nn.Parameter] = []
    for name in ("transformer", "token_embedding", "ln_final"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            params += list(module.parameters())
    for name in ("positional_embedding", "text_projection"):
        p = getattr(model, name, None)
        if isinstance(p, nn.Parameter):
            params.append(p)
        elif isinstance(p, nn.Linear):
            params += list(p.parameters())
    return params


def freeze(params: list[nn.Parameter]) -> int:
    for p in params:
        p.requires_grad_(False)
    return sum(p.numel() for p in params)


def freeze_text_tower(model: nn.Module) -> int:
    return freeze(text_tower_parameters(model))


def freeze_vision_tower(model: nn.Module) -> int:
    return freeze(list(model.visual.parameters()))


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def clamp_logit_scale(model: nn.Module) -> None:
    """CLIP caps the learned temperature at 100 to keep training stable."""
    model.logit_scale.clamp_(0, MAX_LOGIT_SCALE)
