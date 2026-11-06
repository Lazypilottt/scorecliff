import json
import math
import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class TinyCLIP(nn.Module):
    """Stand-in exposing the open_clip interface used by the engine."""

    def __init__(self, dim: int = 16, vocab: int = 97):
        super().__init__()
        self.visual = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, dim))
        self.token_embedding = nn.EmbeddingBag(vocab, dim)
        self.text_projection = nn.Parameter(torch.eye(dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, x):
        return self.visual(x)

    def encode_text(self, tokens):
        return self.token_embedding(tokens) @ self.text_projection


def tiny_tokenizer(texts, length: int = 6):
    out = torch.zeros(len(texts), length, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [sum(map(ord, w)) % 96 + 1 for w in t.lower().split()][:length]
        out[i, : len(ids)] = torch.tensor(ids)
    return out


@pytest.fixture
def tiny_tf():
    return transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])


@pytest.fixture
def coco_dir(tmp_path):
    """A miniature COCO layout: 12 train / 6 val images, 2-3 captions each."""
    rng = np.random.default_rng(0)
    words = ["dog", "cat", "red", "blue", "car", "tree", "beach", "pizza", "man", "kite"]
    for split, n in (("train2017", 12), ("val2017", 6)):
        os.makedirs(tmp_path / split)
        images, anns = [], []
        for i in range(n):
            img_id = 1000 * (split == "val2017") + i + 1
            fname = f"{img_id:012d}.jpg"
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(tmp_path / split / fname)
            images.append({"id": img_id, "file_name": fname})
            for c in range(2 + i % 2):
                anns.append({"id": len(anns) + 1, "image_id": img_id,
                             "caption": " ".join(rng.choice(words, 4)) + f"  {c} "})
        os.makedirs(tmp_path / "annotations", exist_ok=True)
        with open(tmp_path / "annotations" / f"captions_{split}.json", "w") as f:
            json.dump({"images": images, "annotations": anns}, f)
    return tmp_path
