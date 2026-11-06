"""COCO caption tables, subsets and torch datasets."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

COLUMNS = ["image_path", "image_id", "text"]


# --------------------------------------------------------------------------- #
# Caption tables
# --------------------------------------------------------------------------- #
def build_caption_table(images_dir: str, annotation_file: str) -> pd.DataFrame:
    """One row per (image, caption) pair, read straight from the COCO JSON.

    Reading the JSON directly is ~100x faster than iterating
    ``torchvision.datasets.CocoCaptions``, which decodes every image.
    """
    with open(annotation_file) as f:
        ann = json.load(f)
    files = {img["id"]: img["file_name"] for img in ann["images"]}
    rows = sorted(
        (a["image_id"], a["id"], " ".join(a["caption"].split()))
        for a in ann["annotations"]
    )
    return pd.DataFrame(
        [(os.path.join(images_dir, files[img_id]), img_id, text) for img_id, _, text in rows],
        columns=COLUMNS,
    )


def prepare_coco(coco_root: str, out_dir: str) -> dict[str, str]:
    """Write ``train2017.csv`` and ``val2017.csv`` caption tables to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for split in ("train2017", "val2017"):
        df = build_caption_table(
            os.path.join(coco_root, split),
            os.path.join(coco_root, "annotations", f"captions_{split}.json"),
        )
        path = os.path.join(out_dir, f"{split}.csv")
        df.to_csv(path, index=False)
        written[split] = path
        print(f"{split}: {len(df):,} captions / {df.image_id.nunique():,} images -> {path}")
    return written


def select_subset(
    df: pd.DataFrame,
    *,
    caption_frac: float | None = None,
    num_images: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic subset of a caption table.

    * ``caption_frac``: shuffle caption rows with ``seed`` and keep the first
      fraction (this is how the v1 notebooks built their 1% / 25% splits).
    * ``num_images``: shuffle the unique images with ``seed`` and keep all
      captions of the first ``num_images`` images (COCO 1K / 5K style).
    """
    if caption_frac is not None and num_images is not None:
        raise ValueError("Use either caption_frac or num_images, not both.")
    if caption_frac is not None:
        if not 0 < caption_frac <= 1:
            raise ValueError("caption_frac must be in (0, 1].")
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return shuffled.iloc[: int(len(shuffled) * caption_frac)].reset_index(drop=True)
    if num_images is not None:
        ids = df["image_id"].drop_duplicates().sample(frac=1.0, random_state=seed)
        keep = set(ids.iloc[:num_images])
        return df[df["image_id"].isin(keep)].reset_index(drop=True)
    return df.reset_index(drop=True)


@dataclass
class PairTable:
    """Index structure: unique images + captions pointing at them."""

    image_paths: list[str]
    image_ids: np.ndarray       # [N_images]
    texts: list[str]            # [N_captions]
    caption_image: np.ndarray   # [N_captions] -> index into image_paths

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "PairTable":
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"caption table is missing columns: {sorted(missing)}")
        codes, uniques = pd.factorize(df["image_path"], sort=False)
        first_row = pd.Series(np.arange(len(df))).groupby(codes).first().to_numpy()
        return cls(
            image_paths=list(uniques),
            image_ids=df["image_id"].to_numpy()[first_row].astype(np.int64),
            texts=df["text"].astype(str).tolist(),
            caption_image=codes.astype(np.int64),
        )

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def num_captions(self) -> int:
        return len(self.texts)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class ImageDataset(Dataset):
    """Unique images, used for building the retrieval gallery."""

    def __init__(self, paths: Sequence[str], transform: Callable):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(load_rgb(self.paths[i]))


class TrainPairs(Dataset):
    """(positive image, caption index, positive id, negative images, negative ids).

    Negatives are image indices into ``table.image_paths``; they are replaced
    by the miner before every epoch via :meth:`set_negatives`.
    """

    def __init__(self, table: PairTable, transform: Callable, num_negatives: int):
        self.table = table
        self.transform = transform
        self.num_negatives = num_negatives
        self.negatives: np.ndarray | None = None

    def set_negatives(self, negatives: np.ndarray) -> None:
        expected = (self.table.num_captions, self.num_negatives)
        if negatives.shape != expected:
            raise ValueError(f"negatives has shape {negatives.shape}, expected {expected}")
        if (negatives == self.table.caption_image[:, None]).any():
            raise ValueError("a negative equals its own positive image")
        self.negatives = negatives.astype(np.int64)

    def __len__(self) -> int:
        return self.table.num_captions

    def __getitem__(self, i: int):
        if self.negatives is None:
            raise RuntimeError("call set_negatives() before iterating")
        pos_id = int(self.table.caption_image[i])
        neg_ids = self.negatives[i]
        pos = self.transform(load_rgb(self.table.image_paths[pos_id]))
        negs = torch.stack([self.transform(load_rgb(self.table.image_paths[j])) for j in neg_ids])
        return pos, i, pos_id, negs, torch.from_numpy(neg_ids.copy())
