"""Encoding, mining, training and evaluation loops."""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import losses, miners
from .data import ImageDataset, PairTable, TrainPairs
from .metrics import i2t_ranks, summarize, t2i_ranks
from .model import clamp_logit_scale, count_trainable, freeze_text_tower, freeze_vision_tower
from .utils import autocast, save_json, seed_everything

MINERS = ("random", "qure", "cluster")


@dataclass
class TrainConfig:
    miner: str = "qure"
    loss: str = "infonce"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-4
    num_negatives: int = 1
    warmup_epochs: int = 1          # epochs with random negatives before mining starts
    mine_every: int = 1             # re-mine every N epochs after warm-up
    qure_window: int | None = None  # search only this many ranks below the positive
    cluster_algo: str = "kmeans"
    n_clusters: int | None = 40
    freeze_text: bool = True
    freeze_vision: bool = False
    seed: int = 0
    amp: bool = True
    num_workers: int = 4
    model_name: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"
    data: dict[str, Any] = field(default_factory=dict)  # provenance of the training subset

    def __post_init__(self):
        if self.miner not in MINERS:
            raise ValueError(f"miner must be one of {MINERS}")
        if self.loss not in losses.LOSSES:
            raise ValueError(f"loss must be one of {losses.LOSSES}")
        if self.miner == "cluster" and self.cluster_algo not in miners.CLUSTER_ALGORITHMS:
            raise ValueError(f"cluster_algo must be one of {miners.CLUSTER_ALGORITHMS}")

    @property
    def tag(self) -> str:
        name = f"cluster-{self.cluster_algo}" if self.miner == "cluster" else self.miner
        if self.miner == "qure" and self.qure_window:
            name += f"-w{self.qure_window}"
        return f"{name}+{self.loss}"


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_images(
    model: nn.Module, paths: list[str], transform: Callable, device: torch.device,
    batch_size: int = 256, num_workers: int = 4, amp: bool = True, desc: str = "images",
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    out = []
    for x in tqdm(loader, desc=f"encode {desc}", leave=False):
        with autocast(device, amp):
            out.append(F.normalize(model.encode_image(x.to(device, non_blocking=True)).float(), dim=-1))
    model.train(was_training)
    return torch.cat(out)


@torch.no_grad()
def tokenize(tokenizer: Callable, texts: list[str], batch_size: int = 4096) -> torch.Tensor:
    return torch.cat([tokenizer(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])


@torch.no_grad()
def encode_texts(
    model: nn.Module, tokens: torch.Tensor, device: torch.device,
    batch_size: int = 512, amp: bool = True,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = []
    for i in tqdm(range(0, len(tokens), batch_size), desc="encode texts", leave=False):
        with autocast(device, amp):
            out.append(F.normalize(model.encode_text(tokens[i:i + batch_size].to(device)).float(), dim=-1))
    model.train(was_training)
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _mine(cfg: TrainConfig, epoch: int, model, table: PairTable, tokens, text_cache,
          eval_tf, device, rng) -> tuple[np.ndarray, dict[str, Any]]:
    k = cfg.num_negatives
    if cfg.miner == "random" or epoch < cfg.warmup_epochs:
        return miners.random_negatives(table.caption_image, table.num_images, k, rng), {"miner": "random"}

    text_emb = text_cache if text_cache is not None else encode_texts(model, tokens, device, amp=cfg.amp)
    if cfg.miner == "qure":
        image_emb = encode_images(model, table.image_paths, eval_tf, device,
                                  num_workers=cfg.num_workers, amp=cfg.amp, desc="train gallery")
        negs, stats = miners.mine_qure(text_emb, image_emb, table.caption_image, k,
                                       seed=cfg.seed * 1000 + epoch, window=cfg.qure_window)
    else:
        negs, stats = miners.mine_cluster(text_emb, table.caption_image, table.num_images, k, rng,
                                          algorithm=cfg.cluster_algo, n_clusters=cfg.n_clusters,
                                          seed=cfg.seed)
    stats["miner"] = cfg.miner
    return negs, stats


def train(
    cfg: TrainConfig,
    model: nn.Module,
    train_tf: Callable,
    eval_tf: Callable,
    tokenizer: Callable,
    train_df: pd.DataFrame,
    device: torch.device,
    out_dir: str,
    save_every_epoch: bool = False,
) -> dict[str, Any]:
    """Fine-tune ``model`` with the configured miner. Writes ``model.pt`` and
    ``history.json`` to ``out_dir`` and returns the history."""
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    amp = cfg.amp and device.type == "cuda"
    model.to(device)

    table = PairTable.from_df(train_df)
    tokens = tokenize(tokenizer, table.texts)
    frozen_text = freeze_text_tower(model) if cfg.freeze_text else 0
    frozen_vis = freeze_vision_tower(model) if cfg.freeze_vision else 0
    # with a frozen text tower the caption embeddings never change: encode them once
    text_cache = encode_texts(model, tokens, device, amp=amp) if cfg.freeze_text else None

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    dataset = TrainPairs(table, train_tf, cfg.num_negatives)
    loader_gen = torch.Generator().manual_seed(cfg.seed)

    history: dict[str, Any] = {
        "config": asdict(cfg), "tag": cfg.tag,
        "train_captions": table.num_captions, "train_images": table.num_images,
        "trainable_params": count_trainable(model),
        "frozen_params": {"text": frozen_text, "vision": frozen_vis},
        "epochs": [],
    }
    print(f"[{cfg.tag}] {table.num_captions:,} captions / {table.num_images:,} images, "
          f"{history['trainable_params'] / 1e6:.1f}M trainable params")

    negatives, mining_stats = None, {}
    for epoch in range(cfg.epochs):
        t0 = time.time()
        due = (negatives is None or cfg.miner == "random" or epoch < cfg.warmup_epochs
               or (epoch - cfg.warmup_epochs) % cfg.mine_every == 0)
        if due:
            negatives, mining_stats = _mine(cfg, epoch, model, table, tokens, text_cache,
                                            eval_tf, device, rng)
        dataset.set_negatives(negatives)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=loader_gen,
                            num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                            drop_last=len(dataset) > cfg.batch_size)
        model.train()
        total, steps = 0.0, 0
        for pos, cap_idx, pos_ids, neg_imgs, neg_ids in tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
            b, k = neg_ids.shape
            images = torch.cat([pos, neg_imgs.flatten(0, 1)]).to(device, non_blocking=True)
            pos_ids, neg_ids = pos_ids.to(device), neg_ids.to(device)
            with autocast(device, amp):
                img = model.encode_image(images)
                txt = text_cache[cap_idx.to(device)] if text_cache is not None else \
                    model.encode_text(tokens[cap_idx].to(device))
            img = F.normalize(img.float(), dim=-1)
            txt = F.normalize(txt.float(), dim=-1)
            pos_emb, neg_emb = img[:b], img[b:].view(b, k, -1)
            scale = model.logit_scale.exp().float()
            if cfg.loss == "pairwise":
                loss = losses.pairwise_loss(txt, pos_emb, neg_emb, scale)
            else:
                loss = losses.infonce_loss(txt, pos_emb, neg_emb, scale, pos_ids, neg_ids)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            clamp_logit_scale(model)
            total += loss.item()
            steps += 1

        record = {
            "epoch": epoch + 1,
            "loss": total / max(steps, 1),
            "logit_scale": float(model.logit_scale.exp()),
            "mined_this_epoch": due,
            "mining": mining_stats,
            "seconds": round(time.time() - t0, 1),
        }
        history["epochs"].append(record)
        print(f"[{cfg.tag}] epoch {epoch + 1}: loss {record['loss']:.4f}  "
              f"scale {record['logit_scale']:.1f}  ({record['seconds']}s)")
        save_json(history, os.path.join(out_dir, "history.json"))
        if save_every_epoch:
            save_checkpoint(model, cfg, os.path.join(out_dir, f"model_ep{epoch + 1}.pt"))

    save_checkpoint(model, cfg, os.path.join(out_dir, "model.pt"))
    return history


def save_checkpoint(model: nn.Module, cfg: TrainConfig, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "tag": cfg.tag}, path)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_tf: Callable,
    tokenizer: Callable,
    test_df: pd.DataFrame,
    device: torch.device,
    out_dir: str | None = None,
    amp: bool = True,
    num_workers: int = 4,
    chunk: int = 1024,
    top_k: int = 10,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Text->image and image->text retrieval over the unique images of ``test_df``."""
    amp = amp and device.type == "cuda"
    model.to(device).eval()
    table = PairTable.from_df(test_df)
    img = encode_images(model, table.image_paths, eval_tf, device, num_workers=num_workers, amp=amp, desc="test gallery")
    txt = encode_texts(model, tokenize(tokenizer, table.texts), device, amp=amp)
    cap_img = torch.as_tensor(table.caption_image, device=device)

    t2i, top = [], []
    for s in range(0, len(txt), chunk):
        sim = txt[s:s + chunk] @ img.T
        t2i.append(t2i_ranks(sim, cap_img[s:s + chunk]).cpu())
        top.append(sim.topk(min(top_k, sim.shape[1]), dim=1).indices.cpu())
    i2t = []
    for s in range(0, len(img), chunk):
        sim = img[s:s + chunk] @ txt.T
        i2t.append(i2t_ranks(sim, torch.arange(s, s + len(sim), device=device), cap_img).cpu())
    t2i_r, i2t_r = torch.cat(t2i).numpy(), torch.cat(i2t).numpy()

    result = {
        **(meta or {}),
        "num_queries": table.num_captions,
        "num_images": table.num_images,
        "text_to_image": summarize(t2i_r),
        "image_to_text": summarize(i2t_r),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_json(result, os.path.join(out_dir, "metrics.json"))
        np.savez_compressed(os.path.join(out_dir, "ranks.npz"), t2i=t2i_r, i2t=i2t_r,
                            t2i_top=torch.cat(top).numpy().astype(np.int32))
    return result
