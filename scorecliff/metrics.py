"""Retrieval metrics and paired significance testing."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


@torch.no_grad()
def t2i_ranks(sim: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    """1-based rank of each query's ground-truth image. ``sim``: [Q, N]."""
    gt = sim.gather(1, gt_image.view(-1, 1).long())
    return (sim > gt).sum(dim=1) + 1


@torch.no_grad()
def i2t_ranks(sim: torch.Tensor, image_index: torch.Tensor, caption_image: torch.Tensor) -> torch.Tensor:
    """1-based rank of the *best-ranked* correct caption for each image.

    ``sim``: [n, Q] image-to-caption scores for the images ``image_index``.
    """
    correct = caption_image[None, :] == image_index[:, None]          # [n, Q]
    best = sim.masked_fill(~correct, float("-inf")).max(dim=1, keepdim=True).values
    return (sim > best).sum(dim=1) + 1


def summarize(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """R@k and MRR in percent, plus median / mean rank.

    Each query has exactly one relevant item, so R@k is the hit rate @k.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {f"R@{k}": float(100 * np.mean(ranks <= k)) for k in ks}
    out["MRR"] = float(100 * np.mean(1.0 / ranks))
    out["MedR"] = float(np.median(ranks))
    out["MeanR"] = float(np.mean(ranks))
    return out


def _metric(ranks: np.ndarray, name: str, axis: int = -1) -> np.ndarray:
    if name == "MRR":
        return 100 * np.mean(1.0 / ranks, axis=axis)
    if name.startswith("R@"):
        return 100 * np.mean(ranks <= int(name[2:]), axis=axis)
    raise ValueError(name)


def paired_bootstrap(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    metrics: Sequence[str] = ("R@1", "R@5", "R@10", "MRR"),
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap over queries for ``metric(A) - metric(B)``.

    Both rank arrays must come from the same queries in the same order.
    Returns the observed difference, a 95% CI and a two-sided p-value.
    """
    a = np.asarray(ranks_a, dtype=np.float64)
    b = np.asarray(ranks_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("rank arrays must be aligned (same queries, same order)")
    rng = np.random.default_rng(seed)
    out = {}
    for name in metrics:
        diff_obs = float(_metric(a, name) - _metric(b, name))
        diffs = np.empty(n_boot)
        for i0 in range(0, n_boot, 500):
            idx = rng.integers(0, len(a), size=(min(500, n_boot - i0), len(a)))
            diffs[i0:i0 + len(idx)] = _metric(a[idx], name, axis=1) - _metric(b[idx], name, axis=1)
        p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        out[name] = {
            "diff": diff_obs,
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, p)),
        }
    return out
