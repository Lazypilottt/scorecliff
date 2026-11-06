"""Hard-negative miners.

Every miner returns ``negatives`` with shape ``[num_captions, k]`` holding
indices into the unique-image list, plus a dict of summary statistics.
No miner ever returns a caption's own positive image.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CLUSTER_ALGORITHMS = ("kmeans", "minibatch", "agglomerative", "spectral", "hdbscan")


# --------------------------------------------------------------------------- #
# Random
# --------------------------------------------------------------------------- #
def random_negatives(
    caption_image: np.ndarray, num_images: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform over all images except the caption's own image."""
    if num_images < 2:
        raise ValueError("need at least two images to sample negatives")
    draws = rng.integers(0, num_images - 1, size=(len(caption_image), k))
    return draws + (draws >= caption_image[:, None])


# --------------------------------------------------------------------------- #
# QuRe score-drop band
# --------------------------------------------------------------------------- #
@torch.no_grad()
def qure_negatives(
    scores: torch.Tensor,
    positives: torch.Tensor,
    k: int,
    generator: torch.Generator | None = None,
    window: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample negatives from the band between the two steepest score drops
    *below* the positive image.

    Args:
        scores: ``[Q, N]`` query-image similarities (any monotone scale).
        positives: ``[Q]`` index of each query's positive image.
        k: negatives per query.
        window: if set, only the ``window`` images ranked directly below the
            positive are searched for drops.

    With ``r`` the positive's rank and drop ``j = s[j] - s[j+1]`` in the sorted
    list, only drops ``j >= r + 1`` (strictly inside the suffix after the
    positive) are eligible. The two largest eligible drops ``j1 < j2`` define
    the band of ranks ``[j1 + 1, j2]``. If fewer than two drops are eligible,
    negatives are drawn uniformly from the images below the positive (or, if
    the positive is ranked last, from all other images).
    """
    q, n = scores.shape
    if n < 2:
        raise ValueError("need at least two images")
    device = scores.device
    vals, order = scores.float().sort(dim=1, descending=True)
    pos_rank = (order == positives.to(device)[:, None]).int().argmax(dim=1)  # [Q]

    drops = vals[:, :-1] - vals[:, 1:]                                      # [Q, N-1]
    j = torch.arange(n - 1, device=device)[None, :]
    eligible = j >= (pos_rank[:, None] + 1)
    if window is not None:
        eligible &= j <= (pos_rank[:, None] + window - 1)
    n_eligible = eligible.sum(dim=1)
    has_band = n_eligible >= 2

    masked = drops.masked_fill(~eligible, float("-inf"))
    top2 = masked.topk(2, dim=1).indices if n - 1 >= 2 else torch.zeros(q, 2, dtype=torch.long, device=device)
    j1, j2 = top2.min(dim=1).values, top2.max(dim=1).values

    below = n - 1 - pos_rank                                                 # images ranked below
    if window is not None:
        below = below.clamp(max=window)
    fb_lo = torch.where(below >= 1, pos_rank + 1, torch.zeros_like(pos_rank))
    fb_size = torch.where(below >= 1, below, torch.full_like(below, n - 1))

    lo = torch.where(has_band, j1 + 1, fb_lo)
    size = torch.where(has_band, j2 - j1, fb_size)

    u = torch.rand(q, k, device=device, generator=generator)
    picked = lo[:, None] + (u * size[:, None]).long().clamp(max=size[:, None] - 1)
    # positive-ranked-last fallback samples ranks [0, n-2], which never hits the positive
    negatives = order.gather(1, picked)

    stats = {
        "pos_rank": pos_rank,
        "band_lo": lo,
        "band_size": size,
        "fallback": ~has_band,
        "neg_rank": picked[:, 0],
    }
    return negatives, stats


@torch.no_grad()
def mine_qure(
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    caption_image: np.ndarray,
    k: int,
    seed: int,
    window: int | None = None,
    chunk: int = 1024,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chunked QuRe mining over normalised embeddings (cosine scores)."""
    device = image_emb.device
    gen = torch.Generator(device=device).manual_seed(seed)
    positives = torch.as_tensor(caption_image, device=device)
    negs, parts = [], defaultdict(list)
    for s in range(0, text_emb.shape[0], chunk):
        scores = text_emb[s:s + chunk].to(device).float() @ image_emb.float().T
        n, st = qure_negatives(scores, positives[s:s + chunk], k, generator=gen, window=window)
        negs.append(n.cpu())
        for key, v in st.items():
            parts[key].append(v.cpu())
    arrays = {key: torch.cat(v).numpy() for key, v in parts.items()}
    stats = {
        "mean_pos_rank": float(arrays["pos_rank"].mean() + 1),
        "median_pos_rank": float(np.median(arrays["pos_rank"]) + 1),
        "mean_band_offset": float((arrays["band_lo"] - arrays["pos_rank"]).mean()),
        "mean_band_size": float(arrays["band_size"].mean()),
        "median_band_size": float(np.median(arrays["band_size"])),
        "fallback_frac": float(arrays["fallback"].mean()),
        "mean_neg_rank": float(arrays["neg_rank"].mean() + 1),
    }
    return torch.cat(negs).numpy().astype(np.int64), stats


# --------------------------------------------------------------------------- #
# Query clustering (ICT-Q style)
# --------------------------------------------------------------------------- #
def cluster_embeddings(
    x: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: int | None = 40,
    seed: int = 0,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """Cluster L2-normalised embeddings. Returns labels; ``-1`` means noise."""
    from sklearn import cluster as skc

    x = np.asarray(x, dtype=np.float32)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    if n_clusters is None:
        n_clusters = max(10, int(np.sqrt(len(x))))
    n_clusters = min(n_clusters, len(x))

    if algorithm == "kmeans":       # on unit vectors this approximates spherical k-means
        model = skc.KMeans(n_clusters=n_clusters, n_init=4, random_state=seed)
    elif algorithm == "minibatch":
        model = skc.MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=3, random_state=seed)
    elif algorithm == "agglomerative":
        model = skc.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    elif algorithm == "spectral":
        model = skc.SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors",
            n_neighbors=min(10, len(x) - 1), assign_labels="kmeans", random_state=seed,
        )
    elif algorithm == "hdbscan":
        if hasattr(skc, "HDBSCAN"):  # scikit-learn >= 1.3
            model = skc.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        else:
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        raise ValueError(f"unknown clustering algorithm {algorithm!r}; choose from {CLUSTER_ALGORITHMS}")
    return np.asarray(model.fit_predict(x), dtype=np.int64)


def cluster_negatives(
    labels: np.ndarray,
    caption_image: np.ndarray,
    num_images: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """For each query, sample the *images* of other queries in the same
    cluster (excluding its own image). Noise points (label -1) and queries
    whose cluster has no other image fall back to uniform random negatives.
    """
    negatives = random_negatives(caption_image, num_images, k, rng)
    members: dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        if lab >= 0:
            members[int(lab)] = caption_image[labels == lab]
    in_cluster = np.zeros(len(labels), dtype=bool)
    for qi, (lab, own) in enumerate(zip(labels, caption_image)):
        if lab < 0:
            continue
        cand = members[int(lab)]
        cand = cand[cand != own]
        if len(cand):
            negatives[qi] = rng.choice(cand, size=k, replace=True)
            in_cluster[qi] = True
    sizes = np.array([len(v) for v in members.values()]) if members else np.zeros(1)
    stats = {
        "n_clusters": len(members),
        "noise_frac": float((labels < 0).mean()),
        "in_cluster_frac": float(in_cluster.mean()),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "mean_cluster_size": float(sizes.mean()),
    }
    return negatives, stats


def mine_cluster(
    text_emb: torch.Tensor,
    caption_image: np.ndarray,
    num_images: int,
    k: int,
    rng: np.random.Generator,
    algorithm: str = "kmeans",
    n_clusters: int | None = 40,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = F.normalize(text_emb.float(), dim=-1).cpu().numpy()
    labels = cluster_embeddings(x, algorithm=algorithm, n_clusters=n_clusters, seed=seed)
    negatives, stats = cluster_negatives(labels, caption_image, num_images, k, rng)
    stats["algorithm"] = algorithm
    return negatives, stats
