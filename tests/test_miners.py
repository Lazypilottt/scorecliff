import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scorecliff.miners import (cluster_embeddings, cluster_negatives, mine_cluster, mine_qure,
                               qure_negatives, random_negatives)


def test_random_negatives_never_positive():
    rng = np.random.default_rng(0)
    cap_img = rng.integers(0, 7, size=5000)
    negs = random_negatives(cap_img, 7, k=3, rng=rng)
    assert negs.shape == (5000, 3)
    assert (negs != cap_img[:, None]).all()
    assert negs.min() >= 0 and negs.max() <= 6
    assert len(np.unique(negs)) == 7  # every other image is reachable


def test_qure_band_between_two_largest_drops_below_positive():
    # image index == sorted position here. Positive at rank 2.
    #   rank:   0     1     2(+)  3     4     5     6     7
    scores = torch.tensor([[1.00, 0.99, 0.90, 0.60, 0.58, 0.57, 0.20, 0.19]])
    # A big drop sits *above* the positive (0.99 -> 0.90) and must be ignored.
    # Below the positive: drops 0.30 (rank2->3) is not eligible (it's the positive's own cliff),
    # eligible drops: 3->4 .02, 4->5 .01, 5->6 .37, 6->7 .01 -> top-2 are j=5 (.37) and j=3 (.02)
    # band = ranks [j1+1, j2] = [4, 5]
    negs, st = qure_negatives(scores.repeat(200, 1), torch.full((200,), 2), k=4,
                              generator=torch.Generator().manual_seed(0))
    assert set(negs.flatten().tolist()) == {4, 5}
    assert not st["fallback"].any()
    assert (st["band_lo"] == 4).all() and (st["band_size"] == 2).all()


def test_qure_respects_window():
    scores = torch.tensor([[1.0, 0.9, 0.85, 0.84, 0.5, 0.1]])
    # positive rank 0; window=3 -> only ranks 1..3 searchable, eligible drops j in {1, 2}
    negs, st = qure_negatives(scores.repeat(100, 1), torch.zeros(100, dtype=torch.long), k=2, window=3,
                              generator=torch.Generator().manual_seed(1))
    assert negs.max() <= 3 and negs.min() >= 1
    assert (st["band_lo"] == 2).all() and (st["band_size"] == 1).all()


@pytest.mark.parametrize("pos", [5, 4])
def test_qure_fallback_never_returns_positive(pos):
    scores = torch.linspace(1, 0, 6)[None, :].repeat(100, 1)  # rank == index
    negs, st = qure_negatives(scores, torch.full((100,), pos), k=3,
                              generator=torch.Generator().manual_seed(0))
    assert st["fallback"].all()
    assert (negs != pos).all()
    if pos == 5:  # ranked last -> sample from everything above
        assert set(negs.flatten().tolist()) <= {0, 1, 2, 3, 4}


def test_mine_qure_random_embeddings():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(300, 32), dim=-1)
    cap_img = np.repeat(np.arange(300), 2)
    txt = F.normalize(img[cap_img] + 0.8 * torch.randn(600, 32), dim=-1)
    negs, stats = mine_qure(txt, img, cap_img, k=2, seed=0, chunk=128)
    assert negs.shape == (600, 2)
    assert (negs != cap_img[:, None]).all()
    assert 0 <= stats["fallback_frac"] <= 1
    assert stats["mean_band_size"] >= 1


def test_cluster_negatives_share_cluster_and_skip_noise():
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 0, 1, 1, -1, 2])
    cap_img = np.array([0, 1, 1, 2, 3, 4, 5])
    negs, st = cluster_negatives(labels, cap_img, num_images=6, k=5, rng=rng)
    assert (negs != cap_img[:, None]).all()
    assert set(negs[0]) == {1}             # only other image in cluster 0
    assert set(negs[1]) == {0} and set(negs[2]) == {0}
    assert set(negs[3]) == {3} and set(negs[4]) == {2}
    assert st["noise_frac"] == pytest.approx(1 / 7)
    assert st["in_cluster_frac"] == pytest.approx(5 / 7)  # noise row and singleton cluster fall back


@pytest.mark.parametrize("algo", ["kmeans", "minibatch", "agglomerative", "spectral", "hdbscan"])
def test_all_cluster_algorithms_run(algo):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16))
    x = np.concatenate([c + 0.05 * rng.normal(size=(30, 16)) for c in centers])
    labels = cluster_embeddings(x, algorithm=algo, n_clusters=4, seed=0)
    assert labels.shape == (120,)
    assert len(set(labels[labels >= 0])) >= 2


def test_mine_cluster_end_to_end():
    torch.manual_seed(0)
    txt = torch.randn(200, 16)
    cap_img = np.arange(200) // 2
    negs, st = mine_cluster(txt, cap_img, 100, k=2, rng=np.random.default_rng(0), n_clusters=8)
    assert negs.shape == (200, 2) and (negs != cap_img[:, None]).all()
    assert st["n_clusters"] == 8
