import numpy as np
import pytest
import torch

from scorecliff.metrics import i2t_ranks, paired_bootstrap, summarize, t2i_ranks


def test_t2i_ranks():
    sim = torch.tensor([[0.9, 0.1, 0.5],
                        [0.2, 0.3, 0.8],
                        [0.4, 0.6, 0.1]])
    gt = torch.tensor([0, 1, 2])
    assert t2i_ranks(sim, gt).tolist() == [1, 2, 3]


def test_i2t_ranks_uses_best_caption():
    # 2 images, 4 captions; captions 0,1 -> image 0 ; captions 2,3 -> image 1
    cap_img = torch.tensor([0, 0, 1, 1])
    sim = torch.tensor([[0.1, 0.7, 0.9, 0.2],    # best own caption (0.7) is rank 2
                        [0.5, 0.4, 0.3, 0.9]])   # best own caption (0.9) is rank 1
    assert i2t_ranks(sim, torch.tensor([0, 1]), cap_img).tolist() == [2, 1]


def test_summarize():
    m = summarize(np.array([1, 2, 10, 20]), ks=(1, 10))
    assert m["R@1"] == 25.0
    assert m["R@10"] == 75.0
    assert m["MRR"] == pytest.approx(100 * (1 + 0.5 + 0.1 + 0.05) / 4)
    assert m["MedR"] == 6.0


def test_paired_bootstrap_detects_clear_win_and_null():
    rng = np.random.default_rng(0)
    b = rng.integers(1, 50, size=2000)
    better = np.maximum(1, b - 5)
    res = paired_bootstrap(better, b, n_boot=500)
    assert res["MRR"]["diff"] > 0 and res["MRR"]["p_value"] < 0.01
    assert res["MRR"]["ci_low"] > 0
    same = paired_bootstrap(b, b, n_boot=200)
    assert same["R@1"]["diff"] == 0 and same["R@1"]["p_value"] == 1.0


def test_paired_bootstrap_requires_alignment():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(3), np.ones(4))
