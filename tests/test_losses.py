import math

import pytest
import torch
import torch.nn.functional as F

from scorecliff.losses import infonce_loss, pairwise_loss


def _emb(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def test_pairwise_matches_formula():
    torch.manual_seed(0)
    t, p, n = _emb(4, 8), _emb(4, 8), _emb(4, 1, 8)
    scale = torch.tensor(100.0)
    expected = -torch.log(torch.sigmoid(scale * ((t * p).sum(-1) - (t * n[:, 0]).sum(-1)))).mean()
    assert pairwise_loss(t, p, n, scale) == pytest.approx(expected.item(), rel=1e-5)


def test_pairwise_scale_matters():
    torch.manual_seed(0)
    t, p, n = _emb(16, 8), _emb(16, 8), _emb(16, 2, 8)
    # unscaled cosine differences give a loss close to log(2) (the v1 bug); scaling sharpens it
    assert abs(pairwise_loss(t, p, n, torch.tensor(1.0)).item() - math.log(2)) < 0.3
    assert abs(pairwise_loss(t, p, n, torch.tensor(100.0)).item() - math.log(2)) > 0.3


def _reference_infonce(t, v, neg, scale, ids, neg_ids):
    """Loop implementation: drop every gallery column showing the row's own image."""
    b, k, d = neg.shape
    gallery = torch.cat([v, neg.reshape(-1, d)])
    gids = torch.cat([ids, neg_ids.reshape(-1)])
    t2i = []
    for i in range(b):
        cols = [c for c in range(len(gallery)) if c == i or gids[c] != ids[i]]
        logits = scale * t[i] @ gallery[cols].T
        t2i.append(-torch.log_softmax(logits, 0)[cols.index(i)])
    i2t = []
    for i in range(b):
        cols = [c for c in range(b) if c == i or ids[c] != ids[i]]
        logits = scale * v[i] @ t[cols].T
        i2t.append(-torch.log_softmax(logits, 0)[cols.index(i)])
    return 0.5 * (torch.stack(t2i).mean() + torch.stack(i2t).mean())


def test_infonce_without_duplicates_is_clip_plus_hard_negatives():
    torch.manual_seed(0)
    b, k = 6, 2
    t, v, neg = _emb(b, 8), _emb(b, 8), _emb(b, k, 8)
    ids, neg_ids = torch.arange(b), torch.arange(b, b + b * k).view(b, k)
    scale = torch.tensor(50.0)
    got = infonce_loss(t, v, neg, scale, ids, neg_ids)
    assert got.item() == pytest.approx(_reference_infonce(t, v, neg, scale, ids, neg_ids).item(), rel=1e-5)


def test_infonce_own_image_as_negative_is_masked():
    torch.manual_seed(0)
    b, k = 6, 2
    t, v = _emb(b, 8), _emb(b, 8)
    ids = torch.arange(b)
    neg = v[:, None, :].repeat(1, k, 1)          # every negative is a copy of the row's own positive
    neg_ids = ids[:, None].repeat(1, k)
    scale = torch.tensor(50.0)
    got = infonce_loss(t, v, neg, scale, ids, neg_ids)
    assert got.item() == pytest.approx(_reference_infonce(t, v, neg, scale, ids, neg_ids).item(), rel=1e-5)
    # without masking, the copies would score as high as the positive and inflate the loss
    unmasked = F.cross_entropy(scale * t @ torch.cat([v, neg.reshape(-1, 8)]).T, torch.arange(b))
    assert got.item() < 0.5 * (unmasked.item() + F.cross_entropy(scale * v @ t.T, torch.arange(b)).item())


def test_infonce_masks_duplicate_images_in_batch():
    torch.manual_seed(0)
    t = _emb(4, 8)
    v = _emb(4, 8)
    v[1] = v[0]                       # captions 0 and 1 describe the same image
    ids = torch.tensor([7, 7, 8, 9])
    neg, neg_ids = _emb(4, 1, 8), torch.tensor([[8], [9], [7], [8]])
    loss = infonce_loss(t, v, neg, torch.tensor(30.0), ids, neg_ids)
    assert torch.isfinite(loss)
    # gradient flows and no -inf leaks into the loss
    t.requires_grad_(True)
    infonce_loss(t, v, neg, torch.tensor(30.0), ids, neg_ids).backward()
    assert torch.isfinite(t.grad).all()
