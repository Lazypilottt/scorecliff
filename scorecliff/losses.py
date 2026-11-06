"""Training objectives. All inputs are L2-normalised embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_loss(
    text: torch.Tensor,       # [B, D]
    pos_img: torch.Tensor,    # [B, D]
    neg_img: torch.Tensor,    # [B, k, D]
    scale: torch.Tensor,      # scalar (logit_scale.exp())
) -> torch.Tensor:
    """Bradley-Terry / BPR loss used by QuRe: ``-log sigmoid(s+ - s-)``,
    averaged over the ``k`` negatives of every query.
    Scores are scaled by CLIP's temperature so gradients are not vanishingly small.
    """
    s_pos = scale * (text * pos_img).sum(-1)                     # [B]
    s_neg = scale * torch.einsum("bd,bkd->bk", text, neg_img)    # [B, k]
    return F.softplus(s_neg - s_pos[:, None]).mean()


def infonce_loss(
    text: torch.Tensor,       # [B, D]
    pos_img: torch.Tensor,    # [B, D]
    neg_img: torch.Tensor,    # [B, k, D]
    scale: torch.Tensor,
    pos_ids: torch.Tensor,    # [B]  image index of each positive
    neg_ids: torch.Tensor,    # [B, k]
) -> torch.Tensor:
    """Symmetric CLIP InfoNCE with mined hard negatives appended to the gallery.

    * text -> image: every caption is scored against the B in-batch images
      plus all B*k mined negatives.
    * image -> text: standard in-batch loss.

    Columns that show the *same image* as a row's positive (duplicate captions
    of one image in a batch, or another query's negative that happens to be
    this query's positive) are masked out so they are not treated as negatives.
    """
    b, k, d = neg_img.shape
    targets = torch.arange(b, device=text.device)
    eye = torch.eye(b, dtype=torch.bool, device=text.device)

    gallery = torch.cat([pos_img, neg_img.reshape(b * k, d)])            # [B + Bk, D]
    gallery_ids = torch.cat([pos_ids, neg_ids.reshape(-1)])              # [B + Bk]
    same = pos_ids[:, None] == gallery_ids[None, :]
    same[:, :b] &= ~eye
    logits_t2i = (scale * text @ gallery.T).masked_fill(same, float("-inf"))

    same_i = (pos_ids[:, None] == pos_ids[None, :]) & ~eye
    logits_i2t = (scale * pos_img @ text.T).masked_fill(same_i, float("-inf"))

    return 0.5 * (F.cross_entropy(logits_t2i, targets) + F.cross_entropy(logits_i2t, targets))


LOSSES = ("infonce", "pairwise")
