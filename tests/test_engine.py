"""End-to-end smoke test of training + evaluation on CPU with a tiny model."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from scorecliff.data import prepare_coco
from scorecliff.engine import TrainConfig, evaluate, train
from scorecliff.metrics import paired_bootstrap

from .conftest import TinyCLIP, tiny_tokenizer


@pytest.mark.parametrize("miner,loss,freeze_text", [
    ("random", "infonce", True),
    ("qure", "infonce", True),
    ("qure", "pairwise", False),
    ("cluster", "infonce", False),
])
def test_train_and_evaluate(coco_dir, tmp_path, tiny_tf, miner, loss, freeze_text):
    paths = prepare_coco(str(coco_dir), str(tmp_path / "data"))
    train_df = pd.read_csv(paths["train2017"])
    val_df = pd.read_csv(paths["val2017"])

    torch.manual_seed(0)
    model = TinyCLIP()
    text_before = model.token_embedding.weight.detach().clone()
    cfg = TrainConfig(miner=miner, loss=loss, epochs=3, batch_size=8, lr=1e-2, num_negatives=2,
                      n_clusters=3, freeze_text=freeze_text, num_workers=0, amp=False, seed=1)
    out = str(tmp_path / f"run_{miner}_{loss}")
    hist = train(cfg, model, tiny_tf, tiny_tf, tiny_tokenizer, train_df, torch.device("cpu"), out)

    assert os.path.exists(os.path.join(out, "model.pt"))
    assert len(hist["epochs"]) == 3
    assert all(np.isfinite(e["loss"]) for e in hist["epochs"])
    assert hist["epochs"][0]["mining"]["miner"] == "random"          # warm-up epoch
    assert hist["epochs"][1]["mining"]["miner"] == miner
    if miner == "qure":
        assert "fallback_frac" in hist["epochs"][1]["mining"]
    assert all(e["logit_scale"] <= 100.0 + 1e-3 for e in hist["epochs"])
    frozen = torch.equal(model.token_embedding.weight, text_before)
    assert frozen == freeze_text

    res = evaluate(model, tiny_tf, tiny_tokenizer, val_df, torch.device("cpu"),
                   out_dir=os.path.join(out, "eval"), num_workers=0)
    assert res["num_images"] == 6 and res["num_queries"] == len(val_df)
    for d in ("text_to_image", "image_to_text"):
        assert 0 <= res[d]["R@1"] <= res[d]["R@5"] <= res[d]["R@10"] <= 100
    ranks = np.load(os.path.join(out, "eval", "ranks.npz"))
    assert ranks["t2i"].shape == (len(val_df),) and ranks["t2i_top"].shape == (len(val_df), 6)
    assert paired_bootstrap(ranks["t2i"], ranks["t2i"], n_boot=50)["MRR"]["diff"] == 0


def test_checkpoint_roundtrip(coco_dir, tmp_path, tiny_tf):
    paths = prepare_coco(str(coco_dir), str(tmp_path / "data"))
    model = TinyCLIP()
    cfg = TrainConfig(miner="random", epochs=1, batch_size=8, num_workers=0, amp=False)
    train(cfg, model, tiny_tf, tiny_tf, tiny_tokenizer, pd.read_csv(paths["train2017"]),
          torch.device("cpu"), str(tmp_path / "r"))
    ckpt = torch.load(tmp_path / "r" / "model.pt", map_location="cpu", weights_only=False)
    assert ckpt["tag"] == "random+infonce" and ckpt["config"]["epochs"] == 1
    fresh = TinyCLIP()
    fresh.load_state_dict(ckpt["state_dict"])
    for a, b in zip(fresh.parameters(), model.parameters()):
        assert torch.equal(a, b)
