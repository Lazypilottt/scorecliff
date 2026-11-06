import json
import os

import numpy as np

from scorecliff.cli import main


def _fake_run(root, name, tag, seed, r1):
    d = os.path.join(root, name)
    os.makedirs(d)
    metrics = {"tag": tag, "seed": seed,
               "text_to_image": {"R@1": r1, "R@5": r1 + 20, "R@10": r1 + 30, "MRR": r1 + 10, "MedR": 3.0},
               "image_to_text": {"R@1": r1, "R@5": r1, "R@10": r1, "MRR": r1, "MedR": 3.0}}
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    return d


def test_prepare_cli(coco_dir, tmp_path):
    main(["prepare", "--coco-root", str(coco_dir), "--out", str(tmp_path / "data")])
    assert os.path.exists(tmp_path / "data" / "train2017.csv")
    assert os.path.exists(tmp_path / "data" / "val2017.csv")


def test_summarize_and_compare(tmp_path, capsys):
    for s, r1 in enumerate([30.0, 32.0]):
        _fake_run(tmp_path, f"qure_s{s}", "qure+infonce", s, r1)
    _fake_run(tmp_path, "zs", "zero-shot", None, 28.0)
    main(["summarize", str(tmp_path / "*" / "metrics.json"), "--markdown"])
    out = capsys.readouterr().out
    assert "| qure+infonce | 2 | 31.00 ± 1.41" in out
    assert "| zero-shot | 1 | 28.00" in out

    rng = np.random.default_rng(0)
    b = rng.integers(1, 30, 500)
    np.savez(tmp_path / "a.npz", t2i=np.maximum(1, b - 3), i2t=b)
    np.savez(tmp_path / "b.npz", t2i=b, i2t=b)
    main(["compare", str(tmp_path / "a.npz"), str(tmp_path / "b.npz"), "--n-boot", "300",
          "--out", str(tmp_path / "cmp.json")])
    res = json.load(open(tmp_path / "cmp.json"))
    assert res["MRR"]["diff"] > 0 and res["MRR"]["p_value"] < 0.05
