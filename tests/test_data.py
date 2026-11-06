import numpy as np
import pandas as pd
import pytest

from scorecliff.data import PairTable, TrainPairs, build_caption_table, prepare_coco, select_subset


def test_prepare_coco_and_table(coco_dir, tmp_path):
    paths = prepare_coco(str(coco_dir), str(tmp_path / "data"))
    train = pd.read_csv(paths["train2017"])
    val = pd.read_csv(paths["val2017"])
    assert list(train.columns) == ["image_path", "image_id", "text"]
    assert train.image_id.nunique() == 12 and val.image_id.nunique() == 6
    assert len(train) == 12 * 2 + 6          # images alternate 2 / 3 captions
    assert not set(train.image_id) & set(val.image_id)
    assert not train.text.str.contains("  ").any()   # whitespace normalised

    table = PairTable.from_df(train)
    assert table.num_images == 12 and table.num_captions == len(train)
    for i in range(table.num_captions):
        assert table.image_paths[table.caption_image[i]] == train.image_path[i]


def test_select_subset_is_deterministic(coco_dir):
    df = build_caption_table(str(coco_dir / "train2017"), str(coco_dir / "annotations/captions_train2017.json"))
    a = select_subset(df, caption_frac=0.5, seed=501)
    b = select_subset(df, caption_frac=0.5, seed=501)
    assert a.equals(b) and len(a) == len(df) // 2
    imgs = select_subset(df, num_images=4, seed=0)
    assert imgs.image_id.nunique() == 4
    # all captions of the selected images are kept
    assert len(imgs) == df[df.image_id.isin(imgs.image_id.unique())].shape[0]
    with pytest.raises(ValueError):
        select_subset(df, caption_frac=0.5, num_images=3)


def test_train_pairs_rejects_positive_as_negative(coco_dir, tiny_tf):
    df = build_caption_table(str(coco_dir / "train2017"), str(coco_dir / "annotations/captions_train2017.json"))
    table = PairTable.from_df(df)
    ds = TrainPairs(table, tiny_tf, num_negatives=1)
    with pytest.raises(ValueError):
        ds.set_negatives(table.caption_image[:, None].copy())
    ds.set_negatives(((table.caption_image + 1) % table.num_images)[:, None])
    pos, idx, pos_id, negs, neg_ids = ds[3]
    assert pos.shape == (3, 8, 8) and negs.shape == (1, 3, 8, 8)
    assert idx == 3 and pos_id == table.caption_image[3] and neg_ids.item() != pos_id
