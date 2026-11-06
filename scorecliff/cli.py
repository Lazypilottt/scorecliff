"""Command-line interface: ``python -m scorecliff <command> ...``"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import load_json, save_json


def _subset_args(p: argparse.ArgumentParser, frac_default=None, seed_default=0) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--caption-frac", type=float, default=frac_default,
                   help="keep this fraction of shuffled caption rows")
    g.add_argument("--num-images", type=int, default=None,
                   help="keep all captions of this many shuffled images")
    p.add_argument("--subset-seed", type=int, default=seed_default)


def _load_subset(args) -> tuple[pd.DataFrame, dict]:
    from .data import select_subset

    df = pd.read_csv(args.csv)
    sub = select_subset(df, caption_frac=args.caption_frac, num_images=args.num_images, seed=args.subset_seed)
    info = {"csv": os.path.abspath(args.csv), "caption_frac": args.caption_frac,
            "num_images": args.num_images, "subset_seed": args.subset_seed}
    return sub, info


# --------------------------------------------------------------------------- #
def cmd_prepare(args) -> None:
    from .data import prepare_coco

    prepare_coco(args.coco_root, args.out)


def cmd_train(args) -> None:
    from .engine import TrainConfig, evaluate, train
    from .model import load_clip
    from .utils import get_device

    df, info = _load_subset(args)
    cfg = TrainConfig(
        miner=args.miner, loss=args.loss, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, num_negatives=args.num_negatives,
        warmup_epochs=args.warmup_epochs, mine_every=args.mine_every, qure_window=args.qure_window,
        cluster_algo=args.cluster_algo, n_clusters=args.n_clusters,
        freeze_text=args.freeze_text, freeze_vision=args.freeze_vision, seed=args.seed,
        amp=not args.no_amp, num_workers=args.num_workers,
        model_name=args.model, pretrained=args.pretrained, data=info,
    )
    device = get_device(args.device)
    model, train_tf, eval_tf, tokenizer = load_clip(cfg.model_name, cfg.pretrained, device=device)
    train(cfg, model, train_tf, eval_tf, tokenizer, df, device, args.out,
          save_every_epoch=args.save_every_epoch)

    for spec in args.eval or []:
        name, test_df, test_info = _eval_spec(spec)
        res = evaluate(model, eval_tf, tokenizer, test_df, device, out_dir=os.path.join(args.out, f"eval_{name}"),
                       amp=cfg.amp, num_workers=cfg.num_workers,
                       meta={"tag": cfg.tag, "seed": cfg.seed, "test": test_info})
        _print_result(name, res)


def _eval_spec(spec: str):
    """``name=path.csv[:num_images[:seed]]`` e.g. ``5k=data/val2017.csv`` or ``1k=data/val2017.csv:1000``."""
    from .data import select_subset

    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    path = parts[0]
    n = int(parts[1]) if len(parts) > 1 and parts[1] else None
    seed = int(parts[2]) if len(parts) > 2 else 0
    df = select_subset(pd.read_csv(path), num_images=n, seed=seed)
    return name, df, {"csv": os.path.abspath(path), "num_images": n, "subset_seed": seed}


def cmd_evaluate(args) -> None:
    from .engine import evaluate
    from .model import load_checkpoint, load_clip
    from .utils import get_device

    device = get_device(args.device)
    if args.checkpoint:
        model, _, eval_tf, tokenizer, cfg = load_checkpoint(args.checkpoint, device)
        tag, seed = _tag_from_cfg(cfg), cfg.get("seed")
    else:
        model, _, eval_tf, tokenizer = load_clip(args.model, args.pretrained, device=device)
        tag, seed = "zero-shot", None
    df, info = _load_subset(args)
    res = evaluate(model, eval_tf, tokenizer, df, device, out_dir=args.out, amp=not args.no_amp,
                   num_workers=args.num_workers, meta={"tag": tag, "seed": seed, "test": info})
    _print_result(args.out or "eval", res)


def _tag_from_cfg(cfg: dict) -> str:
    from .engine import TrainConfig

    known = {k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}
    return TrainConfig(**known).tag


def cmd_compare(args) -> None:
    from .metrics import paired_bootstrap

    a, b = np.load(args.a), np.load(args.b)
    key = args.direction
    res = paired_bootstrap(a[key], b[key], n_boot=args.n_boot, seed=args.seed)
    print(f"{args.a}  vs  {args.b}   ({key}, paired bootstrap, n={len(a[key]):,})")
    print(f"{'metric':<7}{'A-B':>8}{'95% CI':>20}{'p':>9}")
    for m, r in res.items():
        print(f"{m:<7}{r['diff']:>+8.2f}   [{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]{r['p_value']:>9.4f}")
    if args.out:
        save_json(res, args.out)


def cmd_summarize(args) -> None:
    files = sorted({f for pattern in args.metrics for f in glob.glob(pattern, recursive=True)})
    if not files:
        sys.exit("no metrics files matched")
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        m = load_json(f)
        groups[m.get("tag", os.path.dirname(f))].append(m[args.direction])
    cols = ["R@1", "R@5", "R@10", "MRR", "MedR"]
    rows = []
    for tag, runs in sorted(groups.items()):
        row = {"method": tag, "runs": len(runs)}
        for c in cols:
            vals = np.array([r[c] for r in runs])
            row[c] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" if len(vals) > 1 else f"{vals.mean():.2f}"
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")
    if args.markdown:
        header = "| method | " + " | ".join(table.columns) + " |"
        print(header + "\n|" + "---|" * (len(table.columns) + 1))
        for tag, r in table.iterrows():
            print(f"| {tag} | " + " | ".join(str(v) for v in r.values) + " |")
    else:
        print(table.to_string())


def _print_result(name: str, res: dict) -> None:
    t, i = res["text_to_image"], res["image_to_text"]
    print(f"[{name}] {res['num_queries']:,} queries / {res['num_images']:,} images | "
          f"T->I R@1 {t['R@1']:.2f} R@5 {t['R@5']:.2f} R@10 {t['R@10']:.2f} MRR {t['MRR']:.2f} | "
          f"I->T R@1 {i['R@1']:.2f} R@5 {i['R@5']:.2f}")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    from .engine import MINERS
    from .losses import LOSSES
    from .miners import CLUSTER_ALGORITHMS
    from .model import DEFAULT_MODEL, DEFAULT_PRETRAINED

    p = argparse.ArgumentParser(prog="scorecliff", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="build caption CSVs from a COCO 2017 folder")
    s.add_argument("--coco-root", required=True, help="folder containing train2017/, val2017/, annotations/")
    s.add_argument("--out", default="data")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="fine-tune CLIP with a hard-negative miner")
    s.add_argument("--csv", default="data/train2017.csv")
    _subset_args(s, frac_default=0.01, seed_default=501)
    s.add_argument("--miner", choices=MINERS, default="qure")
    s.add_argument("--loss", choices=LOSSES, default="infonce")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--weight-decay", type=float, default=1e-4)
    s.add_argument("--num-negatives", type=int, default=1)
    s.add_argument("--warmup-epochs", type=int, default=1)
    s.add_argument("--mine-every", type=int, default=1)
    s.add_argument("--qure-window", type=int, default=None)
    s.add_argument("--cluster-algo", choices=CLUSTER_ALGORITHMS, default="kmeans")
    s.add_argument("--n-clusters", type=int, default=40)
    s.add_argument("--freeze-text", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=False)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    s.add_argument("--eval", nargs="*", metavar="NAME=CSV[:N_IMAGES[:SEED]]",
                   help="evaluate after training, e.g. 5k=data/val2017.csv 1k=data/val2017.csv:1000")
    s.add_argument("--save-every-epoch", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="retrieval metrics for a checkpoint (or zero-shot CLIP)")
    s.add_argument("--checkpoint", default=None, help="omit for zero-shot pretrained CLIP")
    s.add_argument("--csv", default="data/val2017.csv")
    _subset_args(s)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_evaluate)

    for s in (sub.choices["train"], sub.choices["evaluate"]):
        s.add_argument("--device", default=None)
        s.add_argument("--num-workers", type=int, default=4)
        s.add_argument("--no-amp", action="store_true")

    s = sub.add_parser("compare", help="paired bootstrap test between two runs")
    s.add_argument("a", help="ranks.npz of run A")
    s.add_argument("b", help="ranks.npz of run B")
    s.add_argument("--direction", choices=["t2i", "i2t"], default="t2i")
    s.add_argument("--n-boot", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("summarize", help="mean ± std table over seeds")
    s.add_argument("metrics", nargs="+", help="metrics.json files or globs")
    s.add_argument("--direction", choices=["text_to_image", "image_to_text"], default="text_to_image")
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(func=cmd_summarize)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "caption_frac", None) is not None and getattr(args, "num_images", None) is not None:
        args.caption_frac = None
    args.func(args)


if __name__ == "__main__":
    main()
