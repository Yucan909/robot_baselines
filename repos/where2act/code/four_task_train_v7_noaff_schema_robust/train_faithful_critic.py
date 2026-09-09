#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from faithful_common import (
    FEAT_DIM,
    INDEX_DEFAULT,
    ConvertedWhere2ActDataset,
    ShapeBalancedBinaryBatchSampler,
    binary_metrics,
    build_converted_samples,
    load_index,
    sha256_file,
    split_stats,
)

from models import model_3d_critic


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def eval_dataset(model, loader, device):
    model.eval()
    labels, probs, variants = [], [], []
    with torch.no_grad():
        for pc, d1, d2, label, _gi, variant, _shape in loader:
            pc = pc.to(device, non_blocking=True)
            d1 = d1.to(device, non_blocking=True)
            d2 = d2.to(device, non_blocking=True)
            logits, _ = model(pc, d1, d2)
            prob = torch.sigmoid(logits).cpu().numpy().tolist()
            labels.extend(label.numpy().astype(int).tolist())
            probs.extend(prob)
            variants.extend(list(variant))

    overall = binary_metrics(labels, probs)

    out = {"converted_all": overall}
    for name in ("original", "neg_direction"):
        ids = [i for i, x in enumerate(variants) if x == name]
        if ids:
            out[name] = binary_metrics(
                [labels[i] for i in ids],
                [probs[i] for i in ids],
            )
    return out


def balanced_dev_score(metrics):
    m = metrics["converted_all"]
    # Selection uses only the frozen converted dev binary objective.
    return float(m["bce"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(INDEX_DEFAULT))
    p.add_argument(
        "--out-dir",
        default="/home/feng/robot_baselines/repos/where2act/logs/"
                "where2act_faithful_critic",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")

    index = load_index(args.index)
    train_samples = build_converted_samples(index, "train")
    dev_samples = build_converted_samples(index, "dev")

    train_ds = ConvertedWhere2ActDataset(index, train_samples)
    dev_ds = ConvertedWhere2ActDataset(index, dev_samples)

    sampler = ShapeBalancedBinaryBatchSampler(
        train_samples,
        batch_size=args.micro_batch_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print("=" * 108)
    print("WHERE2ACT FAITHFUL BASELINE - STAGE A: ACTION SCORING PRETRAIN")
    print("=" * 108)
    print("GPU                         :", torch.cuda.get_device_name(0))
    print("formal overlap              : 0")
    print("train stats                 :", split_stats(train_samples))
    print("dev stats                   :", split_stats(dev_samples))
    print("architecture                : official model_3d_critic.Network")
    print("training                    : PointNet++ + Critic FROM SCRATCH")
    print("binary balancing            : 50% positive / 50% negative")
    print("shape sampling              : uniform within each class")
    print("opposite hemisphere negative: enabled")
    print("micro batch / accum         :", args.micro_batch_size, "/", args.grad_accum)
    print("effective batch             :", args.micro_batch_size * args.grad_accum)
    print("lr / wd                     :", args.lr, "/", args.weight_decay)

    model = model_3d_critic.Network(FEAT_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        n = 0
        opt_steps = 0

        print()
        print("=" * 108)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 108)

        for step, batch in enumerate(train_loader, 1):
            pc, d1, d2, label, _gi, _variant, _shape = batch
            pc = pc.to(device, non_blocking=True)
            d1 = d1.to(device, non_blocking=True)
            d2 = d2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, _ = model(pc, d1, d2)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            bs = len(pc)
            loss_sum += float(loss.item()) * bs
            n += bs

            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    prob = torch.sigmoid(logits)
                    pos = label > 0.5
                    neg = ~pos
                    pp = float(prob[pos].mean()) if torch.any(pos) else float("nan")
                    pn = float(prob[neg].mean()) if torch.any(neg) else float("nan")
                print(
                    f"  micro {step:5d}/{len(train_loader):5d}"
                    f" | loss={loss.item():.5f}"
                    f" | P+={pp:.3f}"
                    f" | P-={pn:.3f}"
                    f" | opt_steps={opt_steps}"
                )

        train_bce = loss_sum / max(n, 1)
        dev_metrics = eval_dataset(model, dev_loader, device)
        score = balanced_dev_score(dev_metrics)

        print(f"[TRAIN] BCE={train_bce:.6f}")
        print(
            "[DEV converted] "
            f"BCE={dev_metrics['converted_all']['bce']:.6f} "
            f"AUC={dev_metrics['converted_all']['auc']:.6f} "
            f"bal={dev_metrics['converted_all']['balanced_accuracy']:.6f}"
        )
        print(
            "[DEV original ] "
            f"BCE={dev_metrics['original']['bce']:.6f} "
            f"AUC={dev_metrics['original']['auc']:.6f} "
            f"bal={dev_metrics['original']['balanced_accuracy']:.6f} "
            f"P+={dev_metrics['original']['p_positive']} "
            f"P-={dev_metrics['original']['p_negative']}"
        )
        print(
            "[DEV neg-dir  ] "
            f"Pnegative={dev_metrics['neg_direction']['negative_specificity']:.6f}"
        )

        rec = {
            "epoch": epoch,
            "train_bce": train_bce,
            "dev": dev_metrics,
            "selection_score": score,
        }
        history.append(rec)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_bce": train_bce,
            "dev_metrics": dev_metrics,
            "selection_score": score,
            "protocol": "where2act_faithful_baseline_v1",
            "stage": "critic_pretrain",
            "index_sha256": sha256_file(args.index),
            "formal_overlap": 0,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.micro_batch_size * args.grad_accum,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

        torch.save(state, out_dir / "last.pt")
        torch.save(model.state_dict(), out_dir / "last-network.pth")

        if score < best:
            best = score
            best_epoch = epoch
            torch.save(state, out_dir / "best.pt")
            torch.save(model.state_dict(), out_dir / "best-network.pth")
            print(f"[BEST] epoch={epoch} dev BCE={best:.6f}")

        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )

    best_state = torch.load(
        out_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_state["model"], strict=True)
    model.to(device)
    final_dev = eval_dataset(model, dev_loader, device)

    report = {
        "best_epoch": best_epoch,
        "best_dev_bce": best,
        "formal_overlap": 0,
        "train_stats": split_stats(train_samples),
        "dev_stats": split_stats(dev_samples),
        "final_dev": final_dev,
        "checkpoint": str((out_dir / "best-network.pth").resolve()),
        "checkpoint_sha256": sha256_file(out_dir / "best-network.pth"),
    }
    (out_dir / "final_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = [
        "=" * 108,
        "WHERE2ACT FAITHFUL CRITIC FINAL REPORT",
        "=" * 108,
        f"best epoch                  : {best_epoch}",
        f"best converted dev BCE      : {best:.6f}",
        "formal overlap              : 0",
        f"train stats                 : {split_stats(train_samples)}",
        f"dev stats                   : {split_stats(dev_samples)}",
        "",
    ]
    for section, vals in final_dev.items():
        lines.append(section.upper())
        for k, v in vals.items():
            lines.append(f"  {k:<38s}: {v}")
        lines.append("")
    lines += [
        f"checkpoint                   : {out_dir / 'best-network.pth'}",
        f"SHA256                       : {report['checkpoint_sha256']}",
    ]
    (out_dir / "final_report.txt").write_text("\n".join(lines) + "\n")
    print()
    print((out_dir / "final_report.txt").read_text())


if __name__ == "__main__":
    main()
