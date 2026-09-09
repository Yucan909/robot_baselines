#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from faithful_common import (
    FEAT_DIM,
    RV_DIM,
    RV_CNT,
    INDEX_DEFAULT,
    ConvertedWhere2ActDataset,
    ShapeBalancedBinaryBatchSampler,
    binary_metrics,
    build_converted_samples,
    clean_state_dict,
    load_index,
    sha256_file,
    split_stats,
)

from models.model_3d import Network


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def load_pretrained_critic(model, path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    state = clean_state_dict(data)
    pn, cr = {}, {}
    for k, v in state.items():
        if k.startswith("pointnet2."):
            pn[k[len("pointnet2."):]] = v
        elif k.startswith("critic."):
            cr[k[len("critic."):]] = v
    if not pn or not cr:
        raise RuntimeError("pretrained critic missing pointnet2/critic weights")

    model.pointnet2.load_state_dict(pn, strict=True)
    model.critic.load_state_dict(cr, strict=True)
    model.critic_copy.load_state_dict(cr, strict=True)
    print("loaded PointNet++ tensors:", len(pn))
    print("loaded Critic tensors   :", len(cr))


def evaluate(model, loader, device, seed):
    model.eval()
    labels, probs, variants = [], [], []
    total_loss = 0.0
    total_critic = 0.0
    total_actor_pos = 0.0
    total_action = 0.0
    n = 0
    pos_n = 0

    # Network.forward uses torch.randn internally. Fork RNG so validation
    # randomness is fixed and does not perturb training RNG.
    devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            for pc, d1, d2, label, _gi, variant, _shape in loader:
                pc = pc.to(device, non_blocking=True)
                d1 = d1.to(device, non_blocking=True)
                d2 = d2.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                (
                    critic_per,
                    actor_per,
                    action_per,
                    logits,
                    _whole,
                ) = model(pc, d1, d2, label)

                critic_loss = critic_per.mean()
                pos_mask = label
                actor_loss = (
                    (actor_per * pos_mask).sum()
                    / (pos_mask.sum() + 1e-12)
                )
                action_loss = action_per.mean()
                total = critic_loss + actor_loss + 100.0 * action_loss

                bs = len(pc)
                pos_count = int((label > 0.5).sum().item())

                total_loss += float(total.item()) * bs
                total_critic += float(critic_loss.item()) * bs
                total_action += float(action_loss.item()) * bs
                if pos_count:
                    total_actor_pos += float(
                        (actor_per[label > 0.5].sum()).item()
                    )
                    pos_n += pos_count

                pr = torch.sigmoid(logits).cpu().numpy().tolist()
                probs.extend(pr)
                labels.extend(label.cpu().numpy().astype(int).tolist())
                variants.extend(list(variant))
                n += bs

    metrics = {
        "total_loss": total_loss / max(n, 1),
        "critic_loss": total_critic / max(n, 1),
        "actor_positive_rad": total_actor_pos / max(pos_n, 1),
        "actor_positive_deg": (
            total_actor_pos / max(pos_n, 1) * 180.0 / math.pi
        ),
        "action_score_loss": total_action / max(n, 1),
        "converted_binary": binary_metrics(labels, probs),
    }
    ids = [i for i, x in enumerate(variants) if x == "original"]
    metrics["original_binary"] = binary_metrics(
        [labels[i] for i in ids],
        [probs[i] for i in ids],
    )
    ids = [i for i, x in enumerate(variants) if x == "neg_direction"]
    metrics["neg_direction_binary"] = binary_metrics(
        [labels[i] for i in ids],
        [probs[i] for i in ids],
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(INDEX_DEFAULT))
    p.add_argument(
        "--critic-checkpoint",
        default="/home/feng/robot_baselines/repos/where2act/logs/"
                "where2act_faithful_critic/best-network.pth",
    )
    p.add_argument(
        "--out-dir",
        default="/home/feng/robot_baselines/repos/where2act/logs/"
                "where2act_faithful_joint",
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
        seed=args.seed + 77,
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
    print("WHERE2ACT FAITHFUL BASELINE - STAGE B: OFFICIAL THREE-DECODER JOINT TRAINING")
    print("=" * 108)
    print("GPU                         :", torch.cuda.get_device_name(0))
    print("formal overlap              : 0")
    print("train stats                 :", split_stats(train_samples))
    print("dev stats                   :", split_stats(dev_samples))
    print("critic source               :", args.critic_checkpoint)
    print("all Network params trainable: YES")
    print("loss                        : Ls + Lr + 100*La")
    print("Actor loss                  : positives only, Min-of-100")
    print("Actionability target        : mean Critic score over 100 Actor proposals")
    print("effective batch             :", args.micro_batch_size * args.grad_accum)
    print("lr / wd                     :", args.lr, "/", args.weight_decay)

    model = Network(FEAT_DIM, RV_DIM, RV_CNT).to(device)
    load_pretrained_critic(model, args.critic_checkpoint)

    # Fidelity point: official full training optimizes the whole Network.
    for q in model.parameters():
        q.requires_grad = True

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

        sums = {
            "total": 0.0,
            "critic": 0.0,
            "actor_pos_rad": 0.0,
            "action": 0.0,
        }
        n = 0
        pos_n = 0
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

            (
                critic_per,
                actor_per,
                action_per,
                logits,
                _whole,
            ) = model(pc, d1, d2, label)

            critic_loss = critic_per.mean()
            actor_loss = (
                (actor_per * label).sum()
                / (label.sum() + 1e-12)
            )
            action_loss = action_per.mean()
            total_loss = (
                critic_loss
                + actor_loss
                + 100.0 * action_loss
            )

            (total_loss / args.grad_accum).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            bs = len(pc)
            pos_count = int((label > 0.5).sum().item())
            sums["total"] += float(total_loss.item()) * bs
            sums["critic"] += float(critic_loss.item()) * bs
            sums["action"] += float(action_loss.item()) * bs
            if pos_count:
                sums["actor_pos_rad"] += float(
                    actor_per[label > 0.5].sum().item()
                )
                pos_n += pos_count
            n += bs

            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    pmean = torch.sigmoid(logits).mean().item()
                print(
                    f"  micro {step:5d}/{len(train_loader):5d}"
                    f" | total={total_loss.item():.5f}"
                    f" | Ls={critic_loss.item():.5f}"
                    f" | Lr={actor_loss.item():.5f}"
                    f" | La={action_loss.item():.6f}"
                    f" | Cmean={pmean:.3f}"
                    f" | opt_steps={opt_steps}"
                )

        train_m = {
            "total_loss": sums["total"] / max(n, 1),
            "critic_loss": sums["critic"] / max(n, 1),
            "actor_positive_rad": sums["actor_pos_rad"] / max(pos_n, 1),
            "actor_positive_deg": (
                sums["actor_pos_rad"] / max(pos_n, 1)
                * 180.0 / math.pi
            ),
            "action_score_loss": sums["action"] / max(n, 1),
        }

        dev_m = evaluate(
            model,
            dev_loader,
            device,
            seed=2026090200,
        )
        score = float(dev_m["total_loss"])

        print(
            "[TRAIN] "
            f"total={train_m['total_loss']:.6f} "
            f"Ls={train_m['critic_loss']:.6f} "
            f"Lr={train_m['actor_positive_deg']:.2f}deg "
            f"La={train_m['action_score_loss']:.6f}"
        )
        print(
            "[DEV]   "
            f"total={dev_m['total_loss']:.6f} "
            f"Ls={dev_m['critic_loss']:.6f} "
            f"Lr={dev_m['actor_positive_deg']:.2f}deg "
            f"La={dev_m['action_score_loss']:.6f} "
            f"AUC(orig)={dev_m['original_binary']['auc']:.4f}"
        )

        rec = {
            "epoch": epoch,
            "train": train_m,
            "dev": dev_m,
            "selection_score": score,
        }
        history.append(rec)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_metrics": train_m,
            "dev_metrics": dev_m,
            "selection_score": score,
            "protocol": "where2act_faithful_baseline_v1",
            "stage": "joint_three_decoder",
            "loss": "Ls + Lr + 100*La",
            "index_sha256": sha256_file(args.index),
            "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
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
            print(f"[BEST] epoch={epoch} dev total={best:.6f}")

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

    repeats = []
    for seed in [2026090300, 2026090301, 2026090302]:
        repeats.append(evaluate(model, dev_loader, device, seed))

    report = {
        "best_epoch": best_epoch,
        "best_dev_total_loss": best,
        "formal_overlap": 0,
        "train_stats": split_stats(train_samples),
        "dev_stats": split_stats(dev_samples),
        "final_dev_repeats": repeats,
        "checkpoint": str((out_dir / "best-network.pth").resolve()),
        "checkpoint_sha256": sha256_file(out_dir / "best-network.pth"),
    }
    (out_dir / "final_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    actor_deg = np.asarray(
        [x["actor_positive_deg"] for x in repeats], dtype=np.float64
    )
    orig_auc = np.asarray(
        [x["original_binary"]["auc"] for x in repeats], dtype=np.float64
    )
    action_loss = np.asarray(
        [x["action_score_loss"] for x in repeats], dtype=np.float64
    )

    lines = [
        "=" * 108,
        "WHERE2ACT FAITHFUL JOINT TRAINING FINAL REPORT",
        "=" * 108,
        f"best epoch                  : {best_epoch}",
        f"best dev total loss         : {best:.6f}",
        "formal overlap              : 0",
        f"train stats                 : {split_stats(train_samples)}",
        f"dev stats                   : {split_stats(dev_samples)}",
        "",
        "3-SEED DEV DIAGNOSTIC",
        f"actor positive Min100 deg   : {actor_deg.mean():.6f} ± {actor_deg.std():.6f}",
        f"original binary AUC          : {orig_auc.mean():.6f} ± {orig_auc.std():.6f}",
        f"action score loss            : {action_loss.mean():.6f} ± {action_loss.std():.6f}",
        "",
        "No angle-based engineering gate is used for checkpoint acceptance.",
        "Checkpoint selection is frozen dev total loss under official Ls+Lr+100La.",
        "",
        f"checkpoint                   : {out_dir / 'best-network.pth'}",
        f"SHA256                       : {report['checkpoint_sha256']}",
    ]
    (out_dir / "final_report.txt").write_text("\n".join(lines) + "\n")
    print()
    print((out_dir / "final_report.txt").read_text())


if __name__ == "__main__":
    main()
