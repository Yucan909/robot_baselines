#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from final_common import (
    DEFAULT_INDEX,
    DEFAULT_FORMAL,
    FinalInteractionDataset,
    actor_ids,
    check_no_formal,
    generate_proposals,
    geodesic_rotation_loss,
    load_index,
    load_network,
    make_gt_rotation,
    rad2deg,
    sha256_file,
    vector_angle,
)

RV_CNT = 100

class AnchorSelectorHead(nn.Module):
    """
    Explicit geometric selector head.

    Instead of asking the old Critic MLP to learn the bilinear relationship
    between scene feature and arbitrary proposal orientation, first predict
    one preferred 6D orientation anchor from the frozen query feature.

    Runtime then selects the Actor proposal nearest to that predicted anchor.
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 6),
        )

    def forward(self, feat):
        return self.net(feat)


def to_rotation(model, pred6):
    return model.actor.bgs(
        pred6.reshape(-1, 3, 2)
    )


def anchor_metrics(pred_R, d1, d2):
    gt_R = make_gt_rotation(d1, d2)
    pull_rad = vector_angle(
        -pred_R[:, :, 0],
        -d1,
    )
    rot_rad = geodesic_rotation_loss(
        pred_R,
        gt_R,
    )
    d2_rad = vector_angle(
        pred_R[:, :, 1],
        d2,
    )
    return pull_rad, rot_rad, d2_rad


def metric_summary(pull_deg, rot_deg, d2_deg):
    pull = np.asarray(pull_deg, dtype=np.float64)
    rot = np.asarray(rot_deg, dtype=np.float64)
    d2 = np.asarray(d2_deg, dtype=np.float64)
    return {
        "n": int(len(pull)),
        "pull_mean_deg": float(pull.mean()),
        "pull_median_deg": float(np.median(pull)),
        "pull_hit15": float((pull < 15.0).mean()),
        "pull_hit30": float((pull < 30.0).mean()),
        "pull_gt90": float((pull > 90.0).mean()),
        "rot_mean_deg": float(rot.mean()),
        "rot_median_deg": float(np.median(rot)),
        "rot_hit30": float((rot < 30.0).mean()),
        "rot_hit45": float((rot < 45.0).mean()),
        "d2_mean_deg": float(d2.mean()),
    }


def run_head_epoch(
    base_model,
    head,
    loader,
    optimizer,
    device,
    *,
    training,
):
    base_model.pointnet2.eval()
    base_model.actor.eval()
    head.train(training)

    loss_sum = 0.0
    n = 0
    pull_all = []
    rot_all = []
    d2_all = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            pc, d1, d2, _ = batch
            pc = pc.to(device, non_blocking=True)
            d1 = d1.to(device, non_blocking=True)
            d2 = d2.to(device, non_blocking=True)
            B = len(pc)

            with torch.no_grad():
                whole = base_model.pointnet2(
                    pc.repeat(1, 1, 2)
                )
                qf = whole[:, :, 0]

            pred6 = head(qf)
            pred_R = to_rotation(
                base_model,
                pred6,
            )
            pull_rad, rot_rad, d2_rad = anchor_metrics(
                pred_R,
                d1,
                d2,
            )

            # Frozen before training:
            # primary = opening/pull direction;
            # full rotation is a weaker physical grasp-orientation term.
            loss = (
                pull_rad.mean()
                + 0.35 * rot_rad.mean()
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    5.0,
                )
                optimizer.step()

            with torch.no_grad():
                pull_deg = rad2deg(pull_rad)
                rot_deg = rad2deg(rot_rad)
                d2_deg = rad2deg(d2_rad)
                pull_all.extend(
                    pull_deg.detach().cpu().numpy().tolist()
                )
                rot_all.extend(
                    rot_deg.detach().cpu().numpy().tolist()
                )
                d2_all.extend(
                    d2_deg.detach().cpu().numpy().tolist()
                )
                loss_sum += float(loss.item()) * B
                n += B

            if training and (
                step == 0
                or (step + 1) % 100 == 0
            ):
                print(
                    f"  step {step+1:5d}/{len(loader):5d}"
                    f" | loss={loss.item():.4f}"
                    f" | pull={pull_deg.mean().item():6.2f}"
                    f" | rot={rot_deg.mean().item():6.2f}"
                    f" | d2={d2_deg.mean().item():6.2f}"
                )

    result = metric_summary(
        pull_all,
        rot_all,
        d2_all,
    )
    result["loss"] = float(
        loss_sum / max(n, 1)
    )
    # Checkpoint selection is completely independent of Critic.
    result["selection_score"] = float(
        result["pull_mean_deg"]
        + 0.35 * result["rot_mean_deg"]
    )
    return result


@torch.no_grad()
def evaluate_selector_seed(
    base_model,
    head,
    loader,
    device,
    seed,
):
    base_model.eval()
    head.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    selected_pull = []
    selected_rot = []
    selected_d2 = []

    critic_pull = []
    critic_rot = []
    critic_d2 = []

    oracle_pull = []
    oracle_rot = []

    proposal_pull30 = []
    proposal_rot45 = []

    anchor_pull = []
    anchor_rot = []
    anchor_d2 = []

    for pc, d1, d2, _ in loader:
        pc = pc.to(device, non_blocking=True)
        d1 = d1.to(device, non_blocking=True)
        d2 = d2.to(device, non_blocking=True)
        B = len(pc)

        whole = base_model.pointnet2(
            pc.repeat(1, 1, 2)
        )
        qf = whole[:, :, 0]

        anchor_R = to_rotation(
            base_model,
            head(qf),
        )

        a_pull_rad, a_rot_rad, a_d2_rad = (
            anchor_metrics(
                anchor_R,
                d1,
                d2,
            )
        )

        anchor_pull.extend(
            rad2deg(a_pull_rad).cpu().numpy().tolist()
        )
        anchor_rot.extend(
            rad2deg(a_rot_rad).cpu().numpy().tolist()
        )
        anchor_d2.extend(
            rad2deg(a_d2_rad).cpu().numpy().tolist()
        )

        p = generate_proposals(
            base_model,
            qf,
            d1,
            d2,
            generator=gen,
        )

        # ----------------------------------------------------
        # FINAL anchor selector:
        #
        # choose the existing Where2Act Actor proposal nearest
        # to the predicted geometric anchor.
        #
        # cost = pull-angle + 0.35 * full-rotation-angle
        #
        # No GT and no oracle is used here.
        # ----------------------------------------------------

        proposal_pull = (
            -p["proposal_d1"]
        )
        anchor_pull_vec = (
            -anchor_R[:, :, 0]
        )

        pull_to_anchor = vector_angle(
            proposal_pull,
            anchor_pull_vec[:, None, :],
        )

        anchor_expand = (
            anchor_R
            .unsqueeze(1)
            .expand(
                -1,
                RV_CNT,
                -1,
                -1,
            )
            .reshape(
                B * RV_CNT,
                3,
                3,
            )
        )

        proposal_R_flat = (
            p["pred_R"]
            .reshape(
                B * RV_CNT,
                3,
                3,
            )
        )

        rot_to_anchor = (
            geodesic_rotation_loss(
                proposal_R_flat,
                anchor_expand,
            )
            .reshape(
                B,
                RV_CNT,
            )
        )

        selector_cost = (
            pull_to_anchor
            + 0.35 * rot_to_anchor
        )

        selector_idx = (
            selector_cost.argmin(
                dim=1
            )
        )

        bi = torch.arange(
            B,
            device=device,
        )

        selected_pull.extend(
            rad2deg(
                p["pull_err"][
                    bi,
                    selector_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        selected_rot.extend(
            rad2deg(
                p["rot_err"][
                    bi,
                    selector_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        selected_d2.extend(
            rad2deg(
                p["d2_err"][
                    bi,
                    selector_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        # Old Critic argmax is retained only as a baseline.
        query6 = torch.cat(
            [
                p["proposal_d1"].reshape(
                    B * RV_CNT,
                    3,
                ),
                p["proposal_d2"].reshape(
                    B * RV_CNT,
                    3,
                ),
            ],
            dim=1,
        )
        critic_score = torch.sigmoid(
            base_model.critic(
                p["expanded_feat"],
                query6,
            )
        ).reshape(B, RV_CNT)

        critic_idx = (
            critic_score.argmax(
                dim=1
            )
        )

        critic_pull.extend(
            rad2deg(
                p["pull_err"][
                    bi,
                    critic_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        critic_rot.extend(
            rad2deg(
                p["rot_err"][
                    bi,
                    critic_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        critic_d2.extend(
            rad2deg(
                p["d2_err"][
                    bi,
                    critic_idx,
                ]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        op = rad2deg(
            p["pull_err"].min(
                dim=1
            ).values
        )
        ort = rad2deg(
            p["rot_err"].min(
                dim=1
            ).values
        )
        oracle_pull.extend(
            op.cpu().numpy().tolist()
        )
        oracle_rot.extend(
            ort.cpu().numpy().tolist()
        )

        proposal_pull30.extend(
            (
                rad2deg(
                    p["pull_err"]
                )
                < 30.0
            )
            .float()
            .mean(
                dim=1
            )
            .cpu()
            .numpy()
            .tolist()
        )
        proposal_rot45.extend(
            (
                rad2deg(
                    p["rot_err"]
                )
                < 45.0
            )
            .float()
            .mean(
                dim=1
            )
            .cpu()
            .numpy()
            .tolist()
        )

    selected = metric_summary(
        selected_pull,
        selected_rot,
        selected_d2,
    )
    critic = metric_summary(
        critic_pull,
        critic_rot,
        critic_d2,
    )
    anchor = metric_summary(
        anchor_pull,
        anchor_rot,
        anchor_d2,
    )

    op = np.asarray(
        oracle_pull,
        dtype=np.float64,
    )
    ort = np.asarray(
        oracle_rot,
        dtype=np.float64,
    )

    return {
        "seed": int(seed),
        "anchor": anchor,
        "selected": selected,
        "critic_baseline": critic,
        "oracle": {
            "pull_mean_deg":
                float(op.mean()),

            "pull_hit15":
                float(
                    (op < 15.0).mean()
                ),

            "rot_mean_deg":
                float(ort.mean()),

            "rot_hit30":
                float(
                    (ort < 30.0).mean()
                ),

            "proposal_pull30_fraction":
                float(
                    np.mean(
                        proposal_pull30
                    )
                ),

            "proposal_rot45_fraction":
                float(
                    np.mean(
                        proposal_rot45
                    )
                ),
        },
    }


def aggregate_rows(rows, section):
    keys = rows[0][section].keys()
    out = {}
    for key in keys:
        a = np.asarray(
            [
                row[section][key]
                for row in rows
            ],
            dtype=np.float64,
        )
        out[key] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
        }
    return out


def selector_gates(summary):
    return {
        "selected_pull_mean_le_25deg":
            (
                summary[
                    "pull_mean_deg"
                ]["mean"]
                <= 25.0
            ),

        "selected_pull_median_le_15deg":
            (
                summary[
                    "pull_median_deg"
                ]["mean"]
                <= 15.0
            ),

        "selected_pull_hit30_ge_0.80":
            (
                summary[
                    "pull_hit30"
                ]["mean"]
                >= 0.80
            ),

        "selected_pull_gt90_le_0.02":
            (
                summary[
                    "pull_gt90"
                ]["mean"]
                <= 0.02
            ),

        "selected_rot_mean_le_40deg":
            (
                summary[
                    "rot_mean_deg"
                ]["mean"]
                <= 40.0
            ),

        "selected_rot_hit45_ge_0.75":
            (
                summary[
                    "rot_hit45"
                ]["mean"]
                >= 0.75
            ),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--index",
        default=str(
            DEFAULT_INDEX
        ),
    )

    p.add_argument(
        "--base-checkpoint",
        default=(
            "/home/feng/robot_baselines/results/"
            "where2act/v3_frozen/"
            "actor_v3_epoch12_generator.pth"
        ),
    )

    p.add_argument(
        "--out-dir",
        default=(
            "/home/feng/robot_baselines/repos/"
            "where2act/logs/"
            "where2act_anchor_selector"
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=12,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = p.parse_args()

    random.seed(
        args.seed
    )
    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda"
    )

    index = load_index(
        args.index
    )
    check_no_formal(
        index,
        DEFAULT_FORMAL,
    )

    train_ids = actor_ids(
        index,
        "train",
    )
    dev_ids = actor_ids(
        index,
        "dev",
    )

    print("=" * 108)
    print("WHERE2ACT ANCHOR SELECTOR V1")
    print("=" * 108)
    print(
        "base checkpoint:",
        args.base_checkpoint,
    )
    print(
        "train/dev:",
        len(train_ids),
        "/",
        len(dev_ids),
    )
    print(
        "formal overlap: 0"
    )
    print(
        "head loss = pull_angle + 0.35 * SO(3)_angle"
    )
    print(
        "runtime selector = nearest Actor proposal to predicted anchor"
    )

    train_ds = FinalInteractionDataset(
        index,
        train_ids,
        seed=args.seed,
    )
    dev_ds = FinalInteractionDataset(
        index,
        dev_ids,
        seed=args.seed + 100000,
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
        prefetch_factor=(
            2
            if args.num_workers > 0
            else None
        ),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
        prefetch_factor=(
            2
            if args.num_workers > 0
            else None
        ),
    )

    base_model = load_network(
        args.base_checkpoint,
        device,
    )
    base_model.eval()

    for param in (
        base_model.parameters()
    ):
        param.requires_grad = False

    head = AnchorSelectorHead(
        128
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_score = float(
        "inf"
    )
    best_epoch = None
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print("=" * 108)
        print(
            f"EPOCH {epoch}/{args.epochs}"
            f" | lr="
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )
        print("=" * 108)

        tr = run_head_epoch(
            base_model,
            head,
            train_loader,
            optimizer,
            device,
            training=True,
        )

        dv = run_head_epoch(
            base_model,
            head,
            dev_loader,
            None,
            device,
            training=False,
        )

        scheduler.step()

        print(
            "[TRAIN]"
            f" pull={tr['pull_mean_deg']:.2f}"
            f" med={tr['pull_median_deg']:.2f}"
            f" hit30={tr['pull_hit30']:.3f}"
            f" | rot={tr['rot_mean_deg']:.2f}"
            f" rot45={tr['rot_hit45']:.3f}"
            f" d2={tr['d2_mean_deg']:.2f}"
        )

        print(
            "[DEV]  "
            f" pull={dv['pull_mean_deg']:.2f}"
            f" med={dv['pull_median_deg']:.2f}"
            f" hit15={dv['pull_hit15']:.3f}"
            f" hit30={dv['pull_hit30']:.3f}"
            f" | rot={dv['rot_mean_deg']:.2f}"
            f" rot45={dv['rot_hit45']:.3f}"
            f" d2={dv['d2_mean_deg']:.2f}"
            f" | score={dv['selection_score']:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train": tr,
            "dev": dv,
        })

        state = {
            "epoch": epoch,
            "head_state_dict":
                head.state_dict(),

            "head_architecture":
                "128-256-256-128-6",

            "loss":
                "pull_angle + 0.35*rotation_geodesic",

            "selector":
                "pull_to_anchor + 0.35*rotation_to_anchor",

            "base_checkpoint":
                str(
                    Path(
                        args.base_checkpoint
                    ).resolve()
                ),

            "base_checkpoint_sha256":
                sha256_file(
                    args.base_checkpoint
                ),

            "index_sha256":
                sha256_file(
                    args.index
                ),

            "formal_overlap":
                0,
        }

        torch.save(
            state,
            out_dir
            / "last.pt",
        )

        if (
            dv["selection_score"]
            < best_score
        ):
            best_score = float(
                dv[
                    "selection_score"
                ]
            )
            best_epoch = int(
                epoch
            )
            torch.save(
                state,
                out_dir
                / "best.pt",
            )
            print(
                f"[BEST] epoch={epoch}"
                f" score="
                f"{best_score:.6f}"
            )

        (
            out_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
            + "\n"
        )

    best = torch.load(
        out_dir
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    head.load_state_dict(
        best[
            "head_state_dict"
        ],
        strict=True,
    )
    head.to(
        device
    )
    head.eval()

    seeds = [
        9100,
        9101,
        9102,
        9103,
        9104,
    ]

    rows = []

    print()
    print("=" * 108)
    print("FINAL 5-SEED PROPOSAL SELECTION")
    print("=" * 108)

    for seed in seeds:
        row = evaluate_selector_seed(
            base_model,
            head,
            dev_loader,
            device,
            seed,
        )
        rows.append(
            row
        )

        s = row[
            "selected"
        ]
        c = row[
            "critic_baseline"
        ]
        print(
            f"seed={seed}"
            f" | ANCHOR-SELECT pull="
            f"{s['pull_mean_deg']:.2f}"
            f" med={s['pull_median_deg']:.2f}"
            f" hit30={s['pull_hit30']:.3f}"
            f" rot={s['rot_mean_deg']:.2f}"
            f" rot45={s['rot_hit45']:.3f}"
            f" | oldCritic pull="
            f"{c['pull_mean_deg']:.2f}"
        )

    anchor_summary = aggregate_rows(
        rows,
        "anchor",
    )
    selected_summary = aggregate_rows(
        rows,
        "selected",
    )
    critic_summary = aggregate_rows(
        rows,
        "critic_baseline",
    )
    oracle_summary = aggregate_rows(
        rows,
        "oracle",
    )

    anchor_gates = {
        "anchor_pull_mean_le_20deg":
            (
                anchor_summary[
                    "pull_mean_deg"
                ]["mean"]
                <= 20.0
            ),

        "anchor_pull_median_le_12deg":
            (
                anchor_summary[
                    "pull_median_deg"
                ]["mean"]
                <= 12.0
            ),

        "anchor_pull_hit30_ge_0.85":
            (
                anchor_summary[
                    "pull_hit30"
                ]["mean"]
                >= 0.85
            ),

        "anchor_rot_mean_le_35deg":
            (
                anchor_summary[
                    "rot_mean_deg"
                ]["mean"]
                <= 35.0
            ),

        "anchor_rot_hit45_ge_0.80":
            (
                anchor_summary[
                    "rot_hit45"
                ]["mean"]
                >= 0.80
            ),
    }

    selected_gates = selector_gates(
        selected_summary
    )

    gate_pass = (
        all(
            anchor_gates.values()
        )
        and
        all(
            selected_gates.values()
        )
    )

    result = {
        "best_epoch":
            best_epoch,

        "best_head_selection_score":
            best_score,

        "base_checkpoint":
            str(
                Path(
                    args.base_checkpoint
                ).resolve()
            ),

        "base_checkpoint_sha256":
            sha256_file(
                args.base_checkpoint
            ),

        "train_samples":
            len(
                train_ids
            ),

        "dev_samples":
            len(
                dev_ids
            ),

        "formal_overlap":
            0,

        "seeds":
            seeds,

        "anchor_summary":
            anchor_summary,

        "selected_summary":
            selected_summary,

        "critic_baseline_summary":
            critic_summary,

        "oracle_summary":
            oracle_summary,

        "anchor_gates":
            anchor_gates,

        "selected_gates":
            selected_gates,

        "anchor_selector_gate_pass":
            bool(
                gate_pass
            ),
    }

    (
        out_dir
        / "final_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    lines = [
        "=" * 108,
        "WHERE2ACT ANCHOR SELECTOR V1 FINAL REPORT",
        "=" * 108,
        f"best epoch                  : {best_epoch}",
        f"base checkpoint             : {Path(args.base_checkpoint).resolve()}",
        f"train/dev                   : {len(train_ids)} / {len(dev_ids)}",
        "formal overlap              : 0",
        "",
        "DIRECT ANCHOR PREDICTION",
    ]

    for k, v in anchor_summary.items():
        lines.append(
            f"{k:<45s}: "
            f"{v['mean']:.6f}"
            f" ± "
            f"{v['std']:.6f}"
        )

    lines += [
        "",
        "ANCHOR-SELECTED ACTOR PROPOSAL",
    ]

    for k, v in selected_summary.items():
        lines.append(
            f"{k:<45s}: "
            f"{v['mean']:.6f}"
            f" ± "
            f"{v['std']:.6f}"
        )

    lines += [
        "",
        "OLD CRITIC ARGMAX BASELINE",
    ]

    for k, v in critic_summary.items():
        lines.append(
            f"{k:<45s}: "
            f"{v['mean']:.6f}"
            f" ± "
            f"{v['std']:.6f}"
        )

    lines += [
        "",
        "ACTOR ORACLE",
    ]

    for k, v in oracle_summary.items():
        lines.append(
            f"{k:<45s}: "
            f"{v['mean']:.6f}"
            f" ± "
            f"{v['std']:.6f}"
        )

    lines += [
        "",
        "ANCHOR GATES",
    ]

    for k, v in anchor_gates.items():
        lines.append(
            f"  {k:<45s}: "
            f"{'PASS' if v else 'FAIL'}"
        )

    lines += [
        "",
        "SELECTED-PROPOSAL GATES",
    ]

    for k, v in selected_gates.items():
        lines.append(
            f"  {k:<45s}: "
            f"{'PASS' if v else 'FAIL'}"
        )

    lines += [
        "",
        (
            "ANCHOR SELECTOR GATE        : "
            + (
                "PASS"
                if gate_pass
                else "FAIL"
            )
        ),
        "",
        (
            "best selector checkpoint    : "
            f"{out_dir / 'best.pt'}"
        ),
    ]

    report_path = (
        out_dir
        / "final_report.txt"
    )

    report_path.write_text(
        "\n".join(
            lines
        )
        + "\n"
    )

    print()
    print(
        report_path.read_text()
    )


if __name__ == "__main__":
    main()
