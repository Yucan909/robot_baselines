#!/usr/bin/env python3
import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DirectionAnchorHead(nn.Module):
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


def to_rotation(base_model, pred6):
    return base_model.actor.bgs(
        pred6.reshape(-1, 3, 2)
    )


def anchor_errors(pred_R, d1, d2):
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


def summary(pull_deg, rot_deg, d2_deg):
    p = np.asarray(pull_deg, dtype=np.float64)
    r = np.asarray(rot_deg, dtype=np.float64)
    d = np.asarray(d2_deg, dtype=np.float64)
    return {
        "n": int(len(p)),
        "pull_mean_deg": float(p.mean()),
        "pull_median_deg": float(np.median(p)),
        "pull_hit15": float((p < 15.0).mean()),
        "pull_hit30": float((p < 30.0).mean()),
        "pull_gt90": float((p > 90.0).mean()),
        "rot_mean_deg": float(r.mean()),
        "rot_median_deg": float(np.median(r)),
        "rot_hit30": float((r < 30.0).mean()),
        "rot_hit45": float((r < 45.0).mean()),
        "d2_mean_deg": float(d.mean()),
    }


def run_direction_epoch(
    base_model,
    direction_pointnet,
    head,
    loader,
    optimizer,
    device,
    *,
    training,
    static_weight,
):
    # Base Where2Act branch is always frozen.
    base_model.eval()
    base_model.action_score.eval()

    direction_pointnet.train(training)
    head.train(training)

    losses = {
        "loss": 0.0,
        "pull_loss_rad": 0.0,
        "rot_loss_rad": 0.0,
        "static_mse": 0.0,
    }
    n = 0

    pull_all = []
    rot_all = []
    d2_all = []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for step, batch in enumerate(loader):
            pc, d1, d2, _, aff = batch

            pc = pc.to(
                device,
                non_blocking=True,
            )
            d1 = d1.to(
                device,
                non_blocking=True,
            )
            d2 = d2.to(
                device,
                non_blocking=True,
            )
            aff = aff.to(
                device,
                non_blocking=True,
            )

            B, N, _ = pc.shape

            # ------------------------------------------------
            # NEW direction-aware representation.
            #
            # It starts from the clean affordance PointNet++ but is
            # allowed to change under direction supervision.
            # ------------------------------------------------
            dir_whole = direction_pointnet(
                pc.repeat(1, 1, 2)
            )
            qf_dir = dir_whole[:, :, 0]

            pred_R = to_rotation(
                base_model,
                head(qf_dir),
            )
            pull_rad, rot_rad, d2_rad = anchor_errors(
                pred_R,
                d1,
                d2,
            )

            pull_loss = pull_rad.mean()
            rot_loss = rot_rad.mean()

            # Preserve the already-validated dense affordance geometry
            # while making the representation articulation-direction aware.
            dense_feat = (
                dir_whole.permute(0, 2, 1)
                .reshape(B * N, -1)
            )
            dense_aff_pred = (
                base_model.action_score(
                    dense_feat
                )
                .reshape(B, N)
            )
            static_mse = F.mse_loss(
                dense_aff_pred,
                aff,
            )

            # Frozen before training.
            loss = (
                pull_loss
                + 0.35 * rot_loss
                + float(static_weight) * static_mse
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(direction_pointnet.parameters())
                    + list(head.parameters()),
                    5.0,
                )
                optimizer.step()

            with torch.no_grad():
                pdeg = rad2deg(pull_rad)
                rdeg = rad2deg(rot_rad)
                ddeg = rad2deg(d2_rad)

                pull_all.extend(
                    pdeg.detach().cpu().numpy().tolist()
                )
                rot_all.extend(
                    rdeg.detach().cpu().numpy().tolist()
                )
                d2_all.extend(
                    ddeg.detach().cpu().numpy().tolist()
                )

                losses["loss"] += (
                    float(loss.item()) * B
                )
                losses["pull_loss_rad"] += (
                    float(pull_loss.item()) * B
                )
                losses["rot_loss_rad"] += (
                    float(rot_loss.item()) * B
                )
                losses["static_mse"] += (
                    float(static_mse.item()) * B
                )
                n += B

            if training and (
                step == 0
                or (step + 1) % 100 == 0
            ):
                print(
                    f"  step {step+1:5d}/{len(loader):5d}"
                    f" | loss={loss.item():.4f}"
                    f" | pull={pdeg.mean().item():6.2f}"
                    f" | rot={rdeg.mean().item():6.2f}"
                    f" | d2={ddeg.mean().item():6.2f}"
                    f" | static={static_mse.item():.5f}"
                )

    result = summary(
        pull_all,
        rot_all,
        d2_all,
    )
    n = max(n, 1)
    for k, v in losses.items():
        result[k] = float(v / n)

    # Model selection depends only on direction prediction plus
    # a weak affordance-preservation penalty.
    result["selection_score"] = float(
        result["pull_mean_deg"]
        + 0.35 * result["rot_mean_deg"]
        + 50.0 * result["static_mse"]
    )
    return result


@torch.no_grad()
def evaluate_seed(
    base_model,
    direction_pointnet,
    head,
    loader,
    device,
    seed,
):
    base_model.eval()
    direction_pointnet.eval()
    head.eval()

    gen = torch.Generator(
        device=device
    )
    gen.manual_seed(
        int(seed)
    )

    anchor_pull = []
    anchor_rot = []
    anchor_d2 = []

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

    for pc, d1, d2, _, _aff in loader:
        pc = pc.to(
            device,
            non_blocking=True,
        )
        d1 = d1.to(
            device,
            non_blocking=True,
        )
        d2 = d2.to(
            device,
            non_blocking=True,
        )

        B = len(pc)

        # Existing Where2Act representation for Actor proposals.
        base_whole = base_model.pointnet2(
            pc.repeat(1, 1, 2)
        )
        base_qf = base_whole[:, :, 0]

        # New direction representation for anchor prediction.
        dir_whole = direction_pointnet(
            pc.repeat(1, 1, 2)
        )
        dir_qf = dir_whole[:, :, 0]

        anchor_R = to_rotation(
            base_model,
            head(dir_qf),
        )

        ap, ar, ad = anchor_errors(
            anchor_R,
            d1,
            d2,
        )
        anchor_pull.extend(
            rad2deg(ap).cpu().numpy().tolist()
        )
        anchor_rot.extend(
            rad2deg(ar).cpu().numpy().tolist()
        )
        anchor_d2.extend(
            rad2deg(ad).cpu().numpy().tolist()
        )

        p = generate_proposals(
            base_model,
            base_qf,
            d1,
            d2,
            generator=gen,
        )

        # ----------------------------------------------------
        # Select the EXISTING Actor proposal closest to anchor.
        # No GT/oracle is used in this selection.
        # ----------------------------------------------------
        pull_to_anchor = vector_angle(
            -p["proposal_d1"],
            -anchor_R[:, None, :, 0],
        )

        anchor_expand = (
            anchor_R.unsqueeze(1)
            .expand(-1, RV_CNT, -1, -1)
            .reshape(B * RV_CNT, 3, 3)
        )

        rot_to_anchor = (
            geodesic_rotation_loss(
                p["pred_R"].reshape(
                    B * RV_CNT,
                    3,
                    3,
                ),
                anchor_expand,
            )
            .reshape(B, RV_CNT)
        )

        cost = (
            pull_to_anchor
            + 0.35 * rot_to_anchor
        )

        idx = cost.argmin(
            dim=1
        )
        bi = torch.arange(
            B,
            device=device,
        )

        selected_pull.extend(
            rad2deg(
                p["pull_err"][bi, idx]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        selected_rot.extend(
            rad2deg(
                p["rot_err"][bi, idx]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        selected_d2.extend(
            rad2deg(
                p["d2_err"][bi, idx]
            )
            .cpu()
            .numpy()
            .tolist()
        )

        # Old Critic baseline.
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
        c = torch.sigmoid(
            base_model.critic(
                p["expanded_feat"],
                query6,
            )
        ).reshape(
            B,
            RV_CNT,
        )
        ci = c.argmax(
            dim=1
        )

        critic_pull.extend(
            rad2deg(
                p["pull_err"][bi, ci]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        critic_rot.extend(
            rad2deg(
                p["rot_err"][bi, ci]
            )
            .cpu()
            .numpy()
            .tolist()
        )
        critic_d2.extend(
            rad2deg(
                p["d2_err"][bi, ci]
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
            .mean(dim=1)
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
            .mean(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )

    anchor = summary(
        anchor_pull,
        anchor_rot,
        anchor_d2,
    )
    selected = summary(
        selected_pull,
        selected_rot,
        selected_d2,
    )
    critic = summary(
        critic_pull,
        critic_rot,
        critic_d2,
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

        "anchor":
            anchor,

        "selected":
            selected,

        "critic_baseline":
            critic,

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


def aggregate(rows, section):
    out = {}
    for key in rows[0][section]:
        x = np.asarray(
            [
                r[section][key]
                for r in rows
            ],
            dtype=np.float64,
        )
        out[key] = {
            "mean": float(
                x.mean()
            ),
            "std": float(
                x.std()
            ),
        }
    return out


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
            "where2act_direction_selector"
        ),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    p.add_argument(
        "--pointnet-lr",
        type=float,
        default=8e-5,
    )

    p.add_argument(
        "--head-lr",
        type=float,
        default=4e-4,
    )

    p.add_argument(
        "--static-weight",
        type=float,
        default=0.10,
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
    print("WHERE2ACT DIRECTION-AWARE SELECTOR V1")
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
        "base Where2Act branch: frozen"
    )
    print(
        "direction PointNet++: initialized from clean backbone, TRAINABLE"
    )
    print(
        "direction loss: pull + 0.35*SO(3) + 0.10*static-affordance-preservation"
    )

    train_ds = FinalInteractionDataset(
        index,
        train_ids,
        seed=args.seed,
        with_affordance=True,
    )
    dev_ds = FinalInteractionDataset(
        index,
        dev_ids,
        seed=args.seed + 100000,
        with_affordance=True,
    )

    g = torch.Generator()
    g.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
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
    for param in base_model.parameters():
        param.requires_grad = False

    # Critical structural change:
    # a separate direction-aware PointNet++ branch.
    direction_pointnet = copy.deepcopy(
        base_model.pointnet2
    ).to(
        device
    )

    for param in (
        direction_pointnet.parameters()
    ):
        param.requires_grad = True

    head = DirectionAnchorHead(
        128
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    direction_pointnet.parameters(),
                "lr":
                    args.pointnet_lr,
            },
            {
                "params":
                    head.parameters(),
                "lr":
                    args.head_lr,
            },
        ],
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
            f" | pointnet_lr="
            f"{optimizer.param_groups[0]['lr']:.2e}"
            f" | head_lr="
            f"{optimizer.param_groups[1]['lr']:.2e}"
        )
        print("=" * 108)

        tr = run_direction_epoch(
            base_model,
            direction_pointnet,
            head,
            train_loader,
            optimizer,
            device,
            training=True,
            static_weight=(
                args.static_weight
            ),
        )

        dv = run_direction_epoch(
            base_model,
            direction_pointnet,
            head,
            dev_loader,
            None,
            device,
            training=False,
            static_weight=(
                args.static_weight
            ),
        )

        scheduler.step()

        print(
            "[TRAIN]"
            f" pull={tr['pull_mean_deg']:.2f}"
            f" med={tr['pull_median_deg']:.2f}"
            f" hit30={tr['pull_hit30']:.3f}"
            f" | rot={tr['rot_mean_deg']:.2f}"
            f" rot45={tr['rot_hit45']:.3f}"
            f" | static={tr['static_mse']:.5f}"
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
            f" | static={dv['static_mse']:.5f}"
            f" | score={dv['selection_score']:.3f}"
        )

        history.append({
            "epoch":
                epoch,
            "train":
                tr,
            "dev":
                dv,
        })

        state = {
            "epoch":
                epoch,

            "direction_pointnet_state_dict":
                direction_pointnet.state_dict(),

            "head_state_dict":
                head.state_dict(),

            "training_version":
                "where2act_direction_aware_selector_v1",

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

            "pointnet_lr":
                args.pointnet_lr,

            "head_lr":
                args.head_lr,

            "static_weight":
                args.static_weight,

            "formal_overlap":
                0,
        }

        torch.save(
            state,
            out_dir
            / "last.pt",
        )

        if (
            dv[
                "selection_score"
            ]
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
                f" score={best_score:.6f}"
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

    direction_pointnet.load_state_dict(
        best[
            "direction_pointnet_state_dict"
        ],
        strict=True,
    )
    head.load_state_dict(
        best[
            "head_state_dict"
        ],
        strict=True,
    )

    direction_pointnet.to(
        device
    )
    head.to(
        device
    )
    direction_pointnet.eval()
    head.eval()

    seeds = [
        10100,
        10101,
        10102,
        10103,
        10104,
    ]

    rows = []

    print()
    print("=" * 108)
    print("FINAL 5-SEED PROPOSAL SELECTION")
    print("=" * 108)

    for seed in seeds:
        row = evaluate_seed(
            base_model,
            direction_pointnet,
            head,
            dev_loader,
            device,
            seed,
        )
        rows.append(
            row
        )

        a = row[
            "anchor"
        ]
        s = row[
            "selected"
        ]

        print(
            f"seed={seed}"
            f" | anchorPull="
            f"{a['pull_mean_deg']:.2f}"
            f" anchorRot="
            f"{a['rot_mean_deg']:.2f}"
            f" | selectedPull="
            f"{s['pull_mean_deg']:.2f}"
            f" med="
            f"{s['pull_median_deg']:.2f}"
            f" hit30="
            f"{s['pull_hit30']:.3f}"
            f" rot="
            f"{s['rot_mean_deg']:.2f}"
            f" rot45="
            f"{s['rot_hit45']:.3f}"
        )

    anchor_summary = aggregate(
        rows,
        "anchor",
    )
    selected_summary = aggregate(
        rows,
        "selected",
    )
    critic_summary = aggregate(
        rows,
        "critic_baseline",
    )
    oracle_summary = aggregate(
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

    selected_gates = {
        "selected_pull_mean_le_25deg":
            (
                selected_summary[
                    "pull_mean_deg"
                ]["mean"]
                <= 25.0
            ),

        "selected_pull_median_le_15deg":
            (
                selected_summary[
                    "pull_median_deg"
                ]["mean"]
                <= 15.0
            ),

        "selected_pull_hit30_ge_0.80":
            (
                selected_summary[
                    "pull_hit30"
                ]["mean"]
                >= 0.80
            ),

        "selected_pull_gt90_le_0.02":
            (
                selected_summary[
                    "pull_gt90"
                ]["mean"]
                <= 0.02
            ),

        "selected_rot_mean_le_40deg":
            (
                selected_summary[
                    "rot_mean_deg"
                ]["mean"]
                <= 40.0
            ),

        "selected_rot_hit45_ge_0.75":
            (
                selected_summary[
                    "rot_hit45"
                ]["mean"]
                >= 0.75
            ),
    }

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

        "best_selection_score":
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

        "direction_selector_gate_pass":
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
        "WHERE2ACT DIRECTION-AWARE SELECTOR V1 FINAL REPORT",
        "=" * 108,
        f"best epoch                  : {best_epoch}",
        f"base checkpoint             : {Path(args.base_checkpoint).resolve()}",
        f"train/dev                   : {len(train_ids)} / {len(dev_ids)}",
        "formal overlap              : 0",
        "",
        "DIRECTION-AWARE ANCHOR",
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
            "DIRECTION SELECTOR GATE     : "
            + (
                "PASS"
                if gate_pass
                else "FAIL"
            )
        ),
        "",
        (
            "best checkpoint             : "
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
