#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from final_common import (
    DEFAULT_INDEX,
    DEFAULT_FORMAL,
    FinalInteractionDataset,
    actor_ids,
    check_no_formal,
    generate_proposals,
    load_index,
    load_network,
    make_gt_rotation,
    geodesic_rotation_loss,
    rad2deg,
    vector_angle,
)

RV_CNT = 100

def selector_metrics(pull_err_deg, rot_err_deg, d2_err_deg, idx):
    B = pull_err_deg.shape[0]
    bi = torch.arange(B, device=pull_err_deg.device)
    sp = pull_err_deg[bi, idx]
    sr = rot_err_deg[bi, idx]
    sd2 = d2_err_deg[bi, idx]
    return {
        "pull": sp.detach().cpu().numpy(),
        "rot": sr.detach().cpu().numpy(),
        "d2": sd2.detach().cpu().numpy(),
    }

def summarize(values):
    pull = np.concatenate([x["pull"] for x in values]).astype(np.float64)
    rot = np.concatenate([x["rot"] for x in values]).astype(np.float64)
    d2 = np.concatenate([x["d2"] for x in values]).astype(np.float64)
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

def mean_std(seed_rows, selector, key):
    x = np.asarray(
        [row["selectors"][selector][key] for row in seed_rows],
        dtype=np.float64,
    )
    return {"mean": float(x.mean()), "std": float(x.std())}

def pairwise_density(pull_vectors, k):
    """
    pull_vectors: [B,100,3], unit vectors.

    Density = mean cosine similarity to k nearest OTHER proposals.
    This is GT-free and therefore legal at runtime.
    """
    sim = torch.bmm(
        pull_vectors,
        pull_vectors.transpose(1, 2),
    )
    B, N, _ = sim.shape
    eye = torch.eye(N, dtype=torch.bool, device=sim.device).unsqueeze(0)
    sim = sim.masked_fill(eye, -2.0)
    vals = torch.topk(sim, k=min(k, N - 1), dim=2, largest=True).values
    return vals.mean(dim=2)

def top_density_then_critic(density, critic, topn):
    topn = min(int(topn), density.shape[1])
    ids = torch.topk(density, k=topn, dim=1, largest=True).indices
    cand_scores = torch.gather(critic, 1, ids)
    local = cand_scores.argmax(dim=1)
    return torch.gather(ids, 1, local[:, None]).squeeze(1)

@torch.no_grad()
def evaluate_seed(model, loader, device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    per_selector = {
        "critic_argmax": [],
        "pull_mean_consensus": [],
        "pull_density10": [],
        "pull_density20": [],
        "density10_top20_then_critic": [],
        "density20_top20_then_critic": [],
        "density10_top40_then_critic": [],
        "density20_top40_then_critic": [],
    }

    oracle_pull = []
    oracle_rot = []
    proposal_pull30 = []
    proposal_rot45 = []

    for pc, d1, d2, _ in loader:
        pc = pc.to(device, non_blocking=True)
        d1 = d1.to(device, non_blocking=True)
        d2 = d2.to(device, non_blocking=True)
        B = len(pc)

        whole = model.pointnet2(pc.repeat(1, 1, 2))
        qf = whole[:, :, 0]
        p = generate_proposals(model, qf, d1, d2, generator=gen)

        pull_err = rad2deg(p["pull_err"])
        rot_err = rad2deg(p["rot_err"])
        d2_err = rad2deg(p["d2_err"])
        pull_vec = -p["proposal_d1"]

        query6 = torch.cat(
            [
                p["proposal_d1"].reshape(B * RV_CNT, 3),
                p["proposal_d2"].reshape(B * RV_CNT, 3),
            ],
            dim=1,
        )
        critic = torch.sigmoid(
            model.critic(p["expanded_feat"], query6)
        ).reshape(B, RV_CNT)

        # Baseline Critic argmax.
        idx_critic = critic.argmax(dim=1)

        # Spherical-mean pull consensus.
        mean_pull = torch.nn.functional.normalize(
            pull_vec.mean(dim=1),
            dim=1,
        )
        mean_sim = (pull_vec * mean_pull[:, None, :]).sum(dim=2)
        idx_mean = mean_sim.argmax(dim=1)

        density10 = pairwise_density(pull_vec, 10)
        density20 = pairwise_density(pull_vec, 20)
        idx_d10 = density10.argmax(dim=1)
        idx_d20 = density20.argmax(dim=1)

        selectors = {
            "critic_argmax": idx_critic,
            "pull_mean_consensus": idx_mean,
            "pull_density10": idx_d10,
            "pull_density20": idx_d20,
            "density10_top20_then_critic": top_density_then_critic(
                density10, critic, 20
            ),
            "density20_top20_then_critic": top_density_then_critic(
                density20, critic, 20
            ),
            "density10_top40_then_critic": top_density_then_critic(
                density10, critic, 40
            ),
            "density20_top40_then_critic": top_density_then_critic(
                density20, critic, 40
            ),
        }

        for name, idx in selectors.items():
            per_selector[name].append(
                selector_metrics(pull_err, rot_err, d2_err, idx)
            )

        op = pull_err.min(dim=1).values
        ort = rot_err.min(dim=1).values
        oracle_pull.extend(op.cpu().numpy().tolist())
        oracle_rot.extend(ort.cpu().numpy().tolist())
        proposal_pull30.extend(
            (pull_err < 30.0).float().mean(dim=1).cpu().numpy().tolist()
        )
        proposal_rot45.extend(
            (rot_err < 45.0).float().mean(dim=1).cpu().numpy().tolist()
        )

    selectors_summary = {
        name: summarize(values)
        for name, values in per_selector.items()
    }
    op = np.asarray(oracle_pull, dtype=np.float64)
    ort = np.asarray(oracle_rot, dtype=np.float64)
    return {
        "seed": int(seed),
        "oracle": {
            "pull_mean_deg": float(op.mean()),
            "pull_hit15": float((op < 15.0).mean()),
            "rot_mean_deg": float(ort.mean()),
            "rot_hit30": float((ort < 30.0).mean()),
            "proposal_pull30_fraction": float(np.mean(proposal_pull30)),
            "proposal_rot45_fraction": float(np.mean(proposal_rot45)),
        },
        "selectors": selectors_summary,
    }

def gate(stat):
    return {
        "pull_mean_le_25": stat["pull_mean_deg"] <= 25.0,
        "pull_median_le_15": stat["pull_median_deg"] <= 15.0,
        "pull_hit30_ge_080": stat["pull_hit30"] >= 0.80,
        "pull_gt90_le_002": stat["pull_gt90"] <= 0.02,
        "rot_mean_le_40": stat["rot_mean_deg"] <= 40.0,
        "rot_hit45_ge_075": stat["rot_hit45"] >= 0.75,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(DEFAULT_INDEX))
    p.add_argument(
        "--checkpoint",
        default="/home/feng/robot_baselines/results/where2act/v3_frozen/"
                "actor_v3_epoch12_generator.pth",
        help="Frozen epoch12 Actor + V3.1 Critic full Network state_dict.",
    )
    p.add_argument(
        "--out-dir",
        default="/home/feng/robot_baselines/results/where2act/v3_audit/"
                "selector_final_v1",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=12)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")

    index = load_index(args.index)
    check_no_formal(index, DEFAULT_FORMAL)
    dev_ids = actor_ids(index, "dev")
    ds = FinalInteractionDataset(index, dev_ids, seed=20260902)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model = load_network(args.checkpoint, device)
    model.eval()

    seeds = [8100, 8101, 8102, 8103, 8104]
    rows = []
    print("=" * 110)
    print("WHERE2ACT FINAL SELECTOR DECISION AUDIT")
    print("=" * 110)
    print("checkpoint:", args.checkpoint)
    print("dev samples:", len(dev_ids))
    print("formal overlap: 0")
    for seed in seeds:
        row = evaluate_seed(model, loader, device, seed)
        rows.append(row)
        print(
            f"seed={seed}"
            f" oraclePull15={row['oracle']['pull_hit15']:.3f}"
            f" oracleRot30={row['oracle']['rot_hit30']:.3f}"
        )
        for name, stat in row["selectors"].items():
            print(
                f"  {name:<32s}"
                f" pull={stat['pull_mean_deg']:6.2f}"
                f" med={stat['pull_median_deg']:6.2f}"
                f" hit30={stat['pull_hit30']:.3f}"
                f" rot={stat['rot_mean_deg']:6.2f}"
                f" rot45={stat['rot_hit45']:.3f}"
            )

    selector_names = list(rows[0]["selectors"].keys())
    summary = {}
    for name in selector_names:
        summary[name] = {}
        for key in rows[0]["selectors"][name].keys():
            summary[name][key] = mean_std(rows, name, key)
        mean_stat = {
            key: summary[name][key]["mean"]
            for key in rows[0]["selectors"][name].keys()
        }
        summary[name]["gates"] = gate(mean_stat)
        summary[name]["gate_pass"] = all(summary[name]["gates"].values())
        # Fixed selector comparison score, used only if multiple selectors pass.
        summary[name]["selection_score"] = (
            mean_stat["pull_mean_deg"]
            + 0.25 * mean_stat["rot_mean_deg"]
            + 50.0 * (1.0 - mean_stat["pull_hit30"])
        )

    oracle_summary = {}
    for key in rows[0]["oracle"]:
        a = np.asarray([r["oracle"][key] for r in rows], dtype=np.float64)
        oracle_summary[key] = {"mean": float(a.mean()), "std": float(a.std())}

    passing = [
        name for name in selector_names
        if summary[name]["gate_pass"]
    ]
    if passing:
        recommended = min(
            passing,
            key=lambda name: summary[name]["selection_score"],
        )
        decision = "PASS_SELECTOR_FOUND"
    else:
        recommended = min(
            selector_names,
            key=lambda name: summary[name]["selection_score"],
        )
        decision = "NO_SELECTOR_PASSES"
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dev_samples": len(dev_ids),
        "formal_overlap": 0,
        "seeds": seeds,
        "oracle_summary": oracle_summary,
        "selectors": summary,
        "passing_selectors": passing,
        "recommended_selector": recommended,
        "decision": decision,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "=" * 110,
        "WHERE2ACT FINAL SELECTOR DECISION REPORT",
        "=" * 110,
        f"checkpoint                   : {Path(args.checkpoint).resolve()}",
        f"dev samples                  : {len(dev_ids)}",
        "formal overlap              : 0",
        "",
        "ORACLE",
    ]
    for k, v in oracle_summary.items():
        lines.append(f"{k:<46s}: {v['mean']:.6f} ± {v['std']:.6f}")
    for name in selector_names:
        s = summary[name]
        lines += ["", "-" * 110, f"SELECTOR: {name}", "-" * 110]
        for k, v in s.items():
            if isinstance(v, dict) and "mean" in v:
                lines.append(f"{k:<46s}: {v['mean']:.6f} ± {v['std']:.6f}")
        lines.append("GATES:")
        for k, v in s["gates"].items():
            lines.append(f"  {k:<42s}: {'PASS' if v else 'FAIL'}")
        lines.append(f"  selector gate             : {'PASS' if s['gate_pass'] else 'FAIL'}")
        lines.append(f"  comparison score          : {s['selection_score']:.6f}")
    lines += [
        "",
        "=" * 110,
        f"DECISION                     : {decision}",
        f"PASSING SELECTORS            : {passing}",
        f"RECOMMENDED SELECTOR         : {recommended}",
        "=" * 110,
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print()
    print((out_dir / "report.txt").read_text())

if __name__ == "__main__":
    main()
