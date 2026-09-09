#!/usr/bin/env python3
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

THIS = Path(__file__).resolve()
CODE_ROOT = THIS.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

INDEX_DEFAULT = Path(
    "/home/feng/robot_baselines/results/where2act/v3_data/"
    "corrected_opening_index_v3_critic.npz"
)
FORMAL_SHAPES = Path(
    "/home/feng/robot_baselines/configs/where2act_v3/"
    "formal_holdout_shapes.txt"
)
EXPECTED_INDEX_SHA256 = (
    "d0106fedc278663cd9f3cd33726b76278242cf1e2efc1e8484c2a4afcd418fe1"
)

FEAT_DIM = 128
RV_DIM = 10
RV_CNT = 100
NUM_POINTS = 8192


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_shape_set(path):
    return {
        x.strip()
        for x in Path(path).read_text().splitlines()
        if x.strip()
    }


def load_index(path=INDEX_DEFAULT):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        index = {k: z[k].copy() for k in z.files}
    required = {
        "shape", "link", "split", "raw_path", "static_path", "qidx",
        "grasp_gate", "dir1_model", "dir2_model", "trust_strict",
        "pointcloud_path", "pointcloud_key", "pointcloud_frame_index",
        "point_count", "camera_pose",
    }
    missing = sorted(required - set(index))
    if missing:
        raise RuntimeError(f"four-task index missing keys: {missing}: {path}")
    n = len(index["shape"])
    for k, v in index.items():
        if np.asarray(v).ndim >= 1 and len(v) != n:
            raise RuntimeError(f"index first-dim mismatch {k}: {len(v)} != {n}")
    splits = set(np.asarray(index["split"]).astype(str).tolist())
    if not splits or not splits.issubset({"train", "dev"}):
        raise RuntimeError(f"invalid training splits {splits}: {path}")
    if not np.asarray(index["trust_strict"], dtype=bool).all():
        raise RuntimeError(f"saved four-task index must contain trusted trajectory-derived rows only: {path}")
    return index


def normalize_np(v):
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    if (not np.isfinite(n)) or n < 1e-8:
        raise RuntimeError("invalid direction")
    return (v / n).astype(np.float32)


@dataclass(frozen=True)
class ConvertedSample:
    global_idx: int
    label: int
    variant: str  # original | neg_direction
    shape: str


def build_converted_samples(index, split):
    """
    Faithful Where2Act conversion of the available source interactions.

    The supplied source trajectories are not native Where2Act random primitive
    trials. We therefore keep only the previously frozen strict geometry subset.

    Binary source outcome:
      original orientation -> grasp_gate
    Official Where2Act synthetic opposite-hemisphere negative:
      (-dir1, same dir2) -> 0

    This is a DATA conversion needed to train the baseline on the supplied data.
    It does not change the Where2Act network or loss.
    """
    split_arr = np.asarray(index["split"])
    strict = np.asarray(index["trust_strict"], dtype=bool)
    gate = np.asarray(index["grasp_gate"], dtype=bool)

    ids = np.flatnonzero(strict & (split_arr == str(split)))
    samples = []
    for gi in ids.tolist():
        shape = str(index["shape"][gi])
        label = int(bool(gate[gi]))
        samples.append(ConvertedSample(gi, label, "original", shape))
        # Official data loader marks the opposite hemisphere as negative
        # without executing another trial.
        samples.append(ConvertedSample(gi, 0, "neg_direction", shape))
    return samples


class ConvertedWhere2ActDataset(Dataset):
    """Thin source-loading adapter only; model/loss/sample semantics stay unchanged."""
    def __init__(self, index, samples, num_points=NUM_POINTS):
        self.index = index
        self.samples = list(samples)
        self.num_points = num_points  # retained for signature compatibility; task index fixes actual N.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        s = self.samples[item]
        gi = int(s.global_idx)
        source_path = str(self.index["pointcloud_path"][gi])
        pc_key = str(self.index["pointcloud_key"][gi])
        frame_i = int(self.index["pointcloud_frame_index"][gi])
        expected_n = int(self.index["point_count"][gi])

        with np.load(source_path, allow_pickle=False) as z:
            if pc_key not in z.files:
                raise RuntimeError(f"missing {pc_key}: {source_path}")
            pc_world = np.asarray(z[pc_key], dtype=np.float32)

        if pc_world.ndim == 3:
            if not (0 <= frame_i < len(pc_world)):
                raise RuntimeError(f"bad frame {frame_i} for {pc_world.shape}: {source_path}")
            pc_world = pc_world[frame_i]
        elif pc_world.ndim == 2:
            if frame_i not in (-1, 0):
                raise RuntimeError(f"2D point cloud with nontrivial frame {frame_i}: {source_path}")
        else:
            raise RuntimeError(f"bad point cloud rank {pc_world.shape}: {source_path}")

        if pc_world.shape != (expected_n, 3):
            raise RuntimeError(f"bad point cloud {pc_world.shape}, expected {(expected_n,3)}: {source_path}")
        if not np.isfinite(pc_world).all():
            raise RuntimeError(f"nonfinite point cloud: {source_path}")

        camera_pose = np.asarray(self.index["camera_pose"][gi], dtype=np.float32)
        if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
            raise RuntimeError(f"bad camera pose in index: {source_path}")
        R_cam_world = camera_pose[:3, :3]
        pc_model = (pc_world @ R_cam_world).astype(np.float32)

        qidx = int(self.index["qidx"][gi])
        if not (0 <= qidx < len(pc_model)):
            raise RuntimeError(f"bad qidx={qidx}: {source_path}")

        # Official Where2Act contract: interacting point is point 0.
        pc_model = pc_model.copy()
        if qidx != 0:
            tmp = pc_model[0].copy()
            pc_model[0] = pc_model[qidx]
            pc_model[qidx] = tmp

        d1 = normalize_np(self.index["dir1_model"][gi])
        d2 = normalize_np(self.index["dir2_model"][gi])
        if s.variant == "neg_direction":
            d1 = -d1
        elif s.variant != "original":
            raise RuntimeError(f"unknown variant {s.variant}")

        return (
            torch.from_numpy(pc_model),
            torch.from_numpy(d1),
            torch.from_numpy(d2),
            torch.tensor(float(s.label), dtype=torch.float32),
            torch.tensor(gi, dtype=torch.long),
            s.variant,
            s.shape,
        )


class ShapeBalancedBinaryBatchSampler(Sampler):
    """
    One micro-batch contains exactly half positive and half negative records.
    Within each class we sample shapes uniformly, then samples uniformly inside
    that shape. This approximates Where2Act's class balancing and shape
    balancing without introducing method-specific supervision.
    """
    def __init__(self, samples, batch_size=8, seed=42):
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size < 2 or self.batch_size % 2 != 0:
            raise ValueError("batch_size must be positive and even")

        self.by_class_shape = {0: {}, 1: {}}
        for i, s in enumerate(self.samples):
            self.by_class_shape[int(s.label)].setdefault(s.shape, []).append(i)

        for label in (0, 1):
            if not self.by_class_shape[label]:
                raise RuntimeError(f"no class {label} samples")

        self.class_counts = {
            y: sum(len(v) for v in self.by_class_shape[y].values())
            for y in (0, 1)
        }

        # One epoch sees approximately the larger class once.
        per_class_draws = max(self.class_counts.values())
        self.num_batches = int(
            math.ceil((2 * per_class_draws) / float(self.batch_size))
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        half = self.batch_size // 2
        shapes = {
            y: sorted(self.by_class_shape[y])
            for y in (0, 1)
        }

        for _ in range(self.num_batches):
            batch = []
            for y in (1, 0):
                for _j in range(half):
                    sid = rng.choice(shapes[y])
                    idx = rng.choice(self.by_class_shape[y][sid])
                    batch.append(idx)
            rng.shuffle(batch)
            yield batch


def split_stats(samples):
    out = {
        "total_converted": len(samples),
        "original": 0,
        "neg_direction": 0,
        "positive": 0,
        "negative": 0,
        "unique_shapes": len({s.shape for s in samples}),
    }
    for s in samples:
        out[s.variant] += 1
        if s.label:
            out["positive"] += 1
        else:
            out["negative"] += 1
    return out


def original_subset_samples(samples):
    return [s for s in samples if s.variant == "original"]


def auc_rank(labels, scores):
    """
    ROC-AUC with average ranks for ties. No sklearn dependency.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n1 = int(pos.sum())
    n0 = int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)

    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def binary_metrics(labels, probs):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    pred = p >= 0.5

    pos = y == 1
    neg = y == 0
    recall = float(pred[pos].mean()) if pos.any() else float("nan")
    specificity = float((~pred[neg]).mean()) if neg.any() else float("nan")
    balanced = float(0.5 * (recall + specificity))

    eps = 1e-7
    pclip = np.clip(p, eps, 1 - eps)
    bce = float(
        -(y * np.log(pclip) + (1 - y) * np.log(1 - pclip)).mean()
    )

    return {
        "n": int(len(y)),
        "positive": int(pos.sum()),
        "negative": int(neg.sum()),
        "bce": bce,
        "auc": auc_rank(y, p),
        "balanced_accuracy": balanced,
        "positive_recall": recall,
        "negative_specificity": specificity,
        "p_positive": float(p[pos].mean()) if pos.any() else None,
        "p_negative": float(p[neg].mean()) if neg.any() else None,
    }


def clean_state_dict(data):
    if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict):
        data = data["model"]
    elif (
        isinstance(data, dict)
        and "state_dict" in data
        and isinstance(data["state_dict"], dict)
    ):
        data = data["state_dict"]

    state = {}
    for k, v in data.items():
        k = str(k)
        if k.startswith("module."):
            k = k[len("module."):]
        state[k] = v
    return state
