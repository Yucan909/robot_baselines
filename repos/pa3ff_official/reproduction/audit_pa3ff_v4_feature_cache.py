#!/usr/bin/env python3
"""Verify cached V4 fields against fresh official PA3FF inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


HOME = Path("/home/feng")
OFFICIAL = HOME / "robot_baselines/repos/pa3ff_official"
ROOT = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
REPRESENTATION = HOME / "robot_baselines/results/pa3ff/representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth"
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import torch

    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / "libs/sonata"))
    os.chdir(OFFICIAL)
    from pointcept.datasets.transform import Compose
    from pointcept.models.PA3FF import PA3FF

    transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"), dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])
    model = PA3FF(backbone_dim=1088, output_dim=768, freeze_backbone=True,
                  max_grouping_scale=2, use_hierarchy_losses=True,
                  backbone=None).cuda().eval()
    payload = torch.load(REPRESENTATION, map_location="cpu")
    model.instance_net.load_state_dict(payload["instance_net"], strict=True)
    del payload

    rows = []
    with torch.inference_mode():
        for task in TASKS:
            for split in ("train", "dev"):
                points = np.load(ROOT / f"{task}_{split}_pointcloud_base_f32.npy", mmap_mode="r")
                normals = np.load(ROOT / f"{task}_{split}_normal_base_f32.npy", mmap_mode="r")
                cached = np.load(ROOT / f"{task}_{split}_pa3ff_field_f16.npy", mmap_mode="r")
                # Row zero is deterministic for the overwhelmingly common no-voxel-collision case.
                row = 0
                xyz = (np.asarray(points[row], dtype=np.float32) * 10.0).astype(np.float32)
                normal = np.asarray(normals[row], dtype=np.float32)
                color = np.zeros((1024, 3), dtype=np.float32)
                pcd = transform({
                    "coord": xyz.copy(), "color": color.copy(), "normal": normal.copy(),
                    "segment": np.zeros(1024, dtype=np.int64),
                })
                fresh, _ = model([{
                    "point": xyz, "color": color, "pcd": pcd,
                    "obj_id": f"padp_v4_audit_{task}_{split}",
                }])
                fresh = fresh.float().cpu().numpy()
                saved = np.asarray(cached[row], dtype=np.float32)
                error = np.abs(saved - fresh)
                unique = len(np.unique(np.floor(xyz / 0.02).astype(np.int64), axis=0))
                record = {
                    "task": task, "split": split, "row": row,
                    "unique_voxels": unique,
                    "max_abs_error": float(error.max()),
                    "mean_abs_error": float(error.mean()),
                    "cached_norm_min": float(np.linalg.norm(saved, axis=1).min()),
                    "cached_norm_max": float(np.linalg.norm(saved, axis=1).max()),
                }
                if unique == 1024 and record["max_abs_error"] > 0.0005:
                    raise RuntimeError(f"cache mismatch: {record}")
                if not np.isfinite(saved).all():
                    raise RuntimeError(f"nonfinite cache: {record}")
                rows.append(record)

    report = {
        "schema": "pa3ff_v4_feature_cache_fresh_inference_audit_v1",
        "status": "PASS", "official_source": str(OFFICIAL / "pointcept/models/PA3FF.py"),
        "official_source_sha256": sha256(OFFICIAL / "pointcept/models/PA3FF.py"),
        "representation_sha256": sha256(REPRESENTATION), "rows": rows,
    }
    target = ROOT / "PA3FF_FIELD_CACHE_ACCURACY_AUDIT.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
