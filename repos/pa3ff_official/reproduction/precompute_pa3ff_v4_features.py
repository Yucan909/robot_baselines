#!/usr/bin/env python3
"""Cache official frozen PA3FF fields for the V4 static observations.

Each task/split process owns a disjoint output file, so four GPU processes may
run concurrently without changing examples or feature computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HOME = Path("/home/feng")
CODE = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
OFFICIAL = HOME / "robot_baselines/repos/pa3ff_official"
ROOT = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
REPRESENTATION = HOME / "robot_baselines/results/pa3ff/representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth"
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")
SPLITS = ("train", "dev")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract(task: str, split: str) -> None:
    import torch
    sys.path.insert(0, str(OFFICIAL))
    sys.path.insert(0, str(OFFICIAL / "libs/sonata"))
    os.chdir(OFFICIAL)
    from pointcept.datasets.transform import Compose
    from pointcept.models.PA3FF import PA3FF

    point = np.load(ROOT / f"{task}_{split}_pointcloud_base_f32.npy", mmap_mode="r")
    normal = np.load(ROOT / f"{task}_{split}_normal_base_f32.npy", mmap_mode="r")
    if point.shape != normal.shape or point.shape[1:] != (1024, 3):
        raise RuntimeError("geometry cache shape mismatch")
    count = len(point)
    output = ROOT / f"{task}_{split}_pa3ff_field_f16.npy"
    temporary = Path(str(output) + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError(f"stale feature cache: {output} or {temporary}")
    field = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(count, 1024, 768)
    )

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"), dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])
    model = PA3FF(backbone_dim=1088, output_dim=768, freeze_backbone=True,
                  max_grouping_scale=2, use_hierarchy_losses=True, backbone=None).to(device).eval()
    payload = torch.load(REPRESENTATION, map_location="cpu")
    model.instance_net.load_state_dict(payload["instance_net"], strict=True)
    del payload
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    with torch.inference_mode():
        for row in range(count):
            xyz = (np.asarray(point[row], dtype=np.float32) * 10.0).astype(np.float32)
            color = np.zeros((1024, 3), dtype=np.float32)
            pcd = transform({
                "coord": xyz.copy(), "color": color.copy(),
                "normal": np.asarray(normal[row], dtype=np.float32).copy(),
                "segment": np.zeros(1024, dtype=np.int64),
            })
            feature, _ = model([{
                "point": xyz, "color": color, "pcd": pcd,
                "obj_id": f"padp_v4_cache_{task}_{split}",
            }])
            value = feature.float().cpu().numpy()
            if value.shape != (1024, 768) or not np.isfinite(value).all():
                raise RuntimeError(f"bad PA3FF field at row {row}: {value.shape}")
            field[row] = value.astype(np.float16)
            if (row + 1) % 500 == 0 or row + 1 == count:
                elapsed = time.time() - started
                print(json.dumps({
                    "task": task, "split": split, "done": row + 1, "total": count,
                    "rows_per_second": (row + 1) / elapsed,
                    "gpu_mib": torch.cuda.max_memory_allocated() / (1024**2),
                }), flush=True)
    field.flush()
    del field, model
    torch.cuda.empty_cache()
    os.replace(temporary, output)
    check = np.load(output, mmap_mode="r")
    probes = np.linspace(0, count - 1, min(count, 64), dtype=np.int64)
    if check.shape != (count, 1024, 768) or not np.isfinite(check[probes]).all():
        raise RuntimeError("written feature cache validation failed")
    report = {
        "schema": "pa3ff_v4_frozen_field_cache_v1", "status": "complete",
        "task": task, "split": split, "rows": count,
        "shape": list(check.shape), "dtype": "float16", "path": str(output),
        "sha256": sha256(output), "official_pa3ff_source": str(OFFICIAL / "pointcept/models/PA3FF.py"),
        "official_pa3ff_source_sha256": sha256(OFFICIAL / "pointcept/models/PA3FF.py"),
        "representation_checkpoint": str(REPRESENTATION),
        "representation_sha256": sha256(REPRESENTATION),
        "quantization": "float32 official output stored as IEEE float16; TRAIN transfers float16 under AMP, deterministic DEV/runtime may restore float32",
        "method_or_labels_changed": False,
    }
    (ROOT / f"{task}_{split}_pa3ff_field_f16.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def finalize() -> None:
    reports = []
    for task in TASKS:
        for split in SPLITS:
            sidecar = ROOT / f"{task}_{split}_pa3ff_field_f16.json"
            if not sidecar.is_file():
                raise RuntimeError(f"missing {sidecar}")
            report = json.loads(sidecar.read_text(encoding="utf-8"))
            path = Path(report["path"])
            if sha256(path) != report["sha256"]:
                raise RuntimeError(f"hash mismatch {path}")
            reports.append(report)
    manifest = {
        "schema": "pa3ff_v4_frozen_field_cache_manifest_v1",
        "status": "complete", "reports": reports,
        "parallel_processes_change_only_throughput": True,
        "formal_or_test_data_used": False,
    }
    (ROOT / "PA3FF_FIELD_CACHE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    elif args.task and args.split:
        extract(args.task, args.split)
    else:
        parser.error("use --task TASK --split SPLIT, or --finalize")


if __name__ == "__main__":
    main()
