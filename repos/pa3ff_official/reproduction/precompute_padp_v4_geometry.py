#!/usr/bin/env python3
"""Precompute V4 point clouds and normals without changing PA3FF inputs."""
from __future__ import annotations

import concurrent.futures as futures
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path("/home/feng/robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask")
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")
SPLITS = ("train", "dev")
WORKERS = min(20, max(4, (os.cpu_count() or 8) // 2))


def base_to_world(base_pose: np.ndarray) -> np.ndarray:
    x, y, yaw, z = np.asarray(base_pose, dtype=np.float64).reshape(4)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = [x, y, z]
    return transform


def read_cloud(spec) -> tuple[np.ndarray, np.ndarray]:
    path, key, selected, base_pose, camera_world = spec
    with np.load(path, allow_pickle=False) as source:
        raw = np.asarray(source[key], dtype=np.float32)
    if raw.shape == (1024, 3):
        world = raw
    elif raw.shape == (1, 1024, 3):
        world = raw[0]
    elif raw.ndim == 3 and raw.shape[1:] == (1024, 3):
        world = raw[int(selected)]
    else:
        raise RuntimeError(f"{path}: unsupported point cloud shape {raw.shape}")
    if world.shape != (1024, 3) or not np.isfinite(world).all():
        raise RuntimeError(f"{path}: invalid point cloud")

    world_from_base = base_to_world(base_pose)
    base_from_world = np.linalg.inv(world_from_base)
    base = (
        world.astype(np.float64) @ base_from_world[:3, :3].T
        + base_from_world[:3, 3]
    ).astype(np.float32)
    camera_base = base_from_world @ np.asarray(camera_world, dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(base.astype(np.float64))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )
    cloud.orient_normals_towards_camera_location(camera_base[:3, 3])
    normal = np.asarray(cloud.normals, dtype=np.float32)
    if normal.shape != (1024, 3) or not np.isfinite(normal).all():
        raise RuntimeError(f"{path}: invalid normals {normal.shape}")
    return base, normal


def build(task: str, split: str) -> dict:
    index_path = ROOT / f"{task}_{split}_v4.npz"
    with np.load(index_path, allow_pickle=False) as index:
        paths = np.asarray(index["source_static_path"]).astype(str)
        keys = np.asarray(index["source_pointcloud_key"]).astype(str)
        selected = np.asarray(index["source_observation_selected_index"], dtype=np.int64)
        bases = np.asarray(index["source_base_pose"], dtype=np.float32)
        cameras = np.asarray(index["source_camera_pose_world"], dtype=np.float32)
    count = len(paths)
    specs = list(zip(paths.tolist(), keys.tolist(), selected.tolist(), bases, cameras))
    point_path = ROOT / f"{task}_{split}_pointcloud_base_f32.npy"
    normal_path = ROOT / f"{task}_{split}_normal_base_f32.npy"
    if point_path.exists() or normal_path.exists():
        raise RuntimeError(f"stale geometry cache for {task}/{split}")
    point_tmp = Path(str(point_path) + ".tmp")
    normal_tmp = Path(str(normal_path) + ".tmp")
    point = np.lib.format.open_memmap(point_tmp, mode="w+", dtype=np.float32,
                                      shape=(count, 1024, 3))
    normal = np.lib.format.open_memmap(normal_tmp, mode="w+", dtype=np.float32,
                                       shape=(count, 1024, 3))
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row, (pc, n) in enumerate(pool.map(read_cloud, specs, chunksize=16)):
            point[row] = pc
            normal[row] = n
            if (row + 1) % 2000 == 0 or row + 1 == count:
                print(f"{task}/{split}: {row + 1}/{count}", flush=True)
    point.flush()
    normal.flush()
    del point, normal
    os.replace(point_tmp, point_path)
    os.replace(normal_tmp, normal_path)
    # Verify mmap headers and a distributed finite sample.
    pc_check = np.load(point_path, mmap_mode="r")
    n_check = np.load(normal_path, mmap_mode="r")
    probes = np.linspace(0, count - 1, min(count, 256), dtype=np.int64)
    if pc_check.shape != (count, 1024, 3) or n_check.shape != (count, 1024, 3):
        raise RuntimeError(f"cache shape failure {task}/{split}")
    if not np.isfinite(pc_check[probes]).all() or not np.isfinite(n_check[probes]).all():
        raise RuntimeError(f"cache finite failure {task}/{split}")
    return {
        "task": task, "split": split, "source_rows": count,
        "pointcloud": str(point_path), "normal": str(normal_path),
        "dtype": "float32", "coordinate_frame": "panda_base",
    }


def main() -> None:
    reports = []
    for task in TASKS:
        for split in SPLITS:
            reports.append(build(task, split))
    manifest = {
        "schema": "pa3ff_padp_v4_geometry_cache_v1",
        "status": "complete", "workers": WORKERS,
        "inputs_unchanged": True,
        "normal_estimation": "Open3D radius=0.08 max_nn=30 oriented_to_camera",
        "reports": reports,
    }
    (ROOT / "GEOMETRY_CACHE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
