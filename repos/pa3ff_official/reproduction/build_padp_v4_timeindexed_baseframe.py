#!/usr/bin/env python3
"""Build the corrected PA3FF/PADP reproduction dataset.

This is deliberately a new dataset rather than an in-place repair of V3.

Changes relative to V3:
  * every expert chunk contributes an initial sample and four operation-time
    suffix samples (anchors 0, 4, 8, 12);
  * q_t is the real robot state at the corresponding anchor;
  * action poses are absolute panda_grasptarget poses in the Panda base frame,
    so decoding no longer depends on a changing point-cloud centroid;
  * suffixes keep the original keyframe spacing and pad only with their final
    pose, as in standard fixed-horizon diffusion-policy datasets.

The source V3 files already enforce the global object-level split.  This
builder re-audits it against the frozen formal catalog.  It never reads an
affordance array, a formal trajectory, or any formal success result.
"""
from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


HOME = Path("/home/feng")
REPO = HOME / "robot_baselines/repos/pa3ff_official/reproduction"
SRC = HOME / "robot_baselines/results/pa3ff/padp_data_v3_initial_state_aligned_fourtask"
OUT = HOME / "robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask"
FORMAL_CATALOG = HOME / "robot_baselines/configs/pa3ff/reproduction_v1_formal/formal_episode_catalog.jsonl"
DOOR_MAP = HOME / "robot_baselines/results/where2act/v3_data/corrected_opening_index_v3_critic.npz"
DATASET_ROOT = HOME / "robot_baselines/data/where2act_four_task"
LEGACY_DATASET_ROOT = HOME / "下载/a"
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")
SPLITS = ("train", "dev")
ANCHORS = np.asarray([-1, 0, 4, 8, 12], dtype=np.int16)
HORIZON = 16
ACTION_DIM = 10
SCHEMA = "pa3ff_padp_demo_v4_timeindexed_baseframe_fourtask"
WORKERS = min(24, max(4, (os.cpu_count() or 8) // 2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def guard(name: str, condition, detail=None) -> None:
    state = "PASS" if bool(condition) else "FAIL"
    print(f"{name:<92s} {state}", flush=True)
    if detail is not None:
        print("   ", detail, flush=True)
    if not condition:
        raise RuntimeError(f"guard failed: {name}: {detail}")


def base_to_world(base_pose: np.ndarray) -> np.ndarray:
    x, y, yaw, z = np.asarray(base_pose, dtype=np.float64).reshape(4)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = [x, y, z]
    return transform


def rotmat_to_6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


def relocate_dataset_path(value: str) -> Path:
    """Map paths frozen before packaging onto the curated dataset root."""
    path = Path(str(value))
    try:
        relative = path.relative_to(LEGACY_DATASET_ROOT)
    except ValueError:
        return path
    return DATASET_ROOT / relative


def load_door_map() -> dict[str, str]:
    with np.load(DOOR_MAP, allow_pickle=False) as index:
        static = np.asarray(index["static_path"]).astype(str)
        raw = np.asarray(index["raw_path"]).astype(str)
    mapping: dict[str, str] = {}
    for source_raw, target_raw in zip(static.tolist(), raw.tolist()):
        source = str(relocate_dataset_path(source_raw))
        target = str(relocate_dataset_path(target_raw))
        previous = mapping.setdefault(source, target)
        if previous != target:
            raise RuntimeError(f"ambiguous door-open scene pairing: {source}")
    return mapping


def paired_door_close_path(static_path: str) -> Path:
    """Resolve the same shape/link/repeat/base/candidate close trajectory.

    The corrected-opening index intentionally contains only its critic subset,
    while the validated Stage6 PADP set is larger.  For those additional rows
    the frozen directory convention gives an unambiguous same-candidate pair.
    """
    source = relocate_dataset_path(static_path)
    parts = source.parts
    try:
        episodes = parts.index("episodes")
    except ValueError as exc:
        raise RuntimeError(f"not a door-open static path: {source}") from exc
    relative = Path(*parts[episodes + 1 :])
    name = relative.name
    suffix = ".pointcloud.npz"
    if not name.endswith(suffix):
        raise RuntimeError(f"unexpected door-open filename: {name}")
    raw_name = "reverse_" + name[: -len(suffix)] + ".npz"
    return (
        DATASET_ROOT / "door_close/dataset_pc_aff_static/raw/data/single"
        / relative.parent / raw_name
    )


def resolve_base_pose(item: tuple[str, str], door_map: dict[str, str]) -> np.ndarray:
    task, static_path = item
    if task == "door_open":
        source = Path(door_map.get(static_path, paired_door_close_path(static_path)))
    else:
        source = Path(static_path)
    with np.load(source, allow_pickle=False) as trajectory:
        if "base_pose" not in trajectory.files:
            raise RuntimeError(f"{source}: missing base_pose")
        base = np.asarray(trajectory["base_pose"], dtype=np.float64).reshape(-1)
    if base.shape != (4,) or not np.isfinite(base).all():
        raise RuntimeError(f"{source}: invalid base_pose {base}")
    return base.astype(np.float32)


def transform_actions(
    xyz_hand_world: np.ndarray,
    rotation_hand_world: np.ndarray,
    base_pose: np.ndarray,
    hand_to_grasp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(base_pose)
    xyz_base = np.empty_like(xyz_hand_world, dtype=np.float32)
    rotation_base = np.empty_like(rotation_hand_world, dtype=np.float32)
    offset_rotation = np.asarray(hand_to_grasp[:3, :3], dtype=np.float64)
    offset_translation = np.asarray(hand_to_grasp[:3, 3], dtype=np.float64)
    block = 4096
    for start in range(0, count, block):
        stop = min(start + block, count)
        bases = np.stack([base_to_world(x) for x in base_pose[start:stop]])
        rotation_world_base = bases[:, :3, :3]
        rotation_base_world = np.swapaxes(rotation_world_base, 1, 2)
        translation_world_base = bases[:, :3, 3]
        raw_r = rotation_hand_world[start:stop].astype(np.float64)
        raw_p = xyz_hand_world[start:stop].astype(np.float64)
        grasp_p_world = raw_p + np.einsum("nhij,j->nhi", raw_r, offset_translation)
        grasp_r_world = np.matmul(raw_r, offset_rotation)
        xyz_base[start:stop] = np.einsum(
            "nij,nhj->nhi", rotation_base_world,
            grasp_p_world - translation_world_base[:, None, :],
        ).astype(np.float32)
        rotation_base[start:stop] = np.einsum(
            "nij,nhjk->nhik", rotation_base_world, grasp_r_world,
        ).astype(np.float32)
    return xyz_base, rotation_base


def build_one(task: str, split: str, door_map: dict[str, str], formal_shapes: set[str]) -> dict:
    source_path = SRC / f"{task}_{split}_v3.npz"
    guard(f"source exists {task}/{split}", source_path.is_file(), str(source_path))
    with np.load(source_path, allow_pickle=False) as source:
        required = (
            "shape", "link", "static_path", "pointcloud_key", "camera_pose",
            "initial_robot_proprioception", "action_qpos", "action_xyz",
            "action_rotmat", "action_finger", "observation_frame_count",
            "observation_selected_index", "observation_selected_trajectory_step",
            "operation_start_frame_index", "hand_to_grasptarget_T",
            "task_instruction", "part_cls", "primitive",
        )
        missing = [key for key in required if key not in source.files]
        guard(f"required fields {task}/{split}", not missing, missing)
        arrays = {key: np.array(source[key], copy=True) for key in required}

    shape = arrays["shape"].astype(str)
    link = arrays["link"].astype(str)
    static_path = np.asarray(
        [str(relocate_dataset_path(path)) for path in arrays["static_path"].astype(str)],
        dtype=str,
    )
    count = len(shape)
    guard(f"nonempty {task}/{split}", count > 0, count)
    if split == "train":
        leakage = sorted(set(shape.tolist()) & formal_shapes)
        guard(f"formal object leakage zero {task}/{split}", not leakage, leakage)

    if task == "door_open":
        missing_pair = [
            path for path in static_path.tolist()
            if not Path(door_map.get(path, paired_door_close_path(path))).is_file()
        ]
        guard(f"door-open paired base mapping complete {split}", not missing_pair,
              missing_pair[:5])
    items = [(task, path) for path in static_path.tolist()]
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        base_pose = np.stack(list(pool.map(lambda x: resolve_base_pose(x, door_map), items)))
    guard(f"base pose shape/finite {task}/{split}",
          base_pose.shape == (count, 4) and np.isfinite(base_pose).all(), base_pose.shape)

    action_qpos = np.asarray(arrays["action_qpos"], dtype=np.float32)
    action_xyz_world = np.asarray(arrays["action_xyz"], dtype=np.float32)
    action_rotation_world = np.asarray(arrays["action_rotmat"], dtype=np.float32)
    action_finger = np.asarray(arrays["action_finger"], dtype=np.float32)
    hand_to_grasp = np.asarray(arrays["hand_to_grasptarget_T"], dtype=np.float32)
    guard(f"source action shapes {task}/{split}",
          action_qpos.shape == (count, HORIZON, 9)
          and action_xyz_world.shape == (count, HORIZON, 3)
          and action_rotation_world.shape == (count, HORIZON, 3, 3)
          and action_finger.shape == (count, HORIZON))
    action_xyz_base, action_rotation_base = transform_actions(
        action_xyz_world, action_rotation_world, base_pose, hand_to_grasp,
    )

    sample_source = np.repeat(np.arange(count, dtype=np.int32), len(ANCHORS))
    sample_anchor = np.tile(ANCHORS, count)
    sample_proprio = np.empty((len(sample_source), 9), dtype=np.float32)
    initial_proprio = np.asarray(arrays["initial_robot_proprioception"], dtype=np.float32)
    initial_mask = sample_anchor < 0
    sample_proprio[initial_mask] = initial_proprio[sample_source[initial_mask]]
    operation_mask = ~initial_mask
    sample_proprio[operation_mask] = action_qpos[
        sample_source[operation_mask], sample_anchor[operation_mask]
    ]

    output_path = OUT / f"{task}_{split}_v4.npz"
    np.savez_compressed(
        output_path,
        schema=np.asarray(SCHEMA), horizon=np.asarray(HORIZON, dtype=np.int32),
        task_name=np.asarray(task), task_instruction=arrays["task_instruction"],
        part_cls=arrays["part_cls"], primitive=arrays["primitive"],
        source_v3_path=np.asarray(str(source_path)), source_v3_sha256=np.asarray(sha256(source_path)),
        source_shape=shape, source_link=link, source_static_path=static_path,
        source_pointcloud_key=arrays["pointcloud_key"].astype(str),
        source_camera_pose_world=np.asarray(arrays["camera_pose"], dtype=np.float32),
        source_base_pose=base_pose,
        source_observation_frame_count=np.asarray(arrays["observation_frame_count"], dtype=np.int32),
        source_observation_selected_index=np.asarray(arrays["observation_selected_index"], dtype=np.int32),
        source_observation_selected_trajectory_step=np.asarray(
            arrays["observation_selected_trajectory_step"], dtype=np.int32),
        source_operation_start_frame_index=np.asarray(arrays["operation_start_frame_index"], dtype=np.int32),
        source_action_xyz_base_grasptarget=action_xyz_base,
        source_action_rotmat_base_grasptarget=action_rotation_base,
        source_action_finger=action_finger,
        sample_source_index=sample_source, sample_anchor_keyframe=sample_anchor,
        sample_initial_robot_proprioception=sample_proprio,
        suffix_rule=np.asarray("future keyframes anchor+i clipped to 15; final-pose padding only"),
        action_pose_coordinate_frame=np.asarray("panda_base"),
        source_ee_frame=np.asarray("panda_hand"), target_ee_frame=np.asarray("panda_grasptarget"),
        hand_to_grasptarget_T=hand_to_grasp,
        pointcloud_coordinate_frame=np.asarray("world_on_disk_transformed_to_panda_base_at_load"),
        affordance_fields_read=np.asarray(False),
    )
    guard(f"output readable {task}/{split}", output_path.is_file())
    return {
        "task": task, "split": split, "source_rows": count,
        "samples": int(len(sample_source)), "anchors": ANCHORS.astype(int).tolist(),
        "objects": len(set(shape.tolist())), "path": str(output_path),
        "sha256": sha256(output_path),
    }


def iter_encoded_actions(path: Path, block: int = 4096):
    with np.load(path, allow_pickle=False) as index:
        xyz = np.asarray(index["source_action_xyz_base_grasptarget"], dtype=np.float32)
        rotation = np.asarray(index["source_action_rotmat_base_grasptarget"], dtype=np.float32)
        finger = np.asarray(index["source_action_finger"], dtype=np.float32)
        source = np.asarray(index["sample_source_index"], dtype=np.int64)
        anchor = np.asarray(index["sample_anchor_keyframe"], dtype=np.int64)
    offsets = np.arange(HORIZON, dtype=np.int64)[None, :]
    for start in range(0, len(source), block):
        stop = min(start + block, len(source))
        selected_source = source[start:stop]
        selected_anchor = anchor[start:stop]
        first = np.maximum(selected_anchor, 0)[:, None]
        future = np.minimum(first + offsets, HORIZON - 1)
        p = xyz[selected_source[:, None], future]
        r = rotation[selected_source[:, None], future]
        g = finger[selected_source[:, None], future, None]
        yield np.concatenate([p, rotmat_to_6d(r), g], axis=-1).astype(np.float32)


def build_normalization(train_paths: dict[str, Path]) -> dict:
    total = 0
    total_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    total_square = np.zeros(ACTION_DIM, dtype=np.float64)
    rows = {}
    for task, path in train_paths.items():
        task_rows = 0
        for action in iter_encoded_actions(path):
            flat = action.reshape(-1, ACTION_DIM).astype(np.float64)
            total += len(flat)
            task_rows += action.shape[0]
            total_sum += flat.sum(axis=0)
            total_square += np.square(flat).sum(axis=0)
        rows[task] = task_rows
    mean = total_sum / total
    variance = np.maximum(total_square / total - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    std = raw_std.copy()
    std[std < 1e-6] = 1.0
    path = OUT / "ACTION_NORMALIZATION_V4.npz"
    np.savez_compressed(
        path, mean=mean.astype(np.float32), std=std.astype(np.float32),
        raw_std=raw_std.astype(np.float32),
        action_layout=np.asarray("absolute_xyz_in_panda_base,rot6d_col0_col1,gripper"),
        translation_reference=np.asarray("panda_base"), coordinate_frame=np.asarray("panda_base"),
        source_ee_frame=np.asarray("panda_hand"), target_ee_frame=np.asarray("panda_grasptarget"),
        train_only=np.asarray(True), tasks=np.asarray(TASKS),
        samples_per_task=np.asarray([rows[task] for task in TASKS], dtype=np.int64),
    )
    return {
        "path": str(path), "sha256": sha256(path), "total_action_vectors": int(total),
        "samples_per_task": rows, "mean": mean.tolist(), "std": std.tolist(),
    }


def main() -> None:
    guard("conda environment pa3ff", os.environ.get("CONDA_DEFAULT_ENV") == "pa3ff",
          os.environ.get("CONDA_DEFAULT_ENV"))
    guard("V3 source root", SRC.is_dir(), str(SRC))
    guard("frozen formal catalog", FORMAL_CATALOG.is_file(), str(FORMAL_CATALOG))
    guard("door scene pairing map", DOOR_MAP.is_file(), str(DOOR_MAP))
    OUT.mkdir(parents=True, exist_ok=True)
    existing = list(OUT.glob("*.npz"))
    guard("new V4 output root has no stale NPZ", not existing, [str(x) for x in existing])

    formal_rows = [json.loads(line) for line in FORMAL_CATALOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    formal_shapes = {str(row["shape_id"]) for row in formal_rows}
    door_map = load_door_map()
    reports = []
    train_paths: dict[str, Path] = {}
    for task in TASKS:
        for split in SPLITS:
            report = build_one(task, split, door_map, formal_shapes)
            reports.append(report)
            if split == "train":
                train_paths[task] = Path(report["path"])
    normalization = build_normalization(train_paths)

    # Global object-level TRAIN/DEV disjointness is rechecked from V4 source metadata.
    train_objects, dev_objects = set(), set()
    for report in reports:
        with np.load(report["path"], allow_pickle=False) as index:
            objects = set(np.asarray(index["source_shape"]).astype(str).tolist())
        (train_objects if report["split"] == "train" else dev_objects).update(objects)
    guard("global TRAIN/DEV objects disjoint", not (train_objects & dev_objects),
          sorted(train_objects & dev_objects))
    guard("global TRAIN/formal objects disjoint", not (train_objects & formal_shapes),
          sorted(train_objects & formal_shapes))

    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "method": "PA3FF reproduction with paper-equivalent reconstructed PADP",
        "official_status": "PADP training/data implementation is reconstructed; PA3FF feature extractor is official",
        "implementation_choices": [
            "time-indexed suffix samples at keyframe anchors 0,4,8,12 plus the initial sample",
            "fixed H=16 suffixes use final-pose padding",
            "absolute SE(3) actions and point clouds are represented in the Panda base frame",
            "when no per-time visual observation exists, the trajectory's non-future static observation is reused",
        ],
        "unchanged_method_components": [
            "official frozen PA3FF part-aware feature field", "H=16 10D SE(3)+gripper actions",
            "DDPM x0 reconstruction objective", "task and part SigLIP conditioning",
        ],
        "forbidden_data": {
            "formal_trajectory_reads": 0, "formal_success_reads": 0,
            "affordance_array_reads": 0, "test_action_label_reads": 0,
        },
        "source_root": str(SRC), "source_rows": reports,
        "normalization": normalization,
        "split_audit": {
            "train_objects": len(train_objects), "dev_objects": len(dev_objects),
            "formal_objects": len(formal_shapes), "train_dev_overlap": [], "train_formal_overlap": [],
        },
    }
    write_json(OUT / "BUILD_MANIFEST.json", manifest)
    write_json(OUT / "ACTION_NORMALIZATION_V4.json", normalization)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
