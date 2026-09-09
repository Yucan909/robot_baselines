#!/usr/bin/env python3
"""Build scene-only catalogs without opening policy/action/label fields."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HOME = Path.home()
TARGET_ROOT = HOME / "robot_baselines/configs/where2act_four_task_formal_v1"
DATA_ROOT = HOME / "robot_baselines/data/where2act_four_task"
INDEX_MANIFEST = (
    HOME
    / "robot_baselines/results/where2act"
    / "four_task_train_v7_noaff_schema_robust_indices/INDEX_MANIFEST.json"
)
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")
ROBOT_QPOS = [0.0, -0.45, 0.0, -2.35, 0.0, 1.9, 0.8, 0.04, 0.04]
CAMERA_KEYS = (
    "observation_camera_pose",
    "initial_camera_pose",
    "camera_pose_world",
    "camera_pose",
)
ALLOWED_ARRAY_KEYS = set(CAMERA_KEYS) | {"base_pose", "initial_object_qpos"}

_index_manifest = json.loads(INDEX_MANIFEST.read_text())
DRAWER_FALLBACK_CAMERA = np.asarray(
    _index_manifest["drawer_camera_pose"],
    dtype=np.float64,
)
if DRAWER_FALLBACK_CAMERA.shape != (4, 4):
    raise RuntimeError("INDEX_MANIFEST drawer_camera_pose is not 4x4")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def target_key(row):
    link = row.get("target_link") or row.get("link_name") or row.get("link")
    return str(row["shape_id"]), str(link)


def dataset_roots(task):
    base = DATA_ROOT / task / "dataset_pc_aff_static"
    if task == "door_close":
        return [base / "raw/data/single", base / "raw/data", base / "episodes/single"]
    if task == "drawer_close":
        return [base / "episodes/single", base / "episodes"]
    return [base / "episodes", base / "raw/data/single", base / "raw/data"]


def numeric_suffix(name):
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 10**9


def scene_rank(item):
    repeat, base, _ = item
    return (
        0 if repeat == "repeat_2" else 1,
        numeric_suffix(repeat),
        0 if base == "base_0000" else 1,
        numeric_suffix(base),
    )


def find_scenes(task, shape_id, target_link):
    seen = set()
    scenes = []
    for root in dataset_roots(task):
        link_root = root / str(shape_id) / str(target_link)
        if not link_root.is_dir():
            continue
        for trajectory in sorted(link_root.glob("repeat_*/base_*/trajectory")):
            files = sorted(trajectory.glob("*.npz"))
            if not files:
                continue
            repeat = trajectory.parent.parent.name
            base = trajectory.parent.name
            key = (repeat, base)
            if key in seen:
                continue
            seen.add(key)
            scenes.append((repeat, base, files[0].resolve()))
    return sorted(scenes, key=scene_rank)


def read_scene_arrays(path):
    arrays = {}
    with np.load(path, allow_pickle=False) as data:
        for key in ALLOWED_ARRAY_KEYS:
            if key in data.files:
                value = np.asarray(data[key], dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise RuntimeError(f"nonfinite scene field {key}: {path}")
                arrays[key] = value
    return arrays


def camera_from(arrays):
    for key in CAMERA_KEYS:
        if key in arrays:
            value = arrays[key]
            if value.shape != (4, 4):
                raise RuntimeError(f"camera field {key} is not 4x4")
            return value, key
    return None, None


def scene_map(task, shape_id, target_link):
    return {
        (repeat, base): (path, read_scene_arrays(path))
        for repeat, base, path in find_scenes(task, shape_id, target_link)
    }


def require_vector(arrays, key, path):
    if key not in arrays:
        raise RuntimeError(f"missing {key}: {path}")
    value = arrays[key].reshape(-1)
    if not len(value):
        raise RuntimeError(f"empty {key}: {path}")
    return value


def make_open_door_row(source):
    row = dict(source)
    shape, link = target_key(row)
    row.update(
        task="door_open",
        primitive="pull",
        goal="open",
        shape_id=shape,
        target_link=link,
        scene_source=str(TARGET_ROOT / "door_open_targets.jsonl"),
        camera_source="native_formal_door_open_pose_package",
        articulation_state_source="frozen_door_open_seed_uniform_ratio_0.10_0.20",
        initial_object_qpos=None,
        initial_ratio_source="sample_initial_ratio(trial_seed)",
    )
    return row


def make_native_row(task, source):
    row = dict(source)
    shape, link = target_key(row)
    own = scene_map(task, shape, link)
    if not own:
        raise RuntimeError(f"no native scene for {task} {shape}/{link}")

    paired = {}
    if task == "drawer_open":
        paired = scene_map("drawer_close", shape, link)

    chosen = None
    for scene_key, (path, arrays) in sorted(
        own.items(), key=lambda item: scene_rank((*item[0], item[1][0]))
    ):
        own_camera, own_camera_key = camera_from(arrays)
        if own_camera is not None:
            chosen = (scene_key, path, arrays, own_camera, own_camera_key, "own_explicit")
            break
        if task == "drawer_open" and scene_key in paired:
            pair_path, pair_arrays = paired[scene_key]
            pair_camera, pair_camera_key = camera_from(pair_arrays)
            if pair_camera is not None:
                chosen = (
                    scene_key,
                    path,
                    arrays,
                    pair_camera,
                    pair_camera_key,
                    "paired_drawer_close_scene",
                )
                break

    if chosen is None:
        if task == "drawer_open":
            scene_key, (path, arrays) = sorted(
                own.items(), key=lambda item: scene_rank((*item[0], item[1][0]))
            )[0]
            chosen = (
                scene_key,
                path,
                arrays,
                DRAWER_FALLBACK_CAMERA.copy(),
                "drawer_camera_pose",
                "unique_close_train_camera_id:fixed_front_single_view",
            )
        else:
            raise RuntimeError(f"no legal camera source for {task} {shape}/{link}")

    scene_key, own_path, arrays, camera, camera_key, camera_source = chosen
    repeat, base = scene_key
    qpos = require_vector(arrays, "initial_object_qpos", own_path)

    if "base_pose" in arrays:
        base_pose = arrays["base_pose"].reshape(-1)
        base_source = "own_explicit:base_pose"
    elif task == "drawer_open" and scene_key in paired and "base_pose" in paired[scene_key][1]:
        base_pose = paired[scene_key][1]["base_pose"].reshape(-1)
        base_source = "paired_drawer_close_scene:base_pose"
    elif "base_pose" in row:
        base_pose = np.asarray(row["base_pose"], dtype=np.float64).reshape(-1)
        base_source = "formal_target_catalog:base_pose"
    else:
        raise RuntimeError(f"no legal base pose for {task} {shape}/{link}")

    if base_pose.shape != (4,):
        raise RuntimeError(f"base_pose is not length 4: {own_path}")

    pair_path = paired.get(scene_key, (None, None))[0]
    if camera_source == "paired_drawer_close_scene":
        camera_source_path = pair_path
    elif camera_source.startswith("unique_close_train_camera_id"):
        camera_source_path = INDEX_MANIFEST
    else:
        camera_source_path = own_path

    row.update(
        task=task,
        primitive="push" if task.endswith("_close") else "pull",
        goal="close" if task.endswith("_close") else "open",
        shape_id=shape,
        target_link=link,
        base_pose=base_pose.tolist(),
        robot_initial_qpos=list(ROBOT_QPOS),
        camera_pose_world=camera.tolist(),
        initial_object_qpos=qpos.tolist(),
        scene_identity={"shape_id": shape, "target_link": link, "repeat": repeat, "base": base},
        scene_source=str(own_path),
        camera_source=camera_source,
        camera_field=camera_key,
        camera_source_path=str(camera_source_path),
        base_pose_source=base_source,
        articulation_state_source=f"own_{task}_scene:initial_object_qpos",
        scene_metadata_fields_loaded=sorted(arrays),
        forbidden_policy_fields_loaded=[],
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = {"door_open": 56, "door_close": 56, "drawer_open": 36, "drawer_close": 34}
    manifest = {
        "schema": "where2act_four_task_noaff_v7_formal_scene_catalog_v1",
        "policy_input_leakage": False,
        "loaded_npz_fields_allowlist": sorted(ALLOWED_ARRAY_KEYS),
        "tasks": {},
    }

    for task in TASKS:
        source_path = TARGET_ROOT / f"{task}_targets.jsonl"
        source_rows = load_jsonl(source_path)
        if len(source_rows) != expected[task]:
            raise RuntimeError(f"{task}: expected {expected[task]}, got {len(source_rows)}")
        keys = [target_key(row) for row in source_rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate targets in {source_path}")

        if task == "door_open":
            rows = [make_open_door_row(row) for row in source_rows]
        else:
            rows = [make_native_row(task, row) for row in source_rows]

        out = output_dir / f"{task}_formal_scene_catalog.jsonl"
        out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        camera_counts = {}
        for row in rows:
            key = row["camera_source"]
            camera_counts[key] = camera_counts.get(key, 0) + 1
        manifest["tasks"][task] = {
            "source_target_catalog": str(source_path),
            "source_target_catalog_sha256": sha256(source_path),
            "output_catalog": str(out),
            "output_catalog_sha256": sha256(out),
            "targets": len(rows),
            "camera_sources": camera_counts,
        }

    manifest_path = output_dir / "SCENE_CATALOG_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
