#!/usr/bin/env python3
"""Fast structural and checksum verification for the curated baseline bundle."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "repos/flowbot3d/flowbot3d",
    "repos/where2act/code",
    "repos/where2act/code/four_task_train_v7_noaff_schema_robust",
    "repos/pa3ff_official/pointcept/models/PA3FF.py",
    "repos/pa3ff_official/reproduction/run_padp_v5_formal.sh",
    "repos/pa3ff_official/reproduction/motion_planner_collision_worker_pa3ff_fixed_v2.py",
    "repos/articubot/manipulation",
    "repos/articubot/reproduction/unified_open",
    "repos/articubot/reproduction/perception_reach",
    "data/partnet-mobility",
    "data/flowbot3d_custom",
    "data/flowbot3d_pose_and_split_package.zip",
    "data/where2act_four_task/door_open/dataset_pc_aff_static",
    "data/where2act_four_task/door_close/dataset_pc_aff_static",
    "data/where2act_four_task/drawer_open/dataset_pc_aff_static",
    "data/where2act_four_task/drawer_close/dataset_pc_aff_static",
    "repos/where2act/reproduction/v7/train_v7.sh",
    "repos/where2act/reproduction/v7/build_four_task_indices_v7_noaff_schema_robust.py",
    "repos/articubot/data/dataset",
    "repos/articubot/data/low-level-ckpt/checkpoints/low-level.ckpt",
    "repos/articubot/data/high_level_200_obj_ckpt.pth",
    "results/flowbot3d/final/model.ckpt",
    "results/pa3ff/reproduction_v5/training_30000/checkpoints/step020000.pt",
    "results/pa3ff/padp_data_v3_initial_state_aligned_fourtask/BUILD_MANIFEST.json",
    "results/where2act/FAITHFUL_FREEZE_MANIFEST.json",
    "results/where2act/v3_data/corrected_opening_index_v3_critic.npz",
    "repos/where2act/logs/four_task_train_v7_noaff_schema_robust/20260905_032913/TRAINING_MANIFEST.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"缺失: {relative}")

    for directory, names, _ in os.walk(ROOT):
        current = Path(directory)
        if ".git" in names and current != ROOT:
            errors.append(f"嵌套 Git 仓库: {current.relative_to(ROOT)}")
        if "__pycache__" in names:
            errors.append(f"Python 缓存: {current.relative_to(ROOT) / '__pycache__'}")

    checksum_file = ROOT / "manifests/SHA256SUMS"
    if checksum_file.is_file():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"校验文件缺失: {relative}")
            elif sha256(path) != expected:
                errors.append(f"SHA256 不匹配: {relative}")

    broken = []
    for path in ROOT.rglob("*"):
        if path.is_symlink() and not path.exists():
            broken.append(str(path.relative_to(ROOT)))
    errors.extend(f"损坏符号链接: {path}" for path in broken)

    if errors:
        print("VERIFY: FAIL")
        print("\n".join(f"- {item}" for item in errors))
        return 1

    print("VERIFY: PASS")
    print(f"PartNet objects: {sum(p.is_dir() for p in (ROOT / 'data/partnet-mobility').iterdir())}")
    print(f"Bundle root: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
