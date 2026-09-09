import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer
from flowbot3d.datasets.flow_dataset import compute_normalized_flow


HOME = Path.home()

PM_ROOT = HOME / "robot_baselines/data/partnet-mobility"
SPLIT_ROOT = HOME / "robot_baselines/splits"

OUT_ROOT = HOME / "robot_baselines/data/flowbot3d_custom/full"

N_POINTS = 1200
MAX_RETRIES = 50


def read_targets(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)

    targets = []

    for shape_id in split["ids"]:
        obj_info = split["objects"][shape_id]

        for link_info in obj_info["selected_links"]:
            targets.append(
                {
                    "shape_id": shape_id,
                    "category": obj_info["category"],
                    "link_name": link_info["link_name"],
                    "size": link_info.get("size", None),
                }
            )

    return targets


def valid_existing_file(path):
    if not path.exists():
        return False

    try:
        d = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        return (
            d.pos.shape == (N_POINTS, 3)
            and d.flow.shape == (N_POINTS, 3)
            and d.mask.shape == (N_POINTS,)
            and bool(torch.isfinite(d.pos).all())
            and bool(torch.isfinite(d.flow).all())
            and float(d.mask.sum()) > 0
        )

    except Exception:
        return False


def generate_one_target(
    split_name,
    target_index,
    target,
    repeats,
    base_seed,
):
    shape_id = target["shape_id"]
    category = target["category"]
    link_name = target["link_name"]
    size = target["size"]

    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    obj_dir = PM_ROOT / shape_id

    if not obj_dir.exists():
        raise FileNotFoundError(
            f"{shape_id} 不存在: {obj_dir}"
        )

    pm_obj = PMObject(obj_dir)

    target_joint = pm_obj.obj.get_joint_by_child(
        link_name
    )

    renderer = PybulletRenderer()

    generated = 0
    skipped = 0

    try:
        for repeat_idx in range(repeats):

            safe_link = link_name.replace("/", "_")

            filename = (
                f"t{target_index:04d}_"
                f"r{repeat_idx:03d}_"
                f"{shape_id}_"
                f"{safe_link}.pt"
            )

            save_path = split_dir / filename

            if valid_existing_file(save_path):
                skipped += 1
                continue

            success = False
            last_error = None

            for attempt in range(MAX_RETRIES):

                # 不同 target / repeat / retry 都得到确定的随机数
                seed = (
                    base_seed
                    + target_index * 100000
                    + repeat_idx * 100
                    + attempt
                )

                try:
                    data = renderer.render(
                        pm_obj=pm_obj,
                        joints="random",
                        camera_xyz="random",
                        seed=seed,
                    )

                    pos = data["pos"]

                    if len(pos) < N_POINTS:
                        continue

                    if link_name not in data["labelmap"]:
                        raise RuntimeError(
                            f"{link_name} 不在 labelmap 中"
                        )

                    # 只计算指定 link 的 3DAF
                    flow = compute_normalized_flow(
                        P_world=pos,
                        T_world_base=data["T_world_base"],
                        current_jas=data["angles"],
                        pc_seg=data["seg"],
                        labelmap=data["labelmap"],
                        pm_raw_data=pm_obj,
                        linknames=[link_name],
                    )

                    mask = ~np.isclose(
                        flow,
                        0.0,
                        atol=1e-7,
                    ).all(axis=-1)

                    target_before = int(mask.sum())

                    if target_before == 0:
                        continue

                    # 固定 1200 点
                    rng = np.random.default_rng(
                        seed + 17
                    )

                    indices = rng.permutation(
                        len(pos)
                    )[:N_POINTS]

                    pos_sample = pos[indices]
                    flow_sample = flow[indices]
                    mask_sample = mask[indices]

                    target_after = int(
                        mask_sample.sum()
                    )

                    if target_after == 0:
                        continue

                    pyg_data = Data(
                        pos=torch.from_numpy(
                            pos_sample
                        ).float(),
                        flow=torch.from_numpy(
                            flow_sample
                        ).float(),
                        mask=torch.from_numpy(
                            mask_sample.astype(
                                np.float32
                            )
                        ),
                    )

                    # metadata
                    pyg_data.shape_id = shape_id
                    pyg_data.category = category
                    pyg_data.link_name = link_name
                    pyg_data.size = size

                    pyg_data.joint_name = (
                        target_joint.name
                    )

                    pyg_data.joint_type = (
                        target_joint.type
                    )

                    pyg_data.seed = seed
                    pyg_data.repeat_index = repeat_idx
                    pyg_data.target_index = target_index

                    pyg_data.raw_point_count = len(pos)
                    pyg_data.target_points_before = (
                        target_before
                    )
                    pyg_data.target_points_after = (
                        target_after
                    )

                    # 原子写文件，避免程序中断留下半个 pt
                    tmp_path = Path(
                        str(save_path) + ".tmp"
                    )

                    torch.save(
                        pyg_data,
                        tmp_path,
                    )

                    os.replace(
                        tmp_path,
                        save_path,
                    )

                    generated += 1
                    success = True
                    break

                except Exception as e:
                    last_error = repr(e)

            if not success:
                raise RuntimeError(
                    f"{shape_id}/{link_name}/"
                    f"repeat={repeat_idx} "
                    f"连续 {MAX_RETRIES} 次失败。"
                    f"最后错误: {last_error}"
                )

    finally:
        if (
            getattr(
                renderer,
                "_render_env",
                None,
            )
            is not None
        ):
            renderer._render_env.close()

    return {
        "target_index": target_index,
        "shape_id": shape_id,
        "category": category,
        "link_name": link_name,
        "repeats": repeats,
        "generated": generated,
        "skipped": skipped,
    }


def rebuild_manifest(split_name):
    split_dir = OUT_ROOT / split_name

    files = sorted(
        split_dir.glob("*.pt")
    )

    manifest = []

    for path in files:
        d = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        flow_mag = d.flow.norm(dim=1)

        target_mask = d.mask.bool()

        manifest.append(
            {
                "file": path.name,
                "shape_id": d.shape_id,
                "category": d.category,
                "link_name": d.link_name,
                "size": d["size"] if "size" in d else None,
                "joint_name": d.joint_name,
                "joint_type": d.joint_type,
                "seed": int(d.seed),
                "repeat_index": int(
                    d.repeat_index
                ),
                "target_index": int(
                    d.target_index
                ),
                "num_points": int(
                    d.pos.shape[0]
                ),
                "target_points": int(
                    d.mask.sum().item()
                ),
                "target_flow_mag_mean": float(
                    flow_mag[target_mask]
                    .mean()
                    .item()
                ),
                "target_flow_mag_max": float(
                    flow_mag[target_mask]
                    .max()
                    .item()
                ),
            }
        )

    manifest_path = (
        OUT_ROOT
        / f"{split_name}_manifest.json"
    )

    with open(
        manifest_path,
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    return manifest


def generate_split(
    split_name,
    split_file,
    repeats,
    workers,
    base_seed,
):
    targets = read_targets(split_file)

    expected = len(targets) * repeats

    print()
    print("=" * 85)
    print(
        f"{split_name.upper()} 数据生成"
    )
    print("=" * 85)

    print("目标 (shape, link) 数:", len(targets))
    print("每个目标重复:", repeats)
    print("预期样本总数:", expected)
    print("并行进程:", workers)

    errors = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = []

        for target_index, target in enumerate(
            targets
        ):
            futures.append(
                executor.submit(
                    generate_one_target,
                    split_name,
                    target_index,
                    target,
                    repeats,
                    base_seed,
                )
            )

        done = 0

        for future in as_completed(futures):

            try:
                result = future.result()

                done += 1

                print(
                    f"[{done:4d}/{len(targets):4d}] "
                    f"{result['shape_id']}/"
                    f"{result['link_name']} "
                    f"generated="
                    f"{result['generated']} "
                    f"skipped="
                    f"{result['skipped']}"
                )

            except Exception:
                errors.append(
                    traceback.format_exc()
                )

                print(
                    "\n!!! 一个 target 生成失败 !!!"
                )

                print(errors[-1])

    if errors:
        error_file = (
            OUT_ROOT
            / f"{split_name}_errors.txt"
        )

        with open(
            error_file,
            "w",
        ) as f:
            f.write(
                "\n\n".join(errors)
            )

        print()
        print("存在生成失败 target。")
        print("错误记录:", error_file)

        raise SystemExit(1)

    manifest = rebuild_manifest(
        split_name
    )

    print()
    print(
        f"{split_name} 实际样本总数:",
        len(manifest),
    )

    if len(manifest) != expected:
        raise RuntimeError(
            f"{split_name}: "
            f"预期 {expected}，"
            f"实际 {len(manifest)}"
        )

    print(
        f"{split_name} 完整性检查通过"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-repeat",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--val-repeat",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_split(
        "train",
        SPLIT_ROOT / "train_split.json",
        args.train_repeat,
        args.workers,
        base_seed=1_000_000,
    )

    generate_split(
        "val",
        SPLIT_ROOT / "val_split.json",
        args.val_repeat,
        args.workers,
        base_seed=90_000_000,
    )

    print()
    print("=" * 85)
    print(
        "FlowBot3D 完整自定义数据集生成成功"
    )
    print("=" * 85)


if __name__ == "__main__":
    main()
