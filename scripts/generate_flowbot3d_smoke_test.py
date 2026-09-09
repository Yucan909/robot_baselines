import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer

from flowbot3d.datasets.flow_dataset import compute_normalized_flow


# ============================================================
# 路径与参数
# ============================================================

HOME = Path.home()

DATA_ROOT = HOME / "robot_baselines/data/partnet-mobility"
SPLIT_FILE = HOME / "robot_baselines/splits/train_split.json"

OUT_ROOT = (
    HOME
    / "robot_baselines/data/flowbot3d_custom/smoke_test"
)

N_TARGETS = 3
N_REPEAT = 2
N_POINTS = 1200
BASE_SEED = 1000


# ============================================================
# 读取学长给的训练划分
# ============================================================

with open(SPLIT_FILE, "r") as f:
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
            }
        )

# smoke test 只取前 3 个 shape-link target
targets = targets[:N_TARGETS]

print("=" * 80)
print("FlowBot3D 自定义训练集 Smoke Test")
print("=" * 80)

print("\n本次选择的目标：")

for i, t in enumerate(targets):
    print(
        f"[{i}] "
        f"shape={t['shape_id']} "
        f"category={t['category']} "
        f"link={t['link_name']}"
    )

print()


# ============================================================
# 开始生成
# ============================================================

OUT_ROOT.mkdir(parents=True, exist_ok=True)

manifest = []

sample_index = 0

for target_idx, target in enumerate(targets):

    shape_id = target["shape_id"]
    category = target["category"]
    link_name = target["link_name"]

    obj_dir = DATA_ROOT / shape_id

    if not obj_dir.exists():
        raise FileNotFoundError(
            f"PartNet-Mobility shape 不存在: {obj_dir}"
        )

    # 每个物体单独建立 PMObject
    pm_obj = PMObject(obj_dir)

    # 检查指定 link
    target_joint = pm_obj.obj.get_joint_by_child(link_name)

    print("=" * 80)
    print(
        f"Target {target_idx + 1}/{len(targets)}: "
        f"{shape_id} / {link_name}"
    )
    print(
        f"category={category}, "
        f"joint={target_joint.name}, "
        f"type={target_joint.type}"
    )
    print("=" * 80)

    for repeat_idx in range(N_REPEAT):

        seed = BASE_SEED + target_idx * 100 + repeat_idx

        # IMPORTANT：
        # PybulletRenderer 内部会缓存当前物体，
        # 所以不同物体必须新建 renderer。
        renderer = PybulletRenderer()

        data = renderer.render(
            pm_obj=pm_obj,
            joints="random",
            camera_xyz="random",
            seed=seed,
        )

        pos = data["pos"]

        if link_name not in data["labelmap"]:
            raise RuntimeError(
                f"{shape_id}: {link_name} "
                f"没有出现在 PyBullet labelmap 中"
            )

        # ----------------------------------------------------
        # 只给学长指定的这个 link 计算 GT 3DAF
        # ----------------------------------------------------

        flow = compute_normalized_flow(
            P_world=pos,
            T_world_base=data["T_world_base"],
            current_jas=data["angles"],
            pc_seg=data["seg"],
            labelmap=data["labelmap"],
            pm_raw_data=pm_obj,
            linknames=[link_name],
        )

        # 只有目标 link（及其运动学子节点）有非零 flow
        mask = ~np.isclose(
            flow,
            0.0,
            atol=1e-7,
        ).all(axis=-1)

        visible_target_before = int(mask.sum())

        if visible_target_before == 0:
            raise RuntimeError(
                f"{shape_id}/{link_name}/seed={seed}: "
                f"当前视角没有可见目标点"
            )

        # ----------------------------------------------------
        # 按 FlowBot3D 的方式随机下采样
        # ----------------------------------------------------

        rng = np.random.default_rng(seed)

        indices = rng.permutation(len(pos))[:N_POINTS]

        pos = pos[indices]
        flow = flow[indices]
        mask = mask[indices]

        visible_target_after = int(mask.sum())

        if len(pos) != N_POINTS:
            raise RuntimeError(
                f"{shape_id}/{link_name}: "
                f"原始点数只有 {len(pos)}，不足 {N_POINTS}"
            )

        if visible_target_after == 0:
            raise RuntimeError(
                f"{shape_id}/{link_name}/seed={seed}: "
                f"下采样后没有目标活动点"
            )

        # ----------------------------------------------------
        # 转成 FlowBot3D 真正训练时使用的 PyG Data
        # ----------------------------------------------------

        pyg_data = Data(
            pos=torch.from_numpy(pos).float(),
            flow=torch.from_numpy(flow).float(),
            mask=torch.from_numpy(
                mask.astype(np.float32)
            ),
        )

        # 额外 metadata 不参与网络输入，只用于我们追踪样本来源
        pyg_data.shape_id = shape_id
        pyg_data.link_name = link_name
        pyg_data.category = category
        pyg_data.seed = seed

        filename = (
            f"{sample_index:04d}_"
            f"{shape_id}_"
            f"{link_name}_"
            f"seed{seed}.pt"
        )

        save_path = OUT_ROOT / filename

        torch.save(
            pyg_data,
            save_path,
        )

        # ----------------------------------------------------
        # 简单统计
        # ----------------------------------------------------

        flow_mag = np.linalg.norm(
            flow,
            axis=1,
        )

        target_flow_mag = flow_mag[mask]

        entry = {
            "sample_index": sample_index,
            "shape_id": shape_id,
            "category": category,
            "link_name": link_name,
            "joint_name": target_joint.name,
            "joint_type": target_joint.type,
            "seed": seed,
            "num_points": int(len(pos)),
            "target_points": visible_target_after,
            "target_points_before_sampling": visible_target_before,
            "target_flow_mag_max": float(
                target_flow_mag.max()
            ),
            "target_flow_mag_mean": float(
                target_flow_mag.mean()
            ),
            "file": filename,
        }

        manifest.append(entry)

        print(
            f"  repeat={repeat_idx} "
            f"seed={seed} "
            f"raw_points={len(data['pos'])} "
            f"target_before={visible_target_before} "
            f"target_after={visible_target_after} "
            f"→ {filename}"
        )

        sample_index += 1


# ============================================================
# 保存 manifest
# ============================================================

manifest_path = OUT_ROOT / "manifest.json"

with open(
    manifest_path,
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print()
print("=" * 80)
print("Smoke Test 数据生成完成")
print("=" * 80)

print("样本总数:", len(manifest))
print("输出目录:", OUT_ROOT)
print("Manifest :", manifest_path)

print()

expected = N_TARGETS * N_REPEAT

assert len(manifest) == expected

print(
    f"预期 {expected} 个样本，"
    f"实际 {len(manifest)} 个样本：正确"
)
