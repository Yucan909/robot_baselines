import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PybulletRenderer

from flowbot3d.datasets.flow_dataset import compute_normalized_flow
import flowbot3d.models.artflownet as artflownet


# ============================================================
# 配置
# ============================================================

DATA_ROOT = Path.home() / "robot_baselines/data/partnet-mobility"
FLOWBOT_ROOT = Path.home() / "robot_baselines/repos/flowbot3d"

SHAPE_ID = "11712"
TARGET_LINK = "link_0"
SEED = 0
N_POINTS = 1200

OUTPUT = (
    Path.home()
    / "robot_baselines/results/flowbot3d/debug"
    / f"{SHAPE_ID}_{TARGET_LINK}_seed{SEED}.npz"
)


# ============================================================
# 1. 加载真实 PartNet-Mobility 物体
# ============================================================

print("=" * 75)
print("FlowBot3D 指定 link 测试")
print("=" * 75)

print("Shape ID   :", SHAPE_ID)
print("Target link:", TARGET_LINK)
print("Seed       :", SEED)

obj_dir = DATA_ROOT / SHAPE_ID

if not obj_dir.exists():
    raise FileNotFoundError(obj_dir)

obj = PMObject(obj_dir)

# link 对应的关节
target_joint = obj.obj.get_joint_by_child(TARGET_LINK)

print()
print("目标 link 对应关节:")
print("  joint name:", target_joint.name)
print("  joint type:", target_joint.type)
print("  axis      :", target_joint.axis)


# ============================================================
# 2. PyBullet 渲染真实点云
# ============================================================

renderer = PybulletRenderer()

data = renderer.render(
    pm_obj=obj,
    joints="random",
    camera_xyz="random",
    seed=SEED,
)

pos = data["pos"]

print()
print("原始可见点数:", len(pos))

if TARGET_LINK not in data["labelmap"]:
    raise RuntimeError(
        f"{TARGET_LINK} 不在渲染 labelmap 中"
    )

print(
    "目标 link segmentation id:",
    data["labelmap"][TARGET_LINK]
)

print(
    "当前对应关节角:",
    data["angles"][target_joint.name]
)


# ============================================================
# 3. 只计算指定 link 的 Ground Truth 3DAF
# ============================================================

gt_flow = compute_normalized_flow(
    P_world=pos,
    T_world_base=data["T_world_base"],
    current_jas=data["angles"],
    pc_seg=data["seg"],
    labelmap=data["labelmap"],
    pm_raw_data=obj,
    linknames=[TARGET_LINK],
)

mask = ~np.isclose(
    gt_flow,
    0.0,
    atol=1e-7,
).all(axis=-1)

print()
print("目标活动部件可见点数:", int(mask.sum()))
print("其余静止点数:", int((~mask).sum()))

if mask.sum() == 0:
    raise RuntimeError(
        "目标 link 在当前视角下没有得到有效 flow"
    )


# ============================================================
# 4. 和原论文保持一致，下采样到最多 1200 点
# ============================================================

rng = np.random.default_rng(SEED)

if len(pos) > N_POINTS:
    indices = rng.permutation(len(pos))[:N_POINTS]

    pos = pos[indices]
    gt_flow = gt_flow[indices]
    mask = mask[indices]

print()
print("送入网络的点数:", len(pos))
print("其中目标活动点:", int(mask.sum()))

if mask.sum() == 0:
    raise RuntimeError(
        "下采样后目标活动部件没有保留点"
    )


# ============================================================
# 5. 加载作者官方 checkpoint
# ============================================================

# 兼容作者旧 checkpoint 中保存的旧模块路径
sys.modules[
    "flowbot3d.models.flowbot3d"
] = artflownet

ckpt_path = FLOWBOT_ROOT / "pretrained/model.ckpt"

print()
print("加载官方 checkpoint:")
print(ckpt_path)

model = artflownet.ArtFlowNet.load_from_checkpoint(
    str(ckpt_path),
    map_location="cpu",
)

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

print("模型设备:", next(model.parameters()).device)


# ============================================================
# 6. FlowBot3D 推理
# ============================================================

xyz_tensor = torch.from_numpy(pos).float()
mask_tensor = torch.from_numpy(
    mask.astype(np.float32)
)

with torch.no_grad():
    pred_flow = model.predict(
        xyz_tensor,
        mask_tensor,
    )

pred_flow_cpu = pred_flow.cpu()

gt_flow_tensor = torch.from_numpy(
    gt_flow
).float()

target_mask = torch.from_numpy(mask).bool()


# ============================================================
# 7. 计算 FlowBot3D 原论文常用 flow 指标
# ============================================================

pred_target = pred_flow_cpu[target_mask]
gt_target = gt_flow_tensor[target_mask]

rmse = (
    pred_target - gt_target
).norm(
    p=2,
    dim=1
).mean()

cosine = F.cosine_similarity(
    pred_target,
    gt_target,
    dim=1,
).mean()

mag_error = (
    pred_target.norm(dim=1)
    - gt_target.norm(dim=1)
).abs().mean()


# ============================================================
# 8. 输出结果
# ============================================================

print()
print("=" * 75)
print("结果")
print("=" * 75)

print("点云 shape       :", tuple(xyz_tensor.shape))
print("GT Flow shape    :", tuple(gt_flow_tensor.shape))
print("Pred Flow shape  :", tuple(pred_flow_cpu.shape))
print("目标活动点数     :", int(target_mask.sum()))

print()
print("Flow 指标:")
print("RMSE             :", float(rmse))
print("Cosine similarity:", float(cosine))
print("Magnitude error  :", float(mag_error))

print()
print("前 5 个目标点的 GT / Pred:")

target_ids = torch.where(
    target_mask
)[0][:5]

for i in target_ids:
    print()
    print("point:", int(i))
    print(
        "GT  :",
        gt_flow_tensor[i].numpy(),
    )
    print(
        "Pred:",
        pred_flow_cpu[i].numpy(),
    )


# ============================================================
# 9. 保存结果，方便以后可视化/排错
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT,
    shape_id=SHAPE_ID,
    target_link=TARGET_LINK,
    seed=SEED,
    pos=pos,
    mask=mask,
    gt_flow=gt_flow,
    pred_flow=pred_flow_cpu.numpy(),
)

print()
print("结果已保存:")
print(OUTPUT)

print()
print("=" * 75)
print("指定 link 的 FlowBot3D 真实推理链测试成功")
print("=" * 75)
