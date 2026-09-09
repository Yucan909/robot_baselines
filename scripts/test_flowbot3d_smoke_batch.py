from pathlib import Path

import torch
from torch_geometric.data import Batch

from flowbot3d.models.artflownet import (
    ArtFlowNet,
    ArtFlowNetParams,
    artflownet_loss,
)


ROOT = (
    Path.home()
    / "robot_baselines/data/flowbot3d_custom/smoke_test"
)

files = sorted(ROOT.glob("*.pt"))

data_list = [
    torch.load(
        f,
        map_location="cpu",
        weights_only=False,
    )
    for f in files
]


# ============================================================
# 组成训练 batch
# ============================================================

batch = Batch.from_data_list(
    data_list
).cuda()


# ============================================================
# 创建论文模型
#
# mask_input_channel=True：
# 和作者 train.py 中默认 FlowBot3D 设置一致
# ============================================================

model = ArtFlowNet(
    p=ArtFlowNetParams(
        mask_input_channel=True
    )
).cuda()

model.train()


# ============================================================
# 前向传播
# ============================================================

pred = model(batch)

print("=" * 75)
print("6 样本 ArtFlowNet 训练前向测试")
print("=" * 75)

print("Pred shape:", pred.shape)
print("GT shape  :", batch.flow.shape)

assert pred.shape == batch.flow.shape


# ============================================================
# 与 FlowBot3D 原训练损失保持一致
# ============================================================

n_nodes = torch.tensor(
    [
        d.num_nodes
        for d in batch.to_data_list()
    ],
    device="cuda",
)

loss = artflownet_loss(
    pred,
    batch.flow,
    n_nodes,
)

print("Loss:", float(loss))

assert torch.isfinite(loss)


# ============================================================
# 真正反向传播一次
# ============================================================

loss.backward()

grad_count = 0

for p in model.parameters():
    if p.grad is not None:
        grad_count += 1

print(
    "有梯度的参数 Tensor 数:",
    grad_count,
)

assert grad_count > 0

print()
print("=" * 75)
print(
    "数据 → ArtFlowNet → Loss → "
    "Backward 全链路测试成功"
)
print("=" * 75)
