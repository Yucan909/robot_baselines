from pathlib import Path

import torch
from torch_geometric.data import Batch


ROOT = (
    Path.home()
    / "robot_baselines/data/flowbot3d_custom/smoke_test"
)

files = sorted(ROOT.glob("*.pt"))

print("=" * 75)
print("检查 FlowBot3D Smoke Test 数据")
print("=" * 75)

print("找到样本数:", len(files))

assert len(files) == 6


all_data = []

for path in files:

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print(path.name)

    print(
        "  shape/link:",
        data.shape_id,
        data.link_name,
    )

    print(
        "  pos :",
        tuple(data.pos.shape),
    )

    print(
        "  flow:",
        tuple(data.flow.shape),
    )

    print(
        "  mask:",
        tuple(data.mask.shape),
    )

    print(
        "  target points:",
        int(data.mask.sum().item()),
    )

    assert data.pos.shape == (1200, 3)
    assert data.flow.shape == (1200, 3)
    assert data.mask.shape == (1200,)

    assert torch.isfinite(
        data.pos
    ).all()

    assert torch.isfinite(
        data.flow
    ).all()

    assert data.mask.sum() > 0

    # 静止点 GT flow 必须接近 0
    static = data.mask == 0

    if static.any():
        static_flow_max = (
            data.flow[static]
            .abs()
            .max()
            .item()
        )

        print(
            "  static flow max:",
            static_flow_max,
        )

        assert static_flow_max < 1e-5

    all_data.append(data)


# ============================================================
# 测试 PyTorch Geometric 能否把这些数据组成 batch
# ============================================================

batch = Batch.from_data_list(
    all_data
)

print()
print("=" * 75)
print("PyG Batch 测试")
print("=" * 75)

print(
    "batch.pos:",
    tuple(batch.pos.shape),
)

print(
    "batch.flow:",
    tuple(batch.flow.shape),
)

print(
    "batch.mask:",
    tuple(batch.mask.shape),
)

print(
    "batch.batch:",
    tuple(batch.batch.shape),
)

print(
    "图数量:",
    batch.num_graphs,
)

assert batch.num_graphs == 6
assert batch.pos.shape == (7200, 3)
assert batch.flow.shape == (7200, 3)
assert batch.mask.shape == (7200,)

print()
print("=" * 75)
print("6 个训练样本全部检查通过")
print("=" * 75)
