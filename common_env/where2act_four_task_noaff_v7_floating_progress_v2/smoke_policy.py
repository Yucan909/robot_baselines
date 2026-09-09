
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from where2act_policy import (
    Where2ActPolicy,
)


print(
    "=" * 90
)

print(
    "WHERE2ACT POLICY WRAPPER SMOKE"
)

print(
    "=" * 90
)


print(
    "torch:",
    torch.__version__,
)

print(
    "CUDA:",
    torch.version.cuda,
)

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)


# ============================================================
# 构造一份仅用于工程验证的单视角风格点云
# ============================================================

rng = np.random.default_rng(
    20260901
)


N = 35000


# 点云放在相机前方附近，
# 只是为了让 PointNet++ neighborhood 尺度合理。
points_model = rng.normal(
    size=(
        N,
        3,
    )
).astype(
    np.float32
)


points_model *= np.array(
    [
        0.35,
        0.35,
        0.35,
    ],
    dtype=np.float32,
)


# smoke中暂时认为model/world坐标一致
points_world = (
    points_model.copy()
)


# 相机方向与世界方向一致
camera_to_world_R = np.eye(
    3,
    dtype=np.float64,
)


# 仅允许约1/3点作为“目标link”
candidate_mask = np.zeros(
    N,
    dtype=bool,
)

candidate_mask[
    N // 3 :
    2 * N // 3
] = True


# ============================================================
# 随机权重仅用于接口测试
# ============================================================

policy = Where2ActPolicy(
    checkpoint=None,
    device="cuda:0",
    allow_untrained=True,
)


with torch.inference_mode():

    result = policy.predict(
        points_model,
        points_world,
        camera_to_world_R,
        candidate_mask=(
            candidate_mask
        ),
        seed=20260901,
    )


print()
print(
    "=" * 90
)

print(
    "RESULT"
)

print(
    "=" * 90
)


for key in [
    "interaction_point_world",
    "interaction_score",
    "proposal_index",
    "critic_score",
    "grasp_position_world",
    "rotation_orthogonality_error",
    "rotation_determinant",
    "num_input_points",
    "num_model_points",
    "trained",
]:

    print(
        f"{key}:",
        result[
            key
        ],
    )


print()
print(
    "grasp_pose_world:"
)

print(
    result[
        "grasp_pose_world"
    ]
)


# ============================================================
# Assertions
# ============================================================

assert (
    result[
        "num_model_points"
    ]
    == 10000
)


assert (
    result[
        "proposal_index"
    ]
    >= 0
)


assert (
    result[
        "proposal_index"
    ]
    < 100
)


assert np.all(
    np.isfinite(
        result[
            "grasp_pose_world"
        ]
    )
)


assert (
    result[
        "rotation_orthogonality_error"
    ]
    < 1e-4
)


assert abs(
    result[
        "rotation_determinant"
    ]
    - 1.0
) < 1e-4


# 确认选中的点确实来自候选target区域
original_idx = (
    result[
        "original_query_index"
    ]
)

assert candidate_mask[
    original_idx
]


print()
print(
    "=" * 90
)

print(
    "WHERE2ACT POLICY WRAPPER: PASS"
)

print(
    "=" * 90
)
