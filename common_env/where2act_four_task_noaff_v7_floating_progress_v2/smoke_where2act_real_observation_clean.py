import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import sapien.core as sapien

from where2act_observation_adapter import (
    capture_where2act_observation,
)

from where2act_policy import (
    Where2ActPolicy,
    Where2ActPolicyError,
)


# ============================================================
# 固定 benchmark 参数
# ============================================================

PARTNET_ROOT = (
    Path.home()
    / "robot_baselines"
    / "data"
    / "partnet-mobility"
)

OBJECT_SCALE = 0.75

WIDTH = 448
HEIGHT = 448

FOV_DEG = 35.0

NEAR = 0.1
FAR = 100.0

INITIAL_RATIO_MIN = 0.10
INITIAL_RATIO_MAX = 0.20


# ============================================================
# 工具函数
# ============================================================

def load_pose_catalog_row(
    catalog_path,
    shape_id,
    target_link,
):
    catalog_path = (
        Path(catalog_path)
        .expanduser()
        .resolve()
    )

    if not catalog_path.exists():
        raise FileNotFoundError(
            catalog_path
        )

    with catalog_path.open() as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            link_name = (
                row.get("link_name")
                or row.get("target_link")
                or row.get("link")
            )

            if (
                str(row.get("shape_id"))
                == str(shape_id)
                and str(link_name)
                == str(target_link)
            ):
                return row

    raise RuntimeError(
        "pose catalog 找不到 "
        f"{shape_id}/{target_link}"
    )


def find_object_urdf(
    shape_id,
):
    root = (
        PARTNET_ROOT
        / str(shape_id)
    )

    candidates = [
        root / "mobility.urdf",
        root / "mobility_vhacd.urdf",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"找不到 PartNet URDF: {root}"
    )


def raw_joint_limits(
    joint,
):
    """
    读取SAPIEN关节的原始上下限。

    注意：
    PartNet中某些非目标活动关节可能是：
        [-inf, +inf]

    这本身不是错误。
    """

    limits = np.asarray(
        joint.get_limits(),
        dtype=np.float64,
    )

    if (
        limits.ndim != 2
        or limits.shape[0] < 1
        or limits.shape[1] != 2
    ):
        raise RuntimeError(
            f"joint limit shape异常: "
            f"{joint.get_name()} "
            f"{limits.shape}"
        )

    return (
        float(limits[0, 0]),
        float(limits[0, 1]),
    )


def closed_joint_position(
    joint,
):
    """
    统一benchmark的“关闭状态”。

    有有限lower limit：
        使用lower。

    没有有限lower limit：
        使用0。

    这与我们已经跑通的FlowBot统一环境保持一致。
    """

    lower, _ = raw_joint_limits(
        joint
    )

    if not np.isfinite(
        lower
    ):
        lower = 0.0

    return float(
        lower
    )


def finite_target_joint_limits(
    joint,
):
    """
    只有真正被评测的target joint必须有有限范围。

    因为：
        progress = (q-q_min)/(q_max-q_min)

    没有有限范围就无法定义10%-20%和40%开度。
    """

    lower, upper = raw_joint_limits(
        joint
    )

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
    ):
        raise RuntimeError(
            "target joint没有有限range，"
            "无法定义统一benchmark开度: "
            f"{joint.get_name()} "
            f"[{lower}, {upper}]"
        )

    if upper <= lower:
        raise RuntimeError(
            "target joint range非法: "
            f"{joint.get_name()} "
            f"[{lower}, {upper}]"
        )

    return (
        float(lower),
        float(upper),
    )


def create_renderer():
    """
    使用当前统一 benchmark 已经验证过的 SAPIEN renderer。
    当前环境为旧版 sapien.core API。
    """
    return sapien.VulkanRenderer(
        offscreen_only=True
    )


# ============================================================
# Main smoke
# ============================================================

def run(args):

    print()
    print("=" * 100)
    print("PURE WHERE2ACT REAL OBSERVATION SMOKE")
    print("=" * 100)

    # --------------------------------------------------------
    # 1. 固定随机性
    # --------------------------------------------------------

    trial_seed = int(
        args.trial_seed
    )

    np.random.seed(
        trial_seed
    )

    torch.manual_seed(
        trial_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            trial_seed
        )

    rng = np.random.default_rng(
        trial_seed
    )

    # --------------------------------------------------------
    # 2. pose catalog
    # --------------------------------------------------------

    row = load_pose_catalog_row(
        args.pose_catalog,
        args.shape_id,
        args.target_link,
    )

    camera_pose_world = np.asarray(
        row[
            "camera_pose_world"
        ],
        dtype=np.float64,
    )

    if camera_pose_world.shape != (
        4,
        4,
    ):
        raise RuntimeError(
            "camera_pose_world 应为4x4，"
            f"实际 {camera_pose_world.shape}"
        )

    category = row.get(
        "category",
        "unknown",
    )

    print(
        "shape:",
        args.shape_id,
    )

    print(
        "target:",
        args.target_link,
    )

    print(
        "category:",
        category,
    )

    # --------------------------------------------------------
    # 3. initial ratio
    # --------------------------------------------------------

    if args.initial_ratio is None:

        initial_ratio = float(
            rng.uniform(
                INITIAL_RATIO_MIN,
                INITIAL_RATIO_MAX,
            )
        )

    else:

        initial_ratio = float(
            args.initial_ratio
        )

    if not (
        INITIAL_RATIO_MIN
        <= initial_ratio
        <= INITIAL_RATIO_MAX
    ):
        raise ValueError(
            "initial_ratio必须位于"
            "[0.10, 0.20]"
        )

    print(
        "initial ratio:",
        initial_ratio,
    )

    # --------------------------------------------------------
    # 4. SAPIEN engine
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 使用统一 benchmark 已验证的 SAPIEN 初始化
    # --------------------------------------------------------

    engine = sapien.Engine(
        0,
        0.001,
        0.005,
    )

    renderer = create_renderer()

    engine.set_renderer(
        renderer
    )

    scene_config = sapien.SceneConfig()

    scene_config.gravity = [
        0,
        0,
        -9.81,
    ]

    scene_config.solver_iterations = 20
    scene_config.enable_pcm = False
    scene_config.sleep_threshold = 0.0

    scene = engine.create_scene(
        config=scene_config
    )

    scene.set_timestep(
        1.0 / 500.0
    )

    scene.set_ambient_light(
        [0.5, 0.5, 0.5]
    )

    scene.add_point_light(
        [1, 2, 2],
        [1, 1, 1],
    )

    scene.add_point_light(
        [1, -2, 2],
        [1, 1, 1],
    )

    scene.add_point_light(
        [-1, 0, 2],
        [1, 1, 1],
    )

    # --------------------------------------------------------
    # 5. PartNet object
    # --------------------------------------------------------

    urdf_path = find_object_urdf(
        args.shape_id
    )

    print(
        "object URDF:",
        urdf_path,
    )

    loader = (
        scene.create_urdf_loader()
    )

    loader.fix_root_link = True
    loader.scale = OBJECT_SCALE

    obj = loader.load(
        str(
            urdf_path
        )
    )

    if obj is None:
        raise RuntimeError(
            "SAPIEN加载PartNet失败"
        )

    obj.set_root_pose(
        sapien.Pose()
    )

    active_joints = list(
        obj.get_active_joints()
    )

    if len(active_joints) == 0:
        raise RuntimeError(
            "物体没有active joints"
        )

    target_joint = None
    target_link = None
    target_index = None

    for i, joint in enumerate(
        active_joints
    ):

        child = (
            joint.get_child_link()
        )

        if (
            child.get_name()
            == str(
                args.target_link
            )
        ):
            target_joint = joint
            target_link = child
            target_index = i
            break

    if target_joint is None:
        raise RuntimeError(
            "找不到target joint: "
            f"{args.target_link}"
        )

    # --------------------------------------------------------
    # 6. 设置物体初始关节状态
    #
    # 与已经验证过的统一benchmark一致：
    #
    # target joint:
    #     完整range的10%-20%
    #
    # 其他active joints:
    #     有有限lower -> lower
    #     无有限lower -> 0
    # --------------------------------------------------------

    target_lower, target_upper = (
        finite_target_joint_limits(
            target_joint
        )
    )

    target_q = (
        target_lower
        + initial_ratio
        * (
            target_upper
            - target_lower
        )
    )

    qpos = np.zeros(
        len(active_joints),
        dtype=np.float64,
    )

    print()
    print("-" * 100)
    print("OBJECT JOINT INITIALIZATION")
    print("-" * 100)

    for i, joint in enumerate(
        active_joints
    ):

        raw_lower, raw_upper = (
            raw_joint_limits(
                joint
            )
        )

        if i == target_index:

            initial_q = float(
                target_q
            )

            role = "TARGET"

        else:

            initial_q = (
                closed_joint_position(
                    joint
                )
            )

            role = "closed"

        qpos[i] = initial_q

        print(
            f"[{i:02d}] "
            f"{joint.get_name():<20} "
            f"raw=[{raw_lower}, {raw_upper}] "
            f"init={initial_q:.6f} "
            f"{role}"
        )

        # smoke阶段也把关节保持在规定初始位置，
        # 防止后续加入scene.step()时自行漂移。
        try:

            joint.set_drive_property(
                stiffness=5000,
                damping=500,
            )

            joint.set_drive_target(
                float(
                    initial_q
                )
            )

            joint.set_drive_velocity_target(
                0.0
            )

        except Exception:

            # observation smoke主要依赖qpos，
            # 某些特殊joint不支持drive时不阻断渲染测试。
            pass


    obj.set_qpos(
        qpos
    )


    # 硬校验target初始开度。
    actual_target_q = float(
        obj.get_qpos()[
            target_index
        ]
    )

    actual_initial_ratio = (
        actual_target_q
        - target_lower
    ) / (
        target_upper
        - target_lower
    )

    if not np.isfinite(
        actual_initial_ratio
    ):
        raise RuntimeError(
            "实际target initial ratio非有限"
        )

    if abs(
        actual_initial_ratio
        - initial_ratio
    ) > 1e-5:
        raise RuntimeError(
            "target初始开度设置失败: "
            f"requested={initial_ratio}, "
            f"actual={actual_initial_ratio}"
        )

    print()
    print(
        "requested initial ratio:",
        initial_ratio,
    )

    print(
        "actual initial ratio:",
        actual_initial_ratio,
    )

    print(
        "target joint:",
        target_joint.get_name(),
    )

    print(
        "target limits:",
        target_lower,
        target_upper,
    )

    print(
        "target q:",
        target_q,
    )

    print(
        "target link actor id:",
        int(
            target_link.get_id()
        ),
    )

    # --------------------------------------------------------
    # 7. 学长固定 Camera
    #
    # 与已经跑通的 FlowBot benchmark camera 一致：
    #
    # 448x448
    # fov 35°
    # near 0.1
    # far 100
    # pose来自同一个pose catalog
    # --------------------------------------------------------

    camera_builder = (
        scene.create_actor_builder()
    )

    camera_mount = (
        camera_builder
        .build_kinematic(
            "benchmark_camera_mount"
        )
    )

    fov = np.deg2rad(
        FOV_DEG
    )

    camera = (
        scene.add_mounted_camera(
            "benchmark_camera",
            camera_mount,
            sapien.Pose(),
            WIDTH,
            HEIGHT,
            0.0,
            fov,
            NEAR,
            FAR,
        )
    )

    camera_mount.set_pose(
        sapien.Pose
        .from_transformation_matrix(
            camera_pose_world
        )
    )

    # --------------------------------------------------------
    # 8. render几帧
    #
    # 不执行机器人动作。
    # --------------------------------------------------------

    for _ in range(5):
        scene.update_render()

    # --------------------------------------------------------
    # 9. 真正统一相机 observation
    # --------------------------------------------------------

    observation = (
        capture_where2act_observation(
            scene,
            camera,
            obj,
            target_link,
            object_origin_world=np.zeros(
                3,
                dtype=np.float64,
            ),
        )
    )

    diag = observation[
        "diagnostics"
    ]

    print()
    print("-" * 100)
    print("OBSERVATION")
    print("-" * 100)

    print(
        "visible object points:",
        diag[
            "visible_object_points"
        ],
    )

    print(
        "visible target points:",
        diag[
            "visible_target_points"
        ],
    )

    print(
        "target fraction:",
        diag[
            "target_fraction"
        ],
    )

    print(
        "model point min:",
        diag[
            "point_min_model"
        ],
    )

    print(
        "model point max:",
        diag[
            "point_max_model"
        ],
    )

    print(
        "model extent:",
        diag[
            "point_extent_model"
        ],
    )

    print(
        "camera position world:",
        observation[
            "camera_position_world"
        ],
    )

    print(
        "object origin camera:",
        observation[
            "object_origin_camera"
        ],
    )

    print(
        "coordinate consistency error:",
        diag[
            "coordinate_consistency_error"
        ],
    )

    print(
        "camera rotation determinant:",
        diag[
            "camera_rotation_determinant"
        ],
    )

    print(
        "camera rotation orthogonality error:",
        diag[
            "camera_rotation_orthogonality_error"
        ],
    )

    # --------------------------------------------------------
    # 10. Where2Act
    #
    # 暂无训练权重，所以随机权重只做工程forward。
    # --------------------------------------------------------

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA不可用"
        )

    print()
    print(
        "Where2Act device:",
        device,
    )

    policy = Where2ActPolicy(
        checkpoint=None,
        device=str(
            device
        ),
        allow_untrained=True,
    )

    with torch.inference_mode():

        result = policy.predict(
            observation[
                "points_model"
            ],
            observation[
                "points_world"
            ],
            observation[
                "camera_to_world_R"
            ],
            candidate_mask=(
                observation[
                    "candidate_mask"
                ]
            ),
            seed=trial_seed,
        )

    # --------------------------------------------------------
    # 11. 结果
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("WHERE2ACT RESULT")
    print("-" * 100)

    print(
        "interaction point world:",
        result[
            "interaction_point_world"
        ],
    )

    print(
        "interaction score:",
        result[
            "interaction_score"
        ],
    )

    print(
        "proposal index:",
        result[
            "proposal_index"
        ],
    )

    print(
        "critic score:",
        result[
            "critic_score"
        ],
    )

    print(
        "grasp position world:",
        result[
            "grasp_position_world"
        ],
    )

    print(
        "rotation determinant:",
        result[
            "rotation_determinant"
        ],
    )

    print(
        "rotation orthogonality error:",
        result[
            "rotation_orthogonality_error"
        ],
    )

    print(
        "grasp pose world:"
    )

    print(
        result[
            "grasp_pose_world"
        ]
    )

    # --------------------------------------------------------
    # 12. 保存结果
    # --------------------------------------------------------

    output_root = (
        Path(
            args.output_root
        )
        .expanduser()
        .resolve()
    )

    output_dir = (
        output_root
        / (
            f"{args.shape_id}_"
            f"{args.target_link}_"
            f"seed_{trial_seed}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir
        / "points_model.npy",
        observation[
            "points_model"
        ],
    )

    np.save(
        output_dir
        / "points_world.npy",
        observation[
            "points_world"
        ],
    )

    np.save(
        output_dir
        / "candidate_mask.npy",
        observation[
            "candidate_mask"
        ],
    )

    save_result = {
        "shape_id":
            str(
                args.shape_id
            ),

        "target_link":
            str(
                args.target_link
            ),

        "category":
            category,

        "trial_seed":
            trial_seed,

        "initial_ratio":
            initial_ratio,

        "visible_object_points":
            int(
                diag[
                    "visible_object_points"
                ]
            ),

        "visible_target_points":
            int(
                diag[
                    "visible_target_points"
                ]
            ),

        "model_extent":
            np.asarray(
                diag[
                    "point_extent_model"
                ]
            ).tolist(),

        "coordinate_consistency_error":
            float(
                diag[
                    "coordinate_consistency_error"
                ]
            ),

        "interaction_point_world":
            np.asarray(
                result[
                    "interaction_point_world"
                ]
            ).tolist(),

        "interaction_score":
            float(
                result[
                    "interaction_score"
                ]
            ),

        "critic_score":
            float(
                result[
                    "critic_score"
                ]
            ),

        "grasp_pose_world":
            np.asarray(
                result[
                    "grasp_pose_world"
                ]
            ).tolist(),

        "trained":
            bool(
                result[
                    "trained"
                ]
            ),
    }

    with (
        output_dir
        / "result.json"
    ).open(
        "w"
    ) as f:

        json.dump(
            save_result,
            f,
            indent=2,
        )

    print()
    print(
        "saved:",
        output_dir,
    )

    print()
    print("=" * 100)
    print(
        "PURE WHERE2ACT REAL OBSERVATION: PASS"
    )
    print("=" * 100)


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shape-id",
        required=True,
    )

    parser.add_argument(
        "--target-link",
        required=True,
    )

    parser.add_argument(
        "--pose-catalog",
        required=True,
    )

    parser.add_argument(
        "--trial-seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--initial-ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser


if __name__ == "__main__":

    args = (
        build_parser()
        .parse_args()
    )

    try:

        run(
            args
        )

    except Exception:

        print()
        print("=" * 100)
        print(
            "IMPLEMENTATION ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
