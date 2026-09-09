import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault(
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "1",
)

import numpy as np
import sapien.core as sapien
import torch

try:
    import cv2
except Exception:
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# ============================================================
# 路径与 import
# ============================================================

HOME = Path.home()

FLOWBOT_ROOT = (
    HOME
    / "robot_baselines/repos/flowbot3d"
)

CLEAN_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

for p in [
    CLEAN_DIR,
    FLOWBOT_ROOT,
]:
    if str(p) not in sys.path:
        sys.path.insert(
            0,
            str(p),
        )

from panda_controller import PandaTwoFingerController
from grasp_pose_adapter import (
    GraspPoseError,
    build_twofinger_grasp_pose,
)
from contact_monitor import (
    monitor_grasp_establishment,
    monitor_post_pull_contact,
)

# 兼容旧 Lightning checkpoint 中可能存在的模块路径。
import flowbot3d.models.artflownet as artflownet
sys.modules.setdefault(
    "flowbot3d.models.flowbot3d",
    artflownet,
)
from flowbot3d.models.artflownet import ArtFlowNet


# ============================================================
# Benchmark 固定协议
# ============================================================

PROTOCOL_VERSION = "flowbot3d_twofinger_physical_v1"

OBJECT_SCALE = 0.75

WIDTH = 448
HEIGHT = 448
FOV_DEG = 35.0
NEAR = 0.1
FAR = 100.0

NUM_POINTS = 1200

INITIAL_RATIO_MIN = 0.10
INITIAL_RATIO_MAX = 0.20
SUCCESS_RATIO = 0.40

# 接近阶段：
# 沿“当前工具中心 -> 模型接触点”的直线，
# 先到最多 8 cm 的预接触点，再到表面外 5 mm。
PREGRASP_DISTANCE = 0.08
FINAL_CONTACT_STANDOFF = 0.0

# 正式 pull 前，目标关节相对规定初始开度允许 2 个百分点漂移。
START_PROGRESS_TOLERANCE = 0.02

APPROACH_SIM_STEPS = 2000
FINAL_APPROACH_SIM_STEPS = 1200
GRIPPER_WAIT_STEPS = 300

# 真实二指抓取的固定物理判定窗口。
GRASP_SETTLE_STEPS = 300
GRASP_TAIL_STEPS = 100
GRASP_MIN_BILATERAL_FRACTION = 0.50
POST_PULL_CONTACT_STEPS = 50


# 每个闭环高层动作沿当前 FlowBot flow 移动 2 cm。
PULL_DISTANCE = 0.02
PULL_SIM_STEPS = 500

# 总计最多 30 次，即最多 0.6 m 的累计高层平移预算。
MAX_PULL_STEPS = 30

PM_ROOT = (
    HOME
    / "robot_baselines/data/partnet-mobility"
)

PANDA_URDF = (
    HOME
    / "robot_baselines/common_env/assets"
    / "panda_articubot/panda.urdf"
)

CKPT_CANDIDATES = [
    (
        HOME
        / "robot_baselines/results/flowbot3d/final/model.ckpt"
    ),
    (
        HOME
        / "robot_baselines/results/flowbot3d/training"
        / "20260829_012327/final.ckpt"
    ),
]


# ============================================================
# Task failure
# ============================================================

class TaskFailure(Exception):
    def __init__(
        self,
        reason,
        data=None,
    ):
        super().__init__(
            str(reason)
        )
        self.reason = str(reason)
        self.data = (
            {} if data is None
            else dict(data)
        )


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shape-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target-link",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pose-catalog",
        type=str,
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
        type=str,
        required=True,
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# JSON helper
# ============================================================

def jsonable(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        (np.floating,),
    ):
        return float(value)

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): jsonable(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            jsonable(v)
            for v in value
        ]

    return value


def write_json(
    path,
    data,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
    ) as f:
        json.dump(
            jsonable(data),
            f,
            indent=2,
        )


# ============================================================
# 主 trial
# ============================================================

def execute_trial(
    args,
    out_dir,
):
    shape_id = str(
        args.shape_id
    )

    target_link_name = str(
        args.target_link
    )

    trial_seed = int(
        args.trial_seed
    )

    # 所有随机源固定。
    random.seed(
        trial_seed
    )

    np.random.seed(
        trial_seed % (2**32)
    )

    torch.manual_seed(
        trial_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            trial_seed
        )

    trial_rng = np.random.default_rng(
        trial_seed
    )

    if args.initial_ratio is None:
        initial_ratio = float(
            trial_rng.uniform(
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
            "initial-ratio 必须位于 [0.10, 0.20]"
        )

    pose_file = (
        Path(
            args.pose_catalog
        )
        .expanduser()
        .resolve()
    )

    save_video = (
        not args.no_video
    )

    metrics = {
        "method":
            "FlowBot3D",

        "protocol_version":
            PROTOCOL_VERSION,

        "executor_version":
            "panda_physical_twofinger_v1",

        "shape_id":
            shape_id,

        "target_link":
            target_link_name,

        "trial_seed":
            trial_seed,

        "pose_catalog":
            str(
                pose_file
            ),

        "requested_initial_progress":
            initial_ratio,

        "success_threshold":
            SUCCESS_RATIO,

        "start_progress_tolerance":
            START_PROGRESS_TOLERANCE,

        "pull_distance_m":
            PULL_DISTANCE,

        "max_pull_steps":
            MAX_PULL_STEPS,

        "grasp_success":
            False,

        "operation_attempted_after_grasp":
            False,

        "success":
            False,

        "failure_reason":
            None,

        "history":
            [],
    }

    progress_fn = None
    video_writer = None

    def safe_progress():
        if progress_fn is None:
            return None

        try:
            value = float(
                progress_fn()
            )
            if np.isfinite(value):
                return value
        except Exception:
            pass

        return None

    def fail(
        reason,
        **extra,
    ):
        data = dict(
            extra
        )

        if (
            "final_progress"
            not in data
        ):
            data[
                "final_progress"
            ] = safe_progress()

        raise TaskFailure(
            reason,
            data,
        )

    try:
        # ----------------------------------------------------
        # checkpoint
        # ----------------------------------------------------

        ckpt = None

        for candidate in (
            CKPT_CANDIDATES
        ):
            if candidate.exists():
                ckpt = candidate
                break

        if ckpt is None:
            raise FileNotFoundError(
                "找不到 FlowBot3D checkpoint"
            )

        metrics[
            "checkpoint"
        ] = str(
            ckpt
        )

        # ----------------------------------------------------
        # pose catalog
        # ----------------------------------------------------

        records = []

        with open(
            pose_file
        ) as f:
            for line in f:
                if line.strip():
                    records.append(
                        json.loads(
                            line
                        )
                    )

        cfg = None

        for row in records:
            link_name = (
                row.get(
                    "link_name"
                )
                or row.get(
                    "target_link"
                )
                or row.get(
                    "link"
                )
            )

            if (
                str(
                    row["shape_id"]
                )
                == shape_id
                and link_name
                == target_link_name
            ):
                cfg = row
                break

        if cfg is None:
            raise RuntimeError(
                f"pose catalog 找不到 "
                f"{shape_id}/{target_link_name}"
            )

        base_pose = np.asarray(
            cfg[
                "base_pose"
            ],
            dtype=np.float64,
        )

        robot_qpos = np.asarray(
            cfg[
                "robot_initial_qpos"
            ],
            dtype=np.float64,
        )

        camera_pose_world = np.asarray(
            cfg[
                "camera_pose_world"
            ],
            dtype=np.float64,
        )

        metrics[
            "category"
        ] = cfg.get(
            "category"
        )

        # ----------------------------------------------------
        # SAPIEN
        # ----------------------------------------------------

        engine = sapien.Engine(
            0,
            0.001,
            0.005,
        )

        renderer = sapien.VulkanRenderer(
            offscreen_only=True
        )

        engine.set_renderer(
            renderer
        )

        scene_config = (
            sapien.SceneConfig()
        )

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

        # ----------------------------------------------------
        # PartNet object
        # ----------------------------------------------------

        object_urdf = (
            PM_ROOT
            / shape_id
            / "mobility.urdf"
        )

        if not object_urdf.exists():
            raise FileNotFoundError(
                str(
                    object_urdf
                )
            )

        loader = (
            scene.create_urdf_loader()
        )

        loader.fix_root_link = True
        loader.scale = OBJECT_SCALE

        obj = loader.load(
            str(
                object_urdf
            )
        )

        if obj is None:
            raise RuntimeError(
                "PartNet object 加载失败"
            )

        obj.set_root_pose(
            sapien.Pose(
                [0, 0, 0],
                [1, 0, 0, 0],
            )
        )

        active_joints = list(
            obj.get_active_joints()
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
                == target_link_name
            ):
                target_joint = joint
                target_link = child
                target_index = i
                break

        if target_joint is None:
            raise RuntimeError(
                "找不到 target joint"
            )

        q_min = float(
            target_joint
            .get_limits()[0, 0]
        )

        q_max = float(
            target_joint
            .get_limits()[0, 1]
        )

        q_range = (
            q_max - q_min
        )

        if (
            not np.isfinite(
                q_min
            )
            or not np.isfinite(
                q_max
            )
            or q_range <= 1e-8
        ):
            raise RuntimeError(
                f"目标关节范围异常: "
                f"[{q_min}, {q_max}]"
            )

        object_qpos = []

        for i, joint in enumerate(
            active_joints
        ):
            limits = np.asarray(
                joint.get_limits()[0],
                dtype=np.float64,
            )

            lower = float(
                limits[0]
            )

            if not np.isfinite(
                lower
            ):
                lower = 0.0

            object_qpos.append(
                lower
            )

            joint.set_drive_property(
                stiffness=5000,
                damping=500,
            )

            if (
                i != target_index
            ):
                joint.set_drive_target(
                    lower
                )
                joint.set_drive_velocity_target(
                    0.0
                )

        target_q = (
            q_min
            + initial_ratio
            * q_range
        )

        object_qpos[
            target_index
        ] = target_q

        obj.set_qpos(
            np.asarray(
                object_qpos,
                dtype=np.float64,
            )
        )

        target_joint.set_drive_target(
            float(
                target_q
            )
        )

        target_joint.set_drive_velocity_target(
            0.0
        )

        def get_progress():
            q = float(
                obj.get_qpos()[
                    target_index
                ]
            )

            return (
                q - q_min
            ) / q_range

        progress_fn = get_progress

        # ----------------------------------------------------
        # Panda
        # ----------------------------------------------------

        panda = PandaTwoFingerController(
            scene,
            PANDA_URDF,
            # 与 ArticuBot Panda URDF 的 finger contact 参数一致：
            # lateral_friction = 1.0。
            finger_static_friction=1.0,
            finger_dynamic_friction=1.0,
            finger_restitution=0.0,
        )

        panda.set_initial_state(
            base_pose,
            robot_qpos,
        )

        hand_link = (
            panda.hand_link
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FlowBot model
        # ----------------------------------------------------

        device_name = os.environ.get(
            "FLOWBOT_DEVICE",
            (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            ),
        )

        device = torch.device(
            device_name
        )

        print(
            "FlowBot3D device:",
            device,
        )

        model = (
            ArtFlowNet
            .load_from_checkpoint(
                str(
                    ckpt
                )
            )
            .to(
                device
            )
        )

        model.eval()

        # ----------------------------------------------------
        # stable initial state
        # ----------------------------------------------------

        panda.wait(
            100
        )

        metrics[
            "actual_initial_progress"
        ] = float(
            get_progress()
        )

        # ----------------------------------------------------
        # observation
        # ----------------------------------------------------

        object_link_ids = np.asarray(
            [
                int(
                    link.get_id()
                )
                for link
                in obj.get_links()
            ],
            dtype=np.int64,
        )

        target_actor_id = int(
            target_link.get_id()
        )

        def capture_observation(
            sample_index,
        ):
            scene.update_render()
            camera.take_picture()

            rgba = np.asarray(
                camera.get_float_texture(
                    "Color"
                )
            )

            rgb = np.clip(
                rgba[..., :3],
                0,
                1,
            ).astype(
                np.float32
            )

            position = np.asarray(
                camera.get_float_texture(
                    "Position"
                )
            )

            segmentation = np.asarray(
                camera.get_uint32_texture(
                    "Segmentation"
                )
            )

            actor_seg = (
                segmentation[..., 1]
                .astype(
                    np.int64
                )
            )

            valid = (
                position[..., 3]
                > 0
            )

            object_mask = np.isin(
                actor_seg,
                object_link_ids,
            )

            keep = (
                valid
                & object_mask
            )

            points_camera = (
                position[..., :3][
                    keep
                ]
                .astype(
                    np.float64
                )
            )

            actor_ids = (
                actor_seg[
                    keep
                ]
            )

            if (
                len(
                    points_camera
                )
                == 0
            ):
                fail(
                    "object_not_visible"
                )

            model_matrix = np.asarray(
                camera.get_model_matrix(),
                dtype=np.float64,
            )

            R = (
                model_matrix[
                    :3,
                    :3
                ]
            )

            t = (
                model_matrix[
                    :3,
                    3
                ]
            )

            points_world = (
                points_camera
                @ R.T
                + t
            )

            target_mask = (
                actor_ids
                == target_actor_id
            )

            visible_target_points = int(
                target_mask.sum()
            )

            if (
                visible_target_points
                == 0
            ):
                fail(
                    "target_not_visible"
                )

            seed_seq = (
                np.random.SeedSequence(
                    [
                        trial_seed,
                        int(
                            sample_index
                        ),
                    ]
                )
            )

            sample_rng = (
                np.random.default_rng(
                    seed_seq
                )
            )

            replace = (
                len(
                    points_world
                )
                < NUM_POINTS
            )

            ids = (
                sample_rng.choice(
                    len(
                        points_world
                    ),
                    NUM_POINTS,
                    replace=replace,
                )
            )

            pos = (
                points_world[
                    ids
                ]
                .astype(
                    np.float32
                )
            )

            mask = (
                target_mask[
                    ids
                ]
                .astype(
                    np.float32
                )
            )

            sampled_target_points = int(
                mask.sum()
            )

            if (
                sampled_target_points
                == 0
            ):
                fail(
                    "target_not_sampled",

                    visible_target_points=(
                        visible_target_points
                    ),
                )

            return (
                pos,
                mask,
                rgb,
                {
                    "visible_target_points":
                        visible_target_points,

                    "sampled_target_points":
                        sampled_target_points,

                    # 仅供初始二指抓取姿态适配使用。
                    # 仍然只是当前单视角可见 target-link 点。
                    "visible_target_points_world":
                        points_world[target_mask].copy(),
                },
            )

        # ----------------------------------------------------
        # FlowBot inference
        # ----------------------------------------------------

        def predict_flow(
            pos,
            mask,
            ee_center,
        ):
            xyz = (
                pos
                - pos.mean(
                    axis=0,
                    keepdims=True,
                )
            )

            max_abs = float(
                np.abs(
                    xyz
                ).max()
            )

            if (
                not np.isfinite(
                    max_abs
                )
                or max_abs <= 1e-8
            ):
                fail(
                    "invalid_pointcloud_scale"
                )

            xyz = (
                xyz
                * (
                    0.999999
                    / max_abs
                )
            ).astype(
                np.float32
            )

            with torch.no_grad():
                pred = (
                    model.predict(
                        torch.from_numpy(
                            xyz
                        ).to(
                            device
                        ),

                        torch.from_numpy(
                            mask
                        )
                        .float()
                        .to(
                            device
                        ),
                    )
                )

            pred = (
                pred
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            if (
                pred.shape
                != pos.shape
            ):
                raise RuntimeError(
                    f"Flow prediction shape 异常: "
                    f"{pred.shape}, expected {pos.shape}"
                )

            if not np.all(
                np.isfinite(
                    pred
                )
            ):
                fail(
                    "invalid_flow"
                )

            flow_norm = np.linalg.norm(
                pred,
                axis=1,
            )

            dist = np.linalg.norm(
                pos
                - ee_center[
                    None,
                    :
                ],
                axis=1,
            )

            score = (
                flow_norm
                / (
                    dist
                    + 1e-7
                )
            )

            if not np.any(
                np.isfinite(
                    score
                )
            ):
                fail(
                    "invalid_flow_score"
                )

            idx = int(
                np.nanargmax(
                    score
                )
            )

            flow = pred[
                idx
            ]

            norm = float(
                np.linalg.norm(
                    flow
                )
            )

            if (
                not np.isfinite(
                    norm
                )
                or norm <= 1e-8
            ):
                fail(
                    "zero_flow"
                )

            direction = (
                flow
                / norm
            )

            target_values = (
                flow_norm[
                    mask > 0.5
                ]
            )

            target_mean_flow = (
                float(
                    target_values.mean()
                )
                if len(
                    target_values
                ) > 0
                else None
            )

            return {
                "point":
                    pos[
                        idx
                    ].copy(),

                "direction":
                    direction.copy(),

                "score":
                    float(
                        score[
                            idx
                        ]
                    ),

                "is_target":
                    bool(
                        mask[
                            idx
                        ] > 0.5
                    ),

                "target_points":
                    int(
                        mask.sum()
                    ),

                "target_mean_flow":
                    target_mean_flow,
            }

        # ----------------------------------------------------
        # video
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Smooth video recorder
        #
        # Physics = 500 Hz.
        # 每 16 physics steps 一帧：
        # 500 / 16 = 31.25 FPS
        #
        # 输出按 30 FPS 编码。
        # ----------------------------------------------------

        video_path = (
            out_dir
            / "trial.mp4"
        )

        def add_video_frame(
            rgb,
        ):
            nonlocal video_writer

            if not save_video:
                return

            if imageio is None:
                raise RuntimeError(
                    "录像需要 imageio + imageio-ffmpeg"
                )

            image = (
                np.clip(
                    rgb,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(
                np.uint8
            )

            if video_writer is None:
                video_writer = (
                    imageio.get_writer(
                        str(video_path),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        pixelformat="yuv420p",
                    )
                )

            video_writer.append_data(
                image
            )

        def add_current_video_frame():
            if not save_video:
                return

            scene.update_render()
            camera.take_picture()

            rgba_now = np.asarray(
                camera.get_float_texture(
                    "Color"
                )
            )

            rgb_now = np.clip(
                rgba_now[..., :3],
                0.0,
                1.0,
            ).astype(
                np.float32
            )

            add_video_frame(
                rgb_now
            )

        if save_video:
            panda.set_video_step_callback(
                add_current_video_frame,
                every_n_steps=16,
            )

        # ----------------------------------------------------
        # Initial FlowBot decision
        # ----------------------------------------------------

        pos, mask, rgb, obs_info = (
            capture_observation(
                0
            )
        )

        add_video_frame(
            rgb
        )

        initial_ee = (
            panda.get_grasp_center()
        )

        initial_prediction = (
            predict_flow(
                pos,
                mask,
                initial_ee,
            )
        )

        contact_point = np.asarray(
            initial_prediction[
                "point"
            ],
            dtype=np.float64,
        )

        initial_direction = np.asarray(
            initial_prediction[
                "direction"
            ],
            dtype=np.float64,
        )

        initial_contact_distance = float(
            np.linalg.norm(
                contact_point
                - initial_ee
            )
        )

        metrics.update(
            {
                "initial_contact_point":
                    contact_point.copy(),

                "initial_pull_direction":
                    initial_direction.copy(),

                "initial_contact_is_target":
                    bool(
                        initial_prediction[
                            "is_target"
                        ]
                    ),

                "initial_contact_distance_m":
                    initial_contact_distance,

                "initial_target_points":
                    int(
                        initial_prediction[
                            "target_points"
                        ]
                    ),

                "initial_visible_target_points":
                    int(
                        obs_info[
                            "visible_target_points"
                        ]
                    ),
            }
        )

        print("=" * 90)
        print(
            "FlowBot3D clean trial"
        )
        print("=" * 90)
        print(
            "shape:",
            shape_id,
        )
        print(
            "target:",
            target_link_name,
        )
        print(
            "requested initial:",
            initial_ratio,
        )
        print(
            "actual initial:",
            get_progress(),
        )
        print(
            "contact:",
            contact_point,
        )
        print(
            "contact is target:",
            initial_prediction[
                "is_target"
            ],
        )
        print(
            "initial contact distance:",
            initial_contact_distance,
        )
        print(
            "initial flow:",
            initial_direction,
        )

        # 如果模型最终选择的最大 score 点根本不在目标 link，
        # 不能使用目标 link drive 将它“救回来”。
        if not initial_prediction[
            "is_target"
        ]:
            fail(
                "contact_not_on_target",

                predicted_contact_point=(
                    contact_point
                ),

                initial_contact_distance_m=(
                    initial_contact_distance
                ),
            )

        # ----------------------------------------------------
        # Physical two-finger grasp pose
        # ----------------------------------------------------

        # target joint 在抓取建立之前保持规定的随机初始开度。
        target_joint.set_drive_property(
            stiffness=5000,
            damping=500,
        )
        target_joint.set_drive_target(float(target_q))
        target_joint.set_drive_velocity_target(0.0)

        panda.open_gripper()
        panda.wait(150)

        try:
            grasp_pose_info = build_twofinger_grasp_pose(
                contact_point,
                initial_direction,
                obs_info["visible_target_points_world"],
                camera_pose_world[:3, 3],
                current_grasp_rotation=(
                    panda.get_grasp_pose_matrix()[:3, :3]
                ),
            )
        except GraspPoseError as exc:
            fail(
                "grasp_pose_invalid",
                grasp_pose_error=str(exc),
            )

        T_grasp = np.asarray(
            grasp_pose_info["T_world_grasp"],
            dtype=np.float64,
        )
        approach_axis = np.asarray(
            grasp_pose_info["approach_axis_world"],
            dtype=np.float64,
        )

        # 抓取中心最终直接对准 FlowBot contact point。
        # 这里不使用 2mm/1cm/5cm 这类“抓取成功距离容差”；
        # 是否抓住最终由 finger-target 真实接触判断。
        T_pregrasp = T_grasp.copy()
        T_pregrasp[:3, 3] = (
            contact_point
            - PREGRASP_DISTANCE * approach_axis
        )

        pregrasp_diag = panda.move_grasp_pose_to(
            T_pregrasp,
            APPROACH_SIM_STEPS,
            position_tolerance=0.005,
            rotation_tolerance=0.03,
        )
        panda.wait(50)

        final_approach_diag = panda.move_grasp_pose_to(
            T_grasp,
            FINAL_APPROACH_SIM_STEPS,
            position_tolerance=0.005,
            rotation_tolerance=0.03,
        )
        panda.wait(50)

        final_grasp_center = panda.get_grasp_center()
        approach_position_error = float(
            np.linalg.norm(final_grasp_center - contact_point)
        )
        progress_after_approach = float(get_progress())

        metrics.update(
            {
                "grasp_pose": T_grasp.copy(),
                "grasp_pose_info": grasp_pose_info,
                "pregrasp_diagnostics": pregrasp_diag,
                "final_approach_diagnostics": final_approach_diag,
                "approach_position_error_m": approach_position_error,
                "progress_after_approach": progress_after_approach,
            }
        )

        print("two-finger grasp pose generated")
        print("approach position error:", approach_position_error)
        print("progress after approach:", progress_after_approach)

        if abs(
            progress_after_approach - initial_ratio
        ) > START_PROGRESS_TOLERANCE:
            fail(
                "object_disturbed_before_grasp",
                stage="after_approach",
                progress_after_approach=progress_after_approach,
            )

        # ----------------------------------------------------
        # REAL two-finger close + contact validation
        # ----------------------------------------------------

        panda.close_gripper()

        grasp_diag = monitor_grasp_establishment(
            panda,
            target_link,
            settle_steps=GRASP_SETTLE_STEPS,
            tail_steps=GRASP_TAIL_STEPS,
            min_bilateral_fraction=GRASP_MIN_BILATERAL_FRACTION,
        )

        metrics["grasp_diagnostics"] = grasp_diag
        metrics["finger_qpos_after_close"] = panda.get_finger_qpos()

        print(
            "physical grasp: ",
            "firm=",
            grasp_diag["firm_grasp"],
            "bilateral_fraction=",
            grasp_diag["bilateral_fraction"],
            "left_fraction=",
            grasp_diag["left_contact_fraction"],
            "right_fraction=",
            grasp_diag["right_contact_fraction"],
        )

        if not grasp_diag["firm_grasp"]:
            fail(
                "grasp_failed",
                grasp_diagnostics=grasp_diag,
            )

        # 这一刻开始，抓取成功已经确定。后面即使开门失败，
        # grasp_success 仍保持 True，用于学长要求的条件成功率。
        metrics["grasp_success"] = True

        pre_pull_progress = float(get_progress())
        metrics["pre_pull_progress"] = pre_pull_progress

        if abs(
            pre_pull_progress - initial_ratio
        ) > START_PROGRESS_TOLERANCE:
            fail(
                "object_disturbed_before_pull",
                pre_pull_progress=pre_pull_progress,
            )

        # 正式开始物理操作。没有任何 hand-object / finger-object drive。
        target_joint.set_drive_property(
            stiffness=0,
            damping=10,
        )
        target_joint.set_drive_velocity_target(0.0)

        metrics["operation_attempted_after_grasp"] = True

        history = []
        success = False

        # ----------------------------------------------------
        # FlowBot closed-loop physical pulling
        # ----------------------------------------------------

        for step in range(MAX_PULL_STEPS):
            before = float(get_progress())

            if before >= SUCCESS_RATIO:
                success = True
                break

            pos, mask, rgb, obs_info = capture_observation(
                1000 + step
            )
            add_video_frame(rgb)

            ee_center = panda.get_grasp_center()
            prediction = predict_flow(
                pos,
                mask,
                ee_center,
            )

            direction = np.asarray(
                prediction["direction"],
                dtype=np.float64,
            )

            # 持续保持两指闭合 actuator；物体受到的力只能来自
            # collision + contact impulse + friction。
            panda.keep_gripper_closed()

            pull_diag = panda.move_grasp_point_by(
                PULL_DISTANCE * direction,
                PULL_SIM_STEPS,
                control_tolerance=0.005,
            )

            # pull 后短窗口检查真实接触是否完全丢失。
            contact_diag = monitor_post_pull_contact(
                panda,
                target_link,
                steps=POST_PULL_CONTACT_STEPS,
            )

            after = float(get_progress())
            if not np.isfinite(after):
                fail("invalid_target_state")

            record = {
                "step": step,
                "before": before,
                "after": after,
                "direction": direction.copy(),
                "ee_center_before": ee_center.copy(),
                "target_points": int(prediction["target_points"]),
                "direction_source_is_target": bool(
                    prediction["is_target"]
                ),
                "visible_target_points": int(
                    obs_info["visible_target_points"]
                ),
                "pull_diagnostics": pull_diag,
                "physical_contact_after_pull": contact_diag,
            }
            history.append(record)

            print(
                f"[{step:02d}] "
                f"{before:.4f} -> {after:.4f} | "
                f"dir=[{direction[0]:+.3f}, "
                f"{direction[1]:+.3f}, "
                f"{direction[2]:+.3f}] | "
                f"contact_any="
                f"{contact_diag['any_contact_fraction']:.2f}"
            )

            if after >= SUCCESS_RATIO:
                success = True
                break

            if contact_diag["grasp_lost"]:
                fail(
                    "grasp_lost",
                    history=history,
                    grasp_lost_at_step=step,
                    final_progress=after,
                )

        final_progress = float(
            get_progress()
        )

        if (
            final_progress
            >= SUCCESS_RATIO
        ):
            success = True

        metrics.update(
            {
                "history":
                    history,

                "final_progress":
                    final_progress,

                "success":
                    bool(
                        success
                    ),

                "failure_reason":
                    (
                        None
                        if success
                        else "max_steps"
                    ),
            }
        )

        # 最后一帧只用于视频；
        # 如果最后一帧目标不可见，不改变已经完成的任务结果。
        if save_video:
            try:
                (
                    _,
                    _,
                    final_rgb,
                    _,
                ) = capture_observation(
                    99999
                )

                add_video_frame(
                    final_rgb
                )
            except TaskFailure:
                pass

        return metrics

    except TaskFailure as task_failure:
        metrics.update(
            task_failure.data
        )

        metrics[
            "success"
        ] = False

        metrics[
            "failure_reason"
        ] = (
            task_failure.reason
        )

        if (
            metrics.get(
                "final_progress"
            )
            is None
        ):
            metrics[
                "final_progress"
            ] = safe_progress()

        return metrics

    finally:
        try:
            panda.clear_video_step_callback()
        except Exception:
            pass

        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass


# ============================================================
# CLI main
# ============================================================

def main():
    args = parse_args()

    out_root = (
        Path(
            args.output_root
        )
        .expanduser()
        .resolve()
    )

    out_dir = (
        out_root
        / (
            f"{args.shape_id}_"
            f"{args.target_link}_"
            f"seed_{int(args.trial_seed)}"
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        out_dir
        / "result.json"
    )

    # 避免旧文件让 runner 误判本轮成功。
    if result_path.exists():
        result_path.unlink()

    try:
        result = execute_trial(
            args,
            out_dir,
        )

        write_json(
            result_path,
            result,
        )

        print()
        print("=" * 90)
        print(
            "TRIAL RESULT"
        )
        print("=" * 90)
        print(
            "grasp_success:",
            result.get(
                "grasp_success",
                False,
            ),
        )
        print(
            "success:",
            result[
                "success"
            ],
        )
        print(
            "failure_reason:",
            result.get(
                "failure_reason"
            ),
        )
        print(
            "final_progress:",
            result.get(
                "final_progress"
            ),
        )
        print(
            "result:",
            result_path,
        )
        print("=" * 90)

        return 0

    except Exception:
        print()
        print("=" * 90)
        print(
            "IMPLEMENTATION ERROR"
        )
        print("=" * 90)

        traceback.print_exc()

        # implementation error 不写 result.json，
        # 防止正式 benchmark 把工程错误统计成 baseline failure。
        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
