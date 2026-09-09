import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation



HOME = Path.home()

PARTNET_ROOT = (
    HOME
    / "robot_baselines"
    / "data"
    / "partnet-mobility"
)

PANDA_URDF = (
    HOME
    / "robot_baselines"
    / "common_env"
    / "assets"
    / "panda_articubot"
    / "panda.urdf"
)

OBJECT_SCALE = 0.75


class ExecutionError(RuntimeError):
    pass


class TrajectoryRobot:
    """
    仅用于：

        OMPL 7D joint trajectory
        -> SAPIEN Panda

    不负责：
        IK
        Jacobian
        Where2Act推理
        motion planning

    因而与现有 panda_controller.py 完全解耦。
    """

    def __init__(
        self,
        env,
        urdf,
        material=None,
        open_gripper=False,
    ):
        self.env = env
        self.scene = env.scene

        loader = (
            self.scene
            .create_urdf_loader()
        )

        loader.fix_root_link = True

        self.robot = loader.load(
            str(urdf)
        )

        if self.robot is None:
            raise ExecutionError(
                f"Panda URDF加载失败: {urdf}"
            )

        if self.robot.dof != 9:
            raise ExecutionError(
                "Panda应为9自由度，"
                f"实际={self.robot.dof}"
            )

        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        self.links = list(
            self.robot.get_links()
        )

        self.link_dict = {
            link.get_name(): link
            for link in self.links
        }

        required_links = [
            "panda_hand",
            "panda_grasptarget",
        ]

        missing_links = [
            name
            for name in required_links
            if name not in self.link_dict
        ]

        if missing_links:
            raise ExecutionError(
                "Panda URDF缺少links: "
                f"{missing_links}"
            )

        # ----------------------------------------------------
        # Active joints
        # ----------------------------------------------------

        active_joints = list(
            self.robot
            .get_active_joints()
        )

        self.arm_joints = [
            joint
            for joint in active_joints
            if not (
                joint
                .get_name()
                .startswith(
                    "panda_finger_joint"
                )
            )
        ]

        self.gripper_joints = [
            joint
            for joint in active_joints
            if (
                joint
                .get_name()
                .startswith(
                    "panda_finger_joint"
                )
            )
        ]

        if len(
            self.arm_joints
        ) != 7:
            raise ExecutionError(
                "Panda arm joints应为7，"
                f"实际={len(self.arm_joints)}"
            )

        if len(
            self.gripper_joints
        ) != 2:
            raise ExecutionError(
                "Panda finger joints应为2，"
                f"实际={len(self.gripper_joints)}"
            )

        # ----------------------------------------------------
        # 强制检查joint顺序
        #
        # 因为OMPL trajectory的7列就是：
        #
        # panda_joint1 ... panda_joint7
        # ----------------------------------------------------

        arm_names = [
            joint.get_name()
            for joint
            in self.arm_joints
        ]

        expected_arm_names = [
            f"panda_joint{i}"
            for i in range(
                1,
                8,
            )
        ]

        if arm_names != expected_arm_names:
            raise ExecutionError(
                "SAPIEN Panda arm joint顺序"
                "和OMPL假设不一致：\n"
                f"actual={arm_names}\n"
                f"expected={expected_arm_names}"
            )

        finger_names = [
            joint.get_name()
            for joint
            in self.gripper_joints
        ]

        expected_finger_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        if (
            finger_names
            != expected_finger_names
        ):
            raise ExecutionError(
                "SAPIEN Panda finger顺序异常："
                f"{finger_names}"
            )

        # ----------------------------------------------------
        # PD参数
        #
        # 与之前验证过的SAPIEN Panda controller一致
        # ----------------------------------------------------

        for joint in self.arm_joints:

            joint.set_drive_property(
                stiffness=1000,
                damping=400,
            )

            joint.set_drive_velocity_target(
                0.0
            )

        for joint in (
            self.gripper_joints
        ):

            joint.set_drive_property(
                stiffness=200,
                damping=60,
            )

            joint.set_drive_velocity_target(
                0.0
            )

        if open_gripper:

            for joint in (
                self.gripper_joints
            ):

                joint.set_drive_target(
                    0.04
                )


class MinimalEnv:
    def __init__(
        self,
        scene,
    ):
        self.scene = scene

    def step(
        self,
    ):
        self.scene.step()


def load_catalog_row(
    path,
    shape_id,
    target_link,
):
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    with path.open() as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(
                line
            )

            link = (
                row.get("link_name")
                or row.get("target_link")
                or row.get("link")
            )

            if (
                str(row["shape_id"])
                == str(shape_id)
                and str(link)
                == str(target_link)
            ):

                return row

    raise ExecutionError(
        "pose catalog找不到 "
        f"{shape_id}/{target_link}"
    )


def create_scene():

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

    config = sapien.SceneConfig()

    config.gravity = [
        0,
        0,
        -9.81,
    ]

    config.solver_iterations = 20
    config.enable_pcm = False
    config.sleep_threshold = 0.0

    scene = engine.create_scene(
        config=config
    )

    scene.set_timestep(
        1.0 / 500.0
    )

    return (
        engine,
        renderer,
        scene,
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
        root
    )


def initialize_object(
    scene,
    shape_id,
    target_link_name,
    initial_ratio,
):

    loader = (
        scene.create_urdf_loader()
    )

    loader.fix_root_link = True
    loader.scale = OBJECT_SCALE

    urdf = find_object_urdf(
        shape_id
    )

    obj = loader.load(
        str(urdf)
    )

    if obj is None:
        raise ExecutionError(
            "SAPIEN加载PartNet失败"
        )

    obj.set_root_pose(
        sapien.Pose()
    )

    active_joints = list(
        obj.get_active_joints()
    )

    target_joint = None
    target_index = None

    qpos = np.zeros(
        len(active_joints),
        dtype=np.float64,
    )

    for i, joint in enumerate(
        active_joints
    ):

        limits = np.asarray(
            joint.get_limits(),
            dtype=np.float64,
        )

        lower = float(
            limits[0, 0]
        )

        upper = float(
            limits[0, 1]
        )

        child = (
            joint
            .get_child_link()
        )

        if np.isfinite(
            lower
        ):
            q = lower
        else:
            q = 0.0

        if (
            child.get_name()
            == str(
                target_link_name
            )
        ):

            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or upper <= lower
            ):
                raise ExecutionError(
                    "target joint范围非法"
                )

            q = (
                lower
                + float(
                    initial_ratio
                )
                * (
                    upper - lower
                )
            )

            target_joint = joint
            target_index = i

        qpos[i] = q

    if target_joint is None:
        raise ExecutionError(
            "找不到target joint"
        )

    obj.set_qpos(
        qpos
    )

    for i, joint in enumerate(
        active_joints
    ):

        try:

            joint.set_drive_property(
                stiffness=5000,
                damping=500,
            )

            joint.set_drive_target(
                float(
                    qpos[i]
                )
            )

            joint.set_drive_velocity_target(
                0.0
            )

        except Exception:
            pass

    limits = np.asarray(
        target_joint.get_limits(),
        dtype=np.float64,
    )

    q_min = float(
        limits[0, 0]
    )

    q_max = float(
        limits[0, 1]
    )

    def get_progress():

        q = float(
            obj.get_qpos()[
                target_index
            ]
        )

        return (
            q - q_min
        ) / (
            q_max - q_min
        )

    return (
        obj,
        target_joint,
        get_progress,
    )


def wait(
    panda,
    scene,
    n,
):

    for _ in range(
        int(n)
    ):

        passive = (
            panda.robot
            .compute_passive_force()
        )

        panda.robot.set_qf(
            passive
        )

        scene.step()


def set_joint_target(
    panda,
    q_target,
):
    """
    给Panda下发一个7D arm joint target。

    两根手指在pre-grasp阶段始终保持打开。
    """

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    ).reshape(7)

    for i, joint in enumerate(
        panda.arm_joints
    ):

        limits = np.asarray(
            joint.get_limits()[0],
            dtype=np.float64,
        )

        lo = float(
            limits[0]
        )

        hi = float(
            limits[1]
        )

        target = float(
            q_target[i]
        )

        if (
            np.isfinite(lo)
            and target < lo - 1e-5
        ):
            raise ExecutionError(
                f"joint {i} target "
                f"{target} < lower {lo}"
            )

        if (
            np.isfinite(hi)
            and target > hi + 1e-5
        ):
            raise ExecutionError(
                f"joint {i} target "
                f"{target} > upper {hi}"
            )

        joint.set_drive_velocity_target(
            0.0
        )

        joint.set_drive_target(
            target
        )

    # pre-grasp期间夹爪保持打开。
    #
    # 不直接访问gripper_joints，因为正式使用的
    # PandaTwoFingerController通过统一API管理手指。
    panda.open_gripper()


def simulation_step(
    panda,
    scene,
):
    passive = (
        panda.robot
        .compute_passive_force()
    )

    panda.robot.set_qf(
        passive
    )

    scene.step()


def track_joint_target(
    panda,
    scene,
    q_target,
    *,
    min_steps,
    max_steps,
    tolerance_norm,
    tolerance_maxabs,
):
    """
    关键：

    不再“固定跑N步就进入下一个waypoint”。

    而是：
        下发target
        -> 至少运行min_steps
        -> 检查实际关节误差
        -> 达标才结束
        -> 最多max_steps

    返回最终tracking信息。
    """

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    ).reshape(7)

    set_joint_target(
        panda,
        q_target,
    )

    best_norm = float(
        "inf"
    )

    best_maxabs = float(
        "inf"
    )

    for step_index in range(
        int(max_steps)
    ):

        simulation_step(
            panda,
            scene,
        )

        actual_q = np.asarray(
            panda.robot.get_qpos(),
            dtype=np.float64,
        )[:7]

        error = (
            actual_q
            - q_target
        )

        error_norm = float(
            np.linalg.norm(
                error
            )
        )

        error_maxabs = float(
            np.max(
                np.abs(
                    error
                )
            )
        )

        best_norm = min(
            best_norm,
            error_norm,
        )

        best_maxabs = min(
            best_maxabs,
            error_maxabs,
        )

        if (
            step_index + 1
            >= int(min_steps)
            and
            error_norm
            <= float(
                tolerance_norm
            )
            and
            error_maxabs
            <= float(
                tolerance_maxabs
            )
        ):

            return {
                "converged":
                    True,

                "steps":
                    step_index + 1,

                "error_norm":
                    error_norm,

                "error_maxabs":
                    error_maxabs,

                "actual_q":
                    actual_q,
            }

    actual_q = np.asarray(
        panda.robot.get_qpos(),
        dtype=np.float64,
    )[:7]

    error = (
        actual_q
        - q_target
    )

    error_norm = float(
        np.linalg.norm(
            error
        )
    )

    error_maxabs = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    return {
        "converged":
            False,

        "steps":
            int(
                max_steps
            ),

        "error_norm":
            error_norm,

        "error_maxabs":
            error_maxabs,

        "best_norm":
            best_norm,

        "best_maxabs":
            best_maxabs,

        "actual_q":
            actual_q,
    }


def execute_joint_trajectory(
    panda,
    scene,
    trajectory,
    steps_per_waypoint,
    max_steps_per_waypoint=500,
    waypoint_tolerance=0.02,
    waypoint_maxabs_tolerance=0.015,
):
    """
    Adaptive execution of an OMPL joint path.

    OMPL给的是几何路径，不是带时间参数的动力学轨迹。

    因此不能假设：
        每个waypoint固定30ms。

    正确做法是确保机械臂实际跟踪到当前waypoint附近，
    再切换下一个。
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 7
    ):
        raise ExecutionError(
            "trajectory必须为N×7，"
            f"当前={trajectory.shape}"
        )

    max_converged_error = 0.0
    max_converged_maxabs = 0.0

    total_sim_steps = 0
    max_steps_used = 0

    for waypoint_index, q_target in enumerate(
        trajectory
    ):

        tracking = track_joint_target(
            panda,
            scene,
            q_target,
            min_steps=(
                steps_per_waypoint
            ),
            max_steps=(
                max_steps_per_waypoint
            ),
            tolerance_norm=(
                waypoint_tolerance
            ),
            tolerance_maxabs=(
                waypoint_maxabs_tolerance
            ),
        )

        total_sim_steps += int(
            tracking[
                "steps"
            ]
        )

        max_steps_used = max(
            max_steps_used,
            int(
                tracking[
                    "steps"
                ]
            ),
        )

        if not tracking[
            "converged"
        ]:

            raise ExecutionError(
                "waypoint tracking无法收敛: "
                f"waypoint={waypoint_index}, "
                f"steps={tracking['steps']}, "
                f"norm={tracking['error_norm']:.6f}, "
                f"maxabs={tracking['error_maxabs']:.6f}, "
                f"best_norm={tracking.get('best_norm')}"
            )

        max_converged_error = max(
            max_converged_error,
            float(
                tracking[
                    "error_norm"
                ]
            ),
        )

        max_converged_maxabs = max(
            max_converged_maxabs,
            float(
                tracking[
                    "error_maxabs"
                ]
            ),
        )

        if (
            waypoint_index % 10 == 0
            or waypoint_index
            == len(
                trajectory
            ) - 1
        ):

            print(
                f"[{waypoint_index:03d}/"
                f"{len(trajectory)-1:03d}] "
                f"error="
                f"{tracking['error_norm']:.6f} "
                f"maxabs="
                f"{tracking['error_maxabs']:.6f} "
                f"steps="
                f"{tracking['steps']}"
            )

    print()
    print(
        "trajectory total simulation steps:",
        total_sim_steps,
    )

    print(
        "max steps used by one waypoint:",
        max_steps_used,
    )

    print(
        "max converged waypoint error:",
        max_converged_error,
    )

    print(
        "max converged waypoint maxabs:",
        max_converged_maxabs,
    )

    return float(
        max_converged_error
    )



def run(
    args,
):

    request_path = (
        Path(
            args.request
        )
        .expanduser()
        .resolve()
    )

    trajectory_path = (
        Path(
            args.trajectory
        )
        .expanduser()
        .resolve()
    )

    request = json.loads(
        request_path.read_text()
    )

    trajectory = np.load(
        trajectory_path
    ).astype(
        np.float64
    )

    shape_id = str(
        request[
            "shape_id"
        ]
    )

    target_link = str(
        request[
            "target_link"
        ]
    )

    initial_ratio = float(
        request[
            "initial_ratio"
        ]
    )

    T_goal = np.asarray(
        request[
            "pregrasp_pose_world"
        ],
        dtype=np.float64,
    )

    row = load_catalog_row(
        args.pose_catalog,
        shape_id,
        target_link,
    )

    robot_qpos = np.asarray(
        row[
            "robot_initial_qpos"
        ],
        dtype=np.float64,
    )

    base_pose = np.asarray(
        row[
            "base_pose"
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 100)
    print(
        "OMPL TRAJECTORY -> SAPIEN PANDA"
    )
    print("=" * 100)

    print(
        "shape:",
        shape_id,
    )

    print(
        "target:",
        target_link,
    )

    print(
        "initial ratio:",
        initial_ratio,
    )

    print(
        "trajectory:",
        trajectory.shape,
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 7
    ):
        raise ExecutionError(
            "trajectory shape错误"
        )

    start_sync_error = float(
        np.linalg.norm(
            trajectory[0]
            - robot_qpos[:7]
        )
    )

    print(
        "planner/SAPIEN start q error:",
        start_sync_error,
    )

    if start_sync_error > 1e-3:
        raise ExecutionError(
            "planner trajectory起点"
            "和benchmark Panda初始qpos不一致"
        )

    # ========================================================
    # Scene
    # ========================================================

    engine, renderer, scene = (
        create_scene()
    )

    # ========================================================
    # Object
    # ========================================================

    (
        obj,
        target_joint,
        get_progress,
    ) = initialize_object(
        scene,
        shape_id,
        target_link,
        initial_ratio,
    )

    initial_progress_actual = float(
        get_progress()
    )

    print(
        "SAPIEN object actual initial progress:",
        initial_progress_actual,
    )

    # ========================================================
    # Panda
    # ========================================================

    env = MinimalEnv(
        scene
    )

    panda = TrajectoryRobot(
        env,
        str(
            PANDA_URDF
        ),
        None,
        open_gripper=False,
    )

    if (
        "panda_grasptarget"
        not in panda.link_dict
    ):
        raise ExecutionError(
            "找不到 panda_grasptarget"
        )

    grasp_link = (
        panda.link_dict[
            "panda_grasptarget"
        ]
    )

    # benchmark：
    # base_pose = [x, y, yaw, z]
    x = float(
        base_pose[0]
    )

    y = float(
        base_pose[1]
    )

    yaw = float(
        base_pose[2]
    )

    z = float(
        base_pose[3]
    )

    # SAPIEN quaternion:
    # [w,x,y,z]
    base_quat = [
        math.cos(
            yaw / 2.0
        ),
        0.0,
        0.0,
        math.sin(
            yaw / 2.0
        ),
    ]

    panda.robot.set_root_pose(
        sapien.Pose(
            [
                x,
                y,
                z,
            ],
            base_quat,
        )
    )

    panda.robot.set_qpos(
        robot_qpos
    )

    # 初始PD目标与qpos一致
    for i, joint in enumerate(
        panda.robot.get_active_joints()
    ):

        joint.set_drive_velocity_target(
            0.0
        )

        joint.set_drive_target(
            float(
                robot_qpos[i]
            )
        )

    wait(
        panda,
        scene,
        100,
    )

    start_q_actual = np.asarray(
        panda.robot.get_qpos(),
        dtype=np.float64,
    )[:7]

    print()
    print(
        "SAPIEN Panda start q:",
        start_q_actual,
    )

    print(
        "start tracking error:",
        float(
            np.linalg.norm(
                start_q_actual
                - trajectory[0]
            )
        ),
    )

    initial_grasp_pose = (
        grasp_link
        .get_pose()
        .to_transformation_matrix()
    )

    print(
        "initial grasptarget position:",
        initial_grasp_pose[
            :3,
            3
        ],
    )

    # ========================================================
    # Execute OMPL trajectory
    # ========================================================

    print()
    print("-" * 100)
    print(
        "EXECUTING 7D JOINT TRAJECTORY"
    )
    print("-" * 100)

    max_waypoint_error = (
        execute_joint_trajectory(
            panda,
            scene,
            trajectory,
            args.steps_per_waypoint,
            max_steps_per_waypoint=500,
            waypoint_tolerance=0.02,
            waypoint_maxabs_tolerance=0.015,
        )
    )

    # 最终多稳定一段
    final_q_target = (
        trajectory[
            -1
        ]
    )

    for i, joint in enumerate(
        panda.arm_joints
    ):

        joint.set_drive_velocity_target(
            0.0
        )

        joint.set_drive_target(
            float(
                final_q_target[i]
            )
        )

    final_tracking = track_joint_target(
        panda,
        scene,
        final_q_target,
        min_steps=50,
        max_steps=(
            args.final_settle_steps
        ),
        tolerance_norm=0.005,
        tolerance_maxabs=0.003,
    )

    print()
    print(
        "final adaptive tracking:",
        final_tracking[
            "converged"
        ],
    )

    print(
        "final adaptive steps:",
        final_tracking[
            "steps"
        ],
    )

    print(
        "final adaptive error:",
        final_tracking[
            "error_norm"
        ],
    )

    if not final_tracking[
        "converged"
    ]:

        raise ExecutionError(
            "最终OMPL goal无法精确跟踪: "
            f"error={final_tracking['error_norm']}, "
            f"maxabs={final_tracking['error_maxabs']}"
        )

    # ========================================================
    # Validation
    # ========================================================

    final_q_actual = np.asarray(
        panda.robot.get_qpos(),
        dtype=np.float64,
    )[:7]

    final_joint_error = float(
        np.linalg.norm(
            final_q_actual
            - final_q_target
        )
    )

    T_actual = (
        grasp_link
        .get_pose()
        .to_transformation_matrix()
    )

    position_error = float(
        np.linalg.norm(
            T_actual[
                :3,
                3
            ]
            - T_goal[
                :3,
                3
            ]
        )
    )

    rotation_error = float(
        np.linalg.norm(
            Rotation
            .from_matrix(
                T_goal[
                    :3,
                    :3
                ]
                @ T_actual[
                    :3,
                    :3
                ].T
            )
            .as_rotvec()
        )
    )

    final_progress = float(
        get_progress()
    )

    object_progress_drift = float(
        abs(
            final_progress
            - initial_progress_actual
        )
    )

    print()
    print("-" * 100)
    print(
        "SAPIEN EXECUTION RESULT"
    )
    print("-" * 100)

    print(
        "max waypoint joint error:",
        max_waypoint_error,
    )

    print(
        "final joint error:",
        final_joint_error,
    )

    print(
        "target pregrasp position:",
        T_goal[
            :3,
            3
        ],
    )

    print(
        "actual grasptarget position:",
        T_actual[
            :3,
            3
        ],
    )

    print(
        "grasptarget position error:",
        position_error,
    )

    print(
        "grasptarget rotation error:",
        rotation_error,
    )

    print(
        "object progress:",
        initial_progress_actual,
        "->",
        final_progress,
    )

    print(
        "object progress drift:",
        object_progress_drift,
    )

    # ========================================================
    # PASS条件
    #
    # 这里仍是工程smoke，不是最终benchmark阈值。
    # ========================================================

    failures = []

    if (
        final_joint_error
        > 0.05
    ):
        failures.append(
            "final_joint_tracking"
        )

    if (
        position_error
        > 0.02
    ):
        failures.append(
            "grasptarget_position"
        )

    if (
        rotation_error
        > 0.10
    ):
        failures.append(
            "grasptarget_rotation"
        )

    # pregrasp离物体8cm，
    # 正常不应该把target joint明显推动。
    if (
        object_progress_drift
        > 0.03
    ):
        failures.append(
            "unexpected_object_motion"
        )

    # ========================================================
    # Save
    # ========================================================

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "success":
            len(
                failures
            ) == 0,

        "failures":
            failures,

        "shape_id":
            shape_id,

        "target_link":
            target_link,

        "trajectory_shape":
            list(
                trajectory.shape
            ),

        "start_sync_error":
            start_sync_error,

        "max_waypoint_joint_error":
            max_waypoint_error,

        "final_joint_error":
            final_joint_error,

        "grasptarget_position_error":
            position_error,

        "grasptarget_rotation_error":
            rotation_error,

        "initial_progress":
            initial_progress_actual,

        "final_progress":
            final_progress,

        "object_progress_drift":
            object_progress_drift,

        "target_pregrasp_pose":
            T_goal.tolist(),

        "actual_pregrasp_pose":
            T_actual.tolist(),
    }

    with (
        output_dir
        / "result.json"
    ).open(
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "saved:",
        output_dir,
    )

    if failures:

        print()
        print("=" * 100)
        print(
            "SAPIEN TRAJECTORY EXECUTION: FAIL"
        )
        print(
            "failures:",
            failures,
        )
        print("=" * 100)

        raise SystemExit(
            2
        )

    print()
    print("=" * 100)
    print(
        "SAPIEN TRAJECTORY EXECUTION: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--request",
        required=True,
    )

    parser.add_argument(
        "--trajectory",
        required=True,
    )

    parser.add_argument(
        "--pose-catalog",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--steps-per-waypoint",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--final-settle-steps",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    try:

        run(
            args
        )

    except SystemExit:
        raise

    except Exception:

        print()
        print("=" * 100)
        print(
            "SAPIEN EXECUTION ERROR"
        )
        print("=" * 100)

        traceback.print_exc()

        raise SystemExit(
            1
        )
