import json
import math
import os
import shutil
import subprocess
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

WIDTH = 448
HEIGHT = 448
FOV_DEG = 35.0
NEAR = 0.1
FAR = 100.0

INITIAL_RATIO_MIN = 0.10
INITIAL_RATIO_MAX = 0.20

SUCCESS_RATIO = 0.40

# Where2Act官方pulling：
#
# start -> final = 5cm approach
# grasp
# final -> start = 5cm pull
#
# 完整Panda的grasptarget最终停在物体表面外5mm，
# 因此：
#
# contact target = contact - 5mm * approach
# pregrasp      = contact target - 5cm * approach
CONTACT_STANDOFF = 0.005
APPROACH_DISTANCE = 0.05
PREGRASP_DISTANCE = (
    CONTACT_STANDOFF
    + APPROACH_DISTANCE
)

class Where2ActRuntimeError(RuntimeError):
    pass


def jsonify(value):

    if isinstance(value, dict):
        return {
            str(k): jsonify(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            jsonify(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    return value


def save_json(
    path,
    data,
):

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as f:

        json.dump(
            jsonify(data),
            f,
            indent=2,
        )

    return path


def load_catalog_row(
    catalog_path,
    shape_id,
    target_link,
):

    path = (
        Path(catalog_path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open() as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            link = (
                row.get("link_name")
                or row.get("target_link")
                or row.get("link")
            )

            if (
                str(row.get("shape_id"))
                == str(shape_id)
                and str(link)
                == str(target_link)
            ):

                return row

    raise Where2ActRuntimeError(
        "POSE_CATALOG_TARGET_NOT_FOUND: "
        f"{shape_id}/{target_link}"
    )


def sample_initial_ratio(
    trial_seed,
):

    rng = np.random.default_rng(
        int(trial_seed)
    )

    return float(
        rng.uniform(
            INITIAL_RATIO_MIN,
            INITIAL_RATIO_MAX,
        )
    )


def create_scene():

    engine = sapien.Engine(
        0,
        0.001,
        0.005,
    )

    renderer = (
        sapien.VulkanRenderer(
            offscreen_only=True
        )
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
        f"PartNet URDF不存在: {root}"
    )


def initialize_object(
    scene,
    shape_id,
    target_link_name,
    initial_ratio=None,
    initial_object_qpos=None,
):
    exact_qpos = None
    if initial_object_qpos is not None:
        exact_qpos = np.asarray(
            initial_object_qpos,
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(exact_qpos)):
            raise Where2ActRuntimeError(
                "INITIAL_OBJECT_QPOS_NONFINITE"
            )
    else:
        initial_ratio = float(initial_ratio)
        if not (
            INITIAL_RATIO_MIN
            <= initial_ratio
            <= INITIAL_RATIO_MAX
        ):
            raise ValueError(
                "initial_ratio必须位于"
                "[0.10,0.20]"
            )

    loader = (
        scene.create_urdf_loader()
    )

    loader.fix_root_link = True
    loader.scale = OBJECT_SCALE

    urdf_path = find_object_urdf(
        shape_id
    )

    obj = loader.load(
        str(urdf_path)
    )

    if obj is None:
        raise Where2ActRuntimeError(
            "SAPIEN_OBJECT_LOAD_FAILED"
        )

    obj.set_root_pose(
        sapien.Pose()
    )

    active_joints = list(
        obj.get_active_joints()
    )

    target_joint = None
    target_link = None
    target_index = None

    qpos = np.zeros(
        len(active_joints),
        dtype=np.float64,
    )

    if (
        exact_qpos is not None
        and exact_qpos.shape != qpos.shape
    ):
        raise Where2ActRuntimeError(
            "INITIAL_OBJECT_QPOS_LENGTH_MISMATCH: "
            f"expected={len(active_joints)} "
            f"actual={len(exact_qpos)}"
        )

    for i, joint in enumerate(
        active_joints
    ):

        limits = np.asarray(
            joint.get_limits(),
            dtype=np.float64,
        )

        if (
            limits.ndim != 2
            or limits.shape[0] < 1
            or limits.shape[1] != 2
        ):
            raise Where2ActRuntimeError(
                "INVALID_JOINT_LIMIT_SHAPE"
            )

        lower = float(
            limits[0, 0]
        )

        upper = float(
            limits[0, 1]
        )

        child = (
            joint.get_child_link()
        )

        if exact_qpos is not None:
            q = float(exact_qpos[i])
            if np.isfinite(lower) and q < lower - 1e-5:
                raise Where2ActRuntimeError(
                    "INITIAL_OBJECT_QPOS_BELOW_LIMIT"
                )
            if np.isfinite(upper) and q > upper + 1e-5:
                raise Where2ActRuntimeError(
                    "INITIAL_OBJECT_QPOS_ABOVE_LIMIT"
                )
        elif np.isfinite(lower):
            q = lower
        else:
            q = 0.0

        if (
            child.get_name()
            == str(target_link_name)
        ):

            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or upper <= lower
            ):
                raise Where2ActRuntimeError(
                    "TARGET_JOINT_RANGE_INVALID"
                )

            if exact_qpos is None:
                q = (
                    lower
                    + initial_ratio
                    * (
                        upper - lower
                    )
                )

            target_joint = joint
            target_link = child
            target_index = i

        qpos[i] = q

    if target_joint is None:
        raise Where2ActRuntimeError(
            "TARGET_JOINT_NOT_FOUND"
        )

    obj.set_qpos(
        qpos
    )

    # --------------------------------------------------------
    # benchmark：
    #
    # 所有关节初始保持固定。
    # 真正pull之前只释放target joint。
    # --------------------------------------------------------

    for i, joint in enumerate(
        active_joints
    ):

        try:

            joint.set_drive_property(
                stiffness=5000,
                damping=500,
            )

            joint.set_drive_target(
                float(qpos[i])
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

    return {
        "object":
            obj,

        "urdf_path":
            str(urdf_path),

        "active_joints":
            active_joints,

        "target_joint":
            target_joint,

        "target_link":
            target_link,

        "target_index":
            target_index,

        "q_min":
            q_min,

        "q_max":
            q_max,

        "initial_object_qpos":
            qpos.copy(),

        "initial_ratio":
            float(
                (
                    qpos[target_index]
                    - q_min
                )
                / (q_max - q_min)
            ),
    }


def get_progress(
    object_info,
):

    obj = object_info[
        "object"
    ]

    index = int(
        object_info[
            "target_index"
        ]
    )

    q_min = float(
        object_info[
            "q_min"
        ]
    )

    q_max = float(
        object_info[
            "q_max"
        ]
    )

    q = float(
        obj.get_qpos()[
            index
        ]
    )

    return float(
        (
            q - q_min
        )
        / (
            q_max - q_min
        )
    )


def create_fixed_camera(
    scene,
    camera_pose_world,
):

    T = np.asarray(
        camera_pose_world,
        dtype=np.float64,
    )

    if T.shape != (4, 4):
        raise Where2ActRuntimeError(
            "CAMERA_POSE_NOT_4X4"
        )

    builder = (
        scene.create_actor_builder()
    )

    mount = builder.build_kinematic(
        "benchmark_camera_mount"
    )

    fov = np.deg2rad(
        FOV_DEG
    )

    camera = (
        scene.add_mounted_camera(
            "benchmark_camera",
            mount,
            sapien.Pose(),
            WIDTH,
            HEIGHT,
            0.0,
            fov,
            NEAR,
            FAR,
        )
    )

    mount.set_pose(
        sapien.Pose
        .from_transformation_matrix(
            T
        )
    )

    return (
        mount,
        camera,
    )


def rotation_error(
    R_target,
    R_actual,
):

    return float(
        np.linalg.norm(
            Rotation
            .from_matrix(
                np.asarray(
                    R_target,
                    dtype=np.float64,
                )
                @
                np.asarray(
                    R_actual,
                    dtype=np.float64,
                ).T
            )
            .as_rotvec()
        )
    )


def build_planning_request(
    *,
    task,
    primitive,
    shape_id,
    target_link,
    trial_seed,
    initial_ratio,
    initial_object_qpos,
    policy_result,
):

    contact = np.asarray(
        policy_result[
            "interaction_point_world"
        ],
        dtype=np.float64,
    ).reshape(3)

    T_w2a = np.asarray(
        policy_result[
            "grasp_pose_world"
        ],
        dtype=np.float64,
    )

    if T_w2a.shape != (4, 4):
        raise Where2ActRuntimeError(
            "W2A_GRASP_POSE_NOT_4X4"
        )

    R = T_w2a[
        :3,
        :3
    ].copy()

    det = float(
        np.linalg.det(R)
    )

    orth_error = float(
        np.linalg.norm(
            R.T @ R
            - np.eye(3)
        )
    )

    if abs(det - 1.0) > 1e-4:
        raise Where2ActRuntimeError(
            "W2A_ROTATION_DET_INVALID"
        )

    if orth_error > 1e-4:
        raise Where2ActRuntimeError(
            "W2A_ROTATION_NOT_ORTHOGONAL"
        )

    # R = [forward, left, up]
    approach = R[
        :,
        2
    ].copy()

    approach /= (
        np.linalg.norm(
            approach
        )
        + 1e-12
    )

    T_contact = np.eye(
        4,
        dtype=np.float64,
    )

    T_contact[
        :3,
        :3
    ] = R

    T_contact[
        :3,
        3
    ] = (
        contact
        - CONTACT_STANDOFF
        * approach
    )

    T_pregrasp = np.eye(
        4,
        dtype=np.float64,
    )

    T_pregrasp[
        :3,
        :3
    ] = R

    T_pregrasp[
        :3,
        3
    ] = (
        T_contact[
            :3,
            3
        ]
        - APPROACH_DISTANCE
        * approach
    )

    return {
        "version":
            3,

        "task":
            str(task),

        "primitive":
            str(primitive),

        "shape_id":
            str(shape_id),

        "target_link":
            str(target_link),

        "trial_seed":
            int(trial_seed),

        "initial_ratio":
            float(initial_ratio),

        "initial_object_qpos":
            np.asarray(
                initial_object_qpos,
                dtype=np.float64,
            ).reshape(-1).tolist(),

        "network_trained":
            bool(
                policy_result.get(
                    "trained",
                    False,
                )
            ),

        "goal_frame":
            "panda_grasptarget",

        "interaction_point_world":
            contact.tolist(),

        "approach_axis_world":
            approach.tolist(),

        "pull_direction_world":
            (-approach).tolist(),

        "w2a_gripper_root_pose_world":
            T_w2a.tolist(),

        "pregrasp_pose_world":
            T_pregrasp.tolist(),

        "contact_pose_world":
            T_contact.tolist(),

        "contact_standoff":
            CONTACT_STANDOFF,

        "approach_distance":
            APPROACH_DISTANCE,

        # Operation distance is intentionally not part of the policy/planner
        # request.  The corrected evaluator physically advances in small
        # Cartesian segments and stops from measured articulation progress.
        "operation_endpoint":
            "absolute_articulation_progress_target",

        "diagnostics": {
            "rotation_det":
                det,

            "rotation_orthogonality_error":
                orth_error,
        },
    }


def set_joint_target(
    panda,
    q_target,
):

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    ).reshape(7)

    for i, joint in enumerate(
        panda.arm_joints
    ):

        joint.set_drive_velocity_target(
            0.0
        )

        joint.set_drive_target(
            float(
                q_target[i]
            )
        )

    panda.open_gripper()


def track_joint_target(
    panda,
    q_target,
    *,
    min_steps=15,
    max_steps=500,
    tolerance_norm=0.02,
    tolerance_maxabs=0.015,
):

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    ).reshape(7)

    set_joint_target(
        panda,
        q_target,
    )

    for step_index in range(
        int(max_steps)
    ):

        panda.step()

        actual = np.asarray(
            panda.robot.get_qpos(),
            dtype=np.float64,
        )[:7]

        error = (
            actual
            - q_target
        )

        norm = float(
            np.linalg.norm(
                error
            )
        )

        maxabs = float(
            np.max(
                np.abs(error)
            )
        )

        if (
            step_index + 1
            >= int(min_steps)
            and norm
            <= float(tolerance_norm)
            and maxabs
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
                    norm,

                "error_maxabs":
                    maxabs,
            }

    return {
        "converged":
            False,

        "steps":
            int(max_steps),

        "error_norm":
            norm,

        "error_maxabs":
            maxabs,
    }


def execute_joint_trajectory(
    panda,
    trajectory,
):

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 7
    ):
        raise Where2ActRuntimeError(
            "PLANNER_TRAJECTORY_SHAPE_INVALID"
        )

    max_error = 0.0
    total_steps = 0

    for i, q in enumerate(
        trajectory
    ):

        diag = track_joint_target(
            panda,
            q,
        )

        total_steps += int(
            diag[
                "steps"
            ]
        )

        if not diag[
            "converged"
        ]:

            raise Where2ActRuntimeError(
                "SAPIEN_WAYPOINT_TRACKING_FAILED: "
                f"index={i}, "
                f"error={diag['error_norm']}"
            )

        max_error = max(
            max_error,
            float(
                diag[
                    "error_norm"
                ]
            ),
        )

    # 最终goal更严格跟踪一次。
    final_diag = track_joint_target(
        panda,
        trajectory[-1],
        min_steps=50,
        max_steps=1500,
        tolerance_norm=0.005,
        tolerance_maxabs=0.003,
    )

    if not final_diag[
        "converged"
    ]:

        raise Where2ActRuntimeError(
            "SAPIEN_FINAL_TRACKING_FAILED"
        )

    return {
        "trajectory_states":
            int(
                len(trajectory)
            ),

        "total_sim_steps":
            total_steps,

        "max_waypoint_error":
            max_error,

        "final_tracking":
            final_diag,
    }


def launch_collision_planner(
    *,
    request_path,
    pose_catalog,
    output_dir,
    planner_env,
    planner,
    planning_time,
    ik_attempts,
    seed,
):

    here = Path(__file__).resolve().parent

    worker = (
        here
        / "motion_planner_collision_worker.py"
    )

    if not worker.exists():
        raise FileNotFoundError(
            worker
        )

    output_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    conda_exe = (
        os.environ.get(
            "CONDA_EXE"
        )
        or shutil.which(
            "conda"
        )
    )

    if conda_exe is None:
        raise Where2ActRuntimeError(
            "CONDA_EXECUTABLE_NOT_FOUND"
        )

    cmd = [
        conda_exe,
        "run",
        "-n",
        str(planner_env),
        "python",
        str(worker),
        "--request",
        str(
            Path(request_path)
            .expanduser()
            .resolve()
        ),
        "--pose-catalog",
        str(
            Path(pose_catalog)
            .expanduser()
            .resolve()
        ),
        "--output-dir",
        str(output_dir),
        "--planner",
        str(planner),
        "--planning-time",
        str(
            float(planning_time)
        ),
        "--ik-attempts",
        str(
            int(ik_attempts)
        ),
        "--seed",
        str(
            int(seed)
        ),
    ]

    env = os.environ.copy()

    articubot = (
        HOME
        / "robot_baselines"
        / "repos"
        / "articubot"
    )

    old_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    if old_pythonpath:
        env[
            "PYTHONPATH"
        ] = (
            str(articubot)
            + os.pathsep
            + old_pythonpath
        )
    else:
        env[
            "PYTHONPATH"
        ] = str(
            articubot
        )

    proc = subprocess.run(
        cmd,
        cwd=str(here),
        env=env,
        text=True,
        capture_output=True,
    )

    (
        output_dir
        / "planner_stdout.txt"
    ).write_text(
        proc.stdout
    )

    (
        output_dir
        / "planner_stderr.txt"
    ).write_text(
        proc.stderr
    )

    trajectory_path = (
        output_dir
        / "trajectory.npy"
    )

    result_path = (
        output_dir
        / "result.json"
    )

    text = (
        proc.stdout
        + "\n"
        + proc.stderr
    )

    known_method_failure = any(
        token in text
        for token in [
            "IK_FAILED",
            "COLLISION_AWARE_OMPL_FAILED",
            "No solution found",
            "OMPL路径包含碰撞state",
        ]
    )

    return {
        "returncode":
            int(proc.returncode),

        "success":
            bool(
                proc.returncode == 0
                and trajectory_path.exists()
            ),

        "known_method_failure":
            bool(
                known_method_failure
            ),

        "trajectory_path":
            str(
                trajectory_path
            ),

        "planner_result_path":
            str(
                result_path
            ),

        "stdout_path":
            str(
                output_dir
                / "planner_stdout.txt"
            ),

        "stderr_path":
            str(
                output_dir
                / "planner_stderr.txt"
            ),

        "command":
            cmd,
    }
