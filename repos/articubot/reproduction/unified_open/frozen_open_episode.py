"""One ArticuBot episode inside the frozen OPEN/PHYSICAL_V2 benchmark scene."""

from __future__ import annotations

import ast
import gc
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sapien.core as sapien
import torch

from articubot_action_adapter import execute_articubot_action
from articubot_observation_adapter import ObservationHistory, capture_articubot_observation
from senior_soft_weld_pd import DEFAULT_CONFIG as SOFT_WELD_CONFIG
from senior_soft_weld_pd import (
    SeniorSoftWeldPD,
    enable_backend_finger_lock,
    source_defaults as soft_weld_source_defaults,
)


HOME = Path("/home/feng")
FROZEN_DIR = HOME / "robot_baselines/common_env/flowbot3d_conditionA_physical_v2"
POSE_CATALOG = HOME / "robot_baselines/configs/flowbot3d/eval/eval_pose_catalog.jsonl"
DRAWER_POSE_CATALOG = (
    HOME
    / "robot_baselines/results/where2act/four_task_noaff_v7_formal/20260905_082001"
    / "formal_catalogs/drawer_open_formal_scene_catalog.jsonl"
)
PM_ROOT = HOME / "robot_baselines/data/partnet-mobility"
PANDA_URDF = HOME / "robot_baselines/common_env/assets/panda_articubot/panda.urdf"
RELABEL_EVALUATOR = HOME / "relabel_flowbot_35.py"
if str(FROZEN_DIR) not in sys.path:
    sys.path.insert(0, str(FROZEN_DIR))
from panda_controller import PandaTwoFingerController
from contact_monitor import monitor_grasp_establishment, monitor_post_pull_contact, read_twofinger_contact


def frozen_articulation_threshold() -> float:
    """Read the literal used by the actual senior 35% relabel evaluator."""
    tree = ast.parse(RELABEL_EVALUATOR.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "THRESHOLD" for t in node.targets):
            value = float(ast.literal_eval(node.value))
            if value != 0.35:
                raise RuntimeError(f"frozen threshold changed unexpectedly: {value}")
            return value
    raise RuntimeError(f"THRESHOLD not found in {RELABEL_EVALUATOR}")


THRESHOLD = frozen_articulation_threshold()
START_PROGRESS_TOLERANCE = 0.02
# The official evaluator uses ``horizon=35`` and
# ``for t in range(1, horizon)``, i.e. exactly 34 policy calls.  Each call
# retains the checkpoint's native four-action execution horizon.
MAX_POLICY_CYCLES = 34
ACTION_CONTROL_STEPS = 100
GRIPPER_WAIT_STEPS = 300
TARGET_LOCK_STIFFNESS = 5000.0
TARGET_LOCK_DAMPING = 500.0
TARGET_FREE_STIFFNESS = 0.0
TARGET_FREE_DAMPING = 0.05
PSEUDO_ATTACH_STIFFNESS = 45000.0
PSEUDO_ATTACH_DAMPING = 0.0


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def case_link_name(case: dict) -> str:
    """Normalize the two frozen catalog schemas without rewriting either catalog."""
    return str(case.get("link_name", case.get("target_link")))


def case_task_mode(case: dict) -> str:
    return str(case.get("task", "door_open"))


def catalog_path_for_task(task_mode: str) -> Path:
    if task_mode == "door_open":
        return POSE_CATALOG
    if task_mode == "drawer_open":
        return DRAWER_POSE_CATALOG
    raise ValueError(f"unsupported OPEN task: {task_mode}")


def load_catalog(path: Path = POSE_CATALOG, *, task_mode=None):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if task_mode is None:
        task_mode = "drawer_open" if path.resolve() == DRAWER_POSE_CATALOG.resolve() else "door_open"
    expected = {"door_open": 56, "drawer_open": 36}[task_mode]
    keys = [(str(row["shape_id"]), case_link_name(row)) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise RuntimeError(
            f"canonical {task_mode} catalog must contain {expected} unique cases, got {len(rows)}"
        )
    if any(case_task_mode(row) != task_mode for row in rows):
        raise RuntimeError(f"task mismatch in canonical {task_mode} catalog")
    return rows


def initial_ratio_for_seed(seed: int) -> float:
    return float(np.random.default_rng(int(seed)).uniform(0.10, 0.20))


def _create_scene(case: dict, initial_ratio: float):
    engine = sapien.Engine(0, 0.001, 0.005)
    renderer = sapien.VulkanRenderer(offscreen_only=True)
    engine.set_renderer(renderer)
    config = sapien.SceneConfig()
    config.gravity = [0, 0, -9.81]
    config.solver_iterations = 20
    config.enable_pcm = False
    config.sleep_threshold = 0.0
    scene = engine.create_scene(config=config)
    scene.set_timestep(1.0 / 500.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 2], [1, 1, 1])

    object_urdf = PM_ROOT / str(case["shape_id"]) / "mobility.urdf"
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 0.75
    obj = loader.load(str(object_urdf))
    if obj is None:
        raise RuntimeError(f"failed to load {object_urdf}")
    obj.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    joints = list(obj.get_active_joints())
    link_name = case_link_name(case)
    target_index = next(
        (i for i, joint in enumerate(joints) if joint.get_child_link().get_name() == link_name),
        None,
    )
    if target_index is None:
        raise RuntimeError(f"target joint not found: {link_name}")
    target_joint = joints[target_index]
    target_link = target_joint.get_child_link()
    limits = np.asarray(target_joint.get_limits()[0], dtype=np.float64)
    q_min, q_max = float(limits[0]), float(limits[1])
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max - q_min <= 1e-8:
        raise RuntimeError(f"invalid target limits {limits.tolist()}")
    q_range = q_max - q_min
    exact_qpos = case.get("initial_object_qpos")
    if exact_qpos is not None:
        exact_qpos = np.asarray(exact_qpos, dtype=np.float64).reshape(-1)
        if exact_qpos.shape != (len(joints),) or not np.all(np.isfinite(exact_qpos)):
            raise RuntimeError(
                f"invalid frozen initial_object_qpos: expected {(len(joints),)}, got {exact_qpos.shape}"
            )
    qpos = []
    for index, joint in enumerate(joints):
        lower = float(np.asarray(joint.get_limits()[0])[0])
        upper = float(np.asarray(joint.get_limits()[0])[1])
        if not np.isfinite(lower):
            lower = 0.0
        q = float(exact_qpos[index]) if exact_qpos is not None else lower
        if np.isfinite(lower) and q < lower - 1e-5:
            raise RuntimeError(f"frozen qpos below joint limit at index {index}")
        if np.isfinite(upper) and q > upper + 1e-5:
            raise RuntimeError(f"frozen qpos above joint limit at index {index}")
        qpos.append(q)
        joint.set_drive_property(stiffness=5000.0, damping=500.0)
        joint.set_drive_target(q)
        joint.set_drive_velocity_target(0.0)
    target_q = (
        float(exact_qpos[target_index])
        if exact_qpos is not None
        else q_min + float(initial_ratio) * q_range
    )
    qpos[target_index] = target_q
    obj.set_qpos(np.asarray(qpos, dtype=np.float64))
    target_joint.set_drive_target(target_q)

    panda = PandaTwoFingerController(
        scene,
        PANDA_URDF,
        arm_stiffness=1800.0,
        arm_damping=360.0,
        finger_stiffness=8000.0,
        finger_damping=1600.0,
        finger_static_friction=2.0,
        finger_dynamic_friction=2.0,
        finger_restitution=0.0,
    )
    panda.set_initial_state(case["base_pose"], case["robot_initial_qpos"])
    panda.open_gripper()
    for shape in target_link.get_collision_shapes():
        shape.set_physical_material(panda.finger_material)

    camera_mount = scene.create_actor_builder().build_kinematic("benchmark_camera_mount")
    camera = scene.add_mounted_camera(
        "benchmark_camera", camera_mount, sapien.Pose(), 448, 448, 0.0,
        np.deg2rad(35.0), 0.1, 100.0,
    )
    camera_mount.set_pose(sapien.Pose.from_transformation_matrix(np.asarray(case["camera_pose_world"])))

    def progress():
        return float((float(obj.get_qpos()[target_index]) - q_min) / q_range)

    return {
        "engine": engine, "renderer": renderer, "scene": scene, "obj": obj,
        "target_joint": target_joint, "target_link": target_link, "panda": panda,
        "camera": camera, "progress": progress, "q_min": q_min, "q_max": q_max,
        "target_index": target_index, "target_q": target_q,
    }


def _pin_target_state(resources):
    """Apply the frozen PHYSICAL_V2 CLOSE-phase q/qdot stabilization."""
    obj = resources["obj"]
    target_index = int(resources["target_index"])
    target_q = float(resources["target_q"])
    qpos = np.asarray(obj.get_qpos(), dtype=np.float64).copy()
    qpos[target_index] = target_q
    obj.set_qpos(qpos)
    try:
        qvel = np.asarray(obj.get_qvel(), dtype=np.float64).copy()
        qvel[target_index] = 0.0
        obj.set_qvel(qvel)
    except Exception:
        pass


def _set_target_locked(resources):
    _pin_target_state(resources)
    joint = resources["target_joint"]
    joint.set_drive_property(stiffness=TARGET_LOCK_STIFFNESS, damping=TARGET_LOCK_DAMPING)
    joint.set_drive_target(float(resources["target_q"]))
    joint.set_drive_velocity_target(0.0)


def _set_target_free(resources):
    joint = resources["target_joint"]
    joint.set_drive_property(stiffness=TARGET_FREE_STIFFNESS, damping=TARGET_FREE_DAMPING)
    joint.set_drive_velocity_target(0.0)


def run_episode(
    policy,
    case: dict,
    *,
    case_index: int,
    repeat_id: int,
    seed: int,
    pseudo_attachment: bool = False,
    execution_backend: str = "physical_contact_only",
) -> dict:
    started = time.time()
    task_mode = case_task_mode(case)
    catalog_path = catalog_path_for_task(task_mode)
    initial_ratio = None if case.get("initial_object_qpos") is not None else initial_ratio_for_seed(seed)
    link_name = case_link_name(case)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if execution_backend not in ("physical_contact_only", "senior_soft_weld_pd"):
        raise ValueError(f"unsupported execution backend: {execution_backend}")
    if pseudo_attachment and execution_backend != "physical_contact_only":
        raise ValueError("legacy pseudo attachment and senior soft_weld_pd are mutually exclusive")
    use_soft_weld_pd = execution_backend == "senior_soft_weld_pd"
    result = {
        "method": "ArticuBot_official_pretrained",
        "protocol_version": (
            "articubot_senior_physical_v2_pseudo_attach_v1_relabel35"
            if pseudo_attachment
            else (
                "articubot_senior_soft_weld_pd_relabel35"
                if use_soft_weld_pd
                else "flowbot3d_conditionA_physical_v2_relabel35"
            )
        ),
        "case_id": f"{case['shape_id']}_{link_name}",
        "case_index": int(case_index),
        "object_id": str(case["shape_id"]),
        "link_id": link_name,
        "category": str(case.get("category", "UNKNOWN")),
        "trial_id": int(repeat_id),
        "repeat_id": int(repeat_id),
        "trial_seed": int(seed),
        "task_mode": task_mode,
        "requested_initial_articulation_state": initial_ratio,
        "requested_initial_object_qpos": jsonable(case.get("initial_object_qpos")),
        "initial_articulation_state": None,
        "final_articulation_state": None,
        "articulation_progress": None,
        "articulation_success_threshold": THRESHOLD,
        "grasp_success": False,
        "operation_success": False,
        "final_success": False,
        "termination_reason": None,
        "timeout": False,
        "ik_failure": False,
        "policy_failure": False,
        "controller_failure": False,
        "exception": None,
        "runtime_sec": None,
        "high_level_goal": None,
        "low_level_action_summary": None,
        "camera_source": str(catalog_path),
        "frozen_camera_provenance": str(case.get("camera_source", "native_formal_door_open_pose_package")),
        "frozen_articulation_state_source": str(
            case.get("articulation_state_source", "trial_seed_uniform_0.10_0.20")
        ),
        "phase_transitions": [],
        "interaction_mode": (
            "physical_twofinger_grasp_then_temporary_xyz_attachment"
            if pseudo_attachment
            else (
                "senior_soft_weld_pd_after_physical_twofinger_grasp"
                if use_soft_weld_pd
                else "physical_twofinger_contact_friction_only"
            )
        ),
        "pseudo_attachment": {
            "enabled": bool(pseudo_attachment),
            "activation_gate": "frozen_PHYSICAL_V2_firm_bilateral_grasp",
            "active_before_grasp": False,
            "xyz_stiffness": PSEUDO_ATTACH_STIFFNESS if pseudo_attachment else None,
            "xyz_damping": PSEUDO_ATTACH_DAMPING if pseudo_attachment else None,
            "angular_constraint": False,
            "uses_ground_truth_axis": False,
            "uses_ground_truth_handle": False,
            "action_steps_attached": 0,
        },
        "senior_soft_weld_pd": {
            **soft_weld_source_defaults(),
            "enabled": bool(use_soft_weld_pd),
            "created": False,
            "n_welds": 0,
            "fail_reason": "not_reached",
            "source_archive": "/home/feng/robot_baselines/repos/articubot/reproduction/provenance/low_level_execution_logic.zip",
            "source_eval_sha256": "864df4ce14fc2ef6fa13f92a0c1c7528f0ead8ab75cf953d1c7148fd54fd8d18",
            "source_doc_sha256": "f471ca7781ae5e4ee30388a9c303be12fc60e6245bae7582e8132513dfdb341f",
        },
        "finger_grasp_lock": {"enabled": False},
    }
    resources = None
    action_records = []
    last_goal = None
    grasp_monitor = None
    attachment_anchors = None
    active_drive = None
    finger_contact_weld = None
    soft_backend_engaged = False
    soft_finger_target = None
    try:
        resources = _create_scene(case, initial_ratio)
        panda = resources["panda"]
        panda.wait(100)
        actual_initial = float(resources["progress"]())
        result["initial_articulation_state"] = actual_initial
        policy.reset()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 81731]))
        obs = capture_articubot_observation(
            scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
            panda=panda, num_points=policy.num_points, rng=rng,
        )
        history = ObservationHistory(policy.n_obs_steps)
        history.reset(obs)
        grasp_checked = False
        # Frozen senior bottom logic: INIT/PREGRASP is locked, while the
        # policy's open-gripper approach (OPEN/TO_GRASP) is physically free.
        _set_target_free(resources)
        result["phase_transitions"].append(
            {"phase": "OPEN_TO_GRASP", "target_joint": "free", "cycle": 0, "action_index": 0}
        )
        stop = False
        for cycle in range(MAX_POLICY_CYCLES):
            last_goal, actions = policy.infer(history.stack())
            for action_index, action in enumerate(actions):
                current_finger = float(np.asarray(panda.get_finger_qpos()).mean())
                closing_phase = float(action[9]) < 0.0 and current_finger + float(action[9]) <= 0.02
                first_close = bool(not grasp_checked and closing_phase)
                if first_close:
                    # Exact frozen CLOSE phase: restore the prescribed initial
                    # articulation, lock it target q/qdot, and physically close.
                    # The transition is triggered only by ArticuBot's gripper
                    # output; no target geometry or other oracle is consulted.
                    _set_target_locked(resources)
                    result["phase_transitions"].append(
                        {"phase": "CLOSE", "target_joint": "locked", "cycle": cycle,
                         "action_index": action_index, "trigger": "policy_gripper_target_le_0.02"}
                    )
                    panda.set_video_step_callback(lambda: _pin_target_state(resources), every_n_steps=1)
                if pseudo_attachment and result["grasp_success"]:
                    if attachment_anchors is None:
                        raise RuntimeError("pseudo attachment missing post-grasp anchors")
                    panda.keep_gripper_closed()
                    active_drive = resources["scene"].create_drive(
                        panda.hand_link,
                        attachment_anchors["hand_local"],
                        resources["target_link"],
                        attachment_anchors["target_local"],
                    )
                    active_drive.set_x_properties(
                        stiffness=PSEUDO_ATTACH_STIFFNESS, damping=PSEUDO_ATTACH_DAMPING
                    )
                    active_drive.set_y_properties(
                        stiffness=PSEUDO_ATTACH_STIFFNESS, damping=PSEUDO_ATTACH_DAMPING
                    )
                    active_drive.set_z_properties(
                        stiffness=PSEUDO_ATTACH_STIFFNESS, damping=PSEUDO_ATTACH_DAMPING
                    )
                    result["pseudo_attachment"]["action_steps_attached"] += 1
                action_control_steps = (
                    SOFT_WELD_CONFIG.operate_steps_per_action
                    if soft_backend_engaged
                    else ACTION_CONTROL_STEPS
                )
                try:
                    diag = execute_articubot_action(
                        panda,
                        action,
                        num_steps=action_control_steps,
                        finger_target_override=(
                            soft_finger_target
                            if soft_backend_engaged
                            else None
                        ),
                    )
                    if finger_contact_weld is not None:
                        finger_contact_weld.maintain_finger_lock(panda)
                finally:
                    if first_close:
                        panda.clear_video_step_callback()
                    if active_drive is not None:
                        resources["scene"].remove_drive(active_drive)
                        active_drive = None
                ctrl = diag["control"]
                if not ctrl["reached_control_tolerance"] and (
                    ctrl["final_position_error"] > 0.04 or ctrl["final_rotation_error"] > 0.35
                ):
                    result["ik_failure"] = True
                action_records.append(
                    {
                        "cycle": cycle,
                        "action_index": action_index,
                        "min": float(np.min(action)),
                        "max": float(np.max(action)),
                        "mean": float(np.mean(action)),
                        "xyz_norm": float(np.linalg.norm(action[:3])),
                        "target_finger": float(diag["target_finger"]),
                    }
                )
                closing = float(action[9]) < 0.0 and float(diag["target_finger"]) <= 0.02
                just_grasped = False
                if not grasp_checked and closing:
                    grasp_checked = True
                    # Match PHYSICAL_V2's 300-step locked close window exactly.
                    for _ in range(GRIPPER_WAIT_STEPS):
                        _pin_target_state(resources)
                        panda.keep_gripper_closed()
                        panda.step()
                    _set_target_free(resources)
                    result["phase_transitions"].append(
                        {"phase": "HOLD_OPERATE", "target_joint": "free", "cycle": cycle,
                         "action_index": action_index}
                    )
                    grasp_monitor = monitor_grasp_establishment(
                        panda, resources["target_link"], settle_steps=300, tail_steps=100,
                        min_bilateral_fraction=0.50,
                    )
                    result["grasp_success"] = bool(grasp_monitor["firm_grasp"])
                    just_grasped = bool(result["grasp_success"])
                    if result["grasp_success"] and use_soft_weld_pd:
                        soft_backend_engaged = True
                        finger_contact_weld, weld_diag = SeniorSoftWeldPD.try_create(
                            resources["scene"], panda, resources["target_link"], SOFT_WELD_CONFIG
                        )
                        result["senior_soft_weld_pd"] = jsonable(weld_diag)
                        if finger_contact_weld is not None:
                            soft_finger_target = float(finger_contact_weld.finger_target)
                            result["finger_grasp_lock"] = jsonable(
                                finger_contact_weld.enable_finger_lock(panda)
                            )
                        else:
                            # Supplied evaluator locks fully closed fingers and
                            # still enters operate when the contact-gated weld
                            # could not be created.
                            soft_finger_target = 0.0
                            result["finger_grasp_lock"] = jsonable(
                                enable_backend_finger_lock(panda, soft_finger_target)
                            )
                        resources["target_joint"].set_drive_property(
                            SOFT_WELD_CONFIG.operate_target_stiffness,
                            SOFT_WELD_CONFIG.operate_target_damping,
                            SOFT_WELD_CONFIG.operate_target_force_limit,
                        )
                        resources["target_joint"].set_drive_target(
                            float(resources["obj"].get_qpos()[resources["target_index"]])
                        )
                        resources["target_joint"].set_drive_velocity_target(0.0)
                        result["operate_target_drive"] = {
                            "stiffness": SOFT_WELD_CONFIG.operate_target_stiffness,
                            "damping": SOFT_WELD_CONFIG.operate_target_damping,
                            "force_limit": SOFT_WELD_CONFIG.operate_target_force_limit,
                        }
                        result["operate_joint_drives"] = {
                            "arm_stiffness": [SOFT_WELD_CONFIG.operate_arm_stiffness] * 7,
                            "arm_damping": [SOFT_WELD_CONFIG.operate_arm_damping] * 7,
                            "drive_soft_base_joint": SOFT_WELD_CONFIG.soft_arm,
                        }
                        if finger_contact_weld is not None:
                            for _ in range(4):
                                finger_contact_weld.maintain_finger_lock(panda)
                                panda.step()
                            result["senior_soft_weld_pd"].update(
                                {
                                    "weld_position_error_m": finger_contact_weld.max_position_error(),
                                    "weld_angular_error_rad": finger_contact_weld.angular_error(),
                                }
                            )
                            result["phase_transitions"].append(
                                {
                                    "phase": "SENIOR_SOFT_WELD_PD",
                                    "target_joint": "free",
                                    "cycle": cycle,
                                    "action_index": action_index,
                                    "gate": "firm_PHYSICAL_V2_grasp_and_current_twofinger_contact",
                                    "n_welds": int(weld_diag.get("n_welds", 0)),
                                }
                            )
                    if result["grasp_success"] and pseudo_attachment:
                        anchor_world = sapien.Pose(
                            np.asarray(panda.get_grasp_center(), dtype=np.float64).tolist(),
                            [1, 0, 0, 0],
                        )
                        attachment_anchors = {
                            "hand_local": panda.hand_link.get_pose().inv() * anchor_world,
                            "target_local": resources["target_link"].get_pose().inv() * anchor_world,
                        }
                        result["pseudo_attachment"]["activated"] = True
                        result["pseudo_attachment"]["anchor_world_at_activation"] = (
                            np.asarray(anchor_world.p, dtype=np.float64).tolist()
                        )
                        result["phase_transitions"].append(
                            {"phase": "PSEUDO_ATTACH", "target_joint": "free", "cycle": cycle,
                             "action_index": action_index, "gate": "firm_bilateral_grasp"}
                        )
                    if not result["grasp_success"]:
                        result["termination_reason"] = "grasp_failed"
                        stop = True
                current_progress = float(resources["progress"]())
                if result["grasp_success"] and current_progress >= THRESHOLD:
                    result["termination_reason"] = "articulation_threshold_reached"
                    stop = True
                elif (
                    result["grasp_success"]
                    and not just_grasped
                    and not pseudo_attachment
                    and not soft_backend_engaged
                ):
                    post = monitor_post_pull_contact(panda, resources["target_link"], steps=50)
                    if post["grasp_lost"]:
                        result["termination_reason"] = "grasp_lost"
                        stop = True
                obs = capture_articubot_observation(
                    scene=resources["scene"], camera=resources["camera"], obj=resources["obj"],
                    panda=panda, num_points=policy.num_points, rng=rng,
                )
                history.append(obs)
                if stop:
                    break
            if stop:
                break
        if result["termination_reason"] is None:
            result["termination_reason"] = "policy_horizon"
            result["timeout"] = True
        final_progress = float(resources["progress"]())
        result["final_articulation_state"] = final_progress
        result["articulation_progress"] = final_progress
        result["operation_success"] = bool(result["grasp_success"] and final_progress >= THRESHOLD)
        result["final_success"] = bool(result["operation_success"])
        result["grasp_monitor"] = jsonable(grasp_monitor)
        if finger_contact_weld is not None:
            result["senior_soft_weld_pd"]["weld_position_error_m_end"] = jsonable(
                finger_contact_weld.max_position_error()
            )
            result["senior_soft_weld_pd"]["weld_angular_error_rad_end"] = jsonable(
                finger_contact_weld.angular_error()
            )
    except Exception as exc:
        result["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        text = str(exc).lower()
        result["policy_failure"] = any(x in text for x in ("policy", "checkpoint", "cuda", "tensor", "output"))
        result["controller_failure"] = any(x in text for x in ("controller", "jacobian", "rotation", "panda"))
        result["termination_reason"] = "exception"
        if resources is not None:
            try:
                value = float(resources["progress"]())
                result["final_articulation_state"] = value
                result["articulation_progress"] = value
            except Exception:
                pass
    finally:
        if active_drive is not None and resources is not None:
            try:
                resources["scene"].remove_drive(active_drive)
            except Exception:
                pass
        if finger_contact_weld is not None:
            finger_contact_weld.destroy()
        if last_goal is not None:
            result["high_level_goal"] = jsonable(np.asarray(last_goal)[0, 0])
        if action_records:
            xyz = [row["xyz_norm"] for row in action_records]
            result["low_level_action_summary"] = {
                "num_actions": len(action_records),
                "xyz_norm_min": min(xyz), "xyz_norm_max": max(xyz), "xyz_norm_mean": float(np.mean(xyz)),
                "first": action_records[:4], "last": action_records[-4:],
            }
        result["runtime_sec"] = float(time.time() - started)
        resources = None
        gc.collect()
    return jsonable(result)
