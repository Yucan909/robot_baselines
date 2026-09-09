#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sapien.core as sapien
import torch

HOME = Path('/home/feng')
CODE = HOME / 'robot_baselines/repos/pa3ff_official/reproduction'
COMMON = HOME / 'robot_baselines/common_env/flowbot3d_conditionA_physical_v2'
ARTICUBOT = HOME / 'robot_baselines/repos/articubot/reproduction/unified_open'
PM_ROOT = HOME / 'robot_baselines/data/partnet-mobility'
PANDA_URDF = HOME / 'robot_baselines/common_env/assets/panda_articubot/panda.urdf'
CATALOG = HOME / 'robot_baselines/configs/pa3ff/reproduction_v1_formal/formal_episode_catalog.jsonl'
REP_CKPT = HOME / 'robot_baselines/results/pa3ff/representation_native5_balanced_v1/instance_net_snapshots/instance_net_step10000.pth'
for p in (CODE, COMMON, ARTICUBOT):
    sys.path.insert(0, str(p))
from pa3ff_policy_runtime import PA3FFPADPRuntime
from panda_controller import PandaTwoFingerController
from articubot_observation_adapter import capture_articubot_observation

SUCCESS_THRESHOLD = 0.35
COMMAND_TARGET = 0.40
PREGRASP_DISTANCE = 0.08
PREGRASP_STEPS = 2500
APPROACH_STEPS = 800
ACTION_STEPS = 350
ENGAGEMENT_SETTLE_STEPS = 300
ENGAGEMENT_TAIL_STEPS = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def actor_id(actor) -> int:
    return int(actor.get_id()) if hasattr(actor, 'get_id') else int(actor.id)


def effective_contact(scene, robot_links, target_link, eps=1e-8) -> dict:
    robot_ids = {actor_id(link): link.get_name() for link in robot_links}
    target_id = actor_id(target_link)
    links = set()
    points = 0
    total = 0.0
    maximum = 0.0
    for contact in scene.get_contacts():
        a0, a1 = actor_id(contact.actor0), actor_id(contact.actor1)
        if a0 == target_id and a1 in robot_ids:
            name = robot_ids[a1]
        elif a1 == target_id and a0 in robot_ids:
            name = robot_ids[a0]
        else:
            continue
        for point in contact.points:
            impulse = float(np.linalg.norm(np.asarray(point.impulse, dtype=np.float64)))
            if np.isfinite(impulse) and impulse > eps:
                links.add(name); points += 1; total += impulse; maximum = max(maximum, impulse)
    return {'effective': points > 0, 'robot_links': sorted(links), 'points': points,
            'total_impulse': total, 'max_impulse': maximum}


def set_gripper_target(panda, value: float) -> None:
    value = float(np.clip(value, 0.0, 0.04))
    for joint in panda.finger_joints:
        joint.set_drive_velocity_target(0.0)
        joint.set_drive_target(value)


def monitor_engagement(panda, target_link, primitive: str, gripper_target: float) -> dict:
    frames = []
    links = [panda.left_finger_link, panda.right_finger_link]
    if primitive == 'push':
        links = [panda.hand_link, panda.left_finger_link, panda.right_finger_link]
    for _ in range(ENGAGEMENT_SETTLE_STEPS):
        set_gripper_target(panda, gripper_target)
        panda.step()
        frames.append(effective_contact(panda.scene, links, target_link))
    tail = frames[-ENGAGEMENT_TAIL_STEPS:]
    if primitive == 'pull':
        valid = []
        for frame in tail:
            names = set(frame['robot_links'])
            valid.append('panda_leftfinger' in names and 'panda_rightfinger' in names)
        fraction = float(np.mean(valid))
        success = fraction >= 0.50
        definition = 'bilateral_finger_target_contact_fraction>=0.50_in_tail100'
    else:
        valid = [frame['effective'] for frame in tail]
        fraction = float(np.mean(valid))
        success = fraction >= 0.20
        definition = 'correct_target_part_effective_robot_contact_fraction>=0.20_in_tail100'
    return {
        'success': bool(success), 'definition': definition, 'settle_steps': ENGAGEMENT_SETTLE_STEPS,
        'tail_steps': ENGAGEMENT_TAIL_STEPS, 'valid_contact_frames': int(sum(valid)),
        'valid_contact_fraction': fraction, 'last_frame': tail[-1],
        'max_impulse': max((x['max_impulse'] for x in tail), default=0.0),
    }


def create_scene(case: dict):
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

    urdf = PM_ROOT / case['shape_id'] / 'mobility.urdf'
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = float(case['scale'])
    obj = loader.load(str(urdf))
    if obj is None:
        raise RuntimeError(f'object load failed: {urdf}')
    obj.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    joints = list(obj.get_active_joints())
    target_index = next((i for i, j in enumerate(joints)
                         if j.get_child_link().get_name() == case['target_link']), None)
    if target_index is None:
        raise RuntimeError(f"target joint missing: {case['target_link']}")
    target_joint = joints[target_index]
    target_link = target_joint.get_child_link()
    limits = np.asarray(target_joint.get_limits()[0], dtype=np.float64)
    q_min, q_max = float(limits[0]), float(limits[1])
    if not np.isfinite(q_min + q_max) or q_max - q_min <= 1e-8:
        raise RuntimeError(f'invalid target limits {limits.tolist()}')
    lowers = []
    uppers = []
    for joint in joints:
        lim = np.asarray(joint.get_limits()[0], dtype=np.float64)
        lo = float(lim[0]) if np.isfinite(lim[0]) else 0.0
        hi = float(lim[1]) if np.isfinite(lim[1]) else lo
        lowers.append(lo); uppers.append(hi)
    if case['initial_object_qpos'] is None:
        object_qpos = np.asarray(lowers, dtype=np.float64)
        object_qpos[target_index] = q_min + float(case['initial_target_ratio']) * (q_max - q_min)
    else:
        object_qpos = np.asarray(case['initial_object_qpos'], dtype=np.float64)
        if object_qpos.shape != (len(joints),):
            raise RuntimeError(f'initial object qpos shape {object_qpos.shape}, expected {(len(joints),)}')
        object_qpos = np.minimum(np.maximum(object_qpos, np.asarray(lowers)), np.asarray(uppers))
    obj.set_qpos(object_qpos)
    for i, joint in enumerate(joints):
        joint.set_drive_property(stiffness=5000.0, damping=500.0)
        joint.set_drive_target(float(object_qpos[i]))
        joint.set_drive_velocity_target(0.0)

    panda = PandaTwoFingerController(scene, PANDA_URDF, arm_stiffness=1800.0, arm_damping=360.0,
                                      finger_stiffness=8000.0, finger_damping=1600.0,
                                      finger_static_friction=2.0, finger_dynamic_friction=2.0,
                                      finger_restitution=0.0)
    panda.set_initial_state(case['base_pose'], case['robot_initial_qpos'])
    panda.open_gripper()
    for shape in target_link.get_collision_shapes():
        shape.set_physical_material(panda.finger_material)

    mount = scene.create_actor_builder().build_kinematic('pa3ff_camera_mount')
    camera = scene.add_mounted_camera('pa3ff_camera', mount, sapien.Pose(), 448, 448, 0.0,
                                      np.deg2rad(35.0), 0.1, 100.0)
    camera_pose = np.asarray(case['camera_pose_world'], dtype=np.float64)
    mount.set_pose(sapien.Pose.from_transformation_matrix(camera_pose))

    def progress() -> float:
        raw = (float(obj.get_qpos()[target_index]) - q_min) / (q_max - q_min)
        return float(np.clip(raw, 0.0, 1.0))

    return {'engine': engine, 'renderer': renderer, 'scene': scene, 'obj': obj, 'panda': panda,
            'target_joint': target_joint, 'target_link': target_link, 'target_index': target_index,
            'q_min': q_min, 'q_max': q_max, 'camera': camera, 'camera_pose': camera_pose,
            'progress': progress}


def run_episode(runtime: PA3FFPADPRuntime, case: dict, checkpoint_sha: str, catalog_sha: str) -> dict:
    started = time.time()
    seed = int(case['seed'])
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    result = {
        'episode_status': 'running', 'method': 'PA3FF_reproduction_v1', 'task': case['task'],
        'primitive': case['primitive'], 'goal': case['goal'], 'shape_id': case['shape_id'],
        'target_link': case['target_link'], 'target_index': case['target_index'],
        'trial_index': case['trial_index'], 'seed': seed, 'checkpoint': str(runtime.checkpoint),
        'checkpoint_sha256': checkpoint_sha, 'training_step': runtime.training_step,
        'catalog_sha256': catalog_sha, 'grasp_success': False, 'final_success': False,
        'reached_target_40': False, 'initial_progress': None, 'final_progress': None,
        'directional_task_progress': None, 'failure_reason': None, 'exception': None,
        'policy_input_contract': ['object_point_cloud_xyz_1024', 'robot_qpos_9d',
                                  'part_siglip_embedding', 'instruction_siglip_embedding'],
        'forbidden_formal_fields_used_by_policy': [],
    }
    resources = None
    try:
        resources = create_scene(case)
        panda = resources['panda']
        panda.wait(100)
        initial_progress = resources['progress']()
        result['initial_progress'] = initial_progress
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2718]))
        obs = capture_articubot_observation(scene=resources['scene'], camera=resources['camera'],
                                            obj=resources['obj'], panda=panda, num_points=1024, rng=rng)
        prediction = runtime.predict(point_cloud_world=obs['point_cloud'],
                                     camera_pose_world=resources['camera_pose'],
                                     robot_qpos=np.asarray(panda.robot.get_qpos(), dtype=np.float32),
                                     task=case['task'], seed=seed)
        poses = prediction.pop('poses_world_grasptarget')
        gripper = prediction.pop('gripper')
        result['policy_output_diagnostics'] = prediction
        result['first_action'] = {'position': poses[0, :3, 3].tolist(),
                                  'rotation': poses[0, :3, :3].tolist(),
                                  'gripper': float(gripper[0])}

        pregrasp = poses[0].copy()
        pregrasp[:3, 3] -= PREGRASP_DISTANCE * poses[0, :3, 2]
        pre_diag = panda.move_grasp_pose_to(pregrasp, PREGRASP_STEPS)
        approach_diag = panda.move_grasp_pose_to(poses[0], APPROACH_STEPS)
        result['pregrasp_control'] = pre_diag
        result['approach_control'] = approach_diag
        result['progress_before_engagement'] = resources['progress']()
        engagement = monitor_engagement(panda, resources['target_link'], case['primitive'], float(gripper[0]))
        result['engagement_monitor'] = engagement
        result['grasp_success'] = bool(engagement['success'])

        resources['target_joint'].set_drive_property(stiffness=0.0, damping=0.05)
        resources['target_joint'].set_drive_velocity_target(0.0)
        control = []
        for action_i in range(1, 16):
            set_gripper_target(panda, float(gripper[action_i]))
            diag = panda.move_grasp_pose_to(poses[action_i], ACTION_STEPS,
                                             position_tolerance=0.007, rotation_tolerance=0.05)
            control.append({'action_index': action_i, **diag, 'gripper': float(gripper[action_i])})
        panda.wait(100)
        result['operation_control'] = control
        final_progress = resources['progress']()
        directional = (final_progress - initial_progress if case['goal'] == 'open'
                       else initial_progress - final_progress)
        result['final_progress'] = final_progress
        result['directional_task_progress'] = float(directional)
        result['reached_target_40'] = bool(directional >= COMMAND_TARGET)
        result['final_success'] = bool(result['grasp_success'] and directional >= SUCCESS_THRESHOLD)
        if result['final_success']:
            result['failure_reason'] = None
        elif not result['grasp_success']:
            result['failure_reason'] = 'grasp_failed' if case['primitive'] == 'pull' else 'push_engagement_failed'
        elif directional < 0:
            result['failure_reason'] = 'wrong_direction'
        else:
            result['failure_reason'] = 'insufficient_directional_progress'
        result['episode_status'] = 'complete'
    except Exception as exc:
        result['episode_status'] = 'crash'
        result['failure_reason'] = 'infrastructure_exception'
        result['exception'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if resources is not None:
            try:
                result['final_progress'] = resources['progress']()
            except Exception:
                pass
    finally:
        result['runtime_seconds'] = float(time.time() - started)
        resources = None
        gc.collect()
        torch.cuda.empty_cache()
    return jsonable(result)


def valid_existing(path: Path, checkpoint_sha: str, catalog_sha: str) -> bool:
    try:
        row = json.loads(path.read_text(encoding='utf-8'))
        return (row.get('episode_status') == 'complete' and row.get('checkpoint_sha256') == checkpoint_sha
                and row.get('catalog_sha256') == catalog_sha)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--formal-root', type=Path, required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--task', choices=['door_open', 'door_close', 'drawer_open', 'drawer_close'])
    args = parser.parse_args()
    checkpoint_sha = sha256(args.checkpoint)
    catalog_sha = sha256(CATALOG)
    cases = [json.loads(line) for line in CATALOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = list(enumerate(cases))
    if args.task:
        indexed = [(i, c) for i, c in indexed if c['task'] == args.task]
    if args.smoke:
        indexed = indexed[:2]
    else:
        indexed = [(i, c) for i, c in indexed if i % args.num_shards == args.shard_id]
    runtime = PA3FFPADPRuntime(args.checkpoint)
    for ordinal, (global_index, case) in enumerate(indexed, 1):
        episode_dir = (args.formal_root / case['task'] /
                       f"{int(case['target_index']):03d}_{case['shape_id']}_{case['target_link']}" /
                       f"trial_{int(case['trial_index']):02d}_seed_{case['seed']}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        result_path = episode_dir / 'result.json'
        if valid_existing(result_path, checkpoint_sha, catalog_sha):
            print(f'SKIP {ordinal}/{len(indexed)} {case["task"]} {case["shape_id"]}/{case["target_link"]} seed={case["seed"]}', flush=True)
            continue
        result = None
        for attempt in range(1, 3):
            with (episode_dir / f'attempt_{attempt}.log').open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run_episode(runtime, case, checkpoint_sha, catalog_sha)
            result['infrastructure_attempt'] = attempt
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if result['episode_status'] == 'complete':
                break
        print(f'DONE {ordinal}/{len(indexed)} global={global_index} {case["task"]} '
              f'{case["shape_id"]}/{case["target_link"]} seed={case["seed"]} '
              f'status={result["episode_status"]} grasp={result["grasp_success"]} '
              f'progress={result["directional_task_progress"]} final={result["final_success"]} '
              f'sec={result["runtime_seconds"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
