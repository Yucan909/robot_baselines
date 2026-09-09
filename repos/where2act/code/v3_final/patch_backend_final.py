#!/usr/bin/env python3
"""Create an independent final backend from v2.1 start-sync.

Unchanged: physics, PD, friction, contact monitor, phase scheduler, 5 cm pull,
planner start-state synchronization.
Changed: seeded ActionScore sampling, diagnostics, planner taxonomy, oracle-angle
logging, progress thresholds, and known failure classification.
"""
import argparse, hashlib, json, py_compile, shutil
from pathlib import Path
SRC_DEFAULT=Path('/home/feng/robot_baselines/common_env/where2act_backend_v2_1_startsync')
DST_DEFAULT=Path('/home/feng/robot_baselines/common_env/where2act_backend_final')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def once(text,old,new,label):
 n=text.count(old)
 if n!=1: raise RuntimeError(f'{label}: expected 1 match, got {n}')
 return text.replace(old,new,1)

def patch_policy(path):
 t=path.read_text()
 old='''        # Where2Act官方通常直接保存network.state_dict()
        if (
            isinstance(data, dict)
            and "state_dict" in data
            and isinstance(
                data["state_dict"],
                dict,
            )
        ):
            state = data[
                "state_dict"
            ]
        else:
            state = data
'''
 new='''        # Final supports pure state_dict and wrapped checkpoints.
        if (
            isinstance(data, dict)
            and "model" in data
            and isinstance(data["model"], dict)
        ):
            state = data["model"]
        elif (
            isinstance(data, dict)
            and "state_dict" in data
            and isinstance(data["state_dict"], dict)
        ):
            state = data["state_dict"]
        else:
            state = data
'''
 t=once(t,old,new,'checkpoint wrapper')
 old='''        masked_scores = (
            action_scores.clone()
        )

        masked_scores[
            ~mask_tensor
        ] = -torch.inf


        query_idx = int(
            torch.argmax(
                masked_scores
            ).item()
        )


'''
 new='''        # FINAL: deterministic seeded ActionScore sampling on valid target-link points.
        valid_idx_t = torch.nonzero(
            mask_tensor,
            as_tuple=False,
        ).reshape(-1)

        if valid_idx_t.numel() == 0:
            raise Where2ActPolicyError(
                "target_not_sampled"
            )

        valid_scores_t = action_scores[
            valid_idx_t
        ]

        valid_idx_np = (
            valid_idx_t.detach().cpu().numpy().astype(np.int64)
        )

        valid_scores_np = (
            valid_scores_t.detach().cpu().numpy().astype(np.float64)
        )

        weights = np.clip(
            valid_scores_np,
            0.0,
            None,
        )

        if (
            not np.all(np.isfinite(weights))
            or float(weights.sum()) <= 1e-12
        ):
            weights = np.ones_like(
                weights,
                dtype=np.float64,
            )

        probabilities = weights / float(weights.sum())
        query_rng = np.random.default_rng(int(seed) + 23063)
        chosen_local = int(
            query_rng.choice(
                len(valid_idx_np),
                p=probabilities,
            )
        )
        query_idx = int(valid_idx_np[chosen_local])
        chosen_valid_score = float(valid_scores_np[chosen_local])
        query_rank = int(
            1 + np.sum(valid_scores_np > chosen_valid_score)
        )
        top_order = np.argsort(
            -valid_scores_np,
            kind="mergesort",
        )[:min(20, len(valid_scores_np))]
        actionability_top_scores = valid_scores_np[top_order].tolist()
        actionability_top_sampled_indices = valid_idx_np[top_order].tolist()


'''
 t=once(t,old,new,'query sampling')
 old='''            "interaction_score":
                interaction_score,

            "sampled_query_index":
'''
 new='''            "interaction_score":
                interaction_score,

            "query_selection_mode":
                "seeded_actionscore_probability",

            "actionability_candidate_count":
                int(len(valid_idx_np)),

            "actionability_chosen_rank":
                int(query_rank),

            "actionability_top_scores":
                actionability_top_scores,

            "actionability_top_sampled_indices":
                actionability_top_sampled_indices,

            "sampled_query_index":
'''
 t=once(t,old,new,'query diagnostics')
 old='''            "critic_score":
                best_critic_score,

            "up_camera":
'''
 new='''            "critic_score":
                best_critic_score,

            "proposal_critic_scores":
                critic_scores.detach().cpu().numpy().astype(np.float64),

            "proposal_dirs1_camera":
                dirs1.detach().cpu().numpy().astype(np.float64),

            "proposal_dirs2_camera":
                dirs2.detach().cpu().numpy().astype(np.float64),

            "up_camera":
'''
 t=once(t,old,new,'proposal diagnostics')
 path.write_text(t)

def patch_runtime(path):
 t=path.read_text(); old='''        for token in [
            "IK_FAILED",
            "COLLISION_AWARE_OMPL_FAILED",
            "No solution found",
        ]
'''; new='''        for token in [
            "IK_FAILED",
            "COLLISION_AWARE_OMPL_FAILED",
            "No solution found",
            "OMPL路径包含碰撞state",
        ]
'''; path.write_text(once(t,old,new,'planner taxonomy'))

def patch_trial(path):
 t=path.read_text(); t=once(t,'''        "protocol_version":
            "where2act_backend_v2",
''','''        "protocol_version":
            "where2act_final_atomic_v1",
''','protocol')
 anchor='''CLOSE_LOCK_STEPS = 300


def run_trial(args):
'''
 helper=r'''CLOSE_LOCK_STEPS = 300


def _angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-10 or nb <= 1e-10:
        return None
    c = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _oracle_open_tangent_for_logging(object_info, contact_world):
    """Kinematic opening tangent for logging only; never used for control."""
    obj = object_info["object"]
    target_link = object_info["target_link"]
    target_joint = object_info["target_joint"]
    target_index = int(object_info["target_index"])
    q0 = np.asarray(obj.get_qpos(), dtype=np.float64).copy()
    try:
        qv0 = np.asarray(obj.get_qvel(), dtype=np.float64).copy()
    except Exception:
        qv0 = None
    T0 = np.asarray(target_link.get_pose().to_transformation_matrix(), dtype=np.float64)
    p = np.asarray(contact_world, dtype=np.float64).reshape(3)
    local = T0[:3, :3].T @ (p - T0[:3, 3])
    limits = np.asarray(target_joint.get_limits(), dtype=np.float64)
    lo, hi = float(limits[0, 0]), float(limits[0, 1])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        raise RuntimeError("oracle tangent invalid limits")
    eps = max(1e-5, 1e-3 * (hi - lo))
    q1 = q0.copy()
    new_q = min(float(q0[target_index]) + eps, hi - 1e-8)
    if new_q <= float(q0[target_index]) + 1e-10:
        raise RuntimeError("oracle tangent no positive perturbation")
    q1[target_index] = new_q
    try:
        obj.set_qpos(q1)
        T1 = np.asarray(target_link.get_pose().to_transformation_matrix(), dtype=np.float64)
        p1 = T1[:3, :3] @ local + T1[:3, 3]
    finally:
        obj.set_qpos(q0)
        if qv0 is not None:
            try:
                obj.set_qvel(qv0)
            except Exception:
                pass
    tangent = p1 - p
    n = float(np.linalg.norm(tangent))
    if (not np.isfinite(n)) or n <= 1e-10:
        raise RuntimeError("oracle tangent degenerate")
    return tangent / n


def _known_trial_failure(exc):
    text = f"{type(exc).__name__}: {exc}"
    mapping = [
        (("target_not_visible", "object_not_visible"), "target_not_visible"),
        (("target_not_sampled",), "target_not_sampled"),
        (("SAPIEN_WAYPOINT_TRACKING_FAILED", "SAPIEN_FINAL_TRACKING_FAILED"), "pregrasp_execution_failed"),
    ]
    for tokens, reason in mapping:
        if any(token in text for token in tokens):
            return reason
    return None


def run_trial(args):
'''
 t=once(t,anchor,helper,'helpers')
 old='''        result[
            "critic_score"
        ] = policy_result[
            "critic_score"
        ]

        result[
            "policy_success"
        ] = True
'''
 new='''        result[
            "critic_score"
        ] = policy_result[
            "critic_score"
        ]

        result[
            "policy_diagnostics"
        ] = {
            "query_selection_mode": policy_result.get("query_selection_mode"),
            "actionability_candidate_count": policy_result.get("actionability_candidate_count"),
            "actionability_chosen_rank": policy_result.get("actionability_chosen_rank"),
            "actionability_top_scores": policy_result.get("actionability_top_scores"),
            "sampled_query_index": policy_result.get("sampled_query_index"),
            "original_query_index": policy_result.get("original_query_index"),
            "proposal_index": policy_result.get("proposal_index"),
            "proposal_critic_scores": policy_result.get("proposal_critic_scores"),
        }

        result[
            "policy_success"
        ] = True
'''
 t=once(t,old,new,'policy logs')
 old='''        result[
            "pull_direction_world"
        ] = pull_direction

        # ====================================================
        # Collision-aware motion planning
'''
 new='''        result[
            "pull_direction_world"
        ] = pull_direction

        # Senior-requested angle diagnostic. Oracle is logging only.
        try:
            oracle_open_tangent = _oracle_open_tangent_for_logging(
                object_info,
                policy_result["interaction_point_world"],
            )
            result["oracle_open_tangent_world"] = oracle_open_tangent
            result["pull_angle_to_oracle_deg"] = _angle_deg(
                pull_direction,
                oracle_open_tangent,
            )
        except Exception as oracle_exc:
            result["oracle_open_tangent_error"] = (
                f"{type(oracle_exc).__name__}: {oracle_exc}"
            )

        # ====================================================
        # Collision-aware motion planning
'''
 t=once(t,old,new,'oracle logs')
 old='''        result[
            "final_progress"
        ] = final_progress

        result[
            "opening_success"
        ] = opening_success
'''
 new='''        result[
            "final_progress"
        ] = final_progress

        delta_progress = float(final_progress - pre_pull_progress)
        result["delta_progress"] = delta_progress
        result["positive_delta_progress"] = bool(delta_progress > 0.0)
        result["final_ge_20pct"] = bool(final_progress >= 0.20)
        result["final_ge_25pct"] = bool(final_progress >= 0.25)
        result["final_ge_30pct"] = bool(final_progress >= 0.30)
        result["final_ge_40pct"] = bool(final_progress >= 0.40)

        result[
            "opening_success"
        ] = opening_success
'''
 t=once(t,old,new,'progress logs')
 old='''    except Exception as exc:

        result[
            "implementation_error"
        ] = True

        result[
            "failure_reason"
        ] = (
            "implementation_error"
        )

        result[
            "implementation_exception"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        result[
            "runtime_seconds"
        ] = float(
            time.time()
            - start_time
        )

        save_json(
            result_path,
            result,
        )

        raise
'''
 new='''    except Exception as exc:

        known_reason = _known_trial_failure(exc)
        result["implementation_exception"] = f"{type(exc).__name__}: {exc}"
        result["runtime_seconds"] = float(time.time() - start_time)

        if known_reason is not None:
            result["implementation_error"] = False
            result["failure_reason"] = known_reason
            if known_reason == "pregrasp_execution_failed":
                result["pregrasp_execution_success"] = False
            save_json(result_path, result)
            print("TRIAL KNOWN METHOD/EXECUTION FAILURE:", known_reason)
            return result

        result["implementation_error"] = True
        result["failure_reason"] = "implementation_error"
        save_json(result_path, result)
        raise
'''
 t=once(t,old,new,'outer taxonomy'); path.write_text(t)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--src',default=str(SRC_DEFAULT)); ap.add_argument('--dst',default=str(DST_DEFAULT)); a=ap.parse_args(); src=Path(a.src).expanduser().resolve(); dst=Path(a.dst).expanduser().resolve()
 if not src.is_dir(): raise FileNotFoundError(src)
 if dst.exists(): shutil.rmtree(dst)
 shutil.copytree(src,dst); patch_policy(dst/'where2act_policy.py'); patch_runtime(dst/'where2act_runtime.py'); patch_trial(dst/'run_trial.py')
 for f in ['where2act_policy.py','where2act_runtime.py','run_trial.py']: py_compile.compile(str(dst/f),doraise=True)
 changed=['where2act_policy.py','where2act_runtime.py','run_trial.py']; unchanged=['backend_v2_physics.py','panda_controller.py','contact_monitor.py','motion_planner_collision_worker.py','motion_planner_worker.py','where2act_observation_adapter.py']; m={'source':str(src),'destination':str(dst),'protocol_version':'where2act_final_atomic_v1','physics_changed':False,'atomic_pull_distance_m':0.05,'patched_hashes':{f:sha(dst/f) for f in changed},'unchanged_core_hashes':{f:sha(dst/f) for f in unchanged if (dst/f).exists()}}; (dst/'FINAL_BACKEND_PATCH_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2)); print('\nFINAL BACKEND PATCH: PASS')
if __name__=='__main__': main()
