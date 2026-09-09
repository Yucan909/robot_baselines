"""Candidate ranking fitted on TRAIN and selected on object-level Open DEV."""
from __future__ import annotations

import numpy as np

from candidate_rank_devselected_v4 import rank_candidates as rank_v4

SURFACE_TARGET_M = 0.025
DISPLACEMENT_WEIGHT = 0.25
TRAIN_DISPLACEMENT = {
    "door_open": 0.29371519073792934,
    "drawer_open": 0.166679946326512,
}
TRAIN_MEDIAN = {
    "door_open": [0.0009138099384247869, 0.0004931128075580196, 0.0019008799982202383, 1.9459101490553132, 1.0986122886681096, 1.0986122886681096, 0.0210258208300127, 0.0008847534869683132, -0.016976271269369098, 0.0012851235128832454, 1.0137648662698266, -0.008094147784832844, -0.0016176857793289033, -0.0025418266958746384],
    "drawer_open": [0.0001992516644768113, 0.0001102260342728801, 0.01397357787952074, 2.3978952727983707, 1.791759469228055, 1.791759469228055, 0.11767444193427454, 0.00009673414701855042, -0.0006383559791146801, -0.1583510120026454, 1.000008775671882, 0.00003480715274979049, 0.00036190454415113585, -0.002462881278772344],
}
TRAIN_SCALE = {
    "door_open": [0.017281243041159963, 0.01122406541194086, 0.02203851369415821, 0.5288062718835643, 1.2562038078100681, 1.2562038078100681, 0.03117288196257683, 0.03, 0.3177823877683529, 0.18023509063500817, 0.1, 0.8189200932103012, 0.1, 0.1],
    "drawer_open": [0.012965207417557226, 0.01072775979305474, 0.015626805536365692, 0.47213950175633984, 0.6011425692811642, 0.6011425692811642, 0.043829199240374345, 0.03, 0.04210992692191685, 0.042427923198471884, 0.1, 0.1, 0.1, 0.1],
}


def _rotvec(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(matrix).as_rotvec()


def _features(pc: np.ndarray, poses: np.ndarray) -> np.ndarray:
    first = poses[0]
    local = (pc - first[:3, 3]) @ first[:3, :3]
    nearest = local[int(np.argmin(np.linalg.norm(local, axis=1)))]
    mask = ((np.abs(local[:, 0]) < .035) & (local[:, 2] > -.04)
            & (local[:, 2] < .08) & (np.abs(local[:, 1]) < .08))
    inside = local[mask]
    pos = int(np.sum(inside[:, 1] > 0)) if len(inside) else 0
    neg = int(np.sum(inside[:, 1] < 0)) if len(inside) else 0
    span = float(np.ptp(inside[:, 1])) if len(inside) else 0.0
    xyz = poses[:, :3, 3]
    displacement_local = (xyz[-1] - xyz[0]) @ first[:3, :3]
    path = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
    relrot = _rotvec(first[:3, :3].T @ poses[-1, :3, :3])
    return np.concatenate([
        nearest, np.log1p([len(inside), pos, neg]), [span], displacement_local,
        [path / max(chord, 1e-9)], relrot,
    ])


def rank_candidates(point_cloud: np.ndarray, camera_pose: np.ndarray,
                    candidate_poses: np.ndarray, task: str, mode: str = "hybrid"):
    pc = np.asarray(point_cloud, dtype=np.float64).reshape(1024, 3)
    poses = np.asarray(candidate_poses, dtype=np.float64)
    if poses.shape != (32, 16, 4, 4) or task not in TRAIN_MEDIAN:
        raise RuntimeError((poses.shape, task))
    if mode == "v4":
        return rank_v4(pc, camera_pose, poses, task)
    vectors = np.stack([_features(pc, p) for p in poses])
    z = np.clip((vectors - np.asarray(TRAIN_MEDIAN[task])) /
                np.asarray(TRAIN_SCALE[task]), -8.0, 8.0)
    grasp = np.sqrt(np.mean(z[:, :7] ** 2, axis=1))
    all_score = np.sqrt(np.mean(z ** 2, axis=1))
    xyz = poses[:, :, :3, 3]
    surface = np.min(np.linalg.norm(pc[None, :, :] - xyz[:, :1, :], axis=2), axis=1)
    displacement = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
    base = (np.abs(surface - SURFACE_TARGET_M) + DISPLACEMENT_WEIGHT
            * np.abs(displacement - TRAIN_DISPLACEMENT[task]))
    if mode == "pure":
        component = grasp if task == "door_open" else all_score
        score = component
        rule = "TRAIN robust grasp likelihood" if task == "door_open" else "TRAIN robust all-geometry likelihood"
    elif mode == "hybrid":
        component = grasp
        weight = .08 if task == "door_open" else .04
        score = base + weight * component
        rule = f"V4 base + {weight} * TRAIN robust grasp likelihood"
    else:
        raise ValueError(mode)
    order = np.argsort(score, kind="stable").astype(int).tolist()
    selected = order[0]
    return order, {
        "version": "train_geometry_open_devselected_v11", "mode": mode,
        "formal_success_used": False, "rule": rule,
        "selected_candidate_index": selected,
        "selected_score": float(score[selected]),
        "selected_surface_distance_m": float(surface[selected]),
        "selected_displacement_m": float(displacement[selected]),
        "selected_grasp_prior_score": float(grasp[selected]),
        "selected_all_prior_score": float(all_score[selected]),
        "scores": score.tolist(),
    }
