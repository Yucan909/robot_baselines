#!/usr/bin/env bash
set -euo pipefail

PY="$HOME/miniconda3/envs/where2act/bin/python"
BASE="$HOME/robot_baselines"
CODE="$BASE/repos/where2act/code"
SRC="$CODE/faithful_baseline"
TRAIN_CODE="$CODE/four_task_train_v7_noaff_schema_robust"
INDEX_ROOT="$BASE/results/where2act/four_task_train_v7_noaff_schema_robust_indices"
RUN_ROOT_BASE="$BASE/repos/where2act/logs/four_task_train_v7_noaff_schema_robust"
WORK="$BASE/.work/where2act_v7"
MASTER_LOG="$BASE/results/where2act/where2act_v7_build_and_train.log"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$RUN_ROOT_BASE/$RUN_ID"

if [ ! -x "$PY" ]; then
  echo "ERROR: missing where2act python: $PY"
  exit 2
fi

# V5 has its own namespace.  Clean only V5 temporary/index/adapter outputs so a failed
# previous attempt can never leave stale files that are mistaken for this run.
rm -rf "$WORK" "$INDEX_ROOT" "$TRAIN_CODE"
mkdir -p "$WORK" "$INDEX_ROOT" "$RUN_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Use the environment by absolute path; do not activate/deactivate conda inside this script.
"$PY" - <<'PY'
import torch, numpy
print("runtime python:", __import__('sys').executable)
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu:", torch.cuda.get_device_name(0))
PY

BUILDER="$WORK/build_four_task_indices_v7_noaff_schema_robust.py"
PATCHER="$WORK/prepare_training_adapter_v7_noaff_schema_robust.py"
PREFLIGHT="$WORK/preflight_training_loader_v7_noaff_schema_robust.py"
FINALIZER="$WORK/finalize_run_v7_noaff_schema_robust.py"

cat > "$BUILDER" <<'PY'
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, math, gc
import numpy as np
import sapien.core as sapien

HOME = Path.home()
BASE = HOME / "robot_baselines"
DATA = BASE / "data" / "where2act_four_task"
CODE = BASE / "repos" / "where2act" / "code"
PARTNET = BASE / "data" / "partnet-mobility"
PANDA_URDF = BASE / "common_env" / "assets" / "panda_articubot" / "panda.urdf"
V3_SPLIT = BASE / "configs" / "where2act_v3"
FOUR_SPLIT = BASE / "configs" / "where2act_four_task_v1"
DOOR_OPEN_ROOT = DATA / "door_open" / "dataset_pc_aff_static" / "episodes"
DOOR_CLOSE_ROOT = DATA / "door_close" / "dataset_pc_aff_static" / "raw" / "data" / "single"
DRAWER_OPEN_ROOT = DATA / "drawer_open" / "dataset_pc_aff_static" / "episodes"
DRAWER_CLOSE_ROOT = DATA / "drawer_close" / "dataset_pc_aff_static" / "episodes" / "single"
OUT = BASE / "results" / "where2act" / "four_task_train_v7_noaff_schema_robust_indices"
OUT.mkdir(parents=True, exist_ok=True)

OBJECT_SCALE = 0.75
LOCAL_MOTION_DISTANCE = 0.02
DRAWER_CAMERA_ID = "fixed_front_single_view"
TASKS = ("door_open", "door_close", "drawer_open", "drawer_close")


def banner(s):
    print("\n" + "=" * 120)
    print(s)
    print("=" * 120, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_set(path):
    return {x.strip() for x in Path(path).read_text().splitlines() if x.strip()}


def read_id_json(path):
    """Read native object-level split files without assuming one JSON schema.

    Known local datasets have appeared as either:
      {"ids": [...]}
      {"objects": {"shape_id": {...}, ...}}
      [...]
    We only return object/shape IDs; target-link metadata is not consumed here.
    """
    path = Path(path)
    j = json.loads(path.read_text())
    if isinstance(j, list):
        ids = j
    elif isinstance(j, dict) and isinstance(j.get("ids"), list):
        ids = j["ids"]
    elif isinstance(j, dict) and isinstance(j.get("objects"), dict):
        ids = list(j["objects"].keys())
    elif isinstance(j, dict) and isinstance(j.get("objects"), list):
        ids = []
        for x in j["objects"]:
            if isinstance(x, (str, int)):
                ids.append(x)
            elif isinstance(x, dict):
                v = x.get("shape_id", x.get("id", x.get("object_id")))
                if v is not None:
                    ids.append(v)
    else:
        raise RuntimeError(f"unsupported split schema: {path}; top-level type={type(j).__name__}")
    out = {str(x) for x in ids if str(x)}
    if not out:
        raise RuntimeError(f"empty object split: {path}")
    return out


def scalar(z, key, default=None):
    if key not in z.files:
        return default
    x = np.asarray(z[key])
    try:
        return x.item()
    except Exception:
        return x


def decode_result(z):
    if "result_json" not in z.files:
        return None
    x = scalar(z, "result_json")
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    if isinstance(x, dict):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return None


def canonical_task_direction(raw_mode):
    """Normalize generator-specific task_mode strings to physical open/close semantics.

    Local datasets are not schema-consistent: e.g. Drawer Open records `open`, while
    Drawer Close records `door_close`.  The directory/task split is authoritative for object
    category; task_mode is used only to detect a *semantic contradiction* in open-vs-close.
    Missing/unknown spellings are diagnostic, not a reason to throw away an otherwise valid
    trajectory.
    """
    if raw_mode is None:
        return None
    m = str(raw_mode).strip().lower().replace('-', '_').replace(' ', '_')
    if not m:
        return None
    if ('close' in m) or m in {'push', 'pushing'}:
        return 'close'
    if ('open' in m) or m in {'pull', 'pulling'}:
        return 'open'
    return None


def validate_task_mode(task, raw_mode, counter, path):
    expected = 'open' if str(task).endswith('_open') else 'close'
    got = canonical_task_direction(raw_mode)
    counter[f"raw:{raw_mode!s}"] += 1
    if got is None:
        counter['unknown_or_missing_accepted_by_task_root'] += 1
        return
    if got != expected:
        raise RuntimeError(
            f"{task} task_mode contradicts task root: expected {expected}, "
            f"raw={raw_mode!r}, path={path}"
        )
    counter[f"semantic:{got}"] += 1


def canonical_status_bool(raw_status):
    if raw_status is None:
        return None
    s = str(raw_status).strip().lower()
    if s in {'success', 'succeeded', 'pass', 'passed', 'true', '1', 'ok'}:
        return True
    if s in {'failure', 'failed', 'fail', 'false', '0'}:
        return False
    return None


def phase_strings(x):
    out = []
    for v in np.asarray(x).reshape(-1):
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        out.append(str(v))
    return np.asarray(out)


def normalize(v, eps=1e-8):
    try:
        v = np.asarray(v, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n <= eps:
        return None
    return v / n


def angle_deg(a, b):
    a = normalize(a); b = normalize(b)
    if a is None or b is None:
        return None
    return float(np.degrees(np.arccos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))


def project_perpendicular(v, axis):
    v = normalize(v); axis = normalize(axis)
    if v is None or axis is None:
        return None
    return normalize(v - float(np.dot(v, axis)) * axis)


def transform_point(T, p):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def inverse_transform_point(T, p):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return T[:3, :3].T @ (p - T[:3, 3])


def panda_root_pose(base):
    x, y, yaw, z = map(float, np.asarray(base, dtype=np.float64).reshape(4))
    return sapien.Pose([x, y, z], [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def phase_motion(ee, phases, phase_name, distance_m=LOCAL_MOTION_DISTANCE):
    ee = np.asarray(ee, dtype=np.float64)
    phases = phase_strings(phases)
    if ee.ndim != 2 or ee.shape[1] != 3 or len(ee) != len(phases):
        return None
    ids = np.flatnonzero(phases == str(phase_name))
    if len(ids) < 2:
        return None
    p_end = ee[ids[-1]]
    chosen = int(ids[0])
    for j in range(len(ids) - 2, -1, -1):
        idx = int(ids[j])
        if np.linalg.norm(p_end - ee[idx]) >= float(distance_m):
            chosen = idx
            break
    return normalize(p_end - ee[chosen])


def nearest_finite(p_ref, pc):
    p_ref = np.asarray(p_ref, dtype=np.float64).reshape(3)
    pc = np.asarray(pc, dtype=np.float64).reshape(-1, 3)
    ids = np.flatnonzero(np.all(np.isfinite(pc), axis=1))
    if len(ids) == 0:
        return None
    d = np.linalg.norm(pc[ids] - p_ref[None, :], axis=1)
    j = int(np.argmin(d)); idx = int(ids[j])
    return idx, pc[idx].copy(), float(d[j])


def camera_ok(T):
    try:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    if not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=1e-4, rtol=1e-4) and abs(np.linalg.det(R) - 1.0) <= 1e-4)


def camera_key(T, decimals=6):
    if not camera_ok(T):
        return None
    return tuple(np.round(np.asarray(T, dtype=np.float64), decimals).reshape(-1).tolist())


def path_sid(path):
    p = Path(path); parts = p.parts
    if "single" in parts:
        i = max(i for i, x in enumerate(parts) if x == "single")
        if i + 1 < len(parts):
            return parts[i + 1]
    if "episodes" in parts:
        i = parts.index("episodes"); j = i + 1
        if j < len(parts) and parts[j] == "single":
            j += 1
        if j < len(parts):
            return parts[j]
    return None


def path_link(path):
    for x in Path(path).parts:
        if x.startswith("link_"):
            return x
    return None


def scene_identity(path):
    """Open/Close scene identity: same object/link/repeat/base, independent of candidate."""
    p = Path(path)
    sid = path_sid(p); link = path_link(p)
    repeat = next((x for x in p.parts if x.startswith("repeat_")), None)
    base = next((x for x in p.parts if x.startswith("base_") and x != p.name), None)
    if None in (sid, link, repeat, base):
        return None
    return (str(sid), str(link), str(repeat), str(base))


def trajectory_identity(path):
    """Exact Open/Close candidate identity, used only for stronger verification when available."""
    p = Path(path)
    scene = scene_identity(p)
    if scene is None:
        return None
    name = p.name.removeprefix("reverse_")
    return (*scene, str(name))


DOOR_TRAIN = read_set(V3_SPLIT / "train_shapes.txt")
DOOR_DEV = read_set(V3_SPLIT / "dev_shapes.txt")
DOOR_FORMAL = read_set(V3_SPLIT / "formal_holdout_shapes.txt")
# Drawer uses each task's native TRAIN/VAL lists.  The previously frozen drawer_dev_shapes
# file is used only as an internal DEV subset *inside that task's native TRAIN*.  Native
# VAL objects are never opened by this builder.
DRAWER_DEV_CANDIDATES = read_set(FOUR_SPLIT / "drawer_dev_shapes.txt")
DRAWER_OPEN_NATIVE_TRAIN = read_id_json(DATA / "drawer_open" / "dataset_pc_aff_static" / "train_id_list.json")
DRAWER_OPEN_FORMAL = read_id_json(DATA / "drawer_open" / "dataset_pc_aff_static" / "val_id_list.json")
DRAWER_CLOSE_NATIVE_TRAIN = read_id_json(DATA / "drawer_close" / "dataset_pc_aff_static" / "train_id_list.json")
DRAWER_CLOSE_FORMAL = read_id_json(DATA / "drawer_close" / "dataset_pc_aff_static" / "val_id_list.json")
DRAWER_OPEN_DEV = DRAWER_OPEN_NATIVE_TRAIN & DRAWER_DEV_CANDIDATES
DRAWER_CLOSE_DEV = DRAWER_CLOSE_NATIVE_TRAIN & DRAWER_DEV_CANDIDATES
DRAWER_OPEN_TRAIN = DRAWER_OPEN_NATIVE_TRAIN - DRAWER_OPEN_DEV
DRAWER_CLOSE_TRAIN = DRAWER_CLOSE_NATIVE_TRAIN - DRAWER_CLOSE_DEV
if not DRAWER_OPEN_DEV or not DRAWER_CLOSE_DEV:
    raise RuntimeError("task-native Drawer internal DEV subset is empty")
for name, tr, dv, fo in (
    ("drawer_open", DRAWER_OPEN_TRAIN, DRAWER_OPEN_DEV, DRAWER_OPEN_FORMAL),
    ("drawer_close", DRAWER_CLOSE_TRAIN, DRAWER_CLOSE_DEV, DRAWER_CLOSE_FORMAL),
):
    if (tr & dv) or (tr & fo) or (dv & fo):
        raise RuntimeError(f"Drawer split overlap: {name}")

FORMAL_BY_TASK = {
    "door_open": DOOR_FORMAL,
    "door_close": DOOR_FORMAL,
    "drawer_open": DRAWER_OPEN_FORMAL,
    "drawer_close": DRAWER_CLOSE_FORMAL,
}


def split_for(task, sid):
    if task.startswith("door"):
        if sid in DOOR_TRAIN: return "train"
        if sid in DOOR_DEV: return "dev"
        if sid in DOOR_FORMAL: return "formal"
    elif task == "drawer_open":
        if sid in DRAWER_OPEN_TRAIN: return "train"
        if sid in DRAWER_OPEN_DEV: return "dev"
        if sid in DRAWER_OPEN_FORMAL: return "formal"
    elif task == "drawer_close":
        if sid in DRAWER_CLOSE_TRAIN: return "train"
        if sid in DRAWER_CLOSE_DEV: return "dev"
        if sid in DRAWER_CLOSE_FORMAL: return "formal"
    return None


# SAPIEN allows one Engine per process.  Reuse it for all object/Panda scenes.
GLOBAL_ENGINE = sapien.Engine(0, 0.001, 0.005)


def find_object_urdf(sid):
    root = PARTNET / str(sid)
    for p in (root / "mobility_vhacd.urdf", root / "mobility.urdf"):
        if p.is_file():
            return p
    return None


def load_object(sid):
    urdf = find_object_urdf(sid)
    if urdf is None:
        raise FileNotFoundError(f"URDF missing for {sid}")
    eng = GLOBAL_ENGINE
    scene = eng.create_scene()
    loader = scene.create_urdf_loader(); loader.fix_root_link = True; loader.scale = OBJECT_SCALE
    obj = loader.load(str(urdf))
    if obj is None:
        raise RuntimeError(f"object load failed {sid} {urdf}")
    obj.set_root_pose(sapien.Pose())
    links = {x.get_name(): x for x in obj.get_links()}
    return eng, scene, obj, links


def target_joint_index(active_joints, link_name):
    for i, joint in enumerate(active_joints):
        try:
            if joint.get_child_link().get_name() == str(link_name):
                return int(i)
        except Exception:
            pass
    return None


def kinematic_task_tangent(obj, active_joints, jidx, target_link, q0, contact_world, task_sign):
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    if jidx is None or not (0 <= int(jidx) < len(active_joints)) or len(q0) != int(obj.dof):
        return None
    joint = active_joints[int(jidx)]
    limits = np.asarray(joint.get_limits(), dtype=np.float64)
    if limits.shape != (1, 2):
        return None
    lo, hi = float(limits[0, 0]), float(limits[0, 1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    obj.set_qpos(q0)
    T0 = np.asarray(target_link.get_pose().to_transformation_matrix(), dtype=np.float64)
    local = inverse_transform_point(T0, contact_world)
    q1 = q0.copy(); old = float(q1[int(jidx)]); eps = max(1e-5, 1e-3 * (hi - lo))
    if float(task_sign) > 0:
        new = min(old + eps, hi - 1e-8)
        if new <= old + 1e-10: return None
    else:
        new = max(old - eps, lo + 1e-8)
        if new >= old - 1e-10: return None
    q1[int(jidx)] = new
    try:
        obj.set_qpos(q1)
        T1 = np.asarray(target_link.get_pose().to_transformation_matrix(), dtype=np.float64)
        p1 = transform_point(T1, local)
    finally:
        obj.set_qpos(q0)
    return normalize(p1 - np.asarray(contact_world, dtype=np.float64).reshape(3))


def geometry_diagnostics(contact_world, p_ref_world, R_hand_world, task_motion, kin_motion):
    """Diagnostics only.  No geometric threshold is allowed to select/drop a training row."""
    contact_world = np.asarray(contact_world, dtype=np.float64).reshape(3)
    p_ref_world = np.asarray(p_ref_world, dtype=np.float64).reshape(3)
    R = np.asarray(R_hand_world, dtype=np.float64).reshape(3, 3)
    delta = contact_world - p_ref_world
    contact_angle = angle_deg(delta, R[:, 2])
    hand_delta = R.T @ delta
    lateral = float(np.linalg.norm(hand_delta[:2])) if np.all(np.isfinite(hand_delta)) else float("nan")
    zref = float(hand_delta[2]) if np.all(np.isfinite(hand_delta)) else float("nan")
    tangent = angle_deg(task_motion, kin_motion) if kin_motion is not None else None
    return {
        "contact_angle_deg": float(contact_angle) if contact_angle is not None else float("nan"),
        "contact_lateral_m": lateral,
        "contact_z_m": zref,
        "contact_distance_m": float(np.linalg.norm(delta)),
        "tangent_consistency_deg": float(tangent) if tangent is not None else float("nan"),
    }


# Panda FK reconstructs the recorded hand orientation only.
PANDA_ENGINE = GLOBAL_ENGINE
PANDA_SCENE = PANDA_ENGINE.create_scene()
PANDA_LOADER = PANDA_SCENE.create_urdf_loader(); PANDA_LOADER.fix_root_link = True
PANDA = PANDA_LOADER.load(str(PANDA_URDF))
if PANDA is None:
    raise RuntimeError("Panda load failed")
PANDA_LINKS = {x.get_name(): x for x in PANDA.get_links()}
PANDA_HAND = PANDA_LINKS["panda_hand"]


def panda_hand_R(robot_q, base):
    q = np.asarray(robot_q, dtype=np.float64).reshape(9)
    PANDA.set_root_pose(panda_root_pose(base)); PANDA.set_qpos(q)
    return np.asarray(PANDA_HAND.get_pose().to_transformation_matrix(), dtype=np.float64)[:3, :3]


def validate_dirs(d1w, d2w, Tcam):
    d1w = normalize(d1w); d2w = normalize(d2w)
    if d1w is None or d2w is None or abs(float(np.dot(d1w, d2w))) > 1e-4 or not camera_ok(Tcam):
        return None
    R = np.asarray(Tcam, dtype=np.float64)[:3, :3]
    d1m = normalize(d1w @ R); d2m = normalize(d2w @ R)
    if d1m is None or d2m is None:
        return None
    a1 = angle_deg(d1w, d1m @ R.T); a2 = angle_deg(d2w, d2m @ R.T)
    if a1 is None or a2 is None or max(a1, a2) > 1e-3:
        return None
    return d1m, d2m


def select_door_obs(z):
    required = ("observation_point_cloud", "observation_trajectory_step", "operation_start_index", "object_qpos", "observation_camera_pose")
    miss = [k for k in required if k not in z.files]
    if miss:
        raise ValueError(f"missing door observation fields {miss}")
    pc = np.asarray(z["observation_point_cloud"], dtype=np.float64)
    if pc.ndim == 2 and pc.shape[1] == 3:
        pc = pc[None, :, :]
    if pc.ndim != 3 or pc.shape[2] != 3:
        raise ValueError(f"bad observation pc {pc.shape}")
    steps = np.asarray(z["observation_trajectory_step"]).reshape(-1).astype(np.int64)
    F = pc.shape[0]
    if len(steps) != F:
        raise ValueError(f"frame/step mismatch F={F}, steps={steps.shape}")
    op = int(np.asarray(z["operation_start_index"]).item())
    eligible = np.flatnonzero(steps <= op)
    if len(eligible) == 0:
        raise ValueError("no nonfuture observation")
    j = int(eligible[np.argmax(steps[eligible])]); step = int(steps[j])
    oq = np.asarray(z["object_qpos"], dtype=np.float64)
    if oq.ndim != 2 or not (0 <= step < len(oq)):
        raise ValueError("bad qobs step")
    qobs = oq[step].copy()
    Tc = np.asarray(z["observation_camera_pose"], dtype=np.float64)
    if Tc.shape == (4, 4): T = Tc
    elif Tc.ndim == 3 and Tc.shape[0] == F and Tc.shape[1:] == (4, 4): T = Tc[j]
    else: raise ValueError(f"bad observation camera {Tc.shape}")
    return pc[j], qobs, T, j, step


def result_camera_id(rj):
    if not isinstance(rj, dict):
        return None
    pcj = rj.get("pointcloud", None)
    if isinstance(pcj, dict) and pcj.get("camera_id", None) is not None:
        return str(pcj.get("camera_id"))
    return None


def _insert_unique_pose(store, key, value, kind, source_path, atol=1e-7):
    if key is None:
        return
    value = np.asarray(value, dtype=np.float64).copy()
    if key in store and not np.allclose(store[key], value, atol=atol, rtol=atol):
        raise RuntimeError(f"conflicting Drawer Close {kind} for {key}: {source_path}")
    store[key] = value


def resolve_drawer_scene_dictionary():
    """Resolve Drawer scene metadata from Close native TRAIN only.

    The senior's Open/Close contract is scene-level: same shape/link/repeat/base should use
    the same camera/base setup even when the sampled operation candidate differs.  We therefore
    build both scene-level and exact-candidate dictionaries.  Native VAL/formal content is never
    opened; its paths are filtered before np.load.
    """
    by_id = defaultdict(Counter)
    representative = {}
    rows = Counter()
    scene_camera = {}
    scene_base = {}
    exact_camera = {}
    exact_base = {}
    formal_named = 0
    opened_native_train = 0

    for p in sorted(DRAWER_CLOSE_ROOT.rglob("*.npz")):
        if p.parent.name != "trajectory":
            continue
        sid = path_sid(p)
        if sid in DRAWER_CLOSE_FORMAL:
            formal_named += 1
            continue
        if sid not in DRAWER_CLOSE_NATIVE_TRAIN:
            continue

        with np.load(p, allow_pickle=False) as z:
            opened_native_train += 1
            rj = decode_result(z)
            cid = result_camera_id(rj)
            if cid is None:
                raise RuntimeError(f"drawer_close native TRAIN missing camera_id: {p}")
            if "observation_camera_pose" not in z.files:
                raise RuntimeError(f"drawer_close native TRAIN missing explicit camera pose: {p}")
            if "base_pose" not in z.files:
                raise RuntimeError(f"drawer_close native TRAIN missing base_pose: {p}")

            T = np.asarray(z["observation_camera_pose"], dtype=np.float64)
            base = np.asarray(z["base_pose"], dtype=np.float64).reshape(-1)
            if T.shape != (4, 4) or not camera_ok(T):
                raise RuntimeError(f"drawer_close native TRAIN bad camera pose {T.shape}: {p}")
            if base.shape != (4,) or not np.all(np.isfinite(base)):
                raise RuntimeError(f"drawer_close native TRAIN bad base pose {base.shape}: {p}")

            ck = camera_key(T)
            by_id[cid][ck] += 1
            representative.setdefault((cid, ck), T.copy())
            rows[cid] += 1

            scene_key = scene_identity(p)
            exact_key = trajectory_identity(p)
            _insert_unique_pose(scene_camera, scene_key, T, "scene camera", p)
            _insert_unique_pose(scene_base, scene_key, base, "scene base_pose", p)
            _insert_unique_pose(exact_camera, exact_key, T, "exact-candidate camera", p)
            _insert_unique_pose(exact_base, exact_key, base, "exact-candidate base_pose", p)

    if DRAWER_CAMERA_ID not in by_id:
        raise RuntimeError(f"shared drawer camera id absent from Close native TRAIN: {DRAWER_CAMERA_ID}")
    if len(by_id[DRAWER_CAMERA_ID]) != 1:
        raise RuntimeError(
            f"shared drawer camera id is not unique: {DRAWER_CAMERA_ID}: "
            f"{len(by_id[DRAWER_CAMERA_ID])}"
        )

    ck, count = by_id[DRAWER_CAMERA_ID].most_common(1)[0]
    T = representative[(DRAWER_CAMERA_ID, ck)].copy()
    print("Drawer Open/Close scene counterpart resolution:")
    print("  camera_id:", DRAWER_CAMERA_ID)
    print("  Close native-TRAIN trajectories opened:", opened_native_train)
    print("  Close native-TRAIN explicit camera rows:", int(count))
    print("  scene-level camera entries:", len(scene_camera))
    print("  scene-level base entries:", len(scene_base))
    print("  exact same-candidate camera entries:", len(exact_camera))
    print("  exact same-candidate base entries:", len(exact_base))
    print("  unique pose count for camera_id:", len(by_id[DRAWER_CAMERA_ID]))
    print(T)
    print("  native VAL paths named/not opened:", formal_named)

    return (
        {DRAWER_CAMERA_ID: T},
        {k: int(v) for k, v in rows.items()},
        scene_camera,
        scene_base,
        exact_camera,
        exact_base,
    )


def drawer_camera(z, rj, camera_dict, scene_camera, exact_camera, path):
    """Resolve camera using own metadata, then paired Close scene, then unique camera_id.

    Exact-candidate evidence is used as the strongest cross-check when it exists.  Scene-level
    pairing is the actual fallback because camera/view is a property of the scene, not of the
    sampled contact candidate.
    """
    own = None
    own_key = None
    for key in ("observation_camera_pose", "initial_camera_pose", "camera_pose"):
        if key in z.files:
            T = np.asarray(z[key], dtype=np.float64)
            if T.shape == (4, 4) and camera_ok(T):
                own = T.copy()
                own_key = key
                break

    exact_key = trajectory_identity(path)
    scene_key = scene_identity(path)
    exact_pair = exact_camera.get(exact_key) if exact_key is not None else None
    scene_pair = scene_camera.get(scene_key) if scene_key is not None else None

    # Exact and scene dictionaries themselves must agree whenever both exist.
    if exact_pair is not None and scene_pair is not None and not np.allclose(
        exact_pair, scene_pair, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError(f"Drawer exact/scene camera disagreement: {path}")

    counterpart = exact_pair if exact_pair is not None else scene_pair
    counterpart_tag = (
        "exact_same_candidate_close_counterpart"
        if exact_pair is not None
        else "same_scene_close_counterpart"
    )

    if own is not None and counterpart is not None:
        # Each observation must use the camera that actually generated its own point cloud.
        # Open/Close pairing is a fallback / diagnostic relation, not an equality constraint.
        # Different explicit matrices are allowed and never cause replacement of the native camera.
        if np.allclose(own, counterpart, atol=1e-6, rtol=1e-6):
            return own, f"own_explicit+close_scene_matched:{own_key}"
        return own, f"own_explicit+close_scene_diff_diagnostic:{own_key}"

    if own is not None:
        return own, f"own_explicit:{own_key}"

    if counterpart is not None:
        return counterpart.copy(), counterpart_tag

    # Final camera-only fallback: allowed because Close native TRAIN proved this camera_id maps
    # to exactly one 4x4 pose across the dataset.  No analogous base-pose fallback is permitted.
    cid = result_camera_id(rj)
    if cid is not None and cid in camera_dict:
        return camera_dict[cid].copy(), f"unique_close_train_camera_id:{cid}"

    return None, None


def drawer_base_pose(z, scene_base, exact_base, path):
    """Resolve base pose from own metadata or the paired Close scene; never across scenes."""
    own = None
    if "base_pose" in z.files:
        v = np.asarray(z["base_pose"], dtype=np.float64).reshape(-1)
        if v.shape == (4,) and np.all(np.isfinite(v)):
            own = v.copy()

    exact_key = trajectory_identity(path)
    scene_key = scene_identity(path)
    exact_pair = exact_base.get(exact_key) if exact_key is not None else None
    scene_pair = scene_base.get(scene_key) if scene_key is not None else None

    if exact_pair is not None and scene_pair is not None and not np.allclose(
        exact_pair, scene_pair, atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError(f"Drawer exact/scene base-pose disagreement: {path}")

    counterpart = exact_pair if exact_pair is not None else scene_pair
    counterpart_tag = (
        "exact_same_candidate_close_counterpart:base_pose"
        if exact_pair is not None
        else "same_scene_close_counterpart:base_pose"
    )

    if own is not None and counterpart is not None:
        if np.allclose(own, counterpart, atol=1e-6, rtol=1e-6):
            return own, "own_explicit+close_scene_matched:base_pose"
        # Native metadata for the point cloud/trajectory is authoritative.  The counterpart is
        # only a fallback when native metadata is absent, so a difference is diagnostic only.
        return own, "own_explicit+close_scene_diff_diagnostic:base_pose"

    if own is not None:
        return own, "own_explicit:base_pose"

    if counterpart is not None:
        return counterpart.copy(), counterpart_tag

    return None, None


def door_open_static_from_close(raw):
    raw = Path(raw)
    rel = raw.relative_to(DOOR_CLOSE_ROOT)
    name = rel.name
    if not name.startswith("reverse_") or not name.endswith(".npz"):
        return None
    out_name = name[len("reverse_"):-4] + ".pointcloud.npz"
    return DOOR_OPEN_ROOT / rel.parent / out_name


def mkrow(task, primitive, label_semantics, shape, link, split, raw_path, static_path,
          qidx, label, d1w, d2w, d1m, d2m, Tcam, camera_source,
          pointcloud_path, pointcloud_key, pointcloud_frame_index, point_count,
          contact_world, contact_model, geom, extra=None):
    r = {
        "task": str(task), "primitive": str(primitive), "label_semantics": str(label_semantics),
        "shape": str(shape), "link": str(link), "split": str(split),
        "raw_path": str(raw_path), "static_path": str(static_path),
        "pointcloud_path": str(pointcloud_path), "pointcloud_key": str(pointcloud_key),
        "pointcloud_frame_index": int(pointcloud_frame_index), "point_count": int(point_count),
        "qidx": int(qidx), "grasp_gate": bool(label), "trust_strict": True,
        "camera_pose": np.asarray(Tcam, dtype=np.float32).reshape(4, 4),
        "camera_source": str(camera_source),
        "contact_world": np.asarray(contact_world, dtype=np.float32).reshape(3),
        "contact_model": np.asarray(contact_model, dtype=np.float32).reshape(3),
        "dir1_world": np.asarray(d1w, dtype=np.float32).reshape(3),
        "dir2_world": np.asarray(d2w, dtype=np.float32).reshape(3),
        "dir1_model": np.asarray(d1m, dtype=np.float32).reshape(3),
        "dir2_model": np.asarray(d2m, dtype=np.float32).reshape(3),
        "contact_angle_deg": float(geom["contact_angle_deg"]),
        "contact_lateral_m": float(geom["contact_lateral_m"]),
        "contact_z_m": float(geom["contact_z_m"]),
        "contact_distance_m": float(geom["contact_distance_m"]),
        "tangent_consistency_deg": float(geom["tangent_consistency_deg"]),
    }
    if extra:
        r.update(extra)
    return r


def save_rows(task, rows):
    if not rows:
        raise RuntimeError(f"{task}: no rows")
    rows = sorted(rows, key=lambda r: (r["split"], r["shape"], r["link"], r["raw_path"], r["qidx"]))
    keys = [
        "task", "primitive", "label_semantics", "shape", "link", "split", "raw_path", "static_path",
        "pointcloud_path", "pointcloud_key", "pointcloud_frame_index", "point_count", "qidx", "grasp_gate",
        "trust_strict", "camera_pose", "camera_source", "scene_source", "base_pose", "contact_world", "contact_model", "dir1_world",
        "dir2_world", "dir1_model", "dir2_model", "contact_angle_deg", "contact_lateral_m", "contact_z_m",
        "contact_distance_m", "tangent_consistency_deg", "operation_ref_to_observed_nearest_m",
    ]
    arr = {}
    strkeys = {"task","primitive","label_semantics","shape","link","split","raw_path","static_path","pointcloud_path","pointcloud_key","camera_source","scene_source"}
    boolkeys = {"grasp_gate","trust_strict"}
    intkeys = {"pointcloud_frame_index","point_count","qidx"}
    for k in keys:
        vals = [r[k] for r in rows]
        if k in strkeys: arr[k] = np.asarray(vals, dtype=str)
        elif k in boolkeys: arr[k] = np.asarray(vals, dtype=np.bool_)
        elif k in intkeys: arr[k] = np.asarray(vals, dtype=np.int64)
        elif k == "camera_pose": arr[k] = np.asarray(vals, dtype=np.float32).reshape(-1,4,4)
        elif k == "base_pose": arr[k] = np.asarray(vals, dtype=np.float32).reshape(-1,4)
        elif k in {"contact_world","contact_model","dir1_world","dir2_world","dir1_model","dir2_model"}:
            arr[k] = np.asarray(vals, dtype=np.float32).reshape(-1,3)
        else: arr[k] = np.asarray(vals, dtype=np.float32)
    path = OUT / f"{task}_index_v7_noaff_schema_robust.npz"
    np.savez_compressed(path, **arr)
    splitc = Counter(arr["split"].astype(str).tolist())
    labelc = Counter(bool(x) for x in arr["grasp_gate"].tolist())
    pointc = sorted(set(int(x) for x in arr["point_count"].tolist()))
    cams = Counter(arr["camera_source"].astype(str).tolist())
    scenes = Counter(arr["scene_source"].astype(str).tolist())
    if not splitc.get("train") or not splitc.get("dev"):
        raise RuntimeError(f"{task}: train/dev empty {dict(splitc)}")
    if not labelc.get(True):
        raise RuntimeError(f"{task}: source positive empty")
    if pointc not in ([1024], [8192]):
        raise RuntimeError(f"{task}: mixed/unexpected point counts {pointc}")
    print(f"SAVED {task}: {path}")
    print("  split:", dict(splitc), "source labels:", dict(labelc), "point_count:", pointc)
    print("  camera source:", dict(cams))
    print("  scene/base source:", dict(scenes))
    return {
        "path": str(path), "sha256": sha256(path), "rows": len(rows),
        "split": dict(splitc), "source_labels": {str(k): int(v) for k,v in labelc.items()},
        "point_count": pointc, "label_semantics": sorted(set(arr["label_semantics"].astype(str).tolist())),
        "camera_source": dict(cams),
        "scene_source": dict(scenes),
        "nearest_distance_m": {
            "mean": float(np.mean(arr["operation_ref_to_observed_nearest_m"])),
            "median": float(np.median(arr["operation_ref_to_observed_nearest_m"])),
            "p90": float(np.percentile(arr["operation_ref_to_observed_nearest_m"], 90)),
            "p99": float(np.percentile(arr["operation_ref_to_observed_nearest_m"], 99)),
            "max": float(np.max(arr["operation_ref_to_observed_nearest_m"])),
        },
    }


banner("NO-AFF V7 FINAL PAIR-SCENE TRAINING BUILD — LOCKED PRECHECK")
for name, expected in {
    "faithful_common.py": "227c2922d7f1ae99b7ddfa36448a22e4b681186dd52bf91cb00b3958eee78751",
    "train_faithful_critic.py": "4d6f469b82c17b0a500578e302bbfc14154f1601910c987b468c2482d0aa0807",
    "train_faithful_joint.py": "95681d18381ca95c00ed8fe2c2b53c4c0992ba3afb3e6d5d30a8ebe4cd68ae62",
}.items():
    p = CODE / "faithful_baseline" / name
    if sha256(p) != expected:
        raise RuntimeError(f"source training core changed: {p}")
if DOOR_TRAIN & DOOR_DEV or DOOR_TRAIN & DOOR_FORMAL or DOOR_DEV & DOOR_FORMAL:
    raise RuntimeError("door split overlap")
for _name, _tr, _dv, _fo in (
    ("drawer_open", DRAWER_OPEN_TRAIN, DRAWER_OPEN_DEV, DRAWER_OPEN_FORMAL),
    ("drawer_close", DRAWER_CLOSE_TRAIN, DRAWER_CLOSE_DEV, DRAWER_CLOSE_FORMAL),
):
    if (_tr & _dv) or (_tr & _fo) or (_dv & _fo):
        raise RuntimeError(f"drawer split overlap: {_name}")
print("source core + splits: PASS")

(
    DRAWER_CAMERA_DICT,
    DRAWER_CAMERA_TRAIN_ROWS,
    DRAWER_SCENE_CAMERA,
    DRAWER_SCENE_BASE,
    DRAWER_EXACT_CAMERA,
    DRAWER_EXACT_BASE,
) = resolve_drawer_scene_dictionary()

# Enumerate Door raw names without opening formal content.
door_groups = defaultdict(list)
door_formal_named = 0
door_unknown = 0
door_missing_open_static = 0
for p in sorted(DOOR_CLOSE_ROOT.rglob("*.npz")):
    if p.parent.name != "trajectory":
        continue
    sid = path_sid(p); sp = split_for("door_close", sid)
    if sp == "formal":
        door_formal_named += 1
        continue
    if sp not in ("train", "dev"):
        door_unknown += 1
        continue
    static = door_open_static_from_close(p)
    if static is None or not static.is_file():
        door_missing_open_static += 1
        continue
    door_groups[sid].append((p, static, sp))

print("Door train/dev trajectory candidates with Open counterpart:", sum(len(v) for v in door_groups.values()))
print("Door formal paths named/not opened:", door_formal_named)
print("Door unknown paths not opened:", door_unknown)
print("Door train/dev paths missing Open static counterpart:", door_missing_open_static)

rows_by_task = {t: [] for t in TASKS}
skipped = {t: Counter() for t in TASKS}
task_mode_diag = {t: Counter() for t in TASKS}
status_diag = {t: Counter() for t in TASKS}
door_camera_pair_diag = Counter()
door_base_pair_diag = Counter()

banner("BUILD 1/4 + 2/4 door_open / door_close DIRECTLY FROM TRAJECTORY")
for gi, (sid, entries) in enumerate(sorted(door_groups.items()), 1):
    try:
        eng, scene, obj, links = load_object(sid); active = list(obj.get_active_joints())
    except Exception as e:
        raise RuntimeError(f"door object load {sid}: {e}")
    for raw, static, sp in entries:
        link = path_link(raw)
        if link not in links:
            raise RuntimeError(f"door missing link {raw}")
        jidx = target_joint_index(active, link)
        if jidx is None:
            raise RuntimeError(f"door missing target joint {raw}")
        with np.load(raw, allow_pickle=False) as z:
            rj = decode_result(z)
            if not isinstance(rj, dict):
                raise RuntimeError(f"door missing result {raw}")
            # Generator labels are schema-specific; enforce only the physical close/open meaning.
            validate_task_mode("door_close", rj.get("task_mode", None), task_mode_diag["door_close"], raw)
            if str(rj.get("conversion", "")) != "reverse_open_to_close":
                raise RuntimeError(f"door raw conversion mismatch: {raw}: {rj.get('conversion')}")
            required = ("operation_start_object_qpos","final_object_qpos","operation_start_robot_qpos","base_pose",
                        "ee_actual_xyz","action_phase","object_qpos","operation_start_index","observation_point_cloud",
                        "observation_trajectory_step","observation_camera_pose")
            miss = [k for k in required if k not in z.files]
            if miss:
                raise RuntimeError(f"door missing {miss}: {raw}")
            if "local_link_grasp_translation" not in rj:
                raise RuntimeError(f"door missing local operation point: {raw}")
            qop = np.asarray(z["operation_start_object_qpos"], dtype=np.float64).reshape(-1)
            qfinal = np.asarray(z["final_object_qpos"], dtype=np.float64).reshape(-1)
            robot_q = np.asarray(z["operation_start_robot_qpos"], dtype=np.float64).reshape(9)
            base = np.asarray(z["base_pose"], dtype=np.float64).reshape(4)
            Tcam_close_pair = np.asarray(z["observation_camera_pose"], dtype=np.float64)
            if not camera_ok(Tcam_close_pair):
                raise RuntimeError(f"door close paired camera invalid: {raw}")
            local_ref = np.asarray(rj["local_link_grasp_translation"], dtype=np.float64).reshape(3)
            close_motion = phase_motion(z["ee_actual_xyz"], z["action_phase"], "operation_reverse")
            if close_motion is None:
                skipped["door_open"]["bad_operation_motion"] += 1
                skipped["door_close"]["bad_operation_motion"] += 1
                continue
            if len(qop) != int(obj.dof) or len(qfinal) != int(obj.dof):
                raise RuntimeError(f"door dof mismatch: {raw}")

            # Shared recorded hand-to-link orientation at operation start.
            Rhop = panda_hand_R(robot_q, base)
            obj.set_qpos(qop)
            Tlop = np.asarray(links[link].get_pose().to_transformation_matrix(), dtype=np.float64)
            Rrel = Tlop[:3,:3].T @ Rhop

            # -------------------------------------------------------------
            # Door Open / Pull: same trajectory operation point transported
            # to the Open observation state q_final; nearest finite observed
            # point only. No auxiliary point-score array participates.
            # -------------------------------------------------------------
            with np.load(static, allow_pickle=False) as zs:
                if "initial_point_cloud_8192" not in zs.files:
                    raise RuntimeError(f"door_open missing point cloud: {static}")
                pc_open = np.asarray(zs["initial_point_cloud_8192"], dtype=np.float64)
                if pc_open.shape != (8192,3) or not np.all(np.isfinite(pc_open)):
                    raise RuntimeError(f"door_open bad point cloud {pc_open.shape}: {static}")
                if "initial_camera_pose" in zs.files:
                    Tcam_open = np.asarray(zs["initial_camera_pose"], dtype=np.float64)
                    cam_src_open = "own_explicit:initial_camera_pose"
                elif "camera_pose" in zs.files:
                    Tcam_open = np.asarray(zs["camera_pose"], dtype=np.float64)
                    cam_src_open = "own_explicit:camera_pose"
                else:
                    Tcam_open = Tcam_close_pair.copy()
                    cam_src_open = "exact_same_candidate_close_counterpart"

                if "base_pose" in zs.files:
                    base_open_static = np.asarray(zs["base_pose"], dtype=np.float64).reshape(-1)
                    if base_open_static.shape == (4,) and np.all(np.isfinite(base_open_static)):
                        if np.allclose(base_open_static, base, atol=1e-6, rtol=1e-6):
                            door_base_pair_diag["explicit_pair_matched"] += 1
                        else:
                            # Each observation keeps its own recorded geometry; the paired scene is
                            # a cross-check/fallback relation, not an equality constraint.
                            door_base_pair_diag["explicit_pair_different_diagnostic"] += 1
            if not camera_ok(Tcam_open):
                raise RuntimeError(f"door_open bad camera: {static}")
            if cam_src_open.startswith("own_explicit"):
                # The Open point cloud must use its own explicit camera.  The paired Close camera
                # is compared only as a diagnostic; equality is not required because the two
                # datasets may store different native views for the paired reverse interaction.
                if np.allclose(Tcam_open, Tcam_close_pair, atol=1e-6, rtol=1e-6):
                    door_camera_pair_diag["explicit_pair_matched"] += 1
                    cam_src_open += "+close_pair_matched"
                else:
                    door_camera_pair_diag["explicit_pair_different"] += 1
                    cam_src_open += "+close_pair_diff_diagnostic"
            else:
                door_camera_pair_diag["close_pair_used_as_fallback"] += 1
            obj.set_qpos(qfinal)
            Tlopen = np.asarray(links[link].get_pose().to_transformation_matrix(), dtype=np.float64)
            p_ref_open = transform_point(Tlopen, local_ref)
            Rhopen = Tlopen[:3,:3] @ Rrel
            near_open = nearest_finite(p_ref_open, pc_open)
            if near_open is None:
                raise RuntimeError(f"door_open no finite point: {static}")
            qidx_open, cw_open, dist_open = near_open
            d1w_open = normalize(close_motion)  # Pull runtime executes -dir1 = opening.
            d2w_open = project_perpendicular(Rhopen[:,0], d1w_open)
            got = validate_dirs(d1w_open, d2w_open, Tcam_open)
            if got is None:
                raise RuntimeError(f"door_open direction/frame invalid: {raw}")
            d1m_open, d2m_open = got
            # Kinematic consistency is diagnostic only.  The trajectory-defined operation point
            # is not filtered by any hand-written geometric threshold.
            kin_open = kinematic_task_tangent(obj, active, jidx, links[link], qfinal, cw_open, +1.0)
            geom_open = geometry_diagnostics(cw_open, p_ref_open, Rhopen, -close_motion, kin_open)
            if "grasp_gate_pass" not in rj:
                raise RuntimeError(f"door_open trajectory missing grasp_gate_pass: {raw}")
            label_open = bool(rj["grasp_gate_pass"])
            Rcam = Tcam_open[:3,:3]
            rows_by_task["door_open"].append(mkrow(
                "door_open", "pull", "trajectory_grasp_gate_pass", sid, link, sp, str(raw), str(static),
                qidx_open, label_open, d1w_open, d2w_open, d1m_open, d2m_open, Tcam_open, cam_src_open,
                str(static), "initial_point_cloud_8192", -1, 8192, cw_open, cw_open @ Rcam, geom_open,
                {"operation_ref_to_observed_nearest_m": float(dist_open), "scene_source": "exact_same_candidate_close_raw:base_pose", "base_pose": base.astype(np.float32)}
            ))

            # -------------------------------------------------------------
            # Door Close / Push: own operation-start observation; same local
            # trajectory point transformed to q_obs; nearest finite point.
            # -------------------------------------------------------------
            pc_close, qobs, Tcam_close, frame_i, obs_step = select_door_obs(z)
            if len(qobs) != int(obj.dof):
                raise RuntimeError(f"door_close qobs dof mismatch: {raw}")
            if not camera_ok(Tcam_close):
                raise RuntimeError(f"door_close bad camera: {raw}")
            coord = scalar(z, "observation_coordinate_frame", None)
            if coord is not None and str(coord) != "world":
                raise RuntimeError(f"door_close nonworld point cloud: {raw}")
            obj.set_qpos(qobs)
            Tlclose = np.asarray(links[link].get_pose().to_transformation_matrix(), dtype=np.float64)
            p_ref_close = transform_point(Tlclose, local_ref)
            Rhclose = Tlclose[:3,:3] @ Rrel
            near_close = nearest_finite(p_ref_close, pc_close)
            if near_close is None:
                raise RuntimeError(f"door_close no finite point: {raw}")
            qidx_close, cw_close, dist_close = near_close
            d1w_close = normalize(close_motion)
            d2w_close = project_perpendicular(Rhclose[:,0], d1w_close)
            got = validate_dirs(d1w_close, d2w_close, Tcam_close)
            if got is None:
                raise RuntimeError(f"door_close direction/frame invalid: {raw}")
            d1m_close, d2m_close = got
            kin_close = kinematic_task_tangent(obj, active, jidx, links[link], qobs, cw_close, -1.0)
            geom_close = geometry_diagnostics(cw_close, p_ref_close, Rhclose, close_motion, kin_close)
            if "passed" not in rj:
                raise RuntimeError(f"door_close trajectory missing passed: {raw}")
            label_close = bool(rj["passed"]); status = rj.get("status", None)
            sb = canonical_status_bool(status)
            status_diag["door_close"][f"raw:{status!s}"] += 1
            if sb is not None and sb != label_close:
                raise RuntimeError(f"door_close passed/status semantic conflict: {raw}: passed={label_close}, status={status!r}")
            if sb is None and status is not None:
                status_diag["door_close"]["unknown_status_diagnostic"] += 1
            Rcam = Tcam_close[:3,:3]
            rows_by_task["door_close"].append(mkrow(
                "door_close", "push", "trajectory_passed_task_success", sid, link, sp, str(raw), str(static),
                qidx_close, label_close, d1w_close, d2w_close, d1m_close, d2m_close, Tcam_close,
                "own_explicit:observation_camera_pose", str(raw), "observation_point_cloud", frame_i,
                len(pc_close), cw_close, cw_close @ Rcam, geom_close,
                {"operation_ref_to_observed_nearest_m": float(dist_close), "scene_source": "own_explicit:base_pose", "base_pose": base.astype(np.float32)}
            ))
    del obj, links, scene, eng; gc.collect()
    if gi % 25 == 0 or gi == len(door_groups):
        print(f"door shapes {gi}/{len(door_groups)} open_rows={len(rows_by_task['door_open'])} close_rows={len(rows_by_task['door_close'])}", flush=True)

print("door_open skipped:", dict(skipped["door_open"]))
print("door_close skipped:", dict(skipped["door_close"]))


def enumerate_drawer(task, root):
    groups = defaultdict(list); formal_named = 0; unknown = 0
    for p in sorted(Path(root).rglob("*.npz")):
        if p.parent.name != "trajectory":
            continue
        sid = path_sid(p); sp = split_for(task, sid)
        if sp == "formal":
            formal_named += 1
            continue
        if sp in ("train", "dev"):
            groups[sid].append(p)
        else:
            unknown += 1
    return groups, formal_named, unknown


def build_drawer(task, root, phase_name, d1_sign, camera_dict, scene_camera, scene_base, exact_camera, exact_base):
    banner(f"BUILD {'3' if task=='drawer_open' else '4'}/4 {task} DIRECTLY FROM TRAJECTORY")
    groups, formal_named, unknown = enumerate_drawer(task, root)
    rows = []; local_skip = Counter(); cameras = Counter(); scenes = Counter()
    for gi, (sid, paths) in enumerate(sorted(groups.items()), 1):
        try:
            eng, scene, obj, links = load_object(sid); active = list(obj.get_active_joints())
        except Exception as e:
            raise RuntimeError(f"{task} object load {sid}: {e}")
        for p in paths:
            sp = split_for(task, sid); link = path_link(p)
            if link not in links:
                raise RuntimeError(f"{task} missing link {p}")
            jidx = target_joint_index(active, link)
            if jidx is None:
                raise RuntimeError(f"{task} missing target joint {p}")
            with np.load(p, allow_pickle=False) as z:
                rj = decode_result(z)
                if not isinstance(rj, dict):
                    raise RuntimeError(f"{task} missing result {p}")
                # Real local schemas use e.g. drawer_open -> `open` and drawer_close ->
                # `door_close`.  The task root gives the category; task_mode only guards against
                # an actual open-vs-close contradiction.
                validate_task_mode(task, rj.get("task_mode", None), task_mode_diag[task], p)
                required = ("initial_point_cloud_8192","operation_start_object_qpos","operation_start_robot_qpos",
                            "object_qpos","initial_trajectory_step","action_phase")
                miss = [k for k in required if k not in z.files]
                if miss:
                    raise RuntimeError(f"{task} missing {miss}: {p}")
                if "ee_actual_xyz" not in z.files:
                    local_skip["missing_ee_actual_xyz"] += 1
                    continue
                if "local_link_grasp_translation" not in rj:
                    raise RuntimeError(f"{task} missing local operation point {p}")
                pc = np.asarray(z["initial_point_cloud_8192"], dtype=np.float64)
                if pc.shape != (8192,3) or not np.all(np.isfinite(pc)):
                    raise RuntimeError(f"{task} bad point cloud {p} {pc.shape}")
                qop = np.asarray(z["operation_start_object_qpos"], dtype=np.float64).reshape(-1)
                oq = np.asarray(z["object_qpos"], dtype=np.float64)
                step = int(np.asarray(z["initial_trajectory_step"]).item())
                if oq.ndim != 2 or not (0 <= step < len(oq)):
                    raise RuntimeError(f"{task} bad initial step {p}")
                qobs = np.asarray(oq[step], dtype=np.float64).reshape(-1)
                if len(qop) != int(obj.dof) or len(qobs) != int(obj.dof):
                    raise RuntimeError(f"{task} dof mismatch {p}")
                Tcam, csource = drawer_camera(z, rj, camera_dict, scene_camera, exact_camera, p)
                if Tcam is None or not camera_ok(Tcam):
                    raise RuntimeError(f"{task} camera unresolved {p}")
                cameras[csource] += 1
                base, ssource = drawer_base_pose(z, scene_base, exact_base, p)
                if base is None:
                    raise RuntimeError(f"{task} base/scene unresolved {p}")
                scenes[ssource] += 1
                coord = scalar(z, "observation_coordinate_frame", None)
                if coord is None and isinstance(rj.get("pointcloud", None), dict):
                    coord = rj["pointcloud"].get("coordinate_frame", None)
                if coord is not None and str(coord) != "world":
                    raise RuntimeError(f"{task} nonworld point cloud {p}")
                local_ref = np.asarray(rj["local_link_grasp_translation"], dtype=np.float64).reshape(3)
                robot_q = np.asarray(z["operation_start_robot_qpos"], dtype=np.float64).reshape(9)
                Rhop = panda_hand_R(robot_q, base)
                obj.set_qpos(qop)
                Tlop = np.asarray(links[link].get_pose().to_transformation_matrix(), dtype=np.float64)
                Rrel = Tlop[:3,:3].T @ Rhop
                obj.set_qpos(qobs)
                Tlobs = np.asarray(links[link].get_pose().to_transformation_matrix(), dtype=np.float64)
                p_ref_obs = transform_point(Tlobs, local_ref)
                Rhobs = Tlobs[:3,:3] @ Rrel
                near = nearest_finite(p_ref_obs, pc)
                if near is None:
                    raise RuntimeError(f"{task} no finite point cloud {p}")
                qidx, cw, dist = near
                motion = phase_motion(z["ee_actual_xyz"], z["action_phase"], phase_name)
                if motion is None:
                    local_skip["bad_operation_motion"] += 1
                    continue
                d1w = normalize(float(d1_sign) * motion)
                d2w = project_perpendicular(Rhobs[:,0], d1w)
                got = validate_dirs(d1w, d2w, Tcam)
                if got is None:
                    raise RuntimeError(f"{task} direction/frame invalid {p}")
                d1m, d2m = got
                kin_sign = +1.0 if task == "drawer_open" else -1.0
                kin = kinematic_task_tangent(obj, active, jidx, links[link], qobs, cw, kin_sign)
                geom = geometry_diagnostics(cw, p_ref_obs, Rhobs, motion, kin)
                if "passed" not in rj:
                    raise RuntimeError(f"{task} missing passed {p}")
                label = bool(rj["passed"]); status = rj.get("status", None)
                sb = canonical_status_bool(status)
                status_diag[task][f"raw:{status!s}"] += 1
                if sb is not None and sb != label:
                    raise RuntimeError(f"{task} passed/status semantic conflict {p}: passed={label}, status={status!r}")
                if sb is None and status is not None:
                    status_diag[task]["unknown_status_diagnostic"] += 1
                Rcam = Tcam[:3,:3]
                rows.append(mkrow(
                    task, "pull" if task == "drawer_open" else "push", "trajectory_passed_task_success",
                    sid, link, sp, str(p), str(p), qidx, label, d1w, d2w, d1m, d2m, Tcam, csource,
                    str(p), "initial_point_cloud_8192", -1, 8192, cw, cw @ Rcam, geom,
                    {"operation_ref_to_observed_nearest_m": float(dist), "scene_source": ssource, "base_pose": base.astype(np.float32)}
                ))
        del obj, links, scene, eng; gc.collect()
        if gi % 20 == 0 or gi == len(groups):
            print(f"{task} shapes {gi}/{len(groups)} rows={len(rows)}", flush=True)
    print(f"{task} formal paths named/not opened: {formal_named}; unknown: {unknown}; skipped: {dict(local_skip)}; cameras: {dict(cameras)}; scenes: {dict(scenes)}")
    return rows

rows_by_task["drawer_open"] = build_drawer("drawer_open", DRAWER_OPEN_ROOT, "operation", -1.0, DRAWER_CAMERA_DICT, DRAWER_SCENE_CAMERA, DRAWER_SCENE_BASE, DRAWER_EXACT_CAMERA, DRAWER_EXACT_BASE)
rows_by_task["drawer_close"] = build_drawer("drawer_close", DRAWER_CLOSE_ROOT, "operation_reverse", +1.0, DRAWER_CAMERA_DICT, DRAWER_SCENE_CAMERA, DRAWER_SCENE_BASE, DRAWER_EXACT_CAMERA, DRAWER_EXACT_BASE)

print("task_mode schema diagnostics:")
for _t in TASKS:
    print(" ", _t, dict(task_mode_diag[_t]))
print("status schema diagnostics:")
for _t in TASKS:
    print(" ", _t, dict(status_diag[_t]))
print("door base-pair diagnostics:", dict(door_base_pair_diag))

manifest = {}
for task in TASKS:
    manifest[task] = save_rows(task, rows_by_task[task])
    del rows_by_task[task]

# Final leakage/schema lock. Formal paths are filtered by path identity before any np.load.
for task, rec in manifest.items():
    with np.load(rec["path"], allow_pickle=False) as z:
        shapes = set(z["shape"].astype(str).tolist())
        overlap = sorted(shapes & FORMAL_BY_TASK[task])
        if overlap:
            raise RuntimeError(f"{task}: formal overlap {overlap}")
        if set(z["split"].astype(str).tolist()) != {"train", "dev"}:
            raise RuntimeError(f"{task}: invalid split values")

manifest.update({
    "schema": "where2act_four_task_train_index_v7_noaff_schema_robust",
    "contact_point_contract": "trajectory local operation point -> exact articulation-state transform -> nearest finite observed point",
    "door_open_observation_state": "paired reverse trajectory q_final -> own Door Open static point cloud",
    "door_close_observation_state": "latest nonfuture raw observation at operation_start_index",
    "drawer_observation_state": "object_qpos[initial_trajectory_step]",
    "drawer_camera_contract": "own explicit camera is authoritative for its own point cloud; Open/Close counterpart is diagnostic when both exist and fallback only when own camera is absent; then uniquely resolved native camera_id camera-only fallback; otherwise hard fail",
    "door_camera_pair_diagnostics": dict(door_camera_pair_diag),
    "door_base_pair_diagnostics": dict(door_base_pair_diag),
    "task_mode_schema_diagnostics": {t: dict(task_mode_diag[t]) for t in TASKS},
    "status_schema_diagnostics": {t: dict(status_diag[t]) for t in TASKS},
    "drawer_scene_contract": "own base_pose -> verify against exact/scene Close counterpart when available -> same scene (shape,link,repeat,base) Close fallback only; never copied across scenes",
    "geometry_filter_applied": False,
    "geometry_diagnostics_only": True,
    "drawer_camera_train_dictionary_rows": DRAWER_CAMERA_TRAIN_ROWS,
    "drawer_camera_id": DRAWER_CAMERA_ID,
    "drawer_native_split_counts": {
        "drawer_open": {"native_train": len(DRAWER_OPEN_NATIVE_TRAIN), "internal_train": len(DRAWER_OPEN_TRAIN), "internal_dev": len(DRAWER_OPEN_DEV), "native_val_formal": len(DRAWER_OPEN_FORMAL)},
        "drawer_close": {"native_train": len(DRAWER_CLOSE_NATIVE_TRAIN), "internal_train": len(DRAWER_CLOSE_TRAIN), "internal_dev": len(DRAWER_CLOSE_DEV), "native_val_formal": len(DRAWER_CLOSE_FORMAL)},
    },
    "drawer_camera_pose": DRAWER_CAMERA_DICT[DRAWER_CAMERA_ID].tolist(),
    "drawer_scene_camera_rows": len(DRAWER_SCENE_CAMERA),
    "drawer_scene_base_rows": len(DRAWER_SCENE_BASE),
    "drawer_exact_paired_camera_rows": len(DRAWER_EXACT_CAMERA),
    "drawer_exact_paired_base_rows": len(DRAWER_EXACT_BASE),
    "source_point_score_fields_read": 0,
    "source_point_score_fields_in_output": 0,
    "negative_augmentation": "downstream faithful_common: (-dir1, same dir2) -> 0",
    "formal_or_test_trajectory_content_opened": 0,
})
mp = OUT / "INDEX_MANIFEST.json"
mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print("\nINDEX BUILD COMPLETE:", mp)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

# Hard source-code gate: the index builder is not allowed to access the known problematic point-score fields.
FORBIDDEN_RE='affordance|high_aff_nearest'
if grep -En "$FORBIDDEN_RE" "$BUILDER"; then
  echo "ERROR: forbidden point-score field access found in no-aff builder"
  exit 12
fi
echo "no-aff builder source scan: PASS"

cat > "$PATCHER" <<'PY'
from pathlib import Path
import hashlib, re, shutil

HOME=Path.home(); BASE=HOME/'robot_baselines'; CODE=BASE/'repos'/'where2act'/'code'
SRC=CODE/'faithful_baseline'; DST=CODE/'four_task_train_v7_noaff_schema_robust'
DST.mkdir(parents=True,exist_ok=True)
expected={
 'faithful_common.py':'227c2922d7f1ae99b7ddfa36448a22e4b681186dd52bf91cb00b3958eee78751',
 'train_faithful_critic.py':'4d6f469b82c17b0a500578e302bbfc14154f1601910c987b468c2482d0aa0807',
 'train_faithful_joint.py':'95681d18381ca95c00ed8fe2c2b53c4c0992ba3afb3e6d5d30a8ebe4cd68ae62',
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
for n,h in expected.items():
 p=SRC/n
 if sha(p)!=h: raise RuntimeError(f'source hash changed: {p}')
 shutil.copy2(p,DST/n)

p=DST/'faithful_common.py'; text=p.read_text()
start=text.index('def load_index(')
end=text.index('def normalize_np',start)
new_load=r'''def load_index(path=INDEX_DEFAULT):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        index = {k: z[k].copy() for k in z.files}
    required = {
        "shape", "link", "split", "raw_path", "static_path", "qidx",
        "grasp_gate", "dir1_model", "dir2_model", "trust_strict",
        "pointcloud_path", "pointcloud_key", "pointcloud_frame_index",
        "point_count", "camera_pose",
    }
    missing = sorted(required - set(index))
    if missing:
        raise RuntimeError(f"four-task index missing keys: {missing}: {path}")
    n = len(index["shape"])
    for k, v in index.items():
        if np.asarray(v).ndim >= 1 and len(v) != n:
            raise RuntimeError(f"index first-dim mismatch {k}: {len(v)} != {n}")
    splits = set(np.asarray(index["split"]).astype(str).tolist())
    if not splits or not splits.issubset({"train", "dev"}):
        raise RuntimeError(f"invalid training splits {splits}: {path}")
    if not np.asarray(index["trust_strict"], dtype=bool).all():
        raise RuntimeError(f"saved four-task index must contain trusted trajectory-derived rows only: {path}")
    return index


'''
text=text[:start]+new_load+text[end:]

cstart=text.index('class ConvertedWhere2ActDataset')
cend=text.index('class ShapeBalancedBinaryBatchSampler',cstart)
new_class=r'''class ConvertedWhere2ActDataset(Dataset):
    """Thin source-loading adapter only; model/loss/sample semantics stay unchanged."""
    def __init__(self, index, samples, num_points=NUM_POINTS):
        self.index = index
        self.samples = list(samples)
        self.num_points = num_points  # retained for signature compatibility; task index fixes actual N.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        s = self.samples[item]
        gi = int(s.global_idx)
        source_path = str(self.index["pointcloud_path"][gi])
        pc_key = str(self.index["pointcloud_key"][gi])
        frame_i = int(self.index["pointcloud_frame_index"][gi])
        expected_n = int(self.index["point_count"][gi])

        with np.load(source_path, allow_pickle=False) as z:
            if pc_key not in z.files:
                raise RuntimeError(f"missing {pc_key}: {source_path}")
            pc_world = np.asarray(z[pc_key], dtype=np.float32)

        if pc_world.ndim == 3:
            if not (0 <= frame_i < len(pc_world)):
                raise RuntimeError(f"bad frame {frame_i} for {pc_world.shape}: {source_path}")
            pc_world = pc_world[frame_i]
        elif pc_world.ndim == 2:
            if frame_i not in (-1, 0):
                raise RuntimeError(f"2D point cloud with nontrivial frame {frame_i}: {source_path}")
        else:
            raise RuntimeError(f"bad point cloud rank {pc_world.shape}: {source_path}")

        if pc_world.shape != (expected_n, 3):
            raise RuntimeError(f"bad point cloud {pc_world.shape}, expected {(expected_n,3)}: {source_path}")
        if not np.isfinite(pc_world).all():
            raise RuntimeError(f"nonfinite point cloud: {source_path}")

        camera_pose = np.asarray(self.index["camera_pose"][gi], dtype=np.float32)
        if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
            raise RuntimeError(f"bad camera pose in index: {source_path}")
        R_cam_world = camera_pose[:3, :3]
        pc_model = (pc_world @ R_cam_world).astype(np.float32)

        qidx = int(self.index["qidx"][gi])
        if not (0 <= qidx < len(pc_model)):
            raise RuntimeError(f"bad qidx={qidx}: {source_path}")

        # Official Where2Act contract: interacting point is point 0.
        pc_model = pc_model.copy()
        if qidx != 0:
            tmp = pc_model[0].copy()
            pc_model[0] = pc_model[qidx]
            pc_model[qidx] = tmp

        d1 = normalize_np(self.index["dir1_model"][gi])
        d2 = normalize_np(self.index["dir2_model"][gi])
        if s.variant == "neg_direction":
            d1 = -d1
        elif s.variant != "original":
            raise RuntimeError(f"unknown variant {s.variant}")

        return (
            torch.from_numpy(pc_model),
            torch.from_numpy(d1),
            torch.from_numpy(d2),
            torch.tensor(float(s.label), dtype=torch.float32),
            torch.tensor(gi, dtype=torch.long),
            s.variant,
            s.shape,
        )


'''
text=text[:cstart]+new_class+text[cend:]
p.write_text(text)
print('training adapter:',p)
print('adapter sha256:',sha(p))
print('critic source copy:',sha(DST/'train_faithful_critic.py'))
print('joint source copy :',sha(DST/'train_faithful_joint.py'))
PY

cat > "$PREFLIGHT" <<'PY'
from pathlib import Path
import sys, math
import numpy as np
import torch

HOME=Path.home(); BASE=HOME/'robot_baselines'; CODE=BASE/'repos'/'where2act'/'code'/'four_task_train_v7_noaff_schema_robust'
sys.path.insert(0,str(CODE))
from faithful_common import load_index, build_converted_samples, ConvertedWhere2ActDataset
ROOT=BASE/'results'/'where2act'/'four_task_train_v7_noaff_schema_robust_indices'


def pick_positions(n):
    if n <= 0:
        return []
    return sorted(set([0, n//4, n//2, (3*n)//4, n-1]))


def check_one(task, split_name, idx, samples, pos):
    s = samples[pos]
    ds = ConvertedWhere2ActDataset(idx, [s])
    pc,d1,d2,label,gi,variant,shape = ds[0]
    gi = int(gi.item())
    if pc.ndim != 2 or pc.shape[1] != 3 or d1.shape != (3,) or d2.shape != (3,):
        raise RuntimeError(f'{task}/{split_name}: loader shape failure {pc.shape} {d1.shape} {d2.shape}')
    if not torch.isfinite(pc).all() or not torch.isfinite(d1).all() or not torch.isfinite(d2).all():
        raise RuntimeError(f'{task}/{split_name}: nonfinite loader output')
    if abs(float(torch.linalg.norm(d1))-1.0) > 1e-4 or abs(float(torch.linalg.norm(d2))-1.0) > 1e-4:
        raise RuntimeError(f'{task}/{split_name}: nonunit direction')
    if abs(float(torch.dot(d1,d2))) > 1e-4:
        raise RuntimeError(f'{task}/{split_name}: directions not orthogonal')
    if not (0 <= gi < len(idx['shape'])):
        raise RuntimeError(f'{task}/{split_name}: bad global index {gi}')

    # Strong qidx/source regression: adapter point-0 must be exactly the saved qidx point
    # after the same world->model rotation used by the loader.
    src = str(idx['pointcloud_path'][gi]); key = str(idx['pointcloud_key'][gi])
    frame = int(idx['pointcloud_frame_index'][gi]); qidx = int(idx['qidx'][gi])
    with np.load(src, allow_pickle=False) as z:
        if key not in z.files:
            raise RuntimeError(f'{task}/{split_name}: source key missing {key}: {src}')
        raw = np.asarray(z[key], dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[frame]
    elif raw.ndim == 2:
        if frame not in (-1,0):
            raise RuntimeError(f'{task}/{split_name}: bad 2D frame {frame}')
    else:
        raise RuntimeError(f'{task}/{split_name}: bad raw pc rank {raw.shape}')
    T = np.asarray(idx['camera_pose'][gi], dtype=np.float32).reshape(4,4)
    expected_p0 = raw[qidx] @ T[:3,:3]
    if not np.allclose(pc[0].numpy(), expected_p0, atol=2e-5, rtol=2e-5):
        raise RuntimeError(f'{task}/{split_name}: qidx->point0 regression failed gi={gi}')

    base_d1 = np.asarray(idx['dir1_model'][gi], dtype=np.float32)
    base_d1 = base_d1 / np.linalg.norm(base_d1)
    if variant == 'neg_direction':
        if float(label.item()) != 0.0:
            raise RuntimeError(f'{task}/{split_name}: synthetic negative label is not zero')
        expected_d1 = -base_d1
    elif variant == 'original':
        expected_d1 = base_d1
        expected_label = float(bool(idx['grasp_gate'][gi]))
        if float(label.item()) != expected_label:
            raise RuntimeError(f'{task}/{split_name}: original label mismatch')
    else:
        raise RuntimeError(f'{task}/{split_name}: unknown variant {variant}')
    if not np.allclose(d1.numpy(), expected_d1, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f'{task}/{split_name}: direction variant regression failed')


for task in ('door_open','door_close','drawer_open','drawer_close'):
    p=ROOT/f'{task}_index_v7_noaff_schema_robust.npz'
    idx=load_index(p)
    tr=build_converted_samples(idx,'train'); dv=build_converted_samples(idx,'dev')
    if not tr or not dv:
        raise RuntimeError(f'{task}: empty converted train/dev')
    variants_train = {x.variant for x in tr}
    variants_dev = {x.variant for x in dv}
    if not {'original','neg_direction'}.issubset(variants_train):
        raise RuntimeError(f'{task}: train variants missing: {variants_train}')
    if not {'original','neg_direction'}.issubset(variants_dev):
        raise RuntimeError(f'{task}: dev variants missing: {variants_dev}')

    checks=[]
    for split_name,samples in [('train',tr),('dev',dv)]:
        positions = pick_positions(len(samples))
        # Also force one sample of each variant into the checked set.
        for want in ('original','neg_direction'):
            j = next((i for i,x in enumerate(samples) if x.variant == want), None)
            if j is not None:
                positions = sorted(set(positions+[j]))
        for pos in positions:
            check_one(task, split_name, idx, samples, pos)
            checks.append((split_name,pos,samples[pos].variant))

    print(
        f'{task}: PREFLIGHT PASS | source_rows={len(idx["shape"])} '
        f'converted_train={len(tr)} converted_dev={len(dv)} checked={len(checks)} '
        f'point_count={sorted(set(int(x) for x in idx["point_count"].tolist()))}'
    )
print('AUTO PREFLIGHT PASS — starting training')
PY
cat > "$FINALIZER" <<'PY'
from pathlib import Path
import hashlib,json,sys
HOME=Path.home(); BASE=HOME/'robot_baselines'
run=Path(sys.argv[1]); idxroot=BASE/'results'/'where2act'/'four_task_train_v7_noaff_schema_robust_indices'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
out={'protocol':'where2act_four_task_train_v7_noaff_schema_robust','run_root':str(run),'formal_or_test_rollout_used':False,'tasks':{}}
for t in ('door_open','door_close','drawer_open','drawer_close'):
 idx=idxroot/f'{t}_index_v7_noaff_schema_robust.npz'; cr=run/t/'critic'/'best-network.pth'; jt=run/t/'joint'/'best-network.pth'
 for p in (idx,cr,jt):
  if not p.is_file(): raise RuntimeError(f'missing final artifact {p}')
 out['tasks'][t]={'index':str(idx),'index_sha256':sha(idx),'critic':str(cr),'critic_sha256':sha(cr),'joint':str(jt),'joint_sha256':sha(jt)}
p=run/'TRAINING_MANIFEST.json'; p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('\nTRAINING COMPLETE')
print(p)
print(json.dumps(out,indent=2,ensure_ascii=False))
PY

# Compile generated code before doing expensive work.
"$PY" -m py_compile "$BUILDER" "$PATCHER" "$PREFLIGHT" "$FINALIZER"

{
  echo "========================================================================================================================"
  echo "WHERE2ACT NO-AFF V7 FINAL PAIR-SCENE — BUILD FOUR TASK INDICES AND RETRAIN"
  echo "time: $(date '+%F %T %z')"
  echo "run: $RUN_ROOT"
  echo "========================================================================================================================"
  echo "No formal/test trajectory content is opened by the index builder."
  echo "No formal rollout is run. Dev is used only inside the existing trainers for checkpoint selection."
  echo "Contact qidx is built only from trajectory local operation point -> exact state transform -> nearest finite observed point."
  echo "Open/Close scene pairing: own camera/base is verified against the paired Close scene; missing Drawer Open metadata may be filled from the same (shape,link,repeat,base) Close scene, with exact-candidate evidence preferred and fixed camera_id as camera-only final fallback."
  echo "Known problematic point-score fields are never indexed/read and are absent from the output index."
  echo

  echo "[1/4] Build trajectory-only four-task indices (no point-score field access)"
  "$PY" -u "$BUILDER"

  echo
  echo "[2/4] Create thin source-loading adapter (network/loss/trainers remain byte-identical copies)"
  "$PY" -u "$PATCHER"

  echo
  echo "[3/4] Automatic multi-sample loader/qidx/direction preflight"
  "$PY" -u "$PREFLIGHT"

  echo
  echo "[4/4] Retraining: Critic -> full three-decoder model, four independent task-specific models"
  cd "$TRAIN_CODE"

  for TASK in door_open door_close drawer_open drawer_close; do
    IDX="$INDEX_ROOT/${TASK}_index_v7_noaff_schema_robust.npz"
    TROOT="$RUN_ROOT/$TASK"
    CRIT="$TROOT/critic"
    JOINT="$TROOT/joint"
    mkdir -p "$CRIT" "$JOINT"

    echo
    echo "========================================================================================================================"
    echo "TASK $TASK — STAGE A CRITIC PRETRAIN"
    echo "========================================================================================================================"
    "$PY" -u train_faithful_critic.py \
      --index "$IDX" \
      --out-dir "$CRIT" \
      --epochs 12 \
      --micro-batch-size 8 \
      --grad-accum 4 \
      --num-workers 12 \
      --lr 1e-3 \
      --weight-decay 1e-5 \
      --seed 42 \
      2>&1 | tee "$TROOT/critic_console.log"

    if [ ! -f "$CRIT/best-network.pth" ]; then
      echo "ERROR: critic checkpoint missing for $TASK"
      exit 20
    fi

    echo
    echo "========================================================================================================================"
    echo "TASK $TASK — STAGE B FULL WHERE2ACT TRAINING"
    echo "========================================================================================================================"
    "$PY" -u train_faithful_joint.py \
      --index "$IDX" \
      --critic-checkpoint "$CRIT/best-network.pth" \
      --out-dir "$JOINT" \
      --epochs 12 \
      --micro-batch-size 8 \
      --grad-accum 4 \
      --num-workers 12 \
      --lr 1e-3 \
      --weight-decay 1e-5 \
      --seed 42 \
      2>&1 | tee "$TROOT/joint_console.log"

    if [ ! -f "$JOINT/best-network.pth" ]; then
      echo "ERROR: joint checkpoint missing for $TASK"
      exit 21
    fi

    echo "TASK $TASK COMPLETE"
  done

  "$PY" -u "$FINALIZER" "$RUN_ROOT"
  ln -sfn "$RUN_ROOT" "$RUN_ROOT_BASE/latest"
  echo
  echo "LATEST -> $RUN_ROOT_BASE/latest"
} 2>&1 | tee "$MASTER_LOG"
