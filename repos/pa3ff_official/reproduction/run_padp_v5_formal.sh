#!/usr/bin/env bash
set -euo pipefail

CODE=/home/feng/robot_baselines/repos/pa3ff_official/reproduction
CATALOG=/home/feng/robot_baselines/configs/pa3ff/reproduction_v1_formal/formal_episode_catalog.jsonl
SMOKE_CASES=/home/feng/robot_baselines/configs/pa3ff/reproduction_v2_soft_weld_pd_smoke_cases.json
PYTHON=/home/feng/miniconda3/envs/pa3ff/bin/python
CHECKPOINT="$1"
FORMAL_ROOT="$2"
MANIFEST="$FORMAL_ROOT/protocol_manifest.json"
WORKER="$CODE/dev_receding_bottomfaithful_padp_v5_worker_v39.py"
NWORKERS="${PA3FF_V5_FORMAL_WORKERS:-8}"

# Frozen from object-level DEV before smoke/formal.  The task-specific rank
# weights multiply only observation geometry and statistics fitted on TRAIN.
export PA3FF_DEV_CANDIDATE_COUNT=128
export PA3FF_DEV_RANK_MODE=long_motion
export PA3FF_DEV_CLIP_PRED_X0=1.5
export PA3FF_DEV_NOISE_SCALE=0.5
export PA3FF_DEV_DDIM_STEPS=10
export PA3FF_DEV_START_TIMESTEP=90
export PA3FF_DEV_POSITION_PRIOR_WEIGHT=0.02
export PA3FF_DEV_GEOMETRY_PRIOR_WEIGHT=0.02
export PA3FF_DEV_GRASP_GEOMETRY_PRIOR_WEIGHT=0.0
export PA3FF_DOOR_OPEN_POSITION_PRIOR_WEIGHT=0.0
export PA3FF_DOOR_OPEN_GEOMETRY_PRIOR_WEIGHT=0.1
export PA3FF_DOOR_OPEN_GRASP_GEOMETRY_PRIOR_WEIGHT=0.8
export PA3FF_DRAWER_OPEN_POSITION_PRIOR_WEIGHT=0.0
export PA3FF_DRAWER_OPEN_GEOMETRY_PRIOR_WEIGHT=0.0
export PA3FF_DRAWER_OPEN_GRASP_GEOMETRY_PRIOR_WEIGHT=0.8

if [[ "${PA3FF_SKIP_SMOKE:-0}" != "1" ]]; then
  mkdir -p "$FORMAL_ROOT/smoke"
  smoke_pids=()
  for task in door_open door_close drawer_open drawer_close; do
    "$CODE/formal_env.sh" "$WORKER" \
      --checkpoint "$CHECKPOINT" --formal-root "$FORMAL_ROOT/smoke" \
      --catalog "$CATALOG" --protocol-manifest "$MANIFEST" \
      --task "$task" --smoke-cases "$SMOKE_CASES" \
      > "$FORMAL_ROOT/smoke_${task}.log" 2>&1 &
    smoke_pids+=("$!")
  done
  for pid in "${smoke_pids[@]}"; do wait "$pid"; done
  "$PYTHON" "$CODE/validate_padp_v4_smoke.py" --smoke-root "$FORMAL_ROOT/smoke" \
    > "$FORMAL_ROOT/smoke_validation_console.log" 2>&1
fi

formal_pids=()
for ((shard=0; shard<NWORKERS; shard++)); do
  "$CODE/formal_env.sh" "$WORKER" \
    --checkpoint "$CHECKPOINT" --formal-root "$FORMAL_ROOT" \
    --catalog "$CATALOG" --protocol-manifest "$MANIFEST" \
    --num-shards "$NWORKERS" --shard-id "$shard" \
    > "$FORMAL_ROOT/formal_shard_${shard}.log" 2>&1 &
  formal_pids+=("$!")
done
status=0
for pid in "${formal_pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
"$PYTHON" "$CODE/aggregate_formal.py" --formal-root "$FORMAL_ROOT" \
  > "$FORMAL_ROOT/aggregate_console.log" 2>&1
