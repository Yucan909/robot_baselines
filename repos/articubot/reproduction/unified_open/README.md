# ArticuBot unified OPEN adapter

This directory contains a runtime-only adapter for the unmodified official
ArticuBot high- and low-level pretrained checkpoints. It deliberately bypasses
ArticuBot's original-paper evaluation objects, camera setup, initialization,
and metric while retaining the official observation preprocessing, high-level
weighted-displacement inference, diffusion-policy history/horizon/normalizer,
and 10D action decoding.

Formal execution imports the frozen OPEN protocol from:

- `/home/feng/robot_baselines/configs/flowbot3d/eval/eval_pose_catalog.jsonl`
- `/home/feng/robot_baselines/common_env/flowbot3d_conditionA_physical_v2`
- `/home/feng/robot_baselines/scripts/relabel_flowbot_35.py`

The PHYSICAL_V2 target-joint phases are preserved: INIT/PREGRASP locked,
OPEN/TO_GRASP free, CLOSE locked and stabilized, and HOLD/OPERATE free. Grasp
success comes directly from the frozen bilateral physical-contact monitor, and
final success is `grasp_success && final_progress >= 0.35`.

`run_articubot_unified_open.py` is resumable at the per-episode JSONL level.
`supervise_and_finalize.py` watches four formal shards, restarts an interrupted
shard, merges exactly 1120 unique case/repeat pairs, and runs the independent
final audit and aggregation.

`run_articubot_batched_formal.py` was used only for a discarded throughput
experiment. Batched floating-point execution changed the seeded diffusion
trajectory, so no output from it is eligible for the formal metrics.
