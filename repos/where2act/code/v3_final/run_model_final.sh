#!/usr/bin/env bash
set -euo pipefail
FINAL=/home/feng/robot_baselines/repos/where2act/code/v3_final
LOGS=/home/feng/robot_baselines/repos/where2act/logs
mkdir -p "$LOGS"
cd "$FINAL"
echo '========== 1/4 FINAL ACTOR =========='
python -u train_actor_final.py --epochs 8 --lr 2e-4 --batch-size 8 --num-workers 12 2>&1 | tee "$LOGS/where2act_final_actor_console.log"
echo '========== 2/4 FINAL CRITIC =========='
python -u train_critic_final.py --epochs 8 --lr 3e-4 --batch-size 8 --num-workers 12 --quality-weight 1.0 --exact-weight 0.25 --rank-weight 0.50 --rank-margin 1.0 2>&1 | tee "$LOGS/where2act_final_critic_console.log"
echo '========== 3/4 FINAL ACTIONSCORE =========='
python -u train_actionscore_final.py --epochs 8 --lr 5e-4 --static-weight 0.10 --batch-size 8 --num-workers 12 2>&1 | tee "$LOGS/where2act_final_actionscore_console.log"
echo '========== 4/4 FINAL OFFLINE =========='
python -u eval_final_offline.py --batch-size 8 --num-workers 12 2>&1 | tee "$LOGS/where2act_final_offline_console.log"
echo '========== COMPLETE =========='
cat "$LOGS/where2act_final_offline/final_report.txt"
