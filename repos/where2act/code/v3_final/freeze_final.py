#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
HOME=Path.home()
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=str(HOME/'robot_baselines/repos/where2act/logs/where2act_final_actionscore/best-network.pth')); ap.add_argument('--backend',default=str(HOME/'robot_baselines/common_env/where2act_backend_final')); ap.add_argument('--smoke-root',default=str(HOME/'robot_baselines/results/where2act/FINAL_NONFORMAL_SMOKE40')); ap.add_argument('--eval-catalog',default=str(HOME/'robot_baselines/configs/flowbot3d/eval/eval_pose_catalog.jsonl')); ap.add_argument('--output',default=str(HOME/'robot_baselines/results/where2act/FINAL_FREEZE_MANIFEST.json')); a=ap.parse_args(); ck=Path(a.checkpoint).resolve(); be=Path(a.backend).resolve(); smoke=Path(a.smoke_root).resolve()/'SMOKE40_SUMMARY.json'; cat=Path(a.eval_catalog).resolve(); sd=json.loads(smoke.read_text())
 if not sd.get('engineering_smoke_pass'): raise RuntimeError('Smoke40 is not PASS')
 idx=HOME/'robot_baselines/results/where2act/v3_data/corrected_opening_index_v3_critic.npz'; cfg=HOME/'robot_baselines/configs/where2act_v3'; bfs=['run_trial.py','where2act_policy.py','where2act_runtime.py','where2act_observation_adapter.py','backend_v2_physics.py','panda_controller.py','contact_monitor.py','motion_planner_collision_worker.py','motion_planner_worker.py']; m={'protocol_version':'where2act_final_atomic_v1','method':'Where2Act-Final-Atomic','atomic_pull_distance_m':.05,'formal_targets':56,'formal_trials_per_target':20,'expected_formal_trials':1120,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'corrected_index':str(idx),'corrected_index_sha256':sha(idx),'eval_catalog':str(cat),'eval_catalog_sha256':sha(cat),'split_hashes':{n:sha(cfg/n) for n in ['train_shapes.txt','dev_shapes.txt','formal_holdout_shapes.txt']},'backend':str(be),'backend_hashes':{n:sha(be/n) for n in bfs},'smoke40':sd,'formal_data_used_for_tuning':False}; out=Path(a.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(m,indent=2)+'\n'); print(out); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
