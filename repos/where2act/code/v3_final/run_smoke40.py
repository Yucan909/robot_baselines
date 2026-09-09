#!/usr/bin/env python3
import argparse, concurrent.futures, json, os, subprocess
from collections import Counter
from pathlib import Path
HOME=Path.home(); DEFAULT_BACKEND=HOME/'robot_baselines/common_env/where2act_backend_final'; DEFAULT_CKPT=HOME/'robot_baselines/repos/where2act/logs/where2act_final_actionscore/best-network.pth'; DEV=HOME/'robot_baselines/configs/where2act_v3/dev_shapes.txt'; FORMAL=HOME/'robot_baselines/configs/where2act_v3/formal_holdout_shapes.txt'; DEFAULT_OUT=HOME/'robot_baselines/results/where2act/FINAL_NONFORMAL_SMOKE40'
def rset(p): return {x.strip() for x in Path(p).read_text().splitlines() if x.strip()}
def catalog(explicit):
 if explicit:
  p=Path(explicit).expanduser().resolve();
  if not p.exists(): raise FileNotFoundError(p)
  return p
 c=sorted((HOME/'robot_baselines/configs').rglob('train_pose_catalog.jsonl'))
 if not c: raise FileNotFoundError('train_pose_catalog.jsonl not found; pass --pose-catalog')
 return c[0]
def targets(cat):
 dev,formal=rset(DEV),rset(FORMAL); out=[]; seen=set()
 for line in open(cat):
  if not line.strip(): continue
  r=json.loads(line); sid=str(r['shape_id']); link=r.get('link_name') or r.get('target_link') or r.get('link'); key=(sid,str(link))
  if sid in dev and sid not in formal and link is not None and key not in seen: seen.add(key); out.append(key)
  if len(out)>=10: break
 if len(out)<10: raise RuntimeError(f'only {len(out)} nonformal dev targets')
 return out
def one(job):
 backend,ckpt,cat,out,sid,link,seed=job; logdir=out/'logs'; logdir.mkdir(parents=True,exist_ok=True); log=logdir/f'{sid}_{link}_seed_{seed}.log'; cmd=['python',str(backend/'run_trial.py'),'--shape-id',sid,'--target-link',link,'--pose-catalog',str(cat),'--trial-seed',str(seed),'--checkpoint',str(ckpt),'--output-root',str(out),'--device','cuda:0','--planner-env','where2act_planner','--planner','RRTConnect','--planning-time','5','--ik-attempts','100']; env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0'; p=subprocess.run(cmd,cwd=str(backend),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); log.write_text(p.stdout); rp=out/f'{sid}_{link}'/f'seed_{seed:03d}'/'result.json'; return {'sid':sid,'link':link,'seed':seed,'rc':p.returncode,'result':str(rp),'log':str(log)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--backend',default=str(DEFAULT_BACKEND)); ap.add_argument('--checkpoint',default=str(DEFAULT_CKPT)); ap.add_argument('--pose-catalog',default=None); ap.add_argument('--output-root',default=str(DEFAULT_OUT)); ap.add_argument('--workers',type=int,default=2); a=ap.parse_args(); backend=Path(a.backend).resolve(); ckpt=Path(a.checkpoint).resolve(); cat=catalog(a.pose_catalog); out=Path(a.output_root).resolve(); out.mkdir(parents=True,exist_ok=True); ts=targets(cat); seeds=[2026091000+i for i in range(4)]; jobs=[(backend,ckpt,cat,out,s,l,z) for s,l in ts for z in seeds]; rec=[]
 print('catalog:',cat); print('targets:',ts)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  for i,r in enumerate(ex.map(one,jobs),1): rec.append(r); print(f"[{i:02d}/40] {r['sid']}/{r['link']} seed={r['seed']} rc={r['rc']}")
 results=[]; missing=[]
 for r in rec:
  p=Path(r['result']); results.append(json.loads(p.read_text())) if p.exists() else missing.append(r)
 impl=sum(bool(x.get('implementation_error')) for x in results); reasons=Counter(str(x.get('failure_reason')) for x in results); summary={'scope':'nonformal_engineering_smoke_only','catalog':str(cat),'targets':[{'shape_id':s,'target_link':l} for s,l in ts],'trials_expected':40,'results_found':len(results),'missing_results':len(missing),'implementation_errors':impl,'policy_success':sum(bool(x.get('policy_success')) for x in results),'planning_success':sum(bool(x.get('planning_success')) for x in results),'pregrasp_execution_success':sum(bool(x.get('pregrasp_execution_success')) for x in results),'grasp_success':sum(bool(x.get('grasp_success')) for x in results),'final_success':sum(bool(x.get('final_success')) for x in results),'failure_reasons':dict(reasons),'engineering_smoke_pass':bool(not missing and impl==0)}; (out/'SMOKE40_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary['engineering_smoke_pass'] else 2)
if __name__=='__main__': main()
