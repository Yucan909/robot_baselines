#!/usr/bin/env python3
import argparse,concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
HOME=Path.home(); DEFAULT_FREEZE=HOME/'robot_baselines/results/where2act/FINAL_FREEZE_MANIFEST.json'; DEFAULT_OUT=HOME/'robot_baselines/results/where2act/FINAL_ATOMIC_1120'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def targets(cat):
 out=[]; seen=set()
 for line in open(cat):
  if not line.strip(): continue
  r=json.loads(line); link=r.get('link_name') or r.get('target_link') or r.get('link'); key=(str(r['shape_id']),str(link));
  if key in seen: raise RuntimeError(f'duplicate target {key}')
  seen.add(key); out.append(key)
 if len(out)!=56: raise RuntimeError(f'formal targets={len(out)} != 56')
 return out
def one(job):
 be,ck,cat,out,sid,link,seed=job; rp=out/f'{sid}_{link}'/f'seed_{seed:03d}'/'result.json'
 if rp.exists(): return {'sid':sid,'link':link,'seed':seed,'rc':0,'exists':True,'skipped':True}
 ld=out/'logs'; ld.mkdir(parents=True,exist_ok=True); log=ld/f'{sid}_{link}_seed_{seed}.log'; cmd=['python',str(be/'run_trial.py'),'--shape-id',sid,'--target-link',link,'--pose-catalog',str(cat),'--trial-seed',str(seed),'--checkpoint',str(ck),'--output-root',str(out),'--device','cuda:0','--planner-env','where2act_planner','--planner','RRTConnect','--planning-time','5','--ik-attempts','100']; env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0'; p=subprocess.run(cmd,cwd=str(be),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); log.write_text(p.stdout); return {'sid':sid,'link':link,'seed':seed,'rc':p.returncode,'exists':rp.exists(),'log':str(log)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--freeze-manifest',default=str(DEFAULT_FREEZE)); ap.add_argument('--output-root',default=str(DEFAULT_OUT)); ap.add_argument('--workers',type=int,default=2); a=ap.parse_args(); fp=Path(a.freeze_manifest).resolve(); f=json.loads(fp.read_text());
 if f.get('protocol_version')!='where2act_final_atomic_v1': raise RuntimeError('wrong freeze protocol')
 be=Path(f['backend']); ck=Path(f['checkpoint']); cat=Path(f['eval_catalog'])
 if sha(ck)!=f['checkpoint_sha256'] or sha(cat)!=f['eval_catalog_sha256']: raise RuntimeError('frozen checkpoint/catalog changed')
 for n,h in f['backend_hashes'].items():
  if sha(be/n)!=h: raise RuntimeError(f'backend changed: {n}')
 out=Path(a.output_root).resolve(); out.mkdir(parents=True,exist_ok=True); ts=targets(cat); seeds=[2026082900+i for i in range(20)]; jobs=[(be,ck,cat,out,s,l,z) for s,l in ts for z in seeds]; manifest={'protocol_version':'where2act_final_atomic_v1','freeze_manifest':str(fp),'freeze_manifest_sha256':sha(fp),'targets':56,'trials_per_target':20,'expected_trials':1120,'trial_seeds':seeds,'workers':a.workers,'checkpoint_sha256':f['checkpoint_sha256'],'backend_hashes':f['backend_hashes']}; (out/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); fail=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  for i,r in enumerate(ex.map(one,jobs),1):
   if r['rc']!=0 or not r['exists']: fail.append(r)
   if i%20==0 or i==1120: print(f'[{i:4d}/1120] runner failures={len(fail)}')
 (out/'runner_failures.json').write_text(json.dumps(fail,indent=2)+'\n'); print('Formal execution finished; runner failures:',len(fail))
if __name__=='__main__': main()
