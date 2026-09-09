#!/usr/bin/env python3
import argparse,json,math,random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from final_common import *
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision('high')

def run_epoch(model,loader,opt,device,training,seed):
    model.pointnet2.eval(); model.critic.eval(); model.critic_copy.eval(); model.action_score.eval(); model.actor.train(training)
    gen=torch.Generator(device=device); gen.manual_seed(int(seed)); sums={k:0. for k in ['loss','oracle_pull_deg','oracle_rot_deg','oracle_pull_hit15','oracle_rot_hit30','proposal_pull30_fraction','proposal_rot45_fraction','proposal_pull_mean_deg','proposal_rot_mean_deg']}; n=0
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
      for step,(pc,d1,d2,_) in enumerate(loader):
        pc=pc.to(device,non_blocking=True); d1=d1.to(device,non_blocking=True); d2=d2.to(device,non_blocking=True); B=len(pc)
        with torch.no_grad(): qf=model.pointnet2(pc.repeat(1,1,2))[:,:,0]
        p=generate_proposals(model,qf,d1,d2,generator=gen); rot=p['rot_err']; pull=p['pull_err']
        best=rot.min(1).values.mean(); top25=torch.topk(rot,min(25,rot.shape[1]),1,largest=False).values.mean(); top50p=torch.topk(pull,min(50,pull.shape[1]),1,largest=False).values.mean()
        loss=best+.5*top25+.75*top50p+.25*torch.relu(pull-math.radians(30)).mean()+.15*torch.relu(rot-math.radians(45)).mean()
        if training:
          opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.); opt.step()
        with torch.no_grad():
          pd=rad2deg(pull); rd=rad2deg(rot); op=pd.min(1).values; ort=rd.min(1).values
          vals={'loss':loss.item(),'oracle_pull_deg':op.mean().item(),'oracle_rot_deg':ort.mean().item(),'oracle_pull_hit15':(op<15).float().mean().item(),'oracle_rot_hit30':(ort<30).float().mean().item(),'proposal_pull30_fraction':(pd<30).float().mean().item(),'proposal_rot45_fraction':(rd<45).float().mean().item(),'proposal_pull_mean_deg':pd.mean().item(),'proposal_rot_mean_deg':rd.mean().item()}
          for k,v in vals.items(): sums[k]+=v*B
          n+=B
        if training and (step==0 or (step+1)%100==0): print(f"  step {step+1:5d}/{len(loader):5d} loss={loss.item():.4f} oraclePull={op.mean().item():.2f} oracleRot={ort.mean().item():.2f} propPull30={(pd<30).float().mean().item():.3f} propRot45={(rd<45).float().mean().item():.3f}")
    o={k:v/max(n,1) for k,v in sums.items()}; o['selection_score']=o['oracle_rot_deg']+.5*o['oracle_pull_deg']+40*(1-o['proposal_pull30_fraction'])+25*(1-o['proposal_rot45_fraction']); return o

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--index',default=str(DEFAULT_INDEX)); ap.add_argument('--init',default='/home/feng/robot_baselines/results/where2act/v3_frozen/actor_v3_epoch12_generator.pth'); ap.add_argument('--out-dir',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_actor'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--num-workers',type=int,default=12); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
  random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
  if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
  dev=torch.device('cuda'); idx=load_index(a.index); check_no_formal(idx); trids=actor_ids(idx,'train'); dvids=actor_ids(idx,'dev'); print('='*100+'\nWHERE2ACT FINAL ACTOR CONCENTRATION\n'+'='*100); print('train/dev:',len(trids),len(dvids),'formal overlap: 0')
  tr=FinalInteractionDataset(idx,trids,seed=a.seed); dv=FinalInteractionDataset(idx,dvids,seed=a.seed+100000); g=torch.Generator(); g.manual_seed(a.seed)
  kw=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,drop_last=False,persistent_workers=a.num_workers>0,prefetch_factor=2 if a.num_workers>0 else None)
  tl=DataLoader(tr,shuffle=True,generator=g,**kw); dl=DataLoader(dv,shuffle=False,**kw)
  model=load_network(a.init,dev)
  for m in [model.pointnet2,model.critic,model.critic_copy,model.action_score]:
    for q in m.parameters(): q.requires_grad=False
  opt=torch.optim.Adam(model.actor.parameters(),lr=a.lr,weight_decay=1e-5); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); best=float('inf'); be=None; hist=[]
  for ep in range(1,a.epochs+1):
    print(f"\n{'='*100}\nEPOCH {ep}/{a.epochs} lr={opt.param_groups[0]['lr']:.2e}\n{'='*100}"); tm=run_epoch(model,tl,opt,dev,True,a.seed+ep*1000); dm=run_epoch(model,dl,None,dev,False,2026090201); sch.step(); print('[DEV]',dm); hist.append({'epoch':ep,'train':tm,'dev':dm})
    st={'epoch':ep,'model':model.state_dict(),'train_metrics':tm,'dev_metrics':dm,'training_version':'where2act_final_actor_concentration_v1','index_sha256':sha256_file(a.index),'init_sha256':sha256_file(a.init)}; torch.save(st,out/'last.pt'); torch.save(model.state_dict(),out/'last-network.pth')
    if dm['selection_score']<best: best=dm['selection_score']; be=ep; torch.save(st,out/'best.pt'); torch.save(model.state_dict(),out/'best-network.pth'); print('[BEST]',be,best)
    (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
  model=load_network(out/'best-network.pth',dev); model.eval(); runs,s=repeated_runtime_summary(model,dl,dev,[5100,5101,5102,5103,5104]); gates={'oracle_pull_hit15_ge_0.90':s['oracle_pull_hit15']['mean']>=.90,'oracle_rot_hit30_ge_0.90':s['oracle_rot_hit30']['mean']>=.90,'proposal_pull30_fraction_ge_0.50':s['proposal_pull30_fraction']['mean']>=.50,'proposal_rot45_fraction_ge_0.40':s['proposal_rot45_fraction']['mean']>=.40}
  final={'best_epoch':be,'best_selection_score':best,'train_samples':len(trids),'dev_samples':len(dvids),'formal_overlap':0,'repeated_runs':runs,'repeated_summary':s,'engineering_gates':gates,'actor_concentration_gate_pass':all(gates.values())}; (out/'final_metrics.json').write_text(json.dumps(final,indent=2)+'\n')
  lines=['='*100,'WHERE2ACT FINAL ACTOR REPORT','='*100,f'best epoch                  : {be}',f'train/dev                   : {len(trids)} / {len(dvids)}','formal overlap              : 0','','FINAL 5-SEED GENERATOR']+[f"{k:<42s}: {v['mean']:.6f} ± {v['std']:.6f}" for k,v in s.items()]+['','GATES']+[f"  {k:<45s}: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]+['',f"ACTOR CONCENTRATION GATE    : {'PASS' if all(gates.values()) else 'FAIL'}",f"checkpoint                   : {out/'best-network.pth'}"]
  (out/'final_report.txt').write_text('\n'.join(lines)+'\n'); print((out/'final_report.txt').read_text())
if __name__=='__main__': main()
