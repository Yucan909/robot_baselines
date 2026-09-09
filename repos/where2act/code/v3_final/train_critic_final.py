#!/usr/bin/env python3
import argparse,json,math,random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from final_common import *
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision('high')
PULL_SIGMA_DEG=30.; ROT_SIGMA_DEG=60.; HIGH_Q=.65; LOW_Q=.20

def train_epoch(model,loader,opt,device,gen,qw,ew,rw,margin):
  model.pointnet2.eval(); model.actor.eval(); model.action_score.eval(); model.critic_copy.eval(); model.critic.train(); sums={k:0. for k in ['loss','quality','exact','rank']}; n=hi=lo=rv=0
  for step,(pc,d1,d2,_) in enumerate(loader):
    pc=pc.to(device,non_blocking=True); d1=d1.to(device,non_blocking=True); d2=d2.to(device,non_blocking=True); B=len(pc)
    with torch.no_grad():
      qf=model.pointnet2(pc.repeat(1,1,2))[:,:,0]; p=generate_proposals(model,qf,d1,d2,generator=gen); pd=rad2deg(p['pull_err']); rd=rad2deg(p['rot_err']); qp=torch.exp(-(pd/PULL_SIGMA_DEG)**2); qr=torch.exp(-(rd/ROT_SIGMA_DEG)**2); target=qp*(.65+.35*qr)
    q6=torch.cat([p['proposal_d1'].reshape(B*100,3),p['proposal_d2'].reshape(B*100,3)],1); logits=model.critic(p['expanded_feat'],q6).reshape(B,100); qloss=F.binary_cross_entropy_with_logits(logits,target)
    gt=model.critic(qf,torch.cat([d1,d2],1)); rev=model.critic(qf,torch.cat([-d1,d2],1)); exact=.5*F.binary_cross_entropy_with_logits(gt,torch.ones_like(gt))+.5*F.binary_cross_entropy_with_logits(rev,torch.zeros_like(rev))
    high=target>=HIGH_Q; low=target<=LOW_Q; valid=high.any(1)&low.any(1); hi+=int(high.sum()); lo+=int(low.sum()); rv+=int(valid.sum())
    if valid.any():
      neg=torch.tensor(-1e9,device=device,dtype=logits.dtype); hmax=logits.masked_fill(~high,neg).max(1).values; lmax=logits.masked_fill(~low,neg).max(1).values; rank=F.softplus(float(margin)-hmax[valid]+lmax[valid]).mean()
    else: rank=qloss*0
    loss=qw*qloss+ew*exact+rw*rank; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.critic.parameters(),5.); opt.step()
    for k,v in [('loss',loss),('quality',qloss),('exact',exact),('rank',rank)]: sums[k]+=float(v.item())*B
    n+=B
    if step==0 or (step+1)%100==0:
      with torch.no_grad(): c=torch.sigmoid(logits); sel=c.argmax(1); bi=torch.arange(B,device=device); print(f"  step {step+1:5d}/{len(loader):5d} loss={loss.item():.4f} q={qloss.item():.4f} rank={rank.item():.4f} selPull={pd[bi,sel].mean().item():.2f} selRot={rd[bi,sel].mean().item():.2f}")
  o={k:v/max(n,1) for k,v in sums.items()}; o.update(high_quality_proposals=hi,low_quality_proposals=lo,rank_valid_interactions=rv); return o

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--index',default=str(DEFAULT_INDEX)); ap.add_argument('--actor-checkpoint',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_actor/best-network.pth'); ap.add_argument('--critic-init',default='/home/feng/robot_baselines/repos/where2act/logs/v3_critic_v3_1/best-network.pth'); ap.add_argument('--out-dir',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_critic'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--num-workers',type=int,default=12); ap.add_argument('--quality-weight',type=float,default=1.); ap.add_argument('--exact-weight',type=float,default=.25); ap.add_argument('--rank-weight',type=float,default=.5); ap.add_argument('--rank-margin',type=float,default=1.); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
  random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
  if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
  dev=torch.device('cuda'); idx=load_index(a.index); check_no_formal(idx); trids=actor_ids(idx,'train'); dvids=actor_ids(idx,'dev'); print('='*100+'\nWHERE2ACT FINAL CONTINUOUS CRITIC\n'+'='*100); print('train/dev:',len(trids),len(dvids),'formal overlap: 0')
  tr=FinalInteractionDataset(idx,trids,seed=a.seed); dv=FinalInteractionDataset(idx,dvids,seed=a.seed+100000); g=torch.Generator(); g.manual_seed(a.seed); kw=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,drop_last=False,persistent_workers=a.num_workers>0,prefetch_factor=2 if a.num_workers>0 else None); tl=DataLoader(tr,shuffle=True,generator=g,**kw); dl=DataLoader(dv,shuffle=False,**kw)
  model=load_network(a.actor_checkpoint,dev); actor_state={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()}; overwrite_pointnet_and_critic(model,a.critic_init); model.actor.load_state_dict(actor_state,strict=True)
  for m in [model.pointnet2,model.actor,model.action_score,model.critic_copy]:
    for q in m.parameters(): q.requires_grad=False
  opt=torch.optim.Adam(model.critic.parameters(),lr=a.lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); tg=torch.Generator(device=dev); tg.manual_seed(a.seed); qfc,d1c,d2c=cache_query_features(model,dl,dev); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); best=float('inf'); be=None; hist=[]
  for ep in range(1,a.epochs+1):
    print(f"\n{'='*100}\nEPOCH {ep}/{a.epochs} lr={opt.param_groups[0]['lr']:.2e}\n{'='*100}"); tm=train_epoch(model,tl,opt,dev,tg,a.quality_weight,a.exact_weight,a.rank_weight,a.rank_margin); model.critic_copy.load_state_dict(model.critic.state_dict(),strict=True); dm=evaluate_runtime_once(model,qfc,d1c,d2c,dev,seed=2026090202); sch.step(); score=dm['selected_pull_mean_deg']+.5*dm['selected_rot_mean_deg']; print('[DEV]',dm,'score=',score); hist.append({'epoch':ep,'train':tm,'dev':dm,'selection_score':score})
    st={'epoch':ep,'model':model.state_dict(),'train_metrics':tm,'dev_metrics':dm,'selection_score':score,'training_version':'where2act_final_continuous_critic_v1','index_sha256':sha256_file(a.index),'actor_sha256':sha256_file(a.actor_checkpoint),'critic_init_sha256':sha256_file(a.critic_init)}; torch.save(st,out/'last.pt'); torch.save(model.state_dict(),out/'last-network.pth')
    if score<best: best=score; be=ep; torch.save(st,out/'best.pt'); torch.save(model.state_dict(),out/'best-network.pth'); print('[BEST]',be,best)
    (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
  model=load_network(out/'best-network.pth',dev); model.eval(); runs,s=repeated_runtime_summary(model,dl,dev,[6100,6101,6102,6103,6104]); gates={'actor_oracle_pull_hit15_ge_0.90':s['oracle_pull_hit15']['mean']>=.90,'actor_oracle_rot_hit30_ge_0.90':s['oracle_rot_hit30']['mean']>=.90,'selected_pull_mean_le_25deg':s['selected_pull_mean_deg']['mean']<=25,'selected_pull_median_le_15deg':s['selected_pull_median_deg']['mean']<=15,'selected_pull_hit30_ge_0.80':s['selected_pull_hit30']['mean']>=.80,'selected_pull_gt90_le_0.02':s['selected_pull_gt90']['mean']<=.02,'selected_rot_mean_le_40deg':s['selected_rot_mean_deg']['mean']<=40,'selected_rot_hit45_ge_0.75':s['selected_rot_hit45']['mean']>=.75}; final={'best_epoch':be,'best_selection_score':best,'formal_overlap':0,'repeated_runs':runs,'repeated_summary':s,'engineering_gates':gates,'continuous_critic_gate_pass':all(gates.values())}; (out/'final_metrics.json').write_text(json.dumps(final,indent=2)+'\n')
  lines=['='*100,'WHERE2ACT FINAL CONTINUOUS CRITIC REPORT','='*100,f'best epoch                  : {be}','formal overlap              : 0','','FINAL 5-SEED RUNTIME-LIKE DEV']+[f"{k:<42s}: {v['mean']:.6f} ± {v['std']:.6f}" for k,v in s.items()]+['','GATES']+[f"  {k:<45s}: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]+['',f"CONTINUOUS CRITIC GATE      : {'PASS' if all(gates.values()) else 'FAIL'}",f"checkpoint                   : {out/'best-network.pth'}"]
  (out/'final_report.txt').write_text('\n'.join(lines)+'\n'); print((out/'final_report.txt').read_text())
if __name__=='__main__': main()
