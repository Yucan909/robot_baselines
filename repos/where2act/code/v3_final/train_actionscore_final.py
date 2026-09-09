#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from final_common import *
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision('high')

def run_epoch(model,loader,opt,device,training,static_weight,seed):
  model.pointnet2.eval(); model.actor.eval(); model.critic.eval(); model.critic_copy.eval(); model.action_score.train(training); gen=torch.Generator(device=device); gen.manual_seed(int(seed)); sums={k:0. for k in ['loss','query_mse','query_mae','static_mse','static_top10','pred_query_mean','critic_target_mean']}; n=0; ctx=torch.enable_grad() if training else torch.no_grad()
  with ctx:
    for step,(pc,d1,d2,_,aff) in enumerate(loader):
      pc=pc.to(device,non_blocking=True); d1=d1.to(device,non_blocking=True); d2=d2.to(device,non_blocking=True); aff=aff.to(device,non_blocking=True); B,N,_=pc.shape
      with torch.no_grad():
        whole=model.pointnet2(pc.repeat(1,1,2)); qf=whole[:,:,0]; p=generate_proposals(model,qf,d1,d2,generator=gen); q6=torch.cat([p['proposal_d1'].reshape(B*100,3),p['proposal_d2'].reshape(B*100,3)],1); target=torch.sigmoid(model.critic(p['expanded_feat'],q6)).reshape(B,100).mean(1)
      pred=model.action_score(whole.permute(0,2,1).reshape(B*N,-1)).reshape(B,N); qpred=pred[:,0]; qmse=F.mse_loss(qpred,target); smse=F.mse_loss(pred,aff); loss=qmse+float(static_weight)*smse
      if training: opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.action_score.parameters(),5.); opt.step()
      with torch.no_grad():
        vals={'loss':loss.item(),'query_mse':qmse.item(),'query_mae':F.l1_loss(qpred,target).item(),'static_mse':smse.item(),'static_top10':top10_overlap(pred,aff),'pred_query_mean':qpred.mean().item(),'critic_target_mean':target.mean().item()}
        for k,v in vals.items(): sums[k]+=float(v)*B
        n+=B
      if training and (step==0 or (step+1)%100==0): print(f"  step {step+1:5d}/{len(loader):5d} loss={loss.item():.5f} queryMSE={qmse.item():.5f} queryMAE={vals['query_mae']:.5f} static={smse.item():.5f} top10={vals['static_top10']:.3f}")
  return {k:v/max(n,1) for k,v in sums.items()}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--index',default=str(DEFAULT_INDEX)); ap.add_argument('--init',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_critic/best-network.pth'); ap.add_argument('--out-dir',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_actionscore'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--static-weight',type=float,default=.10); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--num-workers',type=int,default=12); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
  random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
  if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
  dev=torch.device('cuda'); idx=load_index(a.index); check_no_formal(idx); trids=actor_ids(idx,'train'); dvids=actor_ids(idx,'dev'); print('='*100+'\nWHERE2ACT FINAL ACTIONSCORE\n'+'='*100); print('train/dev:',len(trids),len(dvids),'formal overlap: 0'); print('loss = query Critic-expectation MSE + 0.10*dense static prior')
  tr=FinalInteractionDataset(idx,trids,seed=a.seed,with_affordance=True); dv=FinalInteractionDataset(idx,dvids,seed=a.seed+100000,with_affordance=True); g=torch.Generator(); g.manual_seed(a.seed); kw=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,drop_last=False,persistent_workers=a.num_workers>0,prefetch_factor=2 if a.num_workers>0 else None); tl=DataLoader(tr,shuffle=True,generator=g,**kw); dl=DataLoader(dv,shuffle=False,**kw)
  model=load_network(a.init,dev)
  for m in [model.pointnet2,model.actor,model.critic,model.critic_copy]:
    for q in m.parameters(): q.requires_grad=False
  opt=torch.optim.Adam(model.action_score.parameters(),lr=a.lr,weight_decay=1e-5); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); best=float('inf'); be=None; hist=[]
  for ep in range(1,a.epochs+1):
    print(f"\n{'='*100}\nEPOCH {ep}/{a.epochs} lr={opt.param_groups[0]['lr']:.2e}\n{'='*100}"); tm=run_epoch(model,tl,opt,dev,True,a.static_weight,a.seed+ep*1000); dm=run_epoch(model,dl,None,dev,False,a.static_weight,2026090203); sch.step(); score=dm['query_mse']+a.static_weight*dm['static_mse']; print('[DEV]',dm); hist.append({'epoch':ep,'train':tm,'dev':dm,'selection_score':score}); st={'epoch':ep,'model':model.state_dict(),'train_metrics':tm,'dev_metrics':dm,'selection_score':score,'training_version':'where2act_final_actionscore_v1','static_weight':a.static_weight,'index_sha256':sha256_file(a.index),'init_sha256':sha256_file(a.init)}; torch.save(st,out/'last.pt'); torch.save(model.state_dict(),out/'last-network.pth')
    if score<best: best=score; be=ep; torch.save(st,out/'best.pt'); torch.save(model.state_dict(),out/'best-network.pth'); print('[BEST]',be,best)
    (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
  model=load_network(out/'best-network.pth',dev); model.eval(); dm=run_epoch(model,dl,None,dev,False,a.static_weight,2026090203); gates={'query_target_mae_le_0.10':dm['query_mae']<=.10,'static_top10_ge_0.25':dm['static_top10']>=.25}; final={'best_epoch':be,'best_selection_score':best,'dev_metrics':dm,'formal_overlap':0,'engineering_gates':gates,'actionscore_gate_pass':all(gates.values()),'final_runtime_checkpoint':str(out/'best-network.pth'),'checkpoint_sha256':sha256_file(out/'best-network.pth')}; (out/'final_metrics.json').write_text(json.dumps(final,indent=2)+'\n')
  lines=['='*100,'WHERE2ACT FINAL ACTIONSCORE REPORT','='*100,f'best epoch                  : {be}',f"query target MSE             : {dm['query_mse']:.6f}",f"query target MAE             : {dm['query_mae']:.6f}",f"dense static MSE             : {dm['static_mse']:.6f}",f"dense static Top10           : {dm['static_top10']:.6f}",'formal overlap              : 0','','GATES']+[f"  {k:<45s}: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]+['',f"ACTIONSCORE GATE             : {'PASS' if all(gates.values()) else 'FAIL'}",f"FINAL CHECKPOINT             : {out/'best-network.pth'}",f"SHA256                       : {sha256_file(out/'best-network.pth')}"]
  (out/'final_report.txt').write_text('\n'.join(lines)+'\n'); print((out/'final_report.txt').read_text())
if __name__=='__main__': main()
