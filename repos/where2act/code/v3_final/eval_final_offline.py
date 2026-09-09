#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from final_common import *

def eval_action(model,loader,device):
  gen=torch.Generator(device=device); gen.manual_seed(2026090204); sums={k:0. for k in ['query_mse','query_mae','static_mse','static_top10','pred_top1_in_static_top10']}; n=0
  with torch.no_grad():
    for pc,d1,d2,_,aff in loader:
      pc=pc.to(device); d1=d1.to(device); d2=d2.to(device); aff=aff.to(device); B,N,_=pc.shape; whole=model.pointnet2(pc.repeat(1,1,2)); qf=whole[:,:,0]; p=generate_proposals(model,qf,d1,d2,generator=gen); q6=torch.cat([p['proposal_d1'].reshape(B*100,3),p['proposal_d2'].reshape(B*100,3)],1); target=torch.sigmoid(model.critic(p['expanded_feat'],q6)).reshape(B,100).mean(1); pred=model.action_score(whole.permute(0,2,1).reshape(B*N,-1)).reshape(B,N); qp=pred[:,0]
      sums['query_mse']+=torch.mean((qp-target)**2).item()*B; sums['query_mae']+=torch.mean(torch.abs(qp-target)).item()*B; sums['static_mse']+=torch.mean((pred-aff)**2).item()*B; sums['static_top10']+=top10_overlap(pred,aff)*B; k=max(1,int(N*.1)); pi=pred.argmax(1); gt=torch.topk(aff,k,dim=1).indices; sums['pred_top1_in_static_top10']+=(pi[:,None]==gt).any(1).float().mean().item()*B; n+=B
  return {k:v/max(n,1) for k,v in sums.items()}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--index',default=str(DEFAULT_INDEX)); ap.add_argument('--checkpoint',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_actionscore/best-network.pth'); ap.add_argument('--out-dir',default='/home/feng/robot_baselines/repos/where2act/logs/where2act_final_offline'); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--num-workers',type=int,default=12); a=ap.parse_args()
  if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
  dev=torch.device('cuda'); idx=load_index(a.index); check_no_formal(idx); ids=actor_ids(idx,'dev'); ds=FinalInteractionDataset(idx,ids,seed=20260902,with_affordance=True); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0,prefetch_factor=2 if a.num_workers>0 else None); model=load_network(a.checkpoint,dev); model.eval(); runs,s=repeated_runtime_summary(model,dl,dev,[7100,7101,7102,7103,7104]); act=eval_action(model,dl,dev)
  gates={'oracle_pull_hit15_ge_0.90':s['oracle_pull_hit15']['mean']>=.90,'oracle_rot_hit30_ge_0.90':s['oracle_rot_hit30']['mean']>=.90,'selected_pull_mean_le_25deg':s['selected_pull_mean_deg']['mean']<=25,'selected_pull_median_le_15deg':s['selected_pull_median_deg']['mean']<=15,'selected_pull_hit30_ge_0.80':s['selected_pull_hit30']['mean']>=.80,'selected_pull_gt90_le_0.02':s['selected_pull_gt90']['mean']<=.02,'selected_rot_mean_le_40deg':s['selected_rot_mean_deg']['mean']<=40,'selected_rot_hit45_ge_0.75':s['selected_rot_hit45']['mean']>=.75,'actionscore_query_mae_le_0.10':act['query_mae']<=.10,'actionscore_static_top10_ge_0.25':act['static_top10']>=.25}; result={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256_file(a.checkpoint),'dev_samples':len(ids),'formal_overlap':0,'runtime_repeated_runs':runs,'runtime_repeated_summary':s,'actionscore':act,'engineering_gates':gates,'offline_gate_pass':all(gates.values())}; out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'final_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
  lines=['='*105,'WHERE2ACT FINAL OFFLINE REPORT','='*105,f'checkpoint                   : {Path(a.checkpoint).resolve()}',f'SHA256                       : {sha256_file(a.checkpoint)}',f'dev samples                  : {len(ids)}','formal overlap              : 0','','ACTOR + CRITIC, 5 proposal seeds']+[f"{k:<44s}: {v['mean']:.6f} ± {v['std']:.6f}" for k,v in s.items()]+['','ACTIONSCORE']+[f"{k:<44s}: {v:.6f}" for k,v in act.items()]+['','ENGINEERING GATES']+[f"  {k:<47s}: {'PASS' if v else 'FAIL'}" for k,v in gates.items()]+['',f"FINAL OFFLINE GATE           : {'PASS' if all(gates.values()) else 'FAIL'}"]
  (out/'final_report.txt').write_text('\n'.join(lines)+'\n'); print((out/'final_report.txt').read_text())
if __name__=='__main__': main()
