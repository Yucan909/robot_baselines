#!/usr/bin/env python3
import hashlib, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from models.model_3d import Network

FEAT_DIM, RV_DIM, RV_CNT = 128, 10, 100
DEFAULT_INDEX = Path('/home/feng/robot_baselines/results/where2act/v3_data/corrected_opening_index_v3_critic.npz')
DEFAULT_FORMAL = Path('/home/feng/robot_baselines/configs/where2act_v3/formal_holdout_shapes.txt')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def read_shapes(path):
    return {x.strip() for x in Path(path).read_text().splitlines() if x.strip()}

def load_index(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def clean_state_dict(data):
    if isinstance(data, dict) and isinstance(data.get('model'), dict): data = data['model']
    elif isinstance(data, dict) and isinstance(data.get('state_dict'), dict): data = data['state_dict']
    out = {}
    for k, v in data.items():
        k = str(k)
        if k.startswith('module.'): k = k[len('module.'):]
        out[k] = v
    return out

def load_network(path, device):
    model = Network(FEAT_DIM, RV_DIM, RV_CNT)
    state = clean_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    model.load_state_dict(state, strict=True)
    return model.to(device)

def overwrite_pointnet_and_critic(model, checkpoint):
    state = clean_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False))
    pn, cr = {}, {}
    for k, v in state.items():
        if k.startswith('pointnet2.'): pn[k[len('pointnet2.'):]] = v
        elif k.startswith('critic.'): cr[k[len('critic.'):]] = v
    if not pn or not cr: raise RuntimeError('critic checkpoint missing pointnet2/critic')
    model.pointnet2.load_state_dict(pn, strict=True)
    model.critic.load_state_dict(cr, strict=True)
    model.critic_copy.load_state_dict(cr, strict=True)

def normalize_np(v):
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8: raise RuntimeError('invalid direction')
    return (v/n).astype(np.float32)

def make_gt_rotation(d1, d2):
    b1 = F.normalize(d1, dim=1)
    b2 = F.normalize(d2 - (b1*d2).sum(dim=1, keepdim=True)*b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1,b2,b3], dim=2)

def geodesic_rotation_loss(pred_R, gt_R):
    diff = torch.bmm(gt_R.transpose(1,2), pred_R)
    tr = diff[:,0,0] + diff[:,1,1] + diff[:,2,2]
    return torch.acos(torch.clamp((tr-1)/2, -1+1e-6, 1-1e-6))

def vector_angle(a,b):
    a,b = F.normalize(a,dim=-1),F.normalize(b,dim=-1)
    return torch.acos(torch.clamp((a*b).sum(dim=-1), -1+1e-7, 1-1e-7))

def rad2deg(x): return x*180.0/math.pi

def check_no_formal(index, formal_path=DEFAULT_FORMAL):
    formal = read_shapes(formal_path)
    leaked = sorted({str(x) for x in index['shape'] if str(x) in formal})
    if leaked: raise RuntimeError(f'FORMAL LEAKAGE: {leaked}')

def actor_ids(index, split_name):
    return np.flatnonzero(
        np.asarray(index['trust_strict'],bool) & np.asarray(index['grasp_gate'],bool)
        & (np.asarray(index['split']) == split_name)
    )

class FinalInteractionDataset(Dataset):
    def __init__(self,index,ids,*,num_points=8192,seed=42,with_affordance=False):
        self.index=index; self.ids=np.asarray(ids,np.int64); self.num_points=int(num_points)
        self.seed=int(seed); self.with_affordance=bool(with_affordance)
    def __len__(self): return len(self.ids)
    def __getitem__(self,item):
        idx=int(self.ids[item]); fn=str(self.index['static_path'][idx])
        with np.load(fn,allow_pickle=False) as z:
            pcw=np.asarray(z['initial_point_cloud_8192'],np.float32)
            cam=np.asarray(z['initial_camera_pose'] if 'initial_camera_pose' in z.files else z['camera_pose'],np.float32)
            if self.with_affordance: aff=np.asarray(z['initial_affordance_8192'],np.float32)
        if pcw.shape!=(8192,3) or not np.isfinite(pcw).all(): raise RuntimeError(f'bad pc: {fn}')
        pc=(pcw@cam[:3,:3]).astype(np.float32)
        qidx=int(self.index['qidx'][idx])
        if not 0<=qidx<len(pc): raise RuntimeError(f'bad qidx {qidx}')
        pc=pc.copy()
        if self.with_affordance: aff=np.clip(aff,0,1).astype(np.float32).copy()
        if qidx!=0:
            t=pc[0].copy(); pc[0]=pc[qidx]; pc[qidx]=t
            if self.with_affordance:
                ta=float(aff[0]); aff[0]=aff[qidx]; aff[qidx]=ta
        if self.num_points<len(pc):
            rng=np.random.default_rng(self.seed+idx*104729)
            choose=rng.choice(np.arange(1,len(pc)),size=self.num_points-1,replace=False)
            pc=np.concatenate([pc[:1],pc[choose]],axis=0)
            if self.with_affordance: aff=np.concatenate([aff[:1],aff[choose]],axis=0)
        d1=normalize_np(self.index['dir1_model'][idx]); d2=normalize_np(self.index['dir2_model'][idx])
        base=(torch.from_numpy(pc),torch.from_numpy(d1),torch.from_numpy(d2),torch.tensor(idx,dtype=torch.long))
        return base+(torch.from_numpy(aff),) if self.with_affordance else base

def generate_proposals(model,qf,d1,d2,*,generator=None,rv_cnt=RV_CNT):
    B=len(qf)
    rvs=torch.randn(B,rv_cnt,RV_DIM,device=qf.device,generator=generator)
    ef=qf.unsqueeze(1).expand(-1,rv_cnt,-1).reshape(B*rv_cnt,-1)
    pred6=model.actor(ef,rvs.reshape(B*rv_cnt,RV_DIM))
    Rflat=model.actor.bgs(pred6.reshape(-1,3,2)); R=Rflat.reshape(B,rv_cnt,3,3)
    gt=make_gt_rotation(d1,d2); gte=gt.unsqueeze(1).expand(-1,rv_cnt,-1,-1).reshape(B*rv_cnt,3,3)
    rot=geodesic_rotation_loss(Rflat,gte).reshape(B,rv_cnt)
    pd1,pd2=R[:,:,:,0],R[:,:,:,1]
    pull=vector_angle(-pd1,-d1.unsqueeze(1)); d2e=vector_angle(pd2,d2.unsqueeze(1))
    return {'expanded_feat':ef,'pred_R':R,'proposal_d1':pd1,'proposal_d2':pd2,'rot_err':rot,'pull_err':pull,'d2_err':d2e}

@torch.no_grad()
def top10_overlap(pred,gt):
    B,N=pred.shape; k=max(1,int(N*.1)); vals=[]
    for b in range(B):
        pi=torch.topk(pred[b],k).indices; gi=torch.topk(gt[b],k).indices
        pm=torch.zeros(N,dtype=torch.bool,device=pred.device); gm=pm.clone(); pm[pi]=True; gm[gi]=True
        vals.append((pm&gm).sum().float()/k)
    return float(torch.stack(vals).mean().item())

@torch.no_grad()
def cache_query_features(model,loader,device):
    model.pointnet2.eval(); qf=[]; d1s=[]; d2s=[]
    for batch in loader:
        pc,d1,d2=batch[:3]; pc=pc.to(device,non_blocking=True)
        feat=model.pointnet2(pc.repeat(1,1,2)); qf.append(feat[:,:,0].cpu()); d1s.append(d1.cpu()); d2s.append(d2.cpu())
    return torch.cat(qf),torch.cat(d1s),torch.cat(d2s)

@torch.no_grad()
def evaluate_runtime_once(model,qfc,d1c,d2c,device,*,seed,batch_size=128):
    gen=torch.Generator(device=device); gen.manual_seed(int(seed))
    acc={k:[] for k in ['sp','sr','sd','op','ort','pp30','pr45','cs','cg']}
    for start in range(0,len(qfc),batch_size):
        end=min(start+batch_size,len(qfc)); qf=qfc[start:end].to(device); d1=d1c[start:end].to(device); d2=d2c[start:end].to(device); B=len(qf)
        p=generate_proposals(model,qf,d1,d2,generator=gen)
        q6=torch.cat([p['proposal_d1'].reshape(B*RV_CNT,3),p['proposal_d2'].reshape(B*RV_CNT,3)],1)
        c=torch.sigmoid(model.critic(p['expanded_feat'],q6)).reshape(B,RV_CNT); sel=c.argmax(1); bi=torch.arange(B,device=device)
        acc['sp']+=rad2deg(p['pull_err'][bi,sel]).cpu().tolist(); acc['sr']+=rad2deg(p['rot_err'][bi,sel]).cpu().tolist(); acc['sd']+=rad2deg(p['d2_err'][bi,sel]).cpu().tolist()
        acc['op']+=rad2deg(p['pull_err'].min(1).values).cpu().tolist(); acc['ort']+=rad2deg(p['rot_err'].min(1).values).cpu().tolist()
        acc['pp30']+=(rad2deg(p['pull_err'])<30).float().mean(1).cpu().tolist(); acc['pr45']+=(rad2deg(p['rot_err'])<45).float().mean(1).cpu().tolist()
        acc['cs']+=c[bi,sel].cpu().tolist(); acc['cg']+=torch.sigmoid(model.critic(qf,torch.cat([d1,d2],1))).cpu().tolist()
    sp=np.asarray(acc['sp']); sr=np.asarray(acc['sr']); sd=np.asarray(acc['sd']); op=np.asarray(acc['op']); ort=np.asarray(acc['ort'])
    return {
      'seed':int(seed),'n':len(sp),'selected_pull_mean_deg':float(sp.mean()),'selected_pull_median_deg':float(np.median(sp)),
      'selected_pull_hit15':float((sp<15).mean()),'selected_pull_hit30':float((sp<30).mean()),'selected_pull_gt90':float((sp>90).mean()),
      'selected_rot_mean_deg':float(sr.mean()),'selected_rot_median_deg':float(np.median(sr)),'selected_rot_hit30':float((sr<30).mean()),'selected_rot_hit45':float((sr<45).mean()),
      'selected_d2_mean_deg':float(sd.mean()),'oracle_pull_mean_deg':float(op.mean()),'oracle_pull_hit15':float((op<15).mean()),'oracle_rot_mean_deg':float(ort.mean()),'oracle_rot_hit30':float((ort<30).mean()),
      'proposal_pull30_fraction':float(np.mean(acc['pp30'])),'proposal_rot45_fraction':float(np.mean(acc['pr45'])),'selected_critic_mean':float(np.mean(acc['cs'])),'gt_critic_mean':float(np.mean(acc['cg']))}

def mean_std(rows,key):
    a=np.asarray([r[key] for r in rows],np.float64); return {'mean':float(a.mean()),'std':float(a.std())}

def repeated_runtime_summary(model,loader,device,seeds):
    qf,d1,d2=cache_query_features(model,loader,device); runs=[evaluate_runtime_once(model,qf,d1,d2,device,seed=s) for s in seeds]
    keys=[k for k in runs[0] if k not in ('seed','n')]
    return runs,{k:mean_std(runs,k) for k in keys}
