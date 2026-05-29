"""Generate eccv-rife-pipeline.ipynb — full Task B pipeline (Kaggle).

Sections:
  1-4  load RIFE, helpers, triplet dataset, fine-tune (2x T4)
  5    scale/TTA diagnostic per delta_s (informational)
  6    GEOMETRY self-check: warp t0->t1 via lidar, compare to real t1 (is path A open?)
  7    SAFE-FLOOR submission: per-sample motion gating, scene-disjoint validation,
       submits the safe-floor only if it beats the base on honest validation.

Writes to the user's Downloads path so they can re-upload directly.
Run:  python3 build_pipeline_nb.py
"""
import json

OUT_PATHS = [
    "/Users/flexonafft/Downloads/eccv-rife-pipeline.ipynb",
    "/Users/flexonafft/MLChallenge2026/NovelView/eccv-rife-pipeline.ipynb",
]

cells = []
def _l(s): return s.splitlines(keepends=True)
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": _l(s)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _l(s)})

md("""# Task B — Novel View Synthesis pipeline

1–4. Fine-tune RIFE on `(t0,t1)->target` triplets (2× T4).
5. Scale/TTA diagnostic per `delta_s`.
6. **Geometry self-check** — warp `t0->t1` via lidar depth + known pose, compare to the
   real `t1`. Tells us (no GT needed) whether depth-based reprojection is healthy on
   this data, i.e. whether the multi-camera/lidar path to ~58–60 is open.
7. **Safe-floor submission** — per-sample motion gating with scene-disjoint validation.
   Submits the gated blend only if it beats the base method on an HONEST (scene-disjoint)
   val split; otherwise keeps the base. Cannot regress on validation.

Setup: **GPU T4 ×2**, **Internet ON**. Attach: dataset (`train/`,`test/`) + RIFE
(`train_log/` with `flownet.pkl` + `RIFE_HDv3.py`).""")

# ---- 1
md("## 1. Imports, paths, load RIFE")
code("""import os, glob, json, math, random, time, sys, shutil
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm.auto import tqdm
try: from scipy import ndimage
except Exception: ndimage = None

device='cuda' if torch.cuda.is_available() else 'cpu'; NGPU=torch.cuda.device_count()
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:',device,'| GPUs:',NGPU,'| torch',torch.__version__)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m): return os.path.dirname(m)
    return None
TRAIN_ROOT=(find_split_root('train') or '')+'/train'
_eb=find_split_root('test'); TEST_ROOT=(_eb+'/test') if _eb else None
print('TRAIN_ROOT:',TRAIN_ROOT,os.path.isdir(TRAIN_ROOT)); print('TEST_ROOT :',TEST_ROOT)

w_dir=str(Path(glob.glob('/kaggle/input/**/flownet.pkl',recursive=True)[0]).parent)   # .../train_log
# RIFE_HDv3.py does `from model.warplayer import warp` -> need the ECCV2022-RIFE repo (has model/ pkg)
_wl=glob.glob('/kaggle/input/**/model/warplayer.py',recursive=True)
if _wl:
    repo_dir=str(Path(_wl[0]).parent.parent); sys.path.insert(0,repo_dir); print('repo:',repo_dir)
else:
    print('WARNING: model/warplayer.py not found — attach the ECCV2022-RIFE repo dataset')
sys.path.insert(0,str(Path(w_dir).parent)); sys.path.insert(0,w_dir)
from train_log.RIFE_HDv3 import Model
_m=Model(); _m.load_model(w_dir,-1); _m.flownet.to(device); flownet=_m.flownet
print('RIFE loaded | %.2fM params'%(sum(p.numel() for p in flownet.parameters())/1e6))""")

# ---- 2
md("## 2. RIFE forward, DIS, helpers")
code("""def rife_forward(net,img0,img1,timestep=0.5,scale=1.0):
    sl=[16/scale,8/scale,4/scale,2/scale,1/scale]
    _,_,merged=net(torch.cat((img0,img1),1),timestep,sl); return merged[-1]
def pad64(t):
    _,_,h,w=t.shape; ph,pw=((h-1)//64+1)*64,((w-1)//64+1)*64
    return F.pad(t,(0,pw-w,0,ph-h),mode='replicate'),h,w
def read01(p): return torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2,0,1).float()/255.
def get_alpha(meta):
    ts=meta.get('timestamps_ns')
    if ts and all(k in ts for k in ('t0','t1','target')): return float((ts['target']-ts['t0'])/(ts['t1']-ts['t0']))
    return 0.5
_DIS=None
def dis_interp(i0u,i1u,a):
    global _DIS
    if _DIS is None: _DIS=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl=_DIS.calc(cv2.cvtColor(i0u,cv2.COLOR_RGB2GRAY),cv2.cvtColor(i1u,cv2.COLOR_RGB2GRAY),None)
    dx,dy=fl[...,0],fl[...,1]
    def w(img,ddx,ddy):
        H,W=img.shape[:2]; ys,xs=np.mgrid[0:H,0:W].astype(np.float32)
        return cv2.remap(img,(xs+ddx).astype(np.float32),(ys+ddy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return (1-a)*w(i0u.astype(np.float32),-a*dx,-a*dy)+a*w(i1u.astype(np.float32),(1-a)*dx,(1-a)*dy)
def _win(ch,s=11,sig=1.5,dev='cpu'):
    c=torch.arange(s,dtype=torch.float32,device=dev)-s//2; g=torch.exp(-(c**2)/(2*sig**2)); g=(g/g.sum())[None]
    return (g.t()@g).expand(ch,1,s,s).contiguous()
def ssim(a,b):
    ch=a.shape[1]; wn=_win(ch,dev=a.device); p=wn.shape[-1]//2
    m1=F.conv2d(a,wn,padding=p,groups=ch); m2=F.conv2d(b,wn,padding=p,groups=ch)
    s1=F.conv2d(a*a,wn,padding=p,groups=ch)-m1*m1; s2=F.conv2d(b*b,wn,padding=p,groups=ch)-m2*m2
    s12=F.conv2d(a*b,wn,padding=p,groups=ch)-m1*m2; c1,c2=0.01**2,0.03**2
    return (((2*m1*m2+c1)*(2*s12+c2))/((m1*m1+m2*m2+c1)*(s1+s2+c2))).mean()
def psnr01(a,b):
    a=np.clip(a,0,1).astype(np.float64); b=np.clip(b,0,1).astype(np.float64)
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(1/np.sqrt(m))
def score_of(psnr): return (min(max(psnr,10),30)-10)/20*100   # the actual contest metric""")

# ---- 3
md("## 3. Triplet dataset (crops + h-flip)")
code("""class Triplets(Dataset):
    def __init__(self,root,crop=256,train=True):
        self.dirs=sorted(str(p) for p in Path(root).iterdir() if p.is_dir()); self.crop,self.train=crop,train
    def __len__(self): return len(self.dirs)
    def _paths(self,sd):
        sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
        return (sd/'input'/'t0'/f'{cam}.jpg',sd/'input'/'t1'/f'{cam}.jpg',sd/'target'/f'{cam}.jpg')
    def __getitem__(self,i):
        p0,p1,pg=self._paths(self.dirs[i]); i0,i1,g=read01(p0),read01(p1),read01(pg)
        if self.train and self.crop:
            _,H,W=i0.shape; ch,cw=min(self.crop,H),min(self.crop,W)
            t=np.random.randint(0,H-ch+1); l=np.random.randint(0,W-cw+1)
            i0=i0[:,t:t+ch,l:l+cw]; i1=i1[:,t:t+ch,l:l+cw]; g=g[:,t:t+ch,l:l+cw]
            if np.random.rand()<0.5: i0=torch.flip(i0,[2]); i1=torch.flip(i1,[2]); g=torch.flip(g,[2])
        return i0,i1,g""")

# ---- 4
md("""## 4. Fine-tune RIFE (2× T4)
Low-LR adaptation; val each epoch on full frames; best -> `/kaggle/working/flownet_ft.pkl`.
(Empirically the gain is small ~+0.3 dB — scale/TTA below matters more. Safe to keep.)""")
code("""CFG=dict(epochs=20,batch=16,crop=256,lr=5e-5,workers=4,val_frac=0.04,ssim_w=0.1)
full=Triplets(TRAIN_ROOT,crop=CFG['crop'],train=True)
nval=max(2,int(len(full)*CFG['val_frac'])); ntr=len(full)-nval
tr,va=random_split(full,[ntr,nval],generator=torch.Generator().manual_seed(42))
va.dataset=Triplets(TRAIN_ROOT,crop=0,train=False)
tl=DataLoader(tr,batch_size=CFG['batch'],shuffle=True,num_workers=CFG['workers'],pin_memory=True,drop_last=True,persistent_workers=CFG['workers']>0)
vl=DataLoader(va,batch_size=1,shuffle=False,num_workers=2); print(f'train={ntr} val={nval}')
net=nn.DataParallel(flownet) if NGPU>1 else flownet
@torch.no_grad()
def val_psnr(use_net):
    was=flownet.training; flownet.eval(); tot=n=0
    for i0,i1,g in vl:
        i0,i1,g=i0.to(device),i1.to(device),g.to(device); x0,h,w=pad64(i0); x1,_,_=pad64(i1)
        pred=rife_forward(use_net,x0,x1)[:,:, :h,:w].clamp(0,1); mse=F.mse_loss(pred,g)
        tot+=(99.0 if mse<1e-9 else (20*torch.log10(1/torch.sqrt(mse))).item()); n+=1
    if was: flownet.train()
    return tot/n
base_psnr=val_psnr(net); print(f'pretrained RIFE val PSNR={base_psnr:.3f} dB')
opt=torch.optim.AdamW(flownet.parameters(),lr=CFG['lr'],weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CFG['epochs'])
best=base_psnr
for ep in range(CFG['epochs']):
    flownet.train(); pbar=tqdm(tl,desc=f'epoch {ep}/{CFG["epochs"]-1}')
    for i0,i1,g in pbar:
        i0,i1,g=i0.to(device,non_blocking=True),i1.to(device,non_blocking=True),g.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True); pred=rife_forward(net,i0,i1)
        loss=F.l1_loss(pred,g)+CFG['ssim_w']*(1-ssim(pred.clamp(0,1),g)); loss.backward(); opt.step()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    sch.step(); vp=val_psnr(net); print(f'epoch {ep}: val={vp:.3f} | gain={vp-base_psnr:+.3f}')
    if vp>best: best=vp; torch.save(flownet.state_dict(),'/kaggle/working/flownet_ft.pkl'); print(f'  -> best {best:.3f} saved')
print('done. best=%.3f (pretrained %.3f)'%(best,base_psnr))""")

# ---- rife_scaled
md("## 5. Scaled RIFE (+TTA) and scale/delta_s diagnostic")
code("""if os.path.exists('/kaggle/working/flownet_ft.pkl'):
    flownet.load_state_dict(torch.load('/kaggle/working/flownet_ft.pkl')); print('loaded fine-tuned weights')
flownet.eval()
@torch.no_grad()
def rife_scaled(i0u,i1u,scale=0.5,tta=True):   # uint8 HxWx3 -> float HxWx3 0..255
    m=64*math.ceil(1.0/scale)
    def padm(t):
        _,_,h,w=t.shape; ph=((h-1)//m+1)*m; pw=((w-1)//m+1)*m
        return F.pad(t,(0,pw-w,0,ph-h),mode='replicate'),h,w
    def run(a0,a1):
        i0=torch.from_numpy(np.ascontiguousarray(a0)).permute(2,0,1)[None].float().to(device)/255.
        i1=torch.from_numpy(np.ascontiguousarray(a1)).permute(2,0,1)[None].float().to(device)/255.
        x0,h,w=padm(i0); x1,_,_=padm(i1)
        return rife_forward(flownet,x0,x1,0.5,scale)[:,:, :h,:w].clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255
    p=run(i0u,i1u)
    if tta: pf=run(i0u[:,::-1],i1u[:,::-1])[:,::-1]; p=0.5*(p+pf)
    return p

from collections import defaultdict
vd=sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir()); random.seed(1); random.shuffle(vd); vd=vd[:120]
res=defaultdict(lambda: defaultdict(float)); cnt=defaultdict(int)
for sd in tqdm(vd):
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']; d=meta.get('delta_s',1.0)
    gt=np.array(Image.open(sd/'target'/f'{cam}.jpg').convert('RGB')).astype(np.float32)/255.
    i0u=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB')); i1u=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    cnt[d]+=1
    for sc in [1.0,0.5,0.25]:
        res[d][f's{sc}']+=psnr01(rife_scaled(i0u,i1u,sc,False)/255.,gt)
        res[d][f's{sc}+tta']+=psnr01(rife_scaled(i0u,i1u,sc,True)/255.,gt)
for d in sorted(res):
    print(f'delta_s={d} (n={cnt[d]}):')
    for k,v in sorted(res[d].items(),key=lambda kv:-kv[1]): print(f'  {k:10s} {v/cnt[d]:.3f} dB')""")

# ---- 6 geometry self-check
md("""## 6. GEOMETRY self-check (no GT needed)
Warp `t0 -> t1`'s view via lidar depth + the known t0->t1 pose change, compare to the
real `t1`. Per camera, pick pinhole vs fisheye by which scores higher.
**Read:** t0->t1 PSNR > ~20 ⇒ geometry healthy ⇒ the lidar/multi-camera path (to ~58-60)
is open. < ~15 ⇒ a projection/axis bug to fix (send me the meta dump).""")
code("""CAMS=['front','left_fwd','left_bwd','right_fwd','right_bwd','rear']
def _K(i): return np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1]],np.float64)
def _dc(i): return np.asarray(i.get('distortion_coeffs') or [],np.float64).reshape(-1)
def _proj(Xc,i,f):
    K,d=_K(i),_dc(i); z=np.zeros(3)
    uv,_=(cv2.fisheye.projectPoints(Xc.reshape(-1,1,3).astype(np.float64),z,z,K,d[:4] if d.size>=4 else np.zeros(4)) if f
          else cv2.projectPoints(Xc.astype(np.float64),z,z,K,d if d.size else None))
    return uv.reshape(-1,2)
def _rays(uv,i,f):
    K,d=_K(i),_dc(i); p=uv.reshape(-1,1,2).astype(np.float64)
    n=(cv2.fisheye.undistortPoints(p,K,d[:4] if d.size>=4 else np.zeros(4)) if f else cv2.undistortPoints(p,K,d if d.size else None)).reshape(-1,2)
    r=np.concatenate([n,np.ones((n.shape[0],1))],1); return r/np.linalg.norm(r,axis=1,keepdims=True)
def _w2c(Xw,c2w): return (Xw-c2w[:3,3][None,:])@c2w[:3,:3]
def _depth(lid,c2w,i,W,H,f,splat=2):
    Xc=_w2c(lid.astype(np.float64),c2w); fr=Xc[:,2]>1e-6; uv=_proj(Xc[fr],i,f); z=Xc[fr,2]
    u=np.round(uv[:,0]).astype(np.int64); v=np.round(uv[:,1]).astype(np.int64)
    m=(u>=0)&(u<W)&(v>=0)&(v<H); u,v,z=u[m],v[m],z[m]
    dep=np.full((H,W),np.inf,np.float32); o=np.argsort(-z); dep[v[o],u[o]]=z[o].astype(np.float32)
    if splat>1:
        k=np.ones((splat,splat),np.uint8); fin=np.isfinite(dep)
        df=cv2.erode(np.where(fin,dep,1e9).astype(np.float32),k); g=(df<1e8)&(~fin); dep[g]=df[g]
    return dep,np.isfinite(dep)
def _fill(dep,val):
    h=~val
    if h.all() or ndimage is None: dep=dep.copy(); dep[h]=30.0 if h.all() else np.median(dep[val]); return dep
    idx=ndimage.distance_transform_edt(h,return_distances=False,return_indices=True); return dep[tuple(idx)].astype(np.float32)
def backward_warp(src,sc2w,si,sf,dep,tc2w,ti,tf):
    H,W=dep.shape; vs,us=np.mgrid[0:H,0:W]; uv=np.stack([us.ravel(),vs.ravel()],1).astype(np.float64)
    r=_rays(uv,ti,tf); z=dep.ravel().astype(np.float64); Xc=r*(z/np.clip(r[:,2],1e-6,None))[:,None]
    Xw=Xc@tc2w[:3,:3].T+tc2w[:3,3][None,:]; Xs=_w2c(Xw,sc2w); fr=Xs[:,2]>1e-6
    uvs=_proj(Xs,si,sf); Hs,Ws=src.shape[:2]
    mx=uvs[:,0].reshape(H,W).astype(np.float32); my=uvs[:,1].reshape(H,W).astype(np.float32)
    wp=cv2.remap(src.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    inb=fr.reshape(H,W)&(mx>=0)&(mx<Ws)&(my>=0)&(my<Hs); return wp,inb
def _r01(p): return np.array(Image.open(p).convert('RGB')).astype(np.float32)/255.

# print one real meta first
sd0=sorted(p for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())[0]; mj=json.loads((sd0/'meta.json').read_text())
print('meta keys:',list(mj.keys()),'| has timestamps_ns:', 'timestamps_ns' in mj)
for c,intr in mj['intrinsics'].items(): print(f'  {c:10s} model={intr.get("distortion_model")!r} coeffs={intr.get("distortion_coeffs")}')

probe=sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir()); random.seed(7); random.shuffle(probe); probe=probe[:25]
CAM_FISH={}
for cam in CAMS:
    pin,fis=[],[]
    for sdp in probe:
        sd=Path(sdp); meta=json.loads((sd/'meta.json').read_text())
        if cam not in meta['intrinsics'] or not (sd/'input'/'t0'/f'{cam}.jpg').exists(): continue
        try:
            intr=meta['intrinsics'][cam]; c0=np.array(meta['poses_c2w']['t0'][cam],np.float64); c1=np.array(meta['poses_c2w']['t1'][cam],np.float64)
            lid=np.load(sd/'input'/'lidar.npz')['xyz']; i0=_r01(sd/'input'/'t0'/f'{cam}.jpg'); i1=_r01(sd/'input'/'t1'/f'{cam}.jpg'); H,W=i1.shape[:2]
            for f,acc in ((False,pin),(True,fis)):
                dep,val=_depth(lid,c1,intr,W,H,f); dep=_fill(dep,val); wp,mk=backward_warp(i0,c0,intr,f,dep,c1,intr,f)
                acc.append(psnr01(wp[mk],i1[mk]) if mk.any() else -1)
        except Exception: pass
    if not pin: print(f'{cam}: no data'); continue
    mp,mf=np.mean(pin),np.mean(fis); CAM_FISH[cam]=mf>mp
    print(f'{cam:10s} pinhole={mp:5.2f}  fisheye={mf:5.2f}  -> {"FISHEYE" if CAM_FISH[cam] else "pinhole"}')
print('\\nCAM_FISH =',CAM_FISH)""")

# ---- 7 safe-floor submission
md("""## 7. SAFE-FLOOR submission (scene-disjoint validation)
Base method = `0.3·DIS + 0.7·RIFE(scale0.5,TTA)`. For high-motion samples (where VFI
ghosts) we blend toward the safe mean `0.5(t0+t1)`. Threshold & weight are picked on a
**scene-disjoint** val split optimizing the ACTUAL contest score. Submits the safe-floor
only if it beats the base on that honest split — else keeps the base (no regression).""")
code("""def base_pred(i0u,i1u,a): return 0.3*dis_interp(i0u,i1u,a)+0.7*rife_scaled(i0u,i1u,0.5,True)
def motion_score(i0u,i1u):
    fl=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(
        cv2.cvtColor(i0u,cv2.COLOR_RGB2GRAY),cv2.cvtColor(i1u,cv2.COLOR_RGB2GRAY),None)
    return float(np.median(np.sqrt((fl**2).sum(-1))))
def load_uu(sd):
    meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    i0=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB')); i1=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    return meta,cam,i0,i1

# scene-disjoint val split
scenes={}
for p in sorted(Path(TRAIN_ROOT).iterdir()):
    if not p.is_dir(): continue
    s=json.loads((p/'meta.json').read_text()).get('scene','?'); scenes.setdefault(s,[]).append(p)
scl=sorted(scenes); random.seed(11); random.shuffle(scl); val_sc=set(scl[:max(1,len(scl)//5)])
val=[p for s in val_sc for p in scenes[s]]; random.shuffle(val); val=val[:120]
print(f'{len(scl)} scenes | honest val on {len(val)} samples from held-out scenes')

rows=[]
for sd in tqdm(val):
    meta,cam,i0u,i1u=load_uu(sd); a=get_alpha(meta)
    gt=np.array(Image.open(sd/'target'/f'{cam}.jpg').convert('RGB')).astype(np.float32)/255.
    b=base_pred(i0u,i1u,a)/255.; mn=0.5*(i0u.astype(np.float32)+i1u.astype(np.float32))/255.
    rows.append((motion_score(i0u,i1u),b,mn,gt))
mot=np.array([r[0] for r in rows])
base_score=np.mean([score_of(psnr01(b,gt)) for _,b,_,gt in rows]); print(f'CURRENT base val score={base_score:.3f}')
best=(base_score,None,None)
for T in np.percentile(mot,[50,60,70,80,90]):
    for w in [0.3,0.5,0.7,1.0]:
        s=np.mean([score_of(psnr01(((1-w)*b+w*mn) if m>=T else b,gt)) for m,b,mn,gt in rows])
        if s>best[0]: best=(s,T,w)
print(f'BEST safe-floor score={best[0]:.3f} (T={best[1]}, w={best[2]})')
USE_SAFE=best[1] is not None and best[0]>base_score+0.05; T_,w_=best[1],best[2]
print('-> submit:', 'SAFE-FLOOR' if USE_SAFE else 'CURRENT base (no reliable gain)')

out='/kaggle/working/submission'
if os.path.exists(out): shutil.rmtree(out)
os.makedirs(out,exist_ok=True)
for sd in tqdm(sorted(p for p in Path(TEST_ROOT).iterdir() if p.is_dir())):
    meta,cam,i0u,i1u=load_uu(sd); a=get_alpha(meta); b=base_pred(i0u,i1u,a)
    if USE_SAFE and motion_score(i0u,i1u)>=T_:
        mn=0.5*(i0u.astype(np.float32)+i1u.astype(np.float32)); b=(1-w_)*b+w_*mn
    od=Path(out)/sd.name; od.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.clip(b,0,255).round().astype(np.uint8)).save(od/'pred.jpg',quality=100,subsampling=0)
shutil.make_archive('/kaggle/working/submission','zip',out)
print('submission ready | mode:', 'SAFE-FLOOR' if USE_SAFE else 'CURRENT')""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
for p in OUT_PATHS:
    with open(p,"w") as f: json.dump(nb,f,indent=1)
    print("wrote",p,"with",len(cells),"cells")
