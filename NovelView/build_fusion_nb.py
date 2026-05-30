"""Generate fusion-refiner.ipynb — learned fusion of RIFE + lidar-geometry for Task B.

Why this can work where the old refiner failed: the old one had only flow-interp input
=> optimum was identity. Now the input carries a GEOMETRY channel (geo, trust) that holds
information the RIFE base lacks, so a small net has real signal: learn the per-pixel
blend of VFI vs geometry (better than the hand-tuned `cap`).

Anti-overfit (the RIFE-finetune trap): small net, predicts a RESIDUAL over the current
hybrid base (starts == LB 49), SCENE-DISJOINT early stopping, augmentation. Submits the
refiner only if it beats the base on held-out scenes.

Pipeline: precompute (rife, geo, trust) per train sample -> cache -> train small UNet ->
validate scene-disjoint -> submit winner.

Attach: competition dataset + RIFE (repo + train_log). GPU T4 x2.
Run: python3 build_fusion_nb.py
"""
import json
OUT_PATHS=["/Users/flexonafft/Downloads/fusion-refiner.ipynb",
           "/Users/flexonafft/MLChallenge2026/NovelView/fusion-refiner.ipynb"]
cells=[]
def _l(s): return s.splitlines(keepends=True)
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":_l(s)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_l(s)})

md("""# Task B — learned fusion refiner (RIFE + lidar geometry)

Small UNet learns the per-pixel blend of **RIFE** and **lidar-geometry warp**, predicting
a **residual over the current hybrid** (so it starts at LB≈49 and can only improve).
Anti-overfit: tiny net · residual · **scene-disjoint** early stopping · augmentation.
Submits the refiner only if it beats the hybrid base on held-out scenes.

Attach: competition dataset + RIFE (repo + train_log). **GPU T4 ×2**.""")

md("## 1. Imports, paths, PRETRAINED RIFE")
code("""import os, glob, json, math, random, sys, shutil, time
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
try: from scipy import ndimage
except Exception: ndimage=None
device='cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:',device)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}',recursive=True)):
        if os.path.isdir(m): return os.path.dirname(m)
    return None
TRAIN_ROOT=(find_split_root('train') or '')+'/train'
_eb=find_split_root('test'); TEST_ROOT=(_eb+'/test') if _eb else None
print('TRAIN_ROOT:',TRAIN_ROOT,os.path.isdir(TRAIN_ROOT)); print('TEST_ROOT :',TEST_ROOT)

w_dir=str(Path(glob.glob('/kaggle/input/**/flownet.pkl',recursive=True)[0]).parent)
_wl=glob.glob('/kaggle/input/**/model/warplayer.py',recursive=True)
if _wl: sys.path.insert(0,str(Path(_wl[0]).parent.parent))
sys.path.insert(0,str(Path(w_dir).parent)); sys.path.insert(0,w_dir)
from train_log.RIFE_HDv3 import Model
_m=Model(); _m.load_model(w_dir,-1); _m.flownet.to(device).eval(); flownet=_m.flownet
print('PRETRAINED RIFE loaded')""")

md("## 2. RIFE, geometry, base hybrid (CAP from hybrid-final)")
code("""CAP=0.9
@torch.no_grad()
def rife_scaled(i0u,i1u,scale=0.5,tta=True):
    m=64*math.ceil(1.0/scale)
    def padm(t):
        _,_,h,w=t.shape; ph=((h-1)//m+1)*m; pw=((w-1)//m+1)*m
        return F.pad(t,(0,pw-w,0,ph-h),mode='replicate'),h,w
    def run(a0,a1):
        i0=torch.from_numpy(np.ascontiguousarray(a0)).permute(2,0,1)[None].float().to(device)/255.
        i1=torch.from_numpy(np.ascontiguousarray(a1)).permute(2,0,1)[None].float().to(device)/255.
        x0,h,w=padm(i0); x1,_,_=padm(i1); sl=[16/scale,8/scale,4/scale,2/scale,1/scale]
        _,_,merged=flownet(torch.cat((x0,x1),1),0.5,sl)
        return merged[-1][:,:, :h,:w].clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    p=run(i0u,i1u)
    if tta: pf=run(i0u[:,::-1],i1u[:,::-1])[:,::-1].copy(); p=0.5*(p+pf)
    return p
def _K(i): return np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1]],np.float64)
def _proj(Xc,i):
    z=np.zeros(3); uv,_=cv2.projectPoints(Xc.astype(np.float64),z,z,_K(i),None); return uv.reshape(-1,2)
def _rays(uv,i):
    n=cv2.undistortPoints(uv.reshape(-1,1,2).astype(np.float64),_K(i),None).reshape(-1,2)
    r=np.concatenate([n,np.ones((n.shape[0],1))],1); return r/np.linalg.norm(r,axis=1,keepdims=True)
def _w2c(Xw,c2w): return (Xw-c2w[:3,3][None,:])@c2w[:3,:3]
def _depth(lid,c2w,i,W,H,splat=2):
    Xc=_w2c(lid.astype(np.float64),c2w); fr=Xc[:,2]>1e-6; uv=_proj(Xc[fr],i); z=Xc[fr,2]
    u=np.round(uv[:,0]).astype(np.int64); v=np.round(uv[:,1]).astype(np.int64)
    m=(u>=0)&(u<W)&(v>=0)&(v<H); u,v,z=u[m],v[m],z[m]
    dep=np.full((H,W),np.inf,np.float32); o=np.argsort(-z); dep[v[o],u[o]]=z[o].astype(np.float32)
    if splat>1:
        k=np.ones((splat,splat),np.uint8); fin=np.isfinite(dep)
        df=cv2.erode(np.where(fin,dep,1e9).astype(np.float32),k); g=(df<1e8)&(~fin); dep[g]=df[g]
    return dep,np.isfinite(dep)
def _fill(dep,val):
    h=~val
    if h.all() or ndimage is None:
        dep=dep.copy(); dep[h]=30.0 if h.all() else np.median(dep[val]); return dep
    idx=ndimage.distance_transform_edt(h,return_distances=False,return_indices=True); return dep[tuple(idx)].astype(np.float32)
def _bw(src,sc2w,si,dep,tc2w,ti):
    H,W=dep.shape; vs,us=np.mgrid[0:H,0:W]; uv=np.stack([us.ravel(),vs.ravel()],1).astype(np.float64)
    r=_rays(uv,ti); z=dep.ravel().astype(np.float64); Xc=r*(z/np.clip(r[:,2],1e-6,None))[:,None]
    Xw=Xc@tc2w[:3,:3].T+tc2w[:3,3][None,:]; Xs=_w2c(Xw,sc2w); fr=Xs[:,2]>1e-6
    uvs=_proj(Xs,si); Hs,Ws=src.shape[:2]
    mx=uvs[:,0].reshape(H,W).astype(np.float32); my=uvs[:,1].reshape(H,W).astype(np.float32)
    wp=cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return wp,(fr.reshape(H,W)&(mx>=0)&(mx<Ws)&(my>=0)&(my<Hs))

def components(sd):
    \"\"\"-> rife(0..1 HxWx3), geo(HxWx3), trust(HxW), i0,i1 (0..1).\"\"\"
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']; intr=meta['intrinsics'][cam]
    ct=np.array(meta['poses_c2w']['target'][cam],np.float64)
    c0=np.array(meta['poses_c2w']['t0'][cam],np.float64); c1=np.array(meta['poses_c2w']['t1'][cam],np.float64)
    lid=np.load(sd/'input'/'lidar.npz')['xyz']
    i0u=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB')); i1u=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    f0,f1=i0u.astype(np.float32)/255.,i1u.astype(np.float32)/255.; H,W=f0.shape[:2]
    dep,val=_depth(lid,ct,intr,W,H); dep=_fill(dep,val)
    w0,m0=_bw(f0,c0,intr,dep,ct,intr); w1,m1=_bw(f1,c1,intr,dep,ct,intr)
    cnt=(m0.astype(np.float32)+m1.astype(np.float32))[...,None]
    geo=np.where(cnt>0,(m0[...,None]*w0+m1[...,None]*w1)/np.clip(cnt,1,None),0.).astype(np.float32)
    agree=np.exp(-np.abs(w0-w1).mean(-1)/0.06); trust=((m0&m1).astype(np.float32)*agree).astype(np.float32)
    rife=rife_scaled(i0u,i1u).astype(np.float32)
    return rife,geo,trust,f0,f1
def base_blend(rife,geo,trust):
    t=np.clip(cv2.GaussianBlur(trust,(0,0),3.0),0,CAP)[...,None]
    return (t*geo+(1-t)*rife).astype(np.float32)
def psnr01(a,b):
    a=np.clip(a,0,1).astype(np.float64); b=np.clip(b,0,1).astype(np.float64)
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(1/np.sqrt(m))
def score_of(p): return (min(max(p,10),30)-10)/20*100""")

md("""## 3. Precompute & cache (rife, geo, trust, gt) for train
One-time (~20-40 min). Caches float16 npz so training epochs are fast. Lower `N_TRAIN`
if disk is tight.""")
code("""N_TRAIN=900
CACHE='/kaggle/working/cache'; os.makedirs(CACHE,exist_ok=True)
alld=[p for p in Path(TRAIN_ROOT).iterdir() if p.is_dir()]
random.seed(11); random.shuffle(alld); use=alld[:N_TRAIN]
meta_scene={}
for sd in tqdm(use, desc='precompute'):
    cf=Path(CACHE)/f'{sd.name}.npz'
    sc=json.loads((sd/'meta.json').read_text()).get('scene','?'); meta_scene[sd.name]=sc
    if cf.exists(): continue
    try:
        cam=json.loads((sd/'meta.json').read_text())['target_camera']
        rife,geo,trust,f0,f1=components(sd)
        gt=np.array(Image.open(sd/'target'/f'{cam}.jpg').convert('RGB')).astype(np.float32)/255.
        np.savez_compressed(cf, rife=rife.astype(np.float16),geo=geo.astype(np.float16),
                            trust=trust.astype(np.float16),t0=f0.astype(np.float16),t1=f1.astype(np.float16),
                            gt=gt.astype(np.float16))
    except Exception as e: print('skip',sd.name,e)
names=[p.stem for p in Path(CACHE).glob('*.npz')]
print('cached:',len(names))""")

md("## 4. Dataset, model (residual over base, zero-init)")
code("""class FuseDS(Dataset):
    def __init__(self,names,crop=256,train=True):
        self.names=names; self.crop=crop; self.train=train
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        d=np.load(Path(CACHE)/f'{self.names[i]}.npz')
        rife=d['rife'].astype(np.float32); geo=d['geo'].astype(np.float32); trust=d['trust'].astype(np.float32)
        t0=d['t0'].astype(np.float32); t1=d['t1'].astype(np.float32); gt=d['gt'].astype(np.float32)
        tt=np.clip(cv2.GaussianBlur(trust,(0,0),3.0),0,CAP)[...,None]
        base=tt*geo+(1-tt)*rife
        x=np.concatenate([t0,t1,rife,geo,trust[...,None]],2)   # H,W,13
        H,W=gt.shape[:2]
        if self.train and self.crop:
            ch,cw=min(self.crop,H),min(self.crop,W); t=np.random.randint(0,H-ch+1); l=np.random.randint(0,W-cw+1)
            sl=(slice(t,t+ch),slice(l,l+cw)); x,base,gt=x[sl],base[sl],gt[sl]
            if np.random.rand()<0.5: x=x[:,::-1].copy(); base=base[:,::-1].copy(); gt=gt[:,::-1].copy()
            g=np.random.uniform(0.9,1.1); b=np.random.uniform(-0.04,0.04)   # shared exposure jitter
            x[...,:12]=np.clip(x[...,:12]*g+b,0,1); base=np.clip(base*g+b,0,1); gt=np.clip(gt*g+b,0,1)
        def tt_(a): return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
        return tt_(x),tt_(base),tt_(gt)

def cb(ci,co):
    return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(8,co),nn.SiLU(),
                         nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(8,co),nn.SiLU())
class Refiner(nn.Module):
    def __init__(self,cin=13,base=24,depth=3):
        super().__init__(); ch=[base*(2**k) for k in range(depth+1)]
        self.inc=cb(cin,ch[0]); self.pool=nn.MaxPool2d(2)
        self.down=nn.ModuleList([cb(ch[k],ch[k+1]) for k in range(depth)])
        self.up=nn.ModuleList([nn.ConvTranspose2d(ch[k+1],ch[k],2,stride=2) for k in range(depth)])
        self.uc=nn.ModuleList([cb(ch[k+1],ch[k]) for k in range(depth)])
        self.outc=nn.Conv2d(ch[0],3,1); nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)
        self.depth=depth
    def forward(self,x,base):
        h=self.inc(x); sk=[h]
        for d in range(self.depth): h=self.down[d](self.pool(h)); sk.append(h)
        for d in reversed(range(self.depth)):
            h=self.up[d](h); s=sk[d]
            dh,dw=s.shape[-2]-h.shape[-2],s.shape[-1]-h.shape[-1]
            if dh or dw: h=F.pad(h,[0,dw,0,dh])
            h=self.uc[d](torch.cat([h,s],1))
        return (base+self.outc(h)).clamp(0,1)
print('model+dataset ready')""")

md("## 5. Train with scene-disjoint early stopping")
code("""scset=sorted(set(meta_scene[n] for n in names)); random.seed(7); random.shuffle(scset)
val_sc=set(scset[:max(1,len(scset)//6)])
val_names=[n for n in names if meta_scene[n] in val_sc]; tr_names=[n for n in names if meta_scene[n] not in val_sc]
random.shuffle(val_names); val_names=val_names[:120]      # cap val for speed (full-frame fwd each epoch)
print(f'scenes={len(scset)} | train={len(tr_names)} val={len(val_names)}')

# RAM-preload the cache: one decompress instead of per-iteration disk reads.
# float16 (~16 GB for 900); workers share it copy-on-write. Falls back to disk on MemoryError.
RAM={}
try:
    for n in tqdm(names, desc='cache->RAM'):
        d=np.load(Path(CACHE)/f'{n}.npz'); RAM[n]={k:d[k] for k in ('rife','geo','trust','t0','t1','gt')}
    print(f'RAM cache: {len(RAM)} samples')
except MemoryError:
    RAM={}; print('MemoryError -> falling back to disk (lower N_TRAIN to use RAM)')

class RAMDS(Dataset):
    def __init__(self,names,crop=256,train=True): self.names=names; self.crop=crop; self.train=train
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        nm=self.names[i]; d=RAM[nm] if nm in RAM else np.load(Path(CACHE)/f'{nm}.npz')
        rife=d['rife'].astype(np.float32); geo=d['geo'].astype(np.float32); trust=d['trust'].astype(np.float32)
        t0=d['t0'].astype(np.float32); t1=d['t1'].astype(np.float32); gt=d['gt'].astype(np.float32)
        tt=np.clip(cv2.GaussianBlur(trust,(0,0),3.0),0,CAP)[...,None]; base=tt*geo+(1-tt)*rife
        x=np.concatenate([t0,t1,rife,geo,trust[...,None]],2); H,W=gt.shape[:2]
        if self.train and self.crop:
            ch,cw=min(self.crop,H),min(self.crop,W); t=np.random.randint(0,H-ch+1); l=np.random.randint(0,W-cw+1)
            sl=(slice(t,t+ch),slice(l,l+cw)); x,base,gt=x[sl],base[sl],gt[sl]
            if np.random.rand()<0.5: x=x[:,::-1].copy(); base=base[:,::-1].copy(); gt=gt[:,::-1].copy()
            g=np.random.uniform(0.9,1.1); b=np.random.uniform(-0.04,0.04)
            x[...,:12]=np.clip(x[...,:12]*g+b,0,1); base=np.clip(base*g+b,0,1); gt=np.clip(gt*g+b,0,1)
        def tt_(a): return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
        return tt_(x),tt_(base),tt_(gt)

NG=torch.cuda.device_count(); print('GPUs:',NG)
tl=DataLoader(RAMDS(tr_names,256,True),batch_size=16*max(1,NG),shuffle=True,num_workers=8,pin_memory=True,drop_last=True,persistent_workers=True)
vl=DataLoader(RAMDS(val_names,0,False),batch_size=1,shuffle=False,num_workers=2)

def ssim(a,b):
    C=a.shape[1]; k=11; s=1.5; c=torch.arange(k,device=a.device,dtype=torch.float32)-k//2
    g=torch.exp(-(c**2)/(2*s*s)); g=(g/g.sum())[None]; win=(g.t()@g).expand(C,1,k,k).contiguous(); p=k//2
    def cv(z): return F.conv2d(z,win,padding=p,groups=C)
    m1,m2=cv(a),cv(b); s1=cv(a*a)-m1*m1; s2=cv(b*b)-m2*m2; s12=cv(a*b)-m1*m2; c1,c2=0.01**2,0.03**2
    return (((2*m1*m2+c1)*(2*s12+c2))/((m1*m1+m2*m2+c1)*(s1+s2+c2))).mean()

@torch.no_grad()
def val_scores(net):
    net.eval(); ref=[]; base=[]
    for x,b,g in vl:
        x,b,g=x.to(device),b.to(device),g.to(device)
        _,_,h,w=x.shape; ph=((h-1)//8+1)*8; pw=((w-1)//8+1)*8
        xp=F.pad(x,(0,pw-w,0,ph-h)); bp=F.pad(b,(0,pw-w,0,ph-h))
        pr=net(xp,bp)[:,:, :h,:w]
        mse=F.mse_loss(pr,g).item(); mseb=F.mse_loss(b,g).item()
        ref.append(score_of(20*math.log10(1/math.sqrt(max(mse,1e-10)))))
        base.append(score_of(20*math.log10(1/math.sqrt(max(mseb,1e-10)))))
    net.train(); return float(np.mean(ref)),float(np.mean(base))

core=Refiner().to(device)
net=nn.DataParallel(core) if NG>1 else core      # split batch across both T4s
opt=torch.optim.AdamW(core.parameters(),lr=3e-4,weight_decay=1e-4)
EPOCHS=40; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
scaler=torch.cuda.amp.GradScaler()                       # mixed precision -> faster on T4
r0,b0=val_scores(net); print(f'init: base={b0:.3f} refiner={r0:.3f} (should match)')
best=b0; bad=0
for ep in range(EPOCHS):
    net.train(); pb=tqdm(tl,desc=f'ep{ep}')
    for x,b,g in pb:
        x,b,g=x.to(device,non_blocking=True),b.to(device,non_blocking=True),g.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pr=net(x,b); loss=F.l1_loss(pr,g)+0.1*(1-ssim(pr,g))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); pb.set_postfix(l=f'{loss.item():.4f}')
    sch.step(); rs,bs=val_scores(net); print(f'ep{ep}: refiner={rs:.3f} base={bs:.3f} gain={rs-bs:+.3f}')
    if rs>best+1e-3:
        best=rs; bad=0; torch.save(core.state_dict(),'/kaggle/working/refiner.pt'); print(f'  -> best {best:.3f} saved')
    else:
        bad+=1
        if bad>=6: print('early stop'); break
print(f'done. best refiner val={best:.3f} | base={b0:.3f} | gain={best-b0:+.3f}')""")

md("## 6. Submission (refiner if it beat base, else base; q100)")
code("""USE_REF=os.path.exists('/kaggle/working/refiner.pt') and best>b0+0.05
if USE_REF:
    core.load_state_dict(torch.load('/kaggle/working/refiner.pt')); core.eval(); print('using REFINER')
else: print('refiner did not beat base -> submitting BASE hybrid')

@torch.no_grad()
def predict(sd):
    rife,geo,trust,f0,f1=components(sd); base=base_blend(rife,geo,trust)
    if not USE_REF: return base
    x=np.concatenate([f0,f1,rife,geo,trust[...,None]],2)
    xt=torch.from_numpy(x.transpose(2,0,1))[None].float().to(device)
    bt=torch.from_numpy(base.transpose(2,0,1))[None].float().to(device)
    _,_,h,w=xt.shape; ph=((h-1)//8+1)*8; pw=((w-1)//8+1)*8
    xp=F.pad(xt,(0,pw-w,0,ph-h)); bp=F.pad(bt,(0,pw-w,0,ph-h))
    return core(xp,bp)[:,:, :h,:w][0].permute(1,2,0).cpu().numpy()

out='/kaggle/working/submission'
if os.path.exists(out): shutil.rmtree(out)
os.makedirs(out,exist_ok=True)
for sd in tqdm(sorted(p for p in Path(TEST_ROOT).iterdir() if p.is_dir())):
    try: pred=predict(sd)
    except Exception:
        meta=json.loads((Path(sd)/'meta.json').read_text()); cam=meta['target_camera']
        i0=np.array(Image.open(Path(sd)/'input'/'t0'/f'{cam}.jpg').convert('RGB')); i1=np.array(Image.open(Path(sd)/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
        pred=rife_scaled(i0,i1)
    img=(np.clip(pred,0,1)*255).round().astype(np.uint8)
    od=Path(out)/Path(sd).name; od.mkdir(parents=True,exist_ok=True)
    Image.fromarray(img).save(od/'pred.jpg',quality=100,subsampling=0)
shutil.make_archive('/kaggle/working/submission','zip',out)
print('submission ready | mode:', 'REFINER' if USE_REF else 'BASE')""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
for p in OUT_PATHS:
    with open(p,"w") as f: json.dump(nb,f,indent=1)
    print("wrote",p,"with",len(cells),"cells")
