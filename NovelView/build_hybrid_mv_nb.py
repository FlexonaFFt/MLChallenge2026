"""Generate hybrid-mv.ipynb — multi-camera geometry + pretrained RIFE.

Builds on hybrid-final (which lifted LB to 49 with target-camera-only geometry).
Now warp ALL 12 source frames (6 cameras x {t0,t1}) into the target view via lidar
depth at the target pose, and fuse them with a robust MEDIAN (kills moving-object
outliers). Trust = where >=2 sources agree (low spread). Blend over pretrained RIFE.
Honest scene-disjoint validation by the real contest score; submit the winner.

Attach: competition dataset + RIFE (repo + train_log). GPU T4 x2.
Run: python3 build_hybrid_mv_nb.py
"""
import json
OUT_PATHS=["/Users/flexonafft/Downloads/hybrid-mv.ipynb",
           "/Users/flexonafft/MLChallenge2026/NovelView/hybrid-mv.ipynb"]
cells=[]
def _l(s): return s.splitlines(keepends=True)
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":_l(s)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_l(s)})

md("""# Task B — multi-camera geometry + RIFE (hybrid-mv)

Target-camera-only geometry already lifted LB to 49. Now use **all 12 source frames**
(6 cameras × {t0,t1}): warp each into the target view via lidar depth at the target pose,
fuse with a robust **median** (drops moving-object outliers), trust where ≥2 sources agree.
Blend over **pretrained** RIFE. Honest scene-disjoint validation; submit the winner.

Attach: competition dataset + RIFE (repo + train_log). **GPU T4 ×2**.""")

md("## 1. Imports, paths, load PRETRAINED RIFE")
code("""import os, glob, json, math, random, sys, shutil
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
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

md("## 2. RIFE + metric helpers")
code("""@torch.no_grad()
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
def psnr01(a,b):
    a=np.clip(a,0,1).astype(np.float64); b=np.clip(b,0,1).astype(np.float64)
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(1/np.sqrt(m))
def score_of(p): return (min(max(p,10),30)-10)/20*100""")

md("## 3. Multi-camera geometry")
code("""CAMS=['front','left_fwd','left_bwd','right_fwd','right_bwd','rear']
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
def _bwarp(src,sc2w,si,Xw_flat,H,W):    # src 0..1; world pts (HW,3) -> warp,mask
    Xs=_w2c(Xw_flat,sc2w); fr=Xs[:,2]>1e-6; uvs=_proj(Xs,si); Hs,Ws=src.shape[:2]
    mx=uvs[:,0].reshape(H,W).astype(np.float32); my=uvs[:,1].reshape(H,W).astype(np.float32)
    wp=cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    inb=fr.reshape(H,W)&(mx>=0)&(mx<Ws)&(my>=0)&(my<Hs); return wp,inb

def mv_geo(sd, agree_sig=0.06):
    \"\"\"warp all 12 sources into target; robust-median fuse; return geo(0..1), trust(0..1), rife(0..1).\"\"\"
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    ti=meta['intrinsics'][cam]; ct=np.array(meta['poses_c2w']['target'][cam],np.float64)
    lid=np.load(sd/'input'/'lidar.npz')['xyz']
    i0t=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB'))
    i1t=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    H,W=i0t.shape[:2]
    dep,val=_depth(lid,ct,ti,W,H); dep=_fill(dep,val)
    # target pixels -> world points (once)
    vs,us=np.mgrid[0:H,0:W]; uv=np.stack([us.ravel(),vs.ravel()],1).astype(np.float64)
    r=_rays(uv,ti); z=dep.ravel().astype(np.float64); Xc=r*(z/np.clip(r[:,2],1e-6,None))[:,None]
    Xw=Xc@ct[:3,:3].T+ct[:3,3][None,:]
    warps=[]; masks=[]
    for c in CAMS:
        if c not in meta['intrinsics']: continue
        si=meta['intrinsics'][c]
        for tname in ('t0','t1'):
            ip=sd/'input'/tname/f'{c}.jpg'
            if not ip.exists() or c not in meta['poses_c2w'][tname]: continue
            sc2w=np.array(meta['poses_c2w'][tname][c],np.float64)
            src=np.array(Image.open(ip).convert('RGB')).astype(np.float32)/255.
            wp,mk=_bwarp(src,sc2w,si,Xw,H,W)
            warps.append(np.where(mk[...,None],wp,np.nan).astype(np.float32)); masks.append(mk)
    S=np.stack(warps,0)                      # (K,H,W,3) nan where invalid
    cnt=np.stack(masks,0).sum(0)             # (H,W)
    geo=np.nanmedian(S,0); geo=np.where(np.isfinite(geo),geo,0.).astype(np.float32)
    with np.errstate(invalid='ignore'):
        spread=np.nanstd(S,0).mean(-1)       # (H,W) color spread across sources
    spread=np.where(np.isfinite(spread),spread,1.0)
    trust=(cnt>=2).astype(np.float32)*np.exp(-spread/agree_sig)
    rife=rife_scaled(i0t,i1t)
    return geo,trust.astype(np.float32),rife""")

md("""## 4. Honest scene-disjoint sweep (cap; cap=0 ⇒ RIFE-only)""")
code("""alld=[p for p in Path(TRAIN_ROOT).iterdir() if p.is_dir()]
random.seed(11); random.shuffle(alld); probe=alld[:450]
scenes={}
for p in probe:
    s=json.loads((p/'meta.json').read_text()).get('scene','?'); scenes.setdefault(s,[]).append(p)
scl=sorted(scenes); random.shuffle(scl); val_sc=set(scl[:max(1,len(scl)//5)])
val=[p for s in val_sc for p in scenes[s]]; random.shuffle(val); val=val[:140]
print(f'{len(scl)} scenes scanned | val on {len(val)} (scene-disjoint)')

cache=[]
for sd in tqdm(val):
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    gt=np.array(Image.open(sd/'target'/f'{cam}.jpg').convert('RGB')).astype(np.float32)/255.
    try: geo,trust,rife=mv_geo(sd); cache.append((geo,trust,rife,gt))
    except Exception as e: print('skip',sd.name,e)

def ev(cap):
    sc=[]
    for geo,trust,rife,gt in cache:
        t=np.clip(cv2.GaussianBlur(trust,(0,0),3.0),0,cap)[...,None]
        sc.append(score_of(psnr01(t*geo+(1-t)*rife,gt)))
    return float(np.mean(sc))
rife_only=ev(0.0); print(f'RIFE-only = {rife_only:.3f}')
best=(rife_only,0.0)
for cap in [0.5,0.7,0.9,1.0]:
    s=ev(cap); print(f'  cap={cap}: {s:.3f}')
    if s>best[0]: best=(s,cap)
BEST_CAP=best[1]; USE_HYBRID=BEST_CAP>0 and best[0]>rife_only+0.05
print(f'BEST cap={BEST_CAP} score={best[0]:.3f} | submit:', 'MV-HYBRID' if USE_HYBRID else 'RIFE-only')""")

md("## 5. Submission (winner, q100)")
code("""def predict(sd):
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    i0=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB')); i1=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    if not USE_HYBRID: return rife_scaled(i0,i1)
    geo,trust,rife=mv_geo(sd)
    t=np.clip(cv2.GaussianBlur(trust,(0,0),3.0),0,BEST_CAP)[...,None]
    return t*geo+(1-t)*rife

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
print('submission ready | mode:', 'MV-HYBRID cap=%.1f'%BEST_CAP if USE_HYBRID else 'RIFE-only')""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
for p in OUT_PATHS:
    with open(p,"w") as f: json.dump(nb,f,indent=1)
    print("wrote",p,"with",len(cells),"cells")
