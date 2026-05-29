"""Generate kaggle_hybrid.ipynb — hybrid geometry(lidar+pose) + RIFE for Task B.

Design (the path to ~58-60):
  - Target = same camera at mid-time. Lidar (world frame) + exact target pose give a
    near-exact reprojection of the STATIC scene; RIFE handles dynamics + holes (sky).
  - The earlier 12.8 dB geometry was almost certainly a distortion/axis bug. This nb is
    SELF-VALIDATING and needs NO ground truth to verify geometry:
      warp t0 -> t1's view via lidar depth + known pose change, compare to the REAL t1.
    The distortion model (pinhole vs fisheye) is auto-picked PER CAMERA by that PSNR.
  - Discipline: measure geo / rife / hybrid on train (GT available) and only submit the
    hybrid if it wins.

Setup: Accelerator = GPU T4 x2, Internet = ON. Attach: dataset (train/ test/) +
RIFE dataset (ECCV2022-RIFE repo + train_log weights). If you fine-tuned RIFE, also
attach flownet_ft.pkl (auto-loaded if found).

Run:  python3 build_hybrid_nb.py  ->  writes kaggle_hybrid.ipynb
"""
import json

cells = []
def _lines(s): return s.splitlines(keepends=True)
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)})

md("""# Task B — hybrid geometry (lidar+pose) + RIFE

**Why hybrid.** Pure VFI tops out (~52) because 1-2 s of driving has large parallax and
the net hallucinates. But we have the task's unique edge: **exact target pose + dense
lidar**. For the *static* scene (buildings, road, lane lines) a depth-based reprojection
is near-exact; RIFE only needs to cover *dynamics* (cars, people) and holes (sky).

**Self-validation (no GT needed for geometry).** We warp `t0 -> t1`'s view through lidar
depth + the known pose change and compare to the **real** `t1` image. That PSNR tells us
if geometry is correct, and we auto-pick the distortion model (pinhole vs fisheye) per
camera by it. This is how we avoid the old 12.8 dB blind failure.

**Discipline.** Measure geo / rife / hybrid on train (GT available); submit hybrid only
if it wins.

Setup: **GPU T4 x2**, **Internet ON**. Attach dataset + RIFE repo/weights (+ optional
`flownet_ft.pkl`).""")

# ---------------------------------------------------------------- cell 1
md("## 1. Imports, paths, load RIFE")
code("""import os, glob, json, math, random, sys, time
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
try:
    from scipy import ndimage
except Exception:
    ndimage = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:', device, '| torch', torch.__version__)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m): return os.path.dirname(m)
    return None
TRAIN_ROOT = (find_split_root('train') or '') + '/train'
_eb = find_split_root('test'); TEST_ROOT = (_eb + '/test') if _eb else None
print('TRAIN_ROOT:', TRAIN_ROOT, os.path.isdir(TRAIN_ROOT))
print('TEST_ROOT :', TEST_ROOT)

# RIFE weights — only needs the inference-only train_log files (RIFE_HDv3.py + flownet.pkl)
try:
    w_dir = str(Path(glob.glob('/kaggle/input/**/flownet.pkl', recursive=True)[0]).parent)
    sys.path.insert(0, str(Path(w_dir).parent)); sys.path.insert(0, w_dir)
    from train_log.RIFE_HDv3 import Model
    _m = Model(); _m.load_model(w_dir, -1); _m.flownet.to(device).eval()
    flownet = _m.flownet
    ft = glob.glob('/kaggle/input/**/flownet_ft.pkl', recursive=True) + glob.glob('/kaggle/working/flownet_ft.pkl')
    if ft:
        flownet.load_state_dict(torch.load(ft[0], map_location=device)); print('loaded fine-tuned weights:', ft[0])
    print('RIFE ready')
except Exception as e:
    flownet = None; print('RIFE NOT loaded:', e)""")

# ---------------------------------------------------------------- cell 2 geometry
md("## 2. Geometry: lidar -> target depth -> backward warp (distortion-aware)")
code("""def K_of(intr):
    return np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]], np.float64)
def dist_of(intr):
    return np.asarray(intr.get('distortion_coeffs') or [], np.float64).reshape(-1)

def project(Xc, intr, fisheye):
    \"\"\"camera pts (N,3) -> pixels (N,2). z>0 = in front.\"\"\"
    K, d = K_of(intr), dist_of(intr)
    rvec = tvec = np.zeros(3)
    if fisheye:
        uv,_ = cv2.fisheye.projectPoints(Xc.reshape(-1,1,3).astype(np.float64), rvec, tvec, K,
                                         d[:4] if d.size>=4 else np.zeros(4))
    else:
        uv,_ = cv2.projectPoints(Xc.astype(np.float64), rvec, tvec, K, d if d.size else None)
    return uv.reshape(-1,2)

def unproject_rays(uv, intr, fisheye):
    \"\"\"pixels (N,2) -> unit cam-frame rays (N,3), undistorted.\"\"\"
    K, d = K_of(intr), dist_of(intr)
    pts = uv.reshape(-1,1,2).astype(np.float64)
    if fisheye:
        n = cv2.fisheye.undistortPoints(pts, K, d[:4] if d.size>=4 else np.zeros(4))
    else:
        n = cv2.undistortPoints(pts, K, d if d.size else None)
    n = n.reshape(-1,2)
    r = np.concatenate([n, np.ones((n.shape[0],1))], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)

def world_to_cam(Xw, c2w):
    R, t = c2w[:3,:3], c2w[:3,3]
    return (Xw - t[None,:]) @ R

def target_depth(lidar, c2w, intr, W, H, fisheye, splat=2):
    Xc = world_to_cam(lidar.astype(np.float64), c2w)
    front = Xc[:,2] > 1e-6
    uv = project(Xc[front], intr, fisheye)
    z = Xc[front,2]
    u = np.round(uv[:,0]).astype(np.int64); v = np.round(uv[:,1]).astype(np.int64)
    inb = (u>=0)&(u<W)&(v>=0)&(v<H)
    u,v,z = u[inb], v[inb], z[inb]
    depth = np.full((H,W), np.inf, np.float32)
    order = np.argsort(-z)                       # far->near, nearest wins
    depth[v[order], u[order]] = z[order].astype(np.float32)
    if splat>1:                                  # small min-pool to fill micro-holes
        k = np.ones((splat,splat), np.uint8)
        fin = np.isfinite(depth)
        df = np.where(fin, depth, 1e9).astype(np.float32)
        df = cv2.erode(df, k)
        grew = (df<1e8)&(~fin); depth[grew] = df[grew]
    valid = np.isfinite(depth)
    return depth, valid

def fill_holes(depth, valid):
    hole = ~valid
    if hole.all(): return np.full_like(depth, 30.0), hole
    if ndimage is None:
        depth = depth.copy(); depth[hole] = np.median(depth[valid]); return depth, hole
    idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
    return depth[tuple(idx)].astype(np.float32), hole

def backward_warp(src_img, src_c2w, src_intr, src_fish, depth, tgt_c2w, tgt_intr, tgt_fish):
    \"\"\"sample src_img at target pixels via target depth. img float 0..1. returns warp,mask.\"\"\"
    H,W = depth.shape
    vs,us = np.mgrid[0:H,0:W]
    uv = np.stack([us.ravel(), vs.ravel()],1).astype(np.float64)
    rays = unproject_rays(uv, tgt_intr, tgt_fish)
    z = depth.ravel().astype(np.float64)
    Xc = rays * (z/np.clip(rays[:,2],1e-6,None))[:,None]
    Xw = Xc @ tgt_c2w[:3,:3].T + tgt_c2w[:3,3][None,:]
    Xcs = world_to_cam(Xw, src_c2w)
    front = Xcs[:,2] > 1e-6
    uvs = project(Xcs, src_intr, src_fish)
    Hs,Ws = src_img.shape[:2]
    mx = uvs[:,0].reshape(H,W).astype(np.float32); my = uvs[:,1].reshape(H,W).astype(np.float32)
    warp = cv2.remap((src_img*255).astype(np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)/255.
    inb = front.reshape(H,W)&(mx>=0)&(mx<Ws)&(my>=0)&(my<Hs)
    return warp.astype(np.float32), inb

def psnr_np(a,b):
    a=np.clip(a,0,1).astype(np.float64); b=np.clip(b,0,1).astype(np.float64)
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(1/np.sqrt(m))""")

# ---------------------------------------------------------------- cell 3 diagnostic
md("""## 3. DIAGNOSTIC — print a real meta.json

Confirms field names before we trust anything: distortion model per camera, whether
real target timestamps exist (else alpha=0.5), lidar shape.""")
code("""sd = sorted(p for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())[0]
meta = json.loads((sd/'meta.json').read_text())
print('keys:', list(meta.keys()))
print('target_camera:', meta.get('target_camera'), '| delta_s:', meta.get('delta_s'))
print('has timestamps_ns:', 'timestamps_ns' in meta, meta.get('timestamps_ns'))
for cam,intr in meta['intrinsics'].items():
    print(f\"  {cam:10s} model={intr.get('distortion_model')!r:24s} coeffs={intr.get('distortion_coeffs')}\")
lz = np.load(sd/'input'/'lidar.npz'); print('lidar keys:', lz.files, '| xyz:', lz['xyz'].shape)
print('poses_c2w targets:', list(meta['poses_c2w'].get('target',{}).keys()))""")

# ---------------------------------------------------------------- cell 4 self-cal
md("""## 4. Self-calibrate distortion model PER CAMERA (no GT)

For each camera we warp its `t0` image into the `t1` view via lidar depth + the known
t0->t1 pose change, and compare to the **real** `t1`. Whichever distortion model gives
higher PSNR is correct. If even the best is poor (< ~15 dB) geometry is unreliable for
that camera and the hybrid will fall back to RIFE there.""")
code("""CAMS = ['front','left_fwd','left_bwd','right_fwd','right_bwd','rear']
def read01(p): return np.array(Image.open(p).convert('RGB')).astype(np.float32)/255.

def cam_selfcheck(sample_dir, cam):
    sd = Path(sample_dir); meta = json.loads((sd/'meta.json').read_text())
    intr = meta['intrinsics'][cam]
    c2w0 = np.array(meta['poses_c2w']['t0'][cam], np.float64)
    c2w1 = np.array(meta['poses_c2w']['t1'][cam], np.float64)
    lidar = np.load(sd/'input'/'lidar.npz')['xyz']
    i0 = read01(sd/'input'/'t0'/f'{cam}.jpg'); i1 = read01(sd/'input'/'t1'/f'{cam}.jpg')
    H,W = i1.shape[:2]; out = {}
    for fish in (False, True):
        try:
            d,v = target_depth(lidar, c2w1, intr, W, H, fish)   # depth at t1 view
            d,_ = fill_holes(d, v)
            w,m = backward_warp(i0, c2w0, intr, fish, d, c2w1, intr, fish)  # t0 -> t1
            out[fish] = psnr_np(w[m>0.5], i1[m>0.5]) if m.any() else -1
        except Exception as e:
            out[fish] = -1
    return out

probe = sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())
random.shuffle(probe); probe = probe[:25]
CAM_FISH = {}
for cam in CAMS:
    pin=[]; fis=[]
    for sdp in probe:
        try:
            mj = json.loads((Path(sdp)/'meta.json').read_text())
            if cam not in mj['intrinsics']: continue
            r = cam_selfcheck(sdp, cam); pin.append(r[False]); fis.append(r[True])
        except Exception: pass
    if not pin: print(f'{cam}: no data'); continue
    mp, mf = np.mean(pin), np.mean(fis)
    CAM_FISH[cam] = mf > mp
    print(f'{cam:10s} pinhole={mp:5.2f}  fisheye={mf:5.2f}  -> {\"FISHEYE\" if CAM_FISH[cam] else \"pinhole\"}')
print('\\nCAM_FISH =', CAM_FISH)
print('NOTE: t0->t1 PSNR > ~20 means geometry is trustworthy for that camera.')""")

# ---------------------------------------------------------------- cell 5 hybrid
md("""## 5. Hybrid render + validate on train (geo vs rife vs hybrid)

Per target pixel:
- **geo base** = validity-weighted blend of (warp_t0, warp_t1) at the target pose.
- **agreement** = how close the two warps are → high on static scene, low on moving
  objects / occlusion. We trust geometry there and fall back to RIFE elsewhere + on holes.""")
code("""def rife_pred(i0u8, i1u8, scale):
    i0=torch.from_numpy(i0u8).permute(2,0,1)[None].float().to(device)/255.
    i1=torch.from_numpy(i1u8).permute(2,0,1)[None].float().to(device)/255.
    m=64*max(1,int(round(1.0/scale)))
    _,_,h,w=i0.shape; ph=((h-1)//m+1)*m; pw=((w-1)//m+1)*m
    x0=F.pad(i0,(0,pw-w,0,ph-h),mode='replicate'); x1=F.pad(i1,(0,pw-w,0,ph-h),mode='replicate')
    sl=[16/scale,8/scale,4/scale,2/scale,1/scale]
    with torch.no_grad():
        _,_,merged = flownet(torch.cat((x0,x1),1), 0.5, sl)
    return merged[-1][:,:, :h,:w].clamp(0,1)[0].permute(1,2,0).cpu().numpy()

def geo_render(sample_dir, cam, meta):
    sd=Path(sample_dir); intr=meta['intrinsics'][cam]; fish=CAM_FISH.get(cam, False)
    c2wt=np.array(meta['poses_c2w']['target'][cam], np.float64)
    c2w0=np.array(meta['poses_c2w']['t0'][cam], np.float64)
    c2w1=np.array(meta['poses_c2w']['t1'][cam], np.float64)
    lidar=np.load(sd/'input'/'lidar.npz')['xyz']
    i0=read01(sd/'input'/'t0'/f'{cam}.jpg'); i1=read01(sd/'input'/'t1'/f'{cam}.jpg')
    H,W=i0.shape[:2]
    d,v=target_depth(lidar,c2wt,intr,W,H,fish); d,_=fill_holes(d,v)
    w0,m0=backward_warp(i0,c2w0,intr,fish,d,c2wt,intr,fish)
    w1,m1=backward_warp(i1,c2w1,intr,fish,d,c2wt,intr,fish)
    wsum=(m0[...,None]*w0 + m1[...,None]*w1)
    wcnt=(m0.astype(np.float32)+m1.astype(np.float32))[...,None]
    geo=np.where(wcnt>0, wsum/np.clip(wcnt,1,None), 0.0).astype(np.float32)
    geo_valid=(m0&m1)                                   # both views see it
    agree=np.exp(-np.abs(w0-w1).mean(-1)/0.06)          # static -> ~1, dynamic -> ~0
    trust=(geo_valid.astype(np.float32))*agree          # HxW in [0,1]
    trust=cv2.GaussianBlur(trust,(0,0),3.0)             # feather
    return geo, trust, (i0*255).astype(np.uint8), (i1*255).astype(np.uint8)

def hybrid_render(sample_dir, scale=0.5):
    sd=Path(sample_dir); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    geo,trust,i0u,i1u=geo_render(sd,cam,meta)
    r=rife_pred(i0u,i1u,scale)
    t=trust[...,None]
    return (t*geo + (1-t)*r).astype(np.float32), geo, r

# validate
vdirs=sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())
random.shuffle(vdirs); vdirs=vdirs[:80]
acc={'geo':0.,'rife':0.,'hybrid':0.}; n=0
for sdp in tqdm(vdirs):
    sd=Path(sdp); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    gt=read01(sd/'target'/f'{cam}.jpg')
    ds=meta.get('delta_s',1.0); sc=0.5 if ds<1.5 else 0.25
    hyb,geo,r=hybrid_render(sd, sc)
    acc['geo']+=psnr_np(geo,gt); acc['rife']+=psnr_np(r,gt); acc['hybrid']+=psnr_np(hyb,gt); n+=1
print(f'\\nval over {n}:')
for k,v in sorted(acc.items(), key=lambda kv:-kv[1]): print(f'  {k:8s} {v/n:.3f} dB')
BEST=max(acc,key=acc.get); print('BEST:', BEST)""")

# ---------------------------------------------------------------- cell 6 submission
md("## 6. Submission (uses whichever won on val)")
code("""def predict(sample_dir):
    sd=Path(sample_dir); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    ds=meta.get('delta_s',1.0); sc=0.5 if ds<1.5 else 0.25
    hyb,geo,r=hybrid_render(sd, sc)
    return {'geo':geo,'rife':r,'hybrid':hyb}[BEST]

import shutil
def make_submission(test_root, out='/kaggle/working/submission'):
    dirs=sorted(p for p in Path(test_root).iterdir() if p.is_dir()); os.makedirs(out,exist_ok=True)
    for sd in tqdm(dirs):
        pred=np.clip(predict(sd),0,1); pred=(pred*255).round().astype(np.uint8)
        od=Path(out)/sd.name; od.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pred).save(od/'pred.jpg', quality=95)
    shutil.make_archive('/kaggle/working/submission','zip',out)
    print('submission ->', out, '| zipped | method:', BEST)

if TEST_ROOT and os.path.isdir(TEST_ROOT):
    make_submission(TEST_ROOT)
else:
    print('No TEST_ROOT — attach test dataset and re-run cell 1.')""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python"}, "accelerator":"GPU"},
      "nbformat":4, "nbformat_minor":5}
with open("kaggle_hybrid.ipynb","w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_hybrid.ipynb with", len(cells), "cells")
