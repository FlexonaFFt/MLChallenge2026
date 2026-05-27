"""Generate kaggle_3dgs.ipynb — per-scene 3D Gaussian Splatting for Task B.

Strategy: for each sample, fit a 3DGS to the 12 input images (poses known),
initialized from the lidar cloud, then render the target camera at its known pose.
We VALIDATE on train samples (with GT) vs a cheap 2D baseline BEFORE running the
full 199-sample test set.

Accelerator: GPU T4 x2 | Internet: ON (to pip install gsplat).
"""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Task B — per-scene 3D Gaussian Splatting

For each scene: initialize Gaussians from the lidar cloud, optimize them to match the
12 input images (known camera poses), then render the **target camera** at its known
pose. This uses the task's unique edge: exact target pose + lidar geometry + no
inference-time limit.

**Discipline:** we first measure 3DGS vs a 2D baseline on a few *train* samples (GT
available). Only if 3DGS wins do we run the full test set.

Setup: **Accelerator = GPU T4 ×2**, **Internet = ON** (Settings panel).""")

md("## 1. Install gsplat (needs Internet ON)")
code("""import subprocess, sys
try:
    import gsplat
    print('gsplat already present')
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gsplat'], check=True)
    import gsplat
print('gsplat', gsplat.__version__)""")

md("## 2. Imports, config, data paths")
code("""import os, glob, json, math, random, time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from gsplat import rasterization

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:', device, '| torch', torch.__version__)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m):
            return os.path.dirname(m)
    return None
TRAIN_ROOT = (find_split_root('train') or '') + '/train'
_eb = find_split_root('test'); TEST_ROOT = (_eb + '/test') if _eb else None
print('TRAIN_ROOT:', TRAIN_ROOT, os.path.isdir(TRAIN_ROOT))
print('TEST_ROOT :', TEST_ROOT)

CAMERAS = ['front','left_fwd','left_bwd','right_fwd','right_bwd','rear']
GS = dict(iters=1500, n_pts=150_000, init_scale=0.10, ssim_w=0.2)""")

md("""## 3. Load a scene

Returns the 12 input views (image + world-to-camera `viewmat` + intrinsics `K`),
the target camera (viewmat/K/size) and the lidar points. gsplat uses the OpenCV
camera convention, which matches this dataset, so `viewmat = inverse(c2w)`.""")
code("""def K_from(intr):
    return torch.tensor([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],
                        dtype=torch.float32)

def read01(p):
    return torch.from_numpy(np.array(Image.open(p).convert('RGB'))).float()/255.

def load_scene(sd, dev=None):
    dev = dev or device
    sd = Path(sd); meta = json.loads((sd/'meta.json').read_text())
    tcam = meta['target_camera']
    imgs, viewmats, Ks = [], [], []
    for ts in ('t0','t1'):
        for cam in CAMERAS:
            p = sd/'input'/ts/f'{cam}.jpg'
            if not p.exists(): continue
            c2w = torch.tensor(meta['poses_c2w'][ts][cam], dtype=torch.float32)
            imgs.append(read01(p))
            viewmats.append(torch.linalg.inv(c2w))
            Ks.append(K_from(meta['intrinsics'][cam]))
    xyz = np.load(sd/'input'/'lidar.npz')['xyz'].astype(np.float32)
    tgt_c2w = torch.tensor(meta['poses_c2w']['target'][tcam], dtype=torch.float32)
    intr = meta['intrinsics'][tcam]
    scene = dict(
        imgs=[im.to(dev) for im in imgs],
        viewmats=torch.stack(viewmats).to(dev),
        Ks=torch.stack(Ks).to(dev),
        xyz=torch.from_numpy(xyz).to(dev),
        tgt_viewmat=torch.linalg.inv(tgt_c2w).to(dev),
        tgt_K=K_from(intr).to(dev),
        H=int(intr['height']), W=int(intr['width']), tcam=tcam,
    )
    gt = sd/'target'/f'{tcam}.jpg'
    scene['gt'] = read01(gt).to(dev) if gt.exists() else None
    return scene""")

md("## 4. SSIM + color init from lidar projection")
code("""def _win(ch, size=11, sigma=1.5, device='cpu'):
    c = torch.arange(size, dtype=torch.float32, device=device)-size//2
    g = torch.exp(-(c**2)/(2*sigma**2)); g=(g/g.sum()).unsqueeze(0)
    return (g.t()@g).expand(ch,1,size,size).contiguous()
def ssim(a,b):  # a,b: (1,3,H,W)
    ch=a.shape[1]; w=_win(ch,device=a.device); p=w.shape[-1]//2
    m1=F.conv2d(a,w,padding=p,groups=ch); m2=F.conv2d(b,w,padding=p,groups=ch)
    s1=F.conv2d(a*a,w,padding=p,groups=ch)-m1*m1
    s2=F.conv2d(b*b,w,padding=p,groups=ch)-m2*m2
    s12=F.conv2d(a*b,w,padding=p,groups=ch)-m1*m2
    c1,c2=0.01**2,0.03**2
    return (((2*m1*m2+c1)*(2*s12+c2))/((m1*m1+m2*m2+c1)*(s1+s2+c2))).mean()
def psnr_t(a,b):
    mse=F.mse_loss(a,b); return 99.0 if mse<1e-9 else (20*torch.log10(1/torch.sqrt(mse))).item()

@torch.no_grad()
def init_colors(means, scene):
    # project each point into every input camera, average the colors where visible
    dev = means.device
    N = means.shape[0]
    acc = torch.zeros(N,3,device=dev); cnt = torch.zeros(N,1,device=dev)
    ones = torch.ones(N,1,device=dev)
    Ph = torch.cat([means, ones],1)                       # (N,4)
    for i in range(scene['viewmats'].shape[0]):
        Xc = (scene['viewmats'][i] @ Ph.T).T[:, :3]       # (N,3)
        z = Xc[:,2]; front = z > 1e-3
        uv = (scene['Ks'][i] @ Xc.T).T
        u = (uv[:,0]/uv[:,2]); v = (uv[:,1]/uv[:,2])
        H,W = scene['imgs'][i].shape[:2]
        ok = front & (u>=0)&(u<W)&(v>=0)&(v<H)
        idx = ok.nonzero(as_tuple=True)[0]
        if idx.numel()==0: continue
        cols = scene['imgs'][i][v[idx].long(), u[idx].long()]
        acc[idx]+=cols; cnt[idx]+=1
    col = torch.where(cnt>0, acc/cnt.clamp_min(1), torch.full_like(acc,0.5))
    return col""")

md("## 5. Fit 3DGS to the 12 views, render the target")
code("""def inv_sig(x): return math.log(x/(1-x))

def fit_and_render(scene, iters=None, verbose=False):
    iters = iters or GS['iters']
    dev = scene['xyz'].device
    xyz = scene['xyz']
    if xyz.shape[0] > GS['n_pts']:
        sel = torch.randperm(xyz.shape[0], device=dev)[:GS['n_pts']]
        xyz = xyz[sel]
    N = xyz.shape[0]
    means = nn.Parameter(xyz.clone())
    quats = nn.Parameter(torch.tensor([1.,0,0,0],device=dev).repeat(N,1))
    scales = nn.Parameter(torch.full((N,3), math.log(GS['init_scale']), device=dev))
    opac = nn.Parameter(torch.full((N,), inv_sig(0.1), device=dev))
    col0 = init_colors(means.detach(), scene).clamp(1e-3,1-1e-3)
    colors = nn.Parameter(torch.log(col0/(1-col0)))     # logit
    opt = torch.optim.Adam([
        {'params':[means],  'lr':1.6e-4},
        {'params':[colors], 'lr':1e-2},
        {'params':[scales], 'lr':5e-3},
        {'params':[quats],  'lr':1e-3},
        {'params':[opac],   'lr':5e-2},
    ])
    C = scene['viewmats'].shape[0]
    for it in range(iters):
        ci = random.randrange(C)
        img = scene['imgs'][ci]                              # per-view size (cameras differ!)
        Hc, Wc = img.shape[0], img.shape[1]
        rc,_,_ = rasterization(means, F.normalize(quats,dim=-1), torch.exp(scales),
                               torch.sigmoid(opac), torch.sigmoid(colors),
                               scene['viewmats'][ci:ci+1], scene['Ks'][ci:ci+1], Wc, Hc)
        pred = rc[0].permute(2,0,1).unsqueeze(0).clamp(0,1)   # (1,3,Hc,Wc)
        tgt = img.permute(2,0,1).unsqueeze(0)
        loss = F.l1_loss(pred,tgt) + GS['ssim_w']*(1-ssim(pred,tgt))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if verbose and it%300==0: print(f'  it{it} loss {loss.item():.4f}')
    with torch.no_grad():
        rc,_,_ = rasterization(means, F.normalize(quats,dim=-1), torch.exp(scales),
                               torch.sigmoid(opac), torch.sigmoid(colors),
                               scene['tgt_viewmat'][None], scene['tgt_K'][None], W, H)
    return rc[0].clamp(0,1)    # (H,W,3)""")

md("""## 6. VALIDATE on a few train samples (vs mean baseline)

If 3DGS PSNR beats the 2D baseline here, it's worth running the full test. If not,
we stop and keep the 2D ensemble (47.1).""")
code("""def mean_pred(sd):
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
    i0=read01(sd/'input'/'t0'/f'{cam}.jpg'); i1=read01(sd/'input'/'t1'/f'{cam}.jpg')
    return (0.5*(i0+i1)).to(device)

val = sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())
random.shuffle(val); val = val[:6]
g=b=0
for sd in val:
    scene = load_scene(sd)
    t0=time.time(); pred = fit_and_render(scene); dt=time.time()-t0
    pg = psnr_t(pred.permute(2,0,1)[None], scene['gt'].permute(2,0,1)[None])
    pm = psnr_t(mean_pred(sd).permute(2,0,1)[None], scene['gt'].permute(2,0,1)[None])
    g+=pg; b+=pm
    print(f'{Path(sd).name[:30]} | 3DGS {pg:.2f} dB | mean {pm:.2f} dB | {dt:.0f}s')
print(f'\\nAVG over {len(val)}: 3DGS {g/len(val):.3f} dB | mean {b/len(val):.3f} dB | gain {(g-b)/len(val):+.3f}')""")

md("""## 7. Full test inference -> submission (run only if 3DGS won)

Scenes are split across all available GPUs (one worker thread per GPU), so 2×T4
processes ~2× faster. Each scene is fit independently on its assigned device.""")
code("""import threading, shutil

def _worker(dirs, dev, out, pbar, lock):
    with torch.cuda.device(dev):
        for sd in dirs:
            scene = load_scene(sd, dev)
            pred = fit_and_render(scene).cpu().numpy()
            od = Path(out)/Path(sd).name; od.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred*255).round().astype(np.uint8)).save(od/'pred.jpg', quality=95)
            with lock: pbar.update(1)

def make_submission(test_root, out='/kaggle/working/submission'):
    dirs = sorted(str(p) for p in Path(test_root).iterdir() if p.is_dir())
    os.makedirs(out, exist_ok=True)
    ngpu = max(1, torch.cuda.device_count())
    devs = [f'cuda:{i}' for i in range(ngpu)] if torch.cuda.is_available() else ['cpu']
    shards = [dirs[i::len(devs)] for i in range(len(devs))]
    print(f'{len(dirs)} scenes across {len(devs)} device(s): {devs}')
    pbar = tqdm(total=len(dirs)); lock = threading.Lock()
    threads = [threading.Thread(target=_worker, args=(shards[i], devs[i], out, pbar, lock))
               for i in range(len(devs))]
    for t in threads: t.start()
    for t in threads: t.join()
    pbar.close()
    shutil.make_archive('/kaggle/working/submission','zip',out)
    print('submission ->', out, '| zipped ->', '/kaggle/working/submission.zip')

# Uncomment to run the full test set (~199 scenes), parallel over both T4s:
# make_submission(TEST_ROOT)""")

nb = {"cells": cells,
      "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python"}, "accelerator":"GPU"},
      "nbformat":4,"nbformat_minor":5}
with open("kaggle_3dgs.ipynb","w") as f:
    json.dump(nb,f,indent=1)
print("wrote kaggle_3dgs.ipynb with", len(cells), "cells")
