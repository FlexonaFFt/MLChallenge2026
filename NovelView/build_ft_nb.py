"""Generate kaggle_rife_finetune.ipynb — fine-tune RIFE v4.25 on the dataset
triplets (t0,t1)->target, on 2x T4 via DataParallel, then ensemble with DIS.

Key facts (verified from the real RIFE_HDv3.py / IFNet_HDv3.py):
  - model.inference(img0,img1,timestep,scale) runs the working (training=False) path
    and IS differentiable -> we fine-tune through it with our own loop.
  - the repo's update()/training=True path is broken in this inference-only release
    (undefined loss_cons / img0) -> we do NOT use it.
  - target is the temporal midpoint -> timestep=0.5.
"""
import json

cells = []
def _lines(s): return s.splitlines(keepends=True)  # canonical nbformat source form
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)})

md("""# Task B — fine-tune RIFE on the dataset (2× T4) + ensemble

The post-hoc refiner was dead (regresses to input) and geometry/3DGS lose. The real
lever: **fine-tune the VFI network end-to-end** on this dataset's `(t0,t1)->target`
triplets. Gradients flow through motion estimation, so the model adapts to the large
1–2 s driving motion that generic RIFE handles poorly.

- 2× T4 via DataParallel (batch split across both GPUs).
- Validate vs pretrained RIFE each epoch; keep best.
- At inference, pick the best of {fine-tuned RIFE, DIS, ensembles} on val.

Setup: **Accelerator = GPU T4 ×2**, **Internet = ON**. Needs the RIFE dataset
(ECCV2022-RIFE repo + train_log weights) attached.""")

md("## 1. Imports, data paths, load RIFE")
code("""import os, glob, json, math, random, time, sys, shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NGPU = torch.cuda.device_count()
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:', device, '| GPUs:', NGPU, '| torch', torch.__version__)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m): return os.path.dirname(m)
    return None
TRAIN_ROOT = (find_split_root('train') or '') + '/train'
_eb = find_split_root('test'); TEST_ROOT = (_eb + '/test') if _eb else None
print('TRAIN_ROOT:', TRAIN_ROOT, os.path.isdir(TRAIN_ROOT))
print('TEST_ROOT :', TEST_ROOT)

# locate RIFE repo (model/) + weights (flownet.pkl)
repo_dir = str(Path(glob.glob('/kaggle/input/**/inference_img.py', recursive=True)[0]).parent)
w_dir = str(Path(glob.glob('/kaggle/input/**/flownet.pkl', recursive=True)[0]).parent)
sys.path.insert(0, repo_dir); sys.path.insert(0, str(Path(w_dir).parent))
print('repo:', repo_dir, '| weights:', w_dir)

from train_log.RIFE_HDv3 import Model
_m = Model(); _m.load_model(w_dir, -1); _m.flownet.to(device)
flownet = _m.flownet                       # IFNet v4.25, trainable
print('RIFE loaded | params: %.2fM' % (sum(p.numel() for p in flownet.parameters())/1e6))""")

md("## 2. RIFE forward, DIS, helpers")
code("""def rife_forward(net, img0, img1, timestep=0.5, scale=1.0):
    imgs = torch.cat((img0, img1), 1)
    sl = [16/scale, 8/scale, 4/scale, 2/scale, 1/scale]
    flow, mask, merged = net(imgs, timestep, sl)
    return merged[-1]

def pad64(t):
    _,_,h,w = t.shape
    ph, pw = ((h-1)//64+1)*64, ((w-1)//64+1)*64
    return F.pad(t, (0, pw-w, 0, ph-h), mode='replicate'), h, w

def read01(p):  # -> (3,H,W) tensor 0..1
    return torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2,0,1).float()/255.

def get_alpha(meta):
    ts = meta.get('timestamps_ns')
    if ts and all(k in ts for k in ('t0','t1','target')):
        return float((ts['target']-ts['t0'])/(ts['t1']-ts['t0']))
    return 0.5

_DIS = None
def dis_interp(i0u, i1u, a):    # uint8 HxWx3 -> float HxWx3 0..255
    global _DIS
    if _DIS is None: _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = _DIS.calc(cv2.cvtColor(i0u,cv2.COLOR_RGB2GRAY), cv2.cvtColor(i1u,cv2.COLOR_RGB2GRAY), None)
    dx,dy = fl[...,0], fl[...,1]
    def w(img,ddx,ddy):
        H,W=img.shape[:2]; ys,xs=np.mgrid[0:H,0:W].astype(np.float32)
        return cv2.remap(img,(xs+ddx).astype(np.float32),(ys+ddy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return (1-a)*w(i0u.astype(np.float32),-a*dx,-a*dy) + a*w(i1u.astype(np.float32),(1-a)*dx,(1-a)*dy)

def _win(ch, s=11, sig=1.5, dev='cpu'):
    c=torch.arange(s,dtype=torch.float32,device=dev)-s//2; g=torch.exp(-(c**2)/(2*sig**2)); g=(g/g.sum())[None]
    return (g.t()@g).expand(ch,1,s,s).contiguous()
def ssim(a,b):
    ch=a.shape[1]; wn=_win(ch,dev=a.device); p=wn.shape[-1]//2
    m1=F.conv2d(a,wn,padding=p,groups=ch); m2=F.conv2d(b,wn,padding=p,groups=ch)
    s1=F.conv2d(a*a,wn,padding=p,groups=ch)-m1*m1; s2=F.conv2d(b*b,wn,padding=p,groups=ch)-m2*m2
    s12=F.conv2d(a*b,wn,padding=p,groups=ch)-m1*m2; c1,c2=0.01**2,0.03**2
    return (((2*m1*m2+c1)*(2*s12+c2))/((m1*m1+m2*m2+c1)*(s1+s2+c2))).mean()
def psnr_np(a,b):
    a=np.clip(a,0,255).astype(np.float64); b=b.astype(np.float64)
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(255/np.sqrt(m))""")

md("## 3. Triplet dataset (crops + h-flip)")
code("""class Triplets(Dataset):
    def __init__(self, root, crop=256, train=True):
        self.dirs = sorted(str(p) for p in Path(root).iterdir() if p.is_dir())
        self.crop, self.train = crop, train
    def __len__(self): return len(self.dirs)
    def _paths(self, sd):
        sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']
        return (sd/'input'/'t0'/f'{cam}.jpg', sd/'input'/'t1'/f'{cam}.jpg', sd/'target'/f'{cam}.jpg')
    def __getitem__(self, i):
        p0,p1,pg = self._paths(self.dirs[i])
        i0,i1,g = read01(p0), read01(p1), read01(pg)
        if self.train and self.crop:
            _,H,W = i0.shape; ch,cw=min(self.crop,H),min(self.crop,W)
            t=np.random.randint(0,H-ch+1); l=np.random.randint(0,W-cw+1)
            i0=i0[:,t:t+ch,l:l+cw]; i1=i1[:,t:t+ch,l:l+cw]; g=g[:,t:t+ch,l:l+cw]
            if np.random.rand()<0.5:
                i0=torch.flip(i0,[2]); i1=torch.flip(i1,[2]); g=torch.flip(g,[2])
        return i0,i1,g""")

md("""## 4. Fine-tune on 2× T4 (DataParallel)

Low LR adaptation of pretrained RIFE. Val each epoch (full frames) vs the pretrained
baseline; best weights saved to `/kaggle/working/flownet_ft.pkl`.""")
code("""CFG = dict(epochs=20, batch=16, crop=256, lr=5e-5, workers=4, val_frac=0.04, ssim_w=0.1)

full = Triplets(TRAIN_ROOT, crop=CFG['crop'], train=True)
nval = max(2, int(len(full)*CFG['val_frac'])); ntr = len(full)-nval
tr, va = random_split(full, [ntr, nval], generator=torch.Generator().manual_seed(42))
va.dataset = Triplets(TRAIN_ROOT, crop=0, train=False)   # full-frame val
tl = DataLoader(tr, batch_size=CFG['batch'], shuffle=True, num_workers=CFG['workers'],
                pin_memory=True, drop_last=True, persistent_workers=CFG['workers']>0)
vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=2)
print(f'train={ntr} val={nval}')

net = nn.DataParallel(flownet) if NGPU > 1 else flownet
if NGPU > 1: print(f'DataParallel over {NGPU} GPUs')

@torch.no_grad()
def val_psnr(use_net):
    use_net_was_training = flownet.training; flownet.eval()
    tot=n=0
    for i0,i1,g in vl:
        i0,i1,g = i0.to(device),i1.to(device),g.to(device)
        x0,h,w = pad64(i0); x1,_,_ = pad64(i1)
        pred = rife_forward(use_net, x0, x1)[:,:, :h,:w].clamp(0,1)
        mse = F.mse_loss(pred,g)
        tot += (99.0 if mse<1e-9 else (20*torch.log10(1/torch.sqrt(mse))).item()); n+=1
    if use_net_was_training: flownet.train()
    return tot/n

base_psnr = val_psnr(net)
print(f'pretrained RIFE val PSNR = {base_psnr:.3f} dB')

opt = torch.optim.AdamW(flownet.parameters(), lr=CFG['lr'], weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'])
# NOTE: train in fp32 (no AMP). RIFE's forward overflows under fp16 autocast ->
# GradScaler skips every step -> weights never update (the "gain=+0.000" bug).
best = base_psnr
for ep in range(CFG['epochs']):
    flownet.train(); pbar=tqdm(tl, desc=f'epoch {ep}/{CFG["epochs"]-1}')
    for i0,i1,g in pbar:
        i0,i1,g = i0.to(device,non_blocking=True),i1.to(device,non_blocking=True),g.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = rife_forward(net, i0, i1)            # no clamp -> keep gradients
        loss = F.l1_loss(pred,g) + CFG['ssim_w']*(1-ssim(pred.clamp(0,1),g))
        loss.backward(); opt.step()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    sch.step()
    vp = val_psnr(net)
    print(f'epoch {ep}: val PSNR={vp:.3f} dB | pretrained={base_psnr:.3f} | gain={vp-base_psnr:+.3f}')
    if vp > best:
        best = vp
        torch.save(flownet.state_dict(), '/kaggle/working/flownet_ft.pkl')
        print(f'  -> best {best:.3f} dB saved')
print('done. best val PSNR = %.3f dB (pretrained %.3f)' % (best, base_psnr))""")

md("## 5. Pick best method on val (fine-tuned RIFE vs DIS vs ensemble)")
code("""# reload best fine-tuned weights
if os.path.exists('/kaggle/working/flownet_ft.pkl'):
    flownet.load_state_dict(torch.load('/kaggle/working/flownet_ft.pkl')); print('loaded fine-tuned weights')
flownet.eval()

@torch.no_grad()
def rife_scaled(i0u, i1u, scale=0.5, tta=True):   # uint8 HxWx3 -> float HxWx3 0..255
    m = 64 * math.ceil(1.0/scale)                 # scale-aware pad: 1.0->64, 0.5->128, 0.25->256
    def padm(t):
        _,_,h,w=t.shape; ph=((h-1)//m+1)*m; pw=((w-1)//m+1)*m
        return F.pad(t,(0,pw-w,0,ph-h),mode='replicate'), h, w
    def run(a0,a1):
        i0=torch.from_numpy(np.ascontiguousarray(a0)).permute(2,0,1)[None].float().to(device)/255.
        i1=torch.from_numpy(np.ascontiguousarray(a1)).permute(2,0,1)[None].float().to(device)/255.
        x0,h,w=padm(i0); x1,_,_=padm(i1)
        out=rife_forward(flownet,x0,x1,0.5,scale)[:,:, :h,:w].clamp(0,1)
        return out[0].permute(1,2,0).cpu().numpy()*255
    p=run(i0u,i1u)
    if tta:                                        # horizontal-flip test-time augmentation
        pf=run(i0u[:,::-1], i1u[:,::-1])[:,::-1]; p=0.5*(p+pf)
    return p

# pick best ensemble on val (fine-tuned RIFE scale0.5+TTA blended with DIS)
val_dirs = sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())
random.shuffle(val_dirs); val_dirs = val_dirs[:100]
acc = {k:0.0 for k in ['ft_tta','dis','0.5dis+0.5ft','0.3dis+0.7ft']}; n=0
for sd in tqdm(val_dirs):
    sd=Path(sd); meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']; a=get_alpha(meta)
    gt = np.array(Image.open(sd/'target'/f'{cam}.jpg').convert('RGB'))
    i0u = np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB'))
    i1u = np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
    r = rife_scaled(i0u,i1u); ds = dis_interp(i0u,i1u,a)
    acc['ft_tta'] += psnr_np(r,gt); acc['dis'] += psnr_np(ds,gt)
    acc['0.5dis+0.5ft'] += psnr_np(0.5*ds+0.5*r, gt)
    acc['0.3dis+0.7ft'] += psnr_np(0.3*ds+0.7*r, gt); n+=1
print(f'\\nval over {n}:')
for k,v in sorted(acc.items(), key=lambda kv:-kv[1]): print(f'  {k:16s} {v/n:.3f} dB')
BEST = max(acc, key=acc.get); print('BEST:', BEST)""")

md("## 6. Build submission with the best method")
code("""def predict(i0u,i1u,a,method):
    r = rife_scaled(i0u,i1u); ds = dis_interp(i0u,i1u,a)
    return {'ft_tta':r, 'dis':ds, '0.5dis+0.5ft':0.5*ds+0.5*r, '0.3dis+0.7ft':0.3*ds+0.7*r}[method]

def make_submission(test_root, method, out='/kaggle/working/submission'):
    dirs=sorted(p for p in Path(test_root).iterdir() if p.is_dir()); os.makedirs(out,exist_ok=True)
    for sd in tqdm(dirs):
        meta=json.loads((sd/'meta.json').read_text()); cam=meta['target_camera']; a=get_alpha(meta)
        i0u=np.array(Image.open(sd/'input'/'t0'/f'{cam}.jpg').convert('RGB'))
        i1u=np.array(Image.open(sd/'input'/'t1'/f'{cam}.jpg').convert('RGB'))
        pred=np.clip(predict(i0u,i1u,a,method),0,255).round().astype(np.uint8)
        od=Path(out)/sd.name; od.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pred).save(od/'pred.jpg', quality=95)
    shutil.make_archive('/kaggle/working/submission','zip',out)
    print('submission ->', out, '| zipped. method:', method)

if TEST_ROOT and os.path.isdir(TEST_ROOT):
    make_submission(TEST_ROOT, BEST)
else:
    print('No TEST_ROOT — attach the test dataset and re-run cell 1.')""")

nb = {"cells": cells,
      "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python"}, "accelerator":"GPU"},
      "nbformat":4,"nbformat_minor":5}
with open("kaggle_rife_finetune.ipynb","w") as f:
    json.dump(nb,f,indent=1)
print("wrote kaggle_rife_finetune.ipynb with", len(cells), "cells")
