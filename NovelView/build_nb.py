"""Generate kaggle_train.ipynb — a clean, structured training notebook for Task B.

Run:  python3 build_nb.py   ->  writes kaggle_train.ipynb
"""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})


md("""# Task B — Novel View Synthesis: flow-based refiner (Kaggle)

**Idea.** Target = the same camera at the mid-time between `t0` and `t1`. Motion over
~0.5 s is small, so a good *motion-compensated* mid-frame is a strong starting point.
We use **DIS optical-flow interpolation** as the base (beats naive mean) and train a
**residual U-Net** to sharpen it toward ground truth (L1 + SSIM, which directly
helps PSNR — the contest metric).

Pipeline per sample:
`t0,t1 -> DIS flow -> warp both to mid -> flow_interp (base)`; the net predicts a
residual on top. Lidar/geometry is intentionally dropped — on this data the
aggregated cloud is contaminated by moving objects and hurt PSNR.

**Run order:** Settings → Accelerator → **GPU T4 ×2**. Then Run All.

> Use **T4**, not P100 — Kaggle's PyTorch dropped Pascal (sm_60) support, so P100
> throws `CUDA error: no kernel image available`. T4 (sm_75) works and the code
> uses both T4s via DataParallel.""")

md("## 1. Imports & config")
code("""import os, glob, json, time, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm.auto import tqdm

CFG = dict(
    train_root=None,   # auto-detected below
    test_root=None,    # auto-detected below
    out_dir='/kaggle/working',
    epochs=30,
    batch=16,          # 8 per GPU on T4 x2; lower to 8 if single GPU / OOM
    crop=512,
    lr=2e-4,
    workers=4,
    base=48,           # U-Net width
    val_frac=0.03,
    amp=True,
    hflip=True,
    seed=42,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
print('device:', device, '| torch:', torch.__version__)""")

md("""## 2. Locate the dataset

Add the data via **+ Add Input** (search `yandex-sorevnovanie-big-train`). Optionally
add a second dataset that contains the `test/` split for submission. The cell finds
the folders that contain `train/` and `test/` automatically.""")
code("""def find_split_root(want):
    # recursive: handles datasets nested several folders deep
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m):
            return os.path.dirname(m)
    return None

tb = find_split_root('train'); eb = find_split_root('test')
CFG['train_root'] = os.path.join(tb, 'train') if tb else None
CFG['test_root']  = os.path.join(eb, 'test')  if eb else None
print('train_root:', CFG['train_root'])
print('test_root :', CFG['test_root'])
assert CFG['train_root'], 'train/ not found under /kaggle/input/* — add the dataset'
print('train samples:', len(os.listdir(CFG['train_root'])))
if CFG['test_root']:
    print('test samples :', len(os.listdir(CFG['test_root'])))""")

md("""## 3. Input builder (flow interpolation)

For each sample: estimate DIS flow `t0->t1`, warp both frames to the mid-time, and
blend into `flow_interp` (the base). The 16-channel network input is:

`raw_t0(3) | raw_t1(3) | warp_t0(3) | warp_t1(3) | flow_interp(3) | flow_mag(1)`

Flow magnitude (not signed flow) is used so horizontal-flip augmentation stays valid.""")
code("""_DIS = None
def dis_flow(g0, g1):
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(g0, g1, None)

def warp(img, dx, dy):
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xs+dx).astype(np.float32), (ys+dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def read_img(p):
    return np.array(Image.open(p).convert('RGB'))

def get_alpha(meta):
    ts = meta.get('timestamps_ns')
    if ts and all(k in ts for k in ('t0','t1','target')):
        return float((ts['target']-ts['t0'])/(ts['t1']-ts['t0']))
    return 0.5

def build_inputs(sd):
    sd = Path(sd)
    meta = json.loads((sd/'meta.json').read_text())
    cam = meta['target_camera']; a = get_alpha(meta)
    i0 = read_img(sd/'input'/'t0'/f'{cam}.jpg')
    i1 = read_img(sd/'input'/'t1'/f'{cam}.jpg')
    g0 = cv2.cvtColor(i0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    fl = dis_flow(g0, g1); dx, dy = fl[...,0], fl[...,1]
    f0 = i0.astype(np.float32)/255.; f1 = i1.astype(np.float32)/255.
    w0 = warp(f0, -a*dx, -a*dy)
    w1 = warp(f1, (1-a)*dx, (1-a)*dy)
    fi = (1-a)*w0 + a*w1
    mag = np.sqrt((fl**2).sum(-1)); mag = (mag/(mag.max()+1e-6)).astype(np.float32)
    x = np.concatenate([
        f0.transpose(2,0,1), f1.transpose(2,0,1),
        w0.transpose(2,0,1), w1.transpose(2,0,1),
        fi.transpose(2,0,1), mag[None],
    ], 0).astype(np.float32)   # (16,H,W); base = channels 12:15
    gt = sd/'target'/f'{cam}.jpg'
    y = read_img(gt).astype(np.float32).transpose(2,0,1)/255. if gt.exists() else None
    return x, y

# quick sanity: flow base vs mean on a few train samples
def _psnr(a,b):
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 20*np.log10(1.0/np.sqrt(m))
_dirs = sorted(str(p) for p in Path(CFG['train_root']).iterdir() if p.is_dir())[:5]
for d in _dirs:
    x,y = build_inputs(d)
    if y is None: continue
    base = x[12:15]; mean = 0.5*(x[0:3]+x[3:6])
    print(os.path.basename(d), '| flow base %.2f dB | mean %.2f dB' % (_psnr(base,y), _psnr(mean,y)))""")

md("## 4. Dataset (random crops + h-flip augmentation)")
code("""class NVSData(Dataset):
    def __init__(self, root, crop=512, train=True, hflip=True):
        self.dirs = sorted(str(p) for p in Path(root).iterdir() if p.is_dir())
        self.crop, self.train, self.hflip = crop, train, hflip
    def __len__(self):
        return len(self.dirs)
    def __getitem__(self, i):
        x, y = build_inputs(self.dirs[i])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y) if y is not None else torch.zeros(3, *x.shape[1:])
        if self.train and self.crop:
            _, H, W = x.shape
            ch, cw = min(self.crop, H), min(self.crop, W)
            t = np.random.randint(0, H-ch+1); l = np.random.randint(0, W-cw+1)
            x = x[:, t:t+ch, l:l+cw]; y = y[:, t:t+ch, l:l+cw]
            if self.hflip and np.random.rand() < 0.5:
                x = torch.flip(x, [2]); y = torch.flip(y, [2])
        return x, y""")

md("## 5. Residual U-Net")
code("""def cba(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=16, base=48, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base*(2**i) for i in range(depth+1)]
        self.inc = cba(in_ch, chs[0])
        self.downs = nn.ModuleList([cba(chs[i], chs[i+1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[i+1], chs[i], 2, 2) for i in range(depth)])
        self.upc = nn.ModuleList([cba(chs[i+1], chs[i]) for i in range(depth)])
        self.outc = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)  # start = base
    def forward(self, x):
        base = x[:, 12:15]                # flow_interp
        h = self.inc(x); skips = [h]
        for d in range(self.depth):
            h = self.downs[d](self.pool(h)); skips.append(h)
        for d in reversed(range(self.depth)):
            h = self.ups[d](h); s = skips[d]
            dh, dw = s.shape[-2]-h.shape[-2], s.shape[-1]-h.shape[-1]
            if dh or dw: h = F.pad(h, [0, dw, 0, dh])
            h = self.upc[d](torch.cat([h, s], 1))
        return (base + self.outc(h)).clamp(0, 1)""")

md("## 6. Losses & metric (L1 + SSIM, PSNR)")
code("""def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    return (20*torch.log10(1.0/torch.sqrt(mse.clamp_min(1e-10)))).mean()

def _win(ch, size=11, sigma=1.5, device='cpu'):
    c = torch.arange(size, dtype=torch.float32, device=device) - size//2
    g = torch.exp(-(c**2)/(2*sigma**2)); g = (g/g.sum()).unsqueeze(0)
    return (g.t()@g).expand(ch,1,size,size).contiguous()

def ssim(pred, target):
    ch = pred.shape[1]; w = _win(ch, device=pred.device); p = w.shape[-1]//2
    mu1 = F.conv2d(pred, w, padding=p, groups=ch); mu2 = F.conv2d(target, w, padding=p, groups=ch)
    m1, m2, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = F.conv2d(pred*pred, w, padding=p, groups=ch)-m1
    s2 = F.conv2d(target*target, w, padding=p, groups=ch)-m2
    s12 = F.conv2d(pred*target, w, padding=p, groups=ch)-m12
    c1, c2 = 0.01**2, 0.03**2
    return (((2*m12+c1)*(2*s12+c2))/((m1+m2+c1)*(s1+s2+c2))).mean()

class RefinerLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.15):
        super().__init__(); self.w_l1, self.w_ssim = w_l1, w_ssim
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target); ss = ssim(pred, target)
        loss = self.w_l1*l1 + self.w_ssim*(1-ss)
        return loss, {'l1': l1.detach(), 'ssim': ss.detach(), 'psnr': psnr(pred.detach(), target)}""")

md("""## 7. Training

Each epoch logs **val refiner PSNR vs base flow PSNR** so you can see whether the
network actually beats the flow base (the gain). Best checkpoint -> `/kaggle/working/best.pt`.""")
code("""@torch.no_grad()
def evaluate(model, loader):
    model.eval(); pr=pb=n=0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pr += psnr(model(x), y).item()*x.size(0)
        pb += psnr(x[:,12:15], y).item()*x.size(0)
        n += x.size(0)
    model.train(); return pr/n, pb/n

def _raw(m):   # state_dict without the DataParallel 'module.' prefix
    return (m.module if isinstance(m, nn.DataParallel) else m).state_dict()

def train():
    full = NVSData(CFG['train_root'], crop=CFG['crop'], train=True, hflip=CFG['hflip'])
    nval = max(1, int(len(full)*CFG['val_frac'])); ntr = len(full)-nval
    tr, va = random_split(full, [ntr, nval], generator=torch.Generator().manual_seed(CFG['seed']))
    va.dataset = NVSData(CFG['train_root'], crop=0, train=False)   # full-frame val
    tl = DataLoader(tr, batch_size=CFG['batch'], shuffle=True, num_workers=CFG['workers'],
                    pin_memory=True, drop_last=True, persistent_workers=CFG['workers']>0)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=2)
    print(f'train={ntr} val={nval}')

    model = UNet(16, CFG['base']).to(device)
    if torch.cuda.device_count() > 1:
        print(f'using {torch.cuda.device_count()} GPUs via DataParallel')
        model = nn.DataParallel(model)
    print('params: %.2fM' % (sum(p.numel() for p in model.parameters())/1e6))
    crit = RefinerLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG['amp'])

    best = -1.0
    for ep in range(CFG['epochs']):
        model.train(); pbar = tqdm(tl, desc=f'epoch {ep}/{CFG["epochs"]-1}')
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=CFG['amp']):
                pred = model(x); loss, parts = crit(pred, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}", psnr=f"{parts['psnr'].item():.2f}")
        sch.step()
        vr, vb = evaluate(model, vl)
        print(f'epoch {ep}: val refiner={vr:.3f} dB | base flow={vb:.3f} dB | gain={vr-vb:+.3f}')
        torch.save({'model': _raw(model), 'cfg': CFG, 'epoch': ep, 'val': vr},
                   f"{CFG['out_dir']}/last.pt")
        if vr > best:
            best = vr
            torch.save({'model': _raw(model), 'cfg': CFG, 'epoch': ep, 'val': vr},
                       f"{CFG['out_dir']}/best.pt")
            print(f'  -> new best {best:.3f} dB saved')
    print('training done. best val PSNR = %.3f dB' % best)
    return model

model = train()""")

md("""## 8. Inference -> submission

Writes `submission/<sample_id>/pred.jpg` and zips it. If `test_root` is not present
(big-train may be train-only), add the dataset that contains `test/` and re-run cell 2.""")
code("""@torch.no_grad()
def make_submission(ckpt, test_root, out='/kaggle/working/submission'):
    ck = torch.load(ckpt, map_location=device)
    m = UNet(16, ck['cfg']['base']).to(device).eval(); m.load_state_dict(ck['model'])
    print('loaded', ckpt, '| val', round(ck.get('val', float('nan')), 3))
    dirs = sorted(p for p in Path(test_root).iterdir() if p.is_dir())
    os.makedirs(out, exist_ok=True)
    for sd in tqdm(dirs):
        x, _ = build_inputs(sd)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        _, _, H, W = x.shape
        ph, pw = ((H-1)//64+1)*64, ((W-1)//64+1)*64
        xp = F.pad(x, [0, pw-W, 0, ph-H], mode='replicate')
        with torch.cuda.amp.autocast(enabled=device=='cuda'):
            pred = m(xp)
        pred = pred[:, :, :H, :W].squeeze(0).clamp(0,1).cpu().numpy().transpose(1,2,0)
        od = Path(out)/sd.name; od.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred*255).round().astype(np.uint8)).save(od/'pred.jpg', quality=95)
    print('submission ->', out, '|', len(dirs), 'samples')
    return out

if CFG['test_root']:
    out = make_submission(f"{CFG['out_dir']}/best.pt", CFG['test_root'])
    import shutil
    shutil.make_archive('/kaggle/working/submission', 'zip', out)
    print('zipped -> /kaggle/working/submission.zip')
else:
    print('No test_root found — add the test dataset and re-run cell 2.')""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("kaggle_train.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_train.ipynb with", len(cells), "cells")
