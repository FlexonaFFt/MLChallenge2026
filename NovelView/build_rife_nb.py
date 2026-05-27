"""Generate kaggle_rife.ipynb — RIFE-based predictor (no training) for Task B.

Compares mean / DIS-flow / RIFE / ensemble on a held-out val subset, picks the best,
then writes the submission. Needs the RIFE code+weights uploaded as a Kaggle dataset.
"""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Task B — RIFE predictor (no training)

The post-hoc refiner had **zero gain** (it regresses to its input), so the lever is a
**better interpolation base**. RIFE is a learned frame-interpolation network that
jointly estimates motion + synthesizes the middle frame — much stronger than DIS on
large motion (the `delta_s=2 s` cases that sank our score).

This notebook is **training-free**: it compares mean / DIS / RIFE / ensemble on a val
subset, picks the best, and writes the submission.

**Accelerator: GPU T4 ×2** (P100 unsupported). Internet off — RIFE comes from an
uploaded dataset.

### One-time setup: upload RIFE as a Kaggle dataset
On your machine:
```
git clone https://github.com/hzwer/ECCV2022-RIFE
# put the HDv3 weights into ECCV2022-RIFE/train_log/ (flownet.pkl + *.py)
# (you already have these in NovelView/baseline_ensemble/train_log/)
```
Zip the `ECCV2022-RIFE` folder, create a Kaggle Dataset from it, then **+ Add Input**
it here. The cell below auto-finds it.""")

md("## 1. Imports, config, dataset paths")
code("""import os, glob, json, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print('device:', device, '| torch:', torch.__version__)

def find_split_root(want):
    for m in sorted(glob.glob(f'/kaggle/input/**/{want}', recursive=True)):
        if os.path.isdir(m):
            return os.path.dirname(m)
    return None

TRAIN_ROOT = (find_split_root('train') or '') + '/train'
_eb = find_split_root('test')
TEST_ROOT = (_eb + '/test') if _eb else None
print('TRAIN_ROOT:', TRAIN_ROOT, '| exists:', os.path.isdir(TRAIN_ROOT))
print('TEST_ROOT :', TEST_ROOT)""")

md("## 2. Load RIFE from the uploaded dataset")
code("""# locate the ECCV2022-RIFE repo (folder that has train_log/RIFE_HDv3.py)
rife_dir = None
for p in glob.glob('/kaggle/input/**/train_log/RIFE_HDv3.py', recursive=True):
    rife_dir = str(Path(p).parent.parent)   # .../ECCV2022-RIFE
    break
assert rife_dir, 'RIFE repo not found — add the dataset containing ECCV2022-RIFE/train_log/'
print('RIFE repo:', rife_dir)

import sys
sys.path.insert(0, rife_dir)
from train_log.RIFE_HDv3 import Model     # noqa: E402
rife = Model()
rife.load_model(os.path.join(rife_dir, 'train_log'), -1)
rife.eval()
print('RIFE loaded')""")

md("## 3. Predictors: mean / DIS-flow / RIFE")
code("""def read_img(p):
    return np.array(Image.open(p).convert('RGB'))

def get_alpha(meta):
    ts = meta.get('timestamps_ns')
    if ts and all(k in ts for k in ('t0','t1','target')):
        return float((ts['target']-ts['t0'])/(ts['t1']-ts['t0']))
    return 0.5

_DIS = None
def _warp(img, dx, dy):
    H, W = img.shape[:2]; ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, (xs+dx).astype(np.float32), (ys+dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def dis_interp(i0, i1, a):
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = _DIS.calc(cv2.cvtColor(i0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY), None)
    dx, dy = fl[..., 0], fl[..., 1]
    w0 = _warp(i0.astype(np.float32), -a*dx, -a*dy)
    w1 = _warp(i1.astype(np.float32), (1-a)*dx, (1-a)*dy)
    return ((1-a)*w0 + a*w1)            # float 0..255

@torch.no_grad()
def rife_interp(i0, i1):
    h, w = i0.shape[:2]
    ph, pw = ((h-1)//64+1)*64, ((w-1)//64+1)*64
    to_t = lambda x: torch.from_numpy(x).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
    t0 = F.pad(to_t(i0), (0, pw-w, 0, ph-h))
    t1 = F.pad(to_t(i1), (0, pw-w, 0, ph-h))
    out = rife.inference(t0, t1)        # mid frame (alpha=0.5)
    return (out[0, :, :h, :w].permute(1,2,0).cpu().numpy()*255)   # float 0..255

def load_pair(sd):
    sd = Path(sd); meta = json.loads((sd/'meta.json').read_text())
    cam = meta['target_camera']; a = get_alpha(meta)
    i0 = read_img(sd/'input'/'t0'/f'{cam}.jpg')
    i1 = read_img(sd/'input'/'t1'/f'{cam}.jpg')
    gt = sd/'target'/f'{cam}.jpg'
    y = read_img(gt) if gt.exists() else None
    return i0, i1, a, y""")

md("""## 4. Compare methods on a val subset

We score mean / DIS / RIFE / ensembles on N held-out train samples and pick the best.
This is free (no submit needed) and tells us which method to ship.""")
code("""def psnr(a, b):
    a = np.clip(a, 0, 255).astype(np.float64); b = b.astype(np.float64)
    m = np.mean((a-b)**2); return 99.0 if m < 1e-9 else 20*np.log10(255/np.sqrt(m))

N_VAL = 80
val_dirs = sorted(str(p) for p in Path(TRAIN_ROOT).iterdir() if p.is_dir())
random.shuffle(val_dirs); val_dirs = val_dirs[:N_VAL]

acc = {k: 0.0 for k in ['mean','dis','rife','0.6dis+0.4rife','0.5dis+0.5rife']}
n = 0
for sd in tqdm(val_dirs):
    i0, i1, a, y = load_pair(sd)
    if y is None: continue
    mean = 0.5*(i0.astype(np.float32)+i1.astype(np.float32))
    dis = dis_interp(i0, i1, a)
    rf = rife_interp(i0, i1)
    acc['mean'] += psnr(mean, y)
    acc['dis'] += psnr(dis, y)
    acc['rife'] += psnr(rf, y)
    acc['0.6dis+0.4rife'] += psnr(0.6*dis+0.4*rf, y)
    acc['0.5dis+0.5rife'] += psnr(0.5*dis+0.5*rf, y)
    n += 1

print(f'\\nval PSNR over {n} samples:')
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f'  {k:18s} {v/n:.3f} dB')
BEST = max(acc, key=acc.get)
print('\\nBEST method:', BEST)""")

md("## 5. Build the submission with the best method")
code("""def predict(i0, i1, a, method):
    if method == 'mean':  return 0.5*(i0.astype(np.float32)+i1.astype(np.float32))
    if method == 'dis':   return dis_interp(i0, i1, a)
    if method == 'rife':  return rife_interp(i0, i1)
    if method == '0.6dis+0.4rife': return 0.6*dis_interp(i0,i1,a)+0.4*rife_interp(i0,i1)
    if method == '0.5dis+0.5rife': return 0.5*dis_interp(i0,i1,a)+0.5*rife_interp(i0,i1)
    raise ValueError(method)

def make_submission(test_root, method, out='/kaggle/working/submission'):
    dirs = sorted(p for p in Path(test_root).iterdir() if p.is_dir())
    os.makedirs(out, exist_ok=True)
    for sd in tqdm(dirs):
        i0, i1, a, _ = load_pair(sd)
        pred = np.clip(predict(i0, i1, a, method), 0, 255).round().astype(np.uint8)
        od = Path(out)/sd.name; od.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).save(od/'pred.jpg', quality=95)
    print('submission ->', out, '|', len(dirs), 'samples')
    return out

if TEST_ROOT and os.path.isdir(TEST_ROOT):
    out = make_submission(TEST_ROOT, BEST)
    import shutil
    shutil.make_archive('/kaggle/working/submission', 'zip', out)
    print('zipped -> /kaggle/working/submission.zip  (method:', BEST, ')')
else:
    print('No TEST_ROOT — add the test dataset and re-run cell 1.')""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("kaggle_rife.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_rife.ipynb with", len(cells), "cells")
