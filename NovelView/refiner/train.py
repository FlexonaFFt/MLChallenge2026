"""Train the geometric refiner with verbose, observable logging.

Observability (so you can watch progress live on the server):
  - stdout + rotating-ish logfile via logging  (follow with `docker logs -f <name>`)
  - tqdm progress bar per epoch with live loss/PSNR
  - TensorBoard scalars (loss parts, PSNR, lr) + image grids (warp/pred/GT)
      -> `tensorboard --logdir runs --host 0.0.0.0 --port 6006`
  - periodic checkpoints (last + best-by-val-PSNR)
  - a heartbeat line every N steps with throughput + ETA

Run:  python -u train.py --data /data --out runs/exp1 --cache-dir /cache
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geo_warp import IN_CHANNELS  # noqa: E402
from refiner.dataset import SampleDataset  # noqa: E402
from refiner.losses import RefinerLoss, psnr  # noqa: E402
from refiner.model import UNetRefiner  # noqa: E402


def setup_logger(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset root (has train/ test/)")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--perceptual", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=1, help="epochs")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", default=None)
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, writer, step, logger):
    model.eval()
    tot, n = 0.0, 0
    logged_img = False
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        tot += psnr(pred, y).item() * x.size(0)
        n += x.size(0)
        if not logged_img and writer is not None:
            base = model._base_blend(x)
            grid = torch.cat([base[:2], pred[:2], y[:2]], dim=0).clamp(0, 1)
            writer.add_images("val/base_pred_gt", grid, step)
            logged_img = True
    val_psnr = tot / max(n, 1)
    logger.info(f"  [val] PSNR={val_psnr:.3f} dB over {n} samples")
    if writer is not None:
        writer.add_scalar("val/psnr", val_psnr, step)
    model.train()
    return val_psnr


def main():
    args = parse_args()
    out = Path(args.out)
    logger = setup_logger(out)
    writer = SummaryWriter(out / "tb")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device} | torch={torch.__version__} | args={vars(args)}")

    full = SampleDataset(args.data, args.split, crop=args.crop, cache_dir=args.cache_dir, train=True)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    # validation uses full frames
    val_ds.dataset = SampleDataset(
        args.data, args.split, crop=0, cache_dir=args.cache_dir, train=False
    )
    logger.info(f"samples: train={n_train} val={n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = UNetRefiner(in_ch=IN_CHANNELS, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"model: UNetRefiner base={args.base} | {n_params:.2f}M params")

    crit = RefinerLoss(w_perc=0.05 if args.perceptual else 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    start_epoch, best = 0, -1.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", -1.0)
        logger.info(f"resumed from {args.resume} at epoch {start_epoch} (best={best:.3f})")

    step = start_epoch * len(train_loader)
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs-1}", dynamic_ncols=True)
        run = {"loss": 0.0, "psnr": 0.0}
        for it, (x, y, _) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                pred = model(x)
                loss, parts = crit(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run["loss"] += loss.item()
            run["psnr"] += parts["psnr"].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{parts['psnr'].item():.2f}")

            if step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/psnr", parts["psnr"].item(), step)
                writer.add_scalar("train/l1", parts["l1"].item(), step)
                writer.add_scalar("train/ssim", parts["ssim"].item(), step)
                writer.add_scalar("lr", sched.get_last_lr()[0], step)
                ips = (step - start_epoch * len(train_loader) + 1) * args.batch / (time.time() - t_start)
                eta_h = (args.epochs - epoch) * len(train_loader) * args.batch / max(ips, 1e-6) / 3600
                logger.info(
                    f"e{epoch} it{it}/{len(train_loader)} step{step} | "
                    f"loss={loss.item():.4f} psnr={parts['psnr'].item():.2f} "
                    f"l1={parts['l1'].item():.4f} ssim={parts['ssim'].item():.3f} | "
                    f"{ips:.1f} img/s | ETA~{eta_h:.1f}h"
                )
            step += 1

        sched.step()
        nb = len(train_loader)
        logger.info(
            f"epoch {epoch} done | train loss={run['loss']/nb:.4f} psnr={run['psnr']/nb:.2f}"
        )

        if (epoch + 1) % args.val_every == 0:
            val_psnr = validate(model, val_loader, device, writer, step, logger)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "epoch": epoch, "best": best, "args": vars(args)}
            torch.save(ck, out / "last.pt")
            if val_psnr > best:
                best = val_psnr
                ck["best"] = best
                torch.save(ck, out / "best.pt")
                logger.info(f"  -> new best {best:.3f} dB saved to {out/'best.pt'}")

    logger.info(f"training finished. best val PSNR={best:.3f} dB. checkpoints in {out}")
    writer.close()


if __name__ == "__main__":
    main()
