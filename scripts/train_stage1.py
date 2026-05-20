"""Stage 1 — train the VO reliability model.

Predicts log1p(|VO twist error|) per timestep from the camera image + scalar
features. After training, runs inference over every VO sample and writes
artifacts/stage1/reliability.npy (311808, 2), which Stage 2 consumes.

  .venv/bin/python scripts/train_stage1.py [--no-image] [--smoke] [--epochs N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vo import data
from vo.dataset import N_SCALAR, Stage1Dataset, bc_runs, prepare_runs
from vo.models import ReliabilityNet

OUT = "artifacts/stage1"


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def evaluate(model, loader, device, use_image):
    model.eval()
    se, n, preds, tgts = 0.0, 0, [], []
    with torch.no_grad():
        for img, feat, tgt in loader:
            img, feat, tgt = img.to(device), feat.to(device), tgt.to(device)
            p = model(img if use_image else None, feat)
            se += torch.sum((p - tgt) ** 2).item()
            n += tgt.numel()
            preds.append(p.cpu().numpy())
            tgts.append(tgt.cpu().numpy())
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    corr = [float(np.corrcoef(preds[:, k], tgts[:, k])[0, 1]) for k in (0, 1)]
    return se / n, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-image", action="store_true", help="scalar features only")
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    use_image = not a.no_image

    os.makedirs(OUT, exist_ok=True)
    fh = open(f"{OUT}/log.txt", "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} use_image={use_image} smoke={a.smoke}", fh)

    scalars = data.load_scalars()
    splits = data.load_splits()
    runs = bc_runs(scalars)
    if a.smoke:
        runs = runs[:3] + [r for r in runs if splits[r] == "val"][:1]
        a.epochs = 2
    log(f"loading {len(runs)} runs (images={use_image}) ...", fh)
    t0 = time.time()
    d = prepare_runs(scalars, runs, with_images=use_image)
    log(f"loaded {len(d['feat'])} samples in {time.time()-t0:.0f}s", fh)

    valid = np.isfinite(d["target"]).all(1)
    run_of = d["run_of"]
    tr = np.where(valid & np.isin(run_of, [r for r in runs if splits[r] == "train"]))[0]
    va = np.where(valid & np.isin(run_of, [r for r in runs if splits[r] == "val"]))[0]
    log(f"train rows={len(tr)}  val rows={len(va)}", fh)

    common = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    tl = DataLoader(Stage1Dataset(d["feat"], d["target"], d["images"], tr),
                    shuffle=True, drop_last=True, **common)
    vl = DataLoader(Stage1Dataset(d["feat"], d["target"], d["images"], va), **common)

    model = ReliabilityNet(N_SCALAR, use_image).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    lossfn = torch.nn.SmoothL1Loss()

    best = float("inf")
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for img, feat, tgt in tl:
            img, feat, tgt = img.to(device), feat.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                p = model(img if use_image else None, feat)
                loss = lossfn(p, tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        vmse, vcorr = evaluate(model, vl, device, use_image)
        log(f"epoch {ep:3d}/{a.epochs}  train_loss={tot/len(tl):.4f}  "
            f"val_mse={vmse:.4f}  corr(v,w)=({vcorr[0]:.3f},{vcorr[1]:.3f})  "
            f"{time.time()-t0:.0f}s", fh)
        if vmse < best:
            best = vmse
            torch.save({"state": model.state_dict(), "use_image": use_image},
                       f"{OUT}/model.pt")
            log(f"  saved best (val_mse={best:.4f})", fh)

    # ---- inference: reliability for every VO sample, written full-length ----
    log("inferring reliability over all VO samples ...", fh)
    ckpt = torch.load(f"{OUT}/model.pt")
    model.load_state_dict(ckpt["state"])
    full = DataLoader(Stage1Dataset(d["feat"], d["target"], d["images"],
                                    np.arange(len(d["feat"]))), **common)
    model.eval()
    out = []
    with torch.no_grad():
        for img, feat, _ in full:
            img, feat = img.to(device), feat.to(device)
            out.append(model(img if use_image else None, feat).cpu().numpy())
    pred = np.concatenate(out)
    rel = np.zeros((len(scalars["run_id"]), 2), np.float32)
    rel[d["gindex"]] = pred
    np.save(f"{OUT}/reliability.npy", rel)
    json.dump({"best_val_mse": best, "use_image": use_image,
               "n_samples": int(len(pred))}, open(f"{OUT}/summary.json", "w"),
              indent=2)
    log(f"done. reliability.npy written ({len(pred)} samples).", fh)
    fh.close()


if __name__ == "__main__":
    main()
