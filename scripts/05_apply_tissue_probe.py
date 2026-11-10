"""Apply the nine-class tissue probe to a fresh tile grid for each slide."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    class _NoBar:
        def __init__(self, it=None, **kw): self._it = it
        def __iter__(self): return iter(self._it if self._it is not None else [])
        def update(self, n=1): pass
        def close(self): pass
    def tqdm(it=None, **kw): return _NoBar(it, **kw)

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# colours for the overlay; tumour deliberately the most salient
CLASS_COLORS = {
    "ADI": "#FFE082", "BACK": "#FFFFFF", "DEB": "#8D6E63", "LYM": "#E53935",
    "MUC": "#4DD0E1", "MUS": "#F48FB1", "NORM": "#43A047", "STR": "#9E9E9E",
    "TUM": "#1E88E5",
}
NCT_PATCH_PX = 224
NCT_MPP = 0.5
LABELS_CSV = "data/labels/msi_tcga.csv"
WSI_ROOT = "data/wsi"


# ---------------------------------------------------------------------------
def load_project_extractor(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[fatal] cannot find {p}; pass --extractor-py")
    spec = importlib.util.spec_from_file_location("proj_extract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_wsi(slide_id: str, root: Path) -> Optional[Path]:
    sid = slide_id.split(".")[0]
    hits = sorted(root.rglob(f"{sid}*.svs"))
    return hits[0] if hits else None


def slide_mpp(sl) -> Optional[float]:
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in sl.properties:
            try:
                return float(sl.properties[k])
            except Exception:
                pass
    return None


def tissue_mask(sl, downsample: int = 32) -> np.ndarray:
    """Otsu on HSV saturation, same approach as the project's tiling step."""
    import cv2
    w0, h0 = sl.dimensions
    tw, th = max(1, w0 // downsample), max(1, h0 // downsample)
    img = np.array(sl.get_thumbnail((tw, th)).convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, m = cv2.threshold(hsv[..., 1], 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


class TileReader:
    """Reads level-0 tiles of patch_l0 px and applies the probe's transform."""
    def __init__(self, svs_path: str, coords: np.ndarray, patch_l0: int, tf):
        self.svs_path, self.coords, self.patch_l0, self.tf = \
            svs_path, coords, patch_l0, tf
        self._sl = None

    def _slide(self):
        if self._sl is None:
            import openslide
            self._sl = openslide.OpenSlide(self.svs_path)
        return self._sl

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = self.coords[i]
        img = self._slide().read_region((int(x), int(y)), 0,
                                        (self.patch_l0, self.patch_l0)
                                        ).convert("RGB")
        return self.tf(img)


# ---------------------------------------------------------------------------
def process_slide(slide_id: str, model, clf, tf, device, args) -> Optional[dict]:
    import openslide
    import torch
    from torch.utils.data import DataLoader

    out_path = Path(args.out_dir) / f"{slide_id}.npz"
    if out_path.exists() and not args.overwrite:
        return {"slide_id": slide_id, "status": "skipped"}

    wsi = resolve_wsi(slide_id, Path(args.wsi_root))
    if wsi is None:
        logging.warning(f"[{slide_id}] no svs found")
        return None
    sl = openslide.OpenSlide(str(wsi))
    mpp = slide_mpp(sl)
    if mpp is None:
        logging.warning(f"[{slide_id}] no mpp metadata -> skip "
                        f"(these are the low-magnification archival scans)")
        sl.close()
        return None

    # tile geometry that reproduces the probe's training field of view
    patch_l0 = int(round(NCT_PATCH_PX * NCT_MPP / mpp))
    W, H = sl.dimensions
    mask = tissue_mask(sl, args.thumb_downsample)
    mh, mw = mask.shape
    sx, sy = mw / W, mh / H

    coords = []
    for y in range(0, H - patch_l0 + 1, patch_l0):
        for x in range(0, W - patch_l0 + 1, patch_l0):
            x1, y1 = int(x * sx), int(y * sy)
            x2, y2 = int((x + patch_l0) * sx), int((y + patch_l0) * sy)
            roi = mask[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
            if roi.size and roi.mean() >= args.min_tissue:
                coords.append((x, y))
    sl.close()
    if not coords:
        logging.warning(f"[{slide_id}] no tissue tiles")
        return None
    coords = np.asarray(coords, dtype=np.int32)

    t0 = time.time()
    ds = TileReader(str(wsi), coords, patch_l0, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, shuffle=False)
    feats = np.empty((len(ds), 1536), dtype=np.float32)
    cur = 0
    with torch.inference_mode():
        for batch in tqdm(dl, desc=f"{slide_id.split('-')[1]}-{slide_id.split('-')[2]}",
                          unit="batch", leave=False, disable=not args.progress):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(batch)
            out = out.float().cpu().numpy()
            feats[cur:cur + len(out)] = out
            cur += len(out)

    # the probe was fitted on L2-normalised features
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
    probs = clf.predict_proba(feats).astype(np.float32)
    pred = probs.argmax(1).astype(np.int8)
    pmax = probs.max(1).astype(np.float16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(coords_l0=coords, pred=pred, prob=pmax,
                   classes=np.array(CLASSES), patch_l0=np.int32(patch_l0),
                   mpp=np.float32(mpp), slide_w=np.int32(W), slide_h=np.int32(H))
    if args.save_probs:
        payload["probs"] = probs.astype(np.float16)
    np.savez_compressed(out_path, **payload)

    comp = {CLASSES[i]: int((pred == i).sum()) for i in range(len(CLASSES))}
    dt = time.time() - t0
    logging.info(f"[{slide_id}] {len(coords)} tiles, patch_l0={patch_l0}, "
                 f"{dt:.0f}s | " +
                 " ".join(f"{k}={100*v/len(pred):.0f}%"
                          for k, v in comp.items() if v > 0))

    if args.overlay:
        render_overlay(wsi, coords, pred, pmax, patch_l0, slide_id,
                       Path(args.out_dir) / "overlays")
    return {"slide_id": slide_id, "status": "ok", "n_tiles": len(coords),
            "seconds": round(dt, 1), **{f"frac_{k}": v / len(pred)
                                        for k, v in comp.items()}}


def render_overlay(wsi: Path, coords, pred, prob, patch_l0, slide_id, out_dir):
    """Thumbnail, tissue map, and a tumour-confidence view side by side."""
    import openslide
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb

    sl = openslide.OpenSlide(str(wsi))
    W, H = sl.dimensions
    tw = 1600
    th = int(round(H / W * tw))
    thumb = np.array(sl.get_thumbnail((tw, th)).convert("RGB"))
    sl.close()
    sx, sy = tw / W, th / H

    # paint the class map at thumbnail resolution
    cmap_img = np.ones((th, tw, 3), dtype=np.float32)
    tum_img = np.ones((th, tw, 3), dtype=np.float32)
    ti = CLASSES.index("TUM")
    for (x, y), p, pr in zip(coords, pred, prob):
        x0, y0 = int(x * sx), int(y * sy)
        x1 = min(tw, int((x + patch_l0) * sx) + 1)
        y1 = min(th, int((y + patch_l0) * sy) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        cmap_img[y0:y1, x0:x1] = to_rgb(CLASS_COLORS[CLASSES[p]])
        if p == ti:
            v = float(pr)
            tum_img[y0:y1, x0:x1] = (1 - v, 1 - v * 0.4, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    axes[0].imshow(thumb); axes[0].set_title("H&E", loc="left", weight="bold")
    axes[1].imshow(cmap_img)
    axes[1].imshow(thumb, alpha=0.35)
    axes[1].set_title("tissue class", loc="left", weight="bold")
    axes[2].imshow(tum_img)
    axes[2].imshow(thumb, alpha=0.30)
    axes[2].set_title("tumour confidence (darker = higher)",
                      loc="left", weight="bold")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="black", label=c)
               for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{slide_id} — tissue map from UNI2 + NCT-CRC linear probe "
                 f"(NOT yet validated on TCGA: check this by eye)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{slide_id}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[{slide_id}] overlay -> {out_dir / (slide_id + '.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels-csv", default=LABELS_CSV)
    ap.add_argument("--wsi-root", default=WSI_ROOT)
    ap.add_argument("--probe", default="data/tissue/probe.joblib")
    ap.add_argument("--uni-dir", default="weights/UNI2-h")
    ap.add_argument("--extractor-py", default=str(Path(__file__).with_name("03_extract_features.py")))
    ap.add_argument("--out-dir", default="data/tissue/maps")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--thumb-downsample", type=int, default=32)
    ap.add_argument("--save-probs", action="store_true",
                    help="store the full 9-class probability vector per tile")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress", action="store_true", default=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch, joblib
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        sys.exit("[fatal] CUDA unavailable")
    logging.info(f"device={device} ({torch.cuda.get_device_name(0)})")

    bundle = joblib.load(args.probe)
    clf = bundle["clf"]
    if list(bundle["classes"]) != CLASSES:
        sys.exit(f"[fatal] class order mismatch: {bundle['classes']}")
    logging.info(f"probe loaded from {args.probe} "
                 f"(L2-normalised={bundle.get('l2_normalised')})")

    mod = load_project_extractor(args.extractor_py)
    model = mod.build_uni2h(args.uni_dir, device)
    model.eval()
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mod.SlidePatchDataset.NORM_MEAN,
                             mod.SlidePatchDataset.NORM_STD),
    ])

    if args.all:
        targets = pd.read_csv(args.labels_csv)["slide_id"].tolist()
    elif args.slides:
        targets = args.slides
    else:
        sys.exit("give --slides <ids...> or --all")
    targets = [s for i, s in enumerate(targets)
               if i % args.shard_total == args.shard_idx]
    logging.info(f"{len(targets)} slides (shard {args.shard_idx}/{args.shard_total})")

    rows = []
    for s in tqdm(targets, desc="slides", unit="slide"):
        try:
            r = process_slide(s, model, clf, tf, device, args)
            if r:
                rows.append(r)
        except Exception as e:
            logging.error(f"[{s}] FAILED: {type(e).__name__}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        p = Path(args.out_dir) / f"_apply_log_shard{args.shard_idx}.csv"
        df.to_csv(p, index=False)
        logging.info(f"\nwrote {p}  ({len(df)} slides)")
        fr = [c for c in df.columns if c.startswith("frac_")]
        if fr:
            logging.info("\nmean tissue composition across slides:")
            for c in sorted(fr, key=lambda c: -df[c].mean()):
                logging.info(f"  {c.replace('frac_',''):<6} {100*df[c].mean():5.1f}%")


if __name__ == "__main__":
    main()
