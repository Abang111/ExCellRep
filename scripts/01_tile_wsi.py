"""Generate coordinate-only HDF5 tile lists from whole-slide images."""
import argparse
import os
import sys
from pathlib import Path
import multiprocessing as mp
from functools import partial

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import openslide
import cv2

TARGET_MPP = 0.5            # 20x equivalent
PATCH_SIZE = 256            # at target mpp
MIN_TISSUE_FRACTION = 0.5
THUMB_DOWNSAMPLE = 32       # for tissue mask


def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    """Return microns-per-pixel at level 0."""
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in slide.properties:
            try:
                return float(slide.properties[k])
            except (ValueError, TypeError):
                pass
    # fallback: assume 0.25 (40x), most TCGA Aperio slides
    return 0.25


def tissue_mask_from_thumb(slide: openslide.OpenSlide, downsample: int = 32) -> np.ndarray:
    """Otsu on saturation, morphological cleanup. Returns boolean mask at thumbnail resolution."""
    w0, h0 = slide.dimensions
    tw, th = max(1, w0 // downsample), max(1, h0 // downsample)
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    img = np.array(thumb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    _, m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return m > 0  # (th, tw)


def find_svs_for_slide(wsi_root: Path, slide_id: str) -> Path | None:
    """GDC structure: each UUID dir contains one .svs whose name starts with slide_id."""
    # walk all subdirs, slow on first call but cached at FS level
    for sub in wsi_root.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*.svs"):
            if f.name.startswith(slide_id):
                return f
    return None


def tile_one_slide(slide_id: str, wsi_root: str, out_dir: str) -> dict:
    out_path = Path(out_dir) / f"{slide_id}.h5"
    if out_path.exists():
        return {"slide_id": slide_id, "status": "skipped", "n_patches": -1}

    svs_path = find_svs_for_slide(Path(wsi_root), slide_id)
    if svs_path is None:
        return {"slide_id": slide_id, "status": "missing_svs", "n_patches": 0}

    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        return {"slide_id": slide_id, "status": f"openslide_error:{e}", "n_patches": 0}

    try:
        mpp = get_slide_mpp(slide)
        # patch size at level 0 to achieve target mpp
        scale = TARGET_MPP / mpp     # e.g. 0.5 / 0.25 = 2.0  ->  patch_l0 = 512
        patch_l0 = int(round(PATCH_SIZE * scale))
        if patch_l0 < 64:
            return {"slide_id": slide_id, "status": "mpp_too_high", "n_patches": 0}

        mask = tissue_mask_from_thumb(slide, THUMB_DOWNSAMPLE)
        mh, mw = mask.shape
        w0, h0 = slide.dimensions
        # mapping from level0 -> mask
        sx = mw / w0
        sy = mh / h0

        coords = []
        # step over level-0 grid
        for y in range(0, h0 - patch_l0 + 1, patch_l0):
            for x in range(0, w0 - patch_l0 + 1, patch_l0):
                # corresponding mask roi
                x1, y1 = int(x * sx), int(y * sy)
                x2, y2 = int((x + patch_l0) * sx), int((y + patch_l0) * sy)
                roi = mask[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
                if roi.size == 0:
                    continue
                if roi.mean() >= MIN_TISSUE_FRACTION:
                    coords.append((x, y))

        if not coords:
            slide.close()
            return {"slide_id": slide_id, "status": "no_tissue", "n_patches": 0}

        coords = np.asarray(coords, dtype=np.int32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("coords", data=coords, compression="gzip")
            f.attrs["patch_size_level0"] = patch_l0
            f.attrs["patch_size_target"] = PATCH_SIZE
            f.attrs["target_mpp"] = TARGET_MPP
            f.attrs["slide_mpp"] = mpp
            f.attrs["n_patches"] = len(coords)
            f.attrs["slide_id"] = slide_id
            f.attrs["svs_path"] = str(svs_path)
        slide.close()
        return {"slide_id": slide_id, "status": "ok", "n_patches": len(coords)}
    except Exception as e:
        slide.close()
        return {"slide_id": slide_id, "status": f"error:{e}", "n_patches": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--wsi-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    slides = df["slide_id"].dropna().unique().tolist()
    print(f"[tile] {len(slides)} slides to process, workers={args.workers}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fn = partial(tile_one_slide, wsi_root=args.wsi_root, out_dir=args.out_dir)

    rows = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, slides), 1):
            rows.append(res)
            if i % 10 == 0 or i == len(slides):
                print(f"  [{i}/{len(slides)}] last={res['slide_id']} "
                      f"status={res['status']} n={res['n_patches']}", flush=True)

    log = pd.DataFrame(rows)
    log_path = Path(args.out_dir) / "_tile_log.csv"
    log.to_csv(log_path, index=False)
    print(f"[done] log -> {log_path}")
    print(log["status"].value_counts())


if __name__ == "__main__":
    main()
