"""Provide shared nuclear contour and distance routines for phenotype and graph construction."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# progress bar (graceful fallback)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _NoBar:
        """Minimal stand-in supporting both `for x in tqdm(it)` and
        `bar = tqdm(total=n); bar.update(1); bar.close()`."""
        def __init__(self, iterable=None, **kw):
            self._it = iterable
        def __iter__(self):
            return iter(self._it if self._it is not None else [])
        def update(self, n=1):
            pass
        def close(self):
            pass

    def tqdm(iterable=None, **kw):
        return _NoBar(iterable, **kw)

# ---- config defaults -------------------------------------------------------
PINST_TO_L0 = 2.0                       # EXACT, forensic-verified. Do not change.
SIGMAS_UM = [5.0, 10.0, 20.0]           # Gaussian scales; 3*sigma must be <= cutoff
DIST_BINS_UM = [0, 5, 10, 20, 40, 60]   # distance-resolved bins
DEFAULT_CUTOFF_UM = 60.0
CONTOUR_MAX_PTS = 20                    # points kept per nucleus contour
DEFAULT_CHUNK = 50_000                  # pairs per distance batch (~80 MB @ K=20)
LABELS_CSV = "data/labels/msi_tcga.csv"
WSI_ROOT = "data/wsi"
CELLS_ROOT = "data/cells"

CLASS_NAMES = ["neutrophil", "epithelial", "lymphocyte", "plasma",
               "eosinophil", "connective", "mitosis"]
EPI = 1   # 0-based epithelial
LYM = 2   # 0-based lymphocyte


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def short_id(name: str) -> str:
    return name.split(".")[0]


def resolve_wsi(slide_id: str, wsi_root: Path) -> Optional[Path]:
    sid = short_id(slide_id)
    hits = sorted(wsi_root.rglob(f"{sid}*.svs"))
    return hits[0] if hits else None


def resolve_cells_dir(slide_id: str, cells_root: Path) -> Optional[Path]:
    sid = short_id(slide_id)
    for j in cells_root.rglob("class_inst.json"):
        if sid in j.parent.name:
            return j.parent
    return None


def read_slide_mpp(wsi: Path) -> float:
    import openslide
    sl = openslide.OpenSlide(str(wsi))
    mpp = None
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in sl.properties:
            try:
                mpp = float(sl.properties[k]); break
            except Exception:
                pass
    sl.close()
    if mpp is None:
        raise RuntimeError(f"mpp not found for {wsi.name}")
    return mpp


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
def load_centroids(cells_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (inst_ids [N], xy_pinst [N,2] as (x,y), cls0 [N])."""
    with open(cells_dir / "class_inst.json") as f:
        d = json.load(f)
    ids = np.empty(len(d), dtype=np.int64)
    xy = np.empty((len(d), 2), dtype=np.float32)
    cls = np.empty(len(d), dtype=np.int64)
    for i, (k, v) in enumerate(d.items()):
        ids[i] = int(k)
        cls[i] = v[0]
        xy[i, 1] = v[1][0]   # y
        xy[i, 0] = v[1][1]   # x
    cls0 = np.clip(cls - 1, 0, 6)
    return ids, xy, cls0


def open_pinst(cells_dir: Path):
    import zarr
    zp = cells_dir / "pinst_pp.zip"
    try:
        store = zarr.ZipStore(str(zp), mode="r")
        arr = zarr.open(store, mode="r")
    except Exception:
        store = zarr.storage.ZipStore(str(zp), mode="r")
        arr = zarr.open(store, mode="r")
    return arr, store


# ---------------------------------------------------------------------------
# FAST contour extraction: bbox per instance, contour on the small crop
# ---------------------------------------------------------------------------
def extract_contours(
    arr, inst_ids: np.ndarray, xy_pinst: np.ndarray,
    block: int = 4096, pad: int = 64, max_pts: int = CONTOUR_MAX_PTS,
    show_progress: bool = True, tag: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        pts   : [N, max_pts, 2] float32, contour points in pinst px
                (padded by repeating the last point; repeats don't change a min)
        valid : [N] bool
    Blocks are read with `pad` overlap; a nucleus is claimed by the block whose
    CORE contains its centroid, so nuclei on block seams are never fragmented
    (nucleus diameter << pad).
    """
    import cv2
    from scipy.ndimage import find_objects

    ph, pw = arr.shape[:2]
    N = len(inst_ids)
    pts = np.zeros((N, max_pts, 2), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    # id -> row lookup
    max_id = int(inst_ids.max())
    id2row = np.full(max_id + 2, -1, dtype=np.int64)
    id2row[inst_ids] = np.arange(N)

    nbx = (pw + block - 1) // block
    nby = (ph + block - 1) // block
    blocks = [(bx, by) for by in range(nby) for bx in range(nbx)]

    it = tqdm(blocks, desc=f"{tag} contours", unit="blk",
              disable=not show_progress, leave=False)
    for bx, by in it:
        x0, y0 = bx * block, by * block
        x1, y1 = min(x0 + block, pw), min(y0 + block, ph)
        # padded read
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(pw, x1 + pad), min(ph, y1 + pad)
        sub = np.asarray(arr[py0:py1, px0:px1])
        if sub.max() == 0:
            continue

        slices = find_objects(sub)          # indexed by label-1
        for lbl_m1, sl in enumerate(slices):
            if sl is None:
                continue
            iid = lbl_m1 + 1
            if iid > max_id:
                continue
            row = id2row[iid]
            if row < 0 or valid[row]:
                continue
            # claim only if the bbox centre falls in this block's CORE
            cy = (sl[0].start + sl[0].stop) * 0.5 + py0
            cx = (sl[1].start + sl[1].stop) * 0.5 + px0
            if not (x0 <= cx < x1 and y0 <= cy < y1):
                continue
            crop = (sub[sl] == iid).astype(np.uint8)
            cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_TC89_L1)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32)
            c[:, 0] += sl[1].start + px0
            c[:, 1] += sl[0].start + py0
            if len(c) >= max_pts:
                idx = np.linspace(0, len(c) - 1, max_pts).astype(int)
                c = c[idx]
            else:
                c = np.vstack([c, np.repeat(c[-1:], max_pts - len(c), axis=0)])
            pts[row] = c
            valid[row] = True
    return pts, valid


# ---------------------------------------------------------------------------
# Vectorised boundary distance over pair chunks (CPU numpy or GPU torch)
# ---------------------------------------------------------------------------
def boundary_distance_chunked(
    pts: np.ndarray, pairs: np.ndarray, pinst_mpp: float,
    device: str = "cpu", chunk: int = DEFAULT_CHUNK,
    show_progress: bool = True, tag: str = "",
) -> np.ndarray:
    """
    Min contour-to-contour distance (um) for each pair. pts: [N,K,2] px.

    Memory: the dominant term is the [chunk, K, K] distance matrix, i.e.
    chunk * K * K * 4 bytes (K=20 -> ~1.6 kB per pair). chunk=50k is ~80 MB,
    which fits comfortably even on a card with only ~2 GB free. On CUDA OOM
    the chunk is halved and retried automatically.

    The pair index stays on the CPU; only the small per-chunk slice is moved
    to the device (moving all ~28M pairs would cost ~450 MB by itself).
    """
    P = len(pairs)
    out = np.empty(P, dtype=np.float32)
    if P == 0:
        return out
    use_torch = device.startswith("cuda")
    if use_torch:
        import torch
        dev = torch.device(device)
        T = torch.from_numpy(pts).to(dev)                 # [N,K,2]  (small)

    s = 0
    cur = int(chunk)
    total_est = (P + cur - 1) // cur
    bar = tqdm(total=total_est, desc=f"{tag} pair-dist", unit="chunk",
               disable=not show_progress, leave=False)
    while s < P:
        e = min(s + cur, P)
        try:
            if use_torch:
                import torch
                pi = torch.from_numpy(pairs[s:e].astype(np.int64)).to(dev)
                a = T[pi[:, 0]]                            # [C,K,2]
                b = T[pi[:, 1]]
                d = torch.cdist(
                    a, b, p=2,
                    compute_mode="donot_use_mm_for_euclid_dist")  # [C,K,K]
                out[s:e] = d.amin(dim=(1, 2)).detach().cpu().numpy()
                del pi, a, b, d
            else:
                a = pts[pairs[s:e, 0]]
                b = pts[pairs[s:e, 1]]
                d2 = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)
                out[s:e] = np.sqrt(d2.min(axis=(1, 2)))
        except Exception as ex:
            is_oom = "out of memory" in str(ex).lower()
            if is_oom and cur > 2000:
                cur = max(2000, cur // 2)
                logging.warning(f"[{tag}] CUDA OOM -> retrying with chunk={cur}")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
        s = e
        bar.update(1)
    bar.close()
    return out * np.float32(pinst_mpp)


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------
def build_for_slide(slide_id: str, msi: str, out_root: Path, cutoff_um: float,
                    device: str, save_graphs: bool, graph_cutoff_um: float,
                    show_progress: bool, chunk: int = DEFAULT_CHUNK,
                    contour_cache: str = "", radius_pct: float = 99.9
                    ) -> Optional[dict]:
    t0 = time.time()
    tag = short_id(slide_id).replace("TCGA-", "")
    wsi = resolve_wsi(slide_id, Path(WSI_ROOT))
    cells_dir = resolve_cells_dir(slide_id, Path(CELLS_ROOT))
    if wsi is None or cells_dir is None:
        logging.warning(f"[{slide_id}] missing wsi or cells dir; skip")
        return None

    mpp_l0 = read_slide_mpp(wsi)
    pinst_mpp = mpp_l0 * PINST_TO_L0
    cutoff_px = cutoff_um / pinst_mpp

    inst_ids, xy_pinst, cls0 = load_centroids(cells_dir)
    n = len(inst_ids)
    is_epi_arr = cls0 == EPI
    is_lym_arr = cls0 == LYM
    n_epi = int(is_epi_arr.sum()); n_lym = int(is_lym_arr.sum())
    logging.info(f"[{tag}] {msi} | cells={n} | pinst mpp={pinst_mpp:.4f} | "
                 f"cutoff={cutoff_um}um={cutoff_px:.1f}px | epi={n_epi} lym={n_lym}")

    # ---- contours (fast) ----
    # Contour extraction is ~60 s per slide and depends only on the
    # segmentation, so reuse the cache written by the compartment analysis.
    cache = (Path(contour_cache) / f"{slide_id}.npz"
             if contour_cache else None)
    pts = valid = None
    if cache is not None and cache.exists():
        try:
            zc = np.load(cache)
            if int(zc["n"]) == n:
                pts, valid = zc["pts"], zc["valid"]
                logging.info(f"[{tag}] contours from cache")
        except Exception:
            pts = valid = None
    if pts is None:
        arr, store = open_pinst(cells_dir)
        pts, valid = extract_contours(arr, inst_ids, xy_pinst,
                                      show_progress=show_progress, tag=tag)
        try:
            store.close()
        except Exception:
            pass
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, pts=pts.astype(np.float32), valid=valid,
                     n=np.int64(n))
    logging.info(f"[{tag}] contours: {int(valid.sum())}/{n} "
                 f"({100*valid.mean():.1f}%)  [{time.time()-t0:.0f}s]")

    # ---- per-nucleus radius, needed to size the candidate search correctly ----
    # Boundary distance is always SMALLER than centroid distance, by up to the
    # sum of the two nuclear radii. Querying the KD-tree at the boundary cutoff
    # therefore misses pairs: two nuclei of radius 5 um whose contours are 55 um
    # apart have centroids 65 um apart and would never be considered. The loss
    # is concentrated in the outermost distance band, which is exactly the band
    # used as a negative control, and it is differential if nuclear size differs
    # between the groups being compared.
    d2c = ((pts - xy_pinst[:, None, :]) ** 2).sum(-1)
    radius_px = np.sqrt(d2c.max(axis=1)).astype(np.float32)
    radius_px[~valid] = 0.0
    rv = radius_px[valid]

    # The search radius must be widened by twice a nuclear radius, but using
    # the MAXIMUM lets one segmentation artefact set it for the whole slide:
    # a merged blob of radius 26 um (a real nucleus is 3-6) pushes the search
    # to 112 um, which triples the candidate count and exhausts memory. A high
    # percentile is used instead, and the nuclei above it are counted so the
    # approximation is visible rather than silent.
    r_cap = float(np.percentile(rv, radius_pct)) if rv.size else 0.0
    n_over = int((rv > r_cap).sum())
    search_px = cutoff_px + 2.0 * r_cap

    from scipy.spatial import cKDTree
    tree = cKDTree(xy_pinst)
    pairs = tree.query_pairs(r=search_px, output_type="ndarray")
    keep = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    pairs = pairs[keep]
    n_wide = len(pairs)

    # exact lower bound on boundary distance, applied in chunks so the
    # intermediate arrays stay small
    keep = np.empty(len(pairs), dtype=bool)
    n_missed_by_old = 0
    STEP = 5_000_000
    for s0 in range(0, len(pairs), STEP):
        s1 = min(s0 + STEP, len(pairs))
        a = pairs[s0:s1, 0]; b = pairs[s0:s1, 1]
        dx = xy_pinst[a, 0] - xy_pinst[b, 0]
        dy = xy_pinst[a, 1] - xy_pinst[b, 1]
        dcs = np.sqrt(dx * dx + dy * dy)
        k = (dcs - radius_px[a] - radius_px[b]) <= cutoff_px
        keep[s0:s1] = k
        n_missed_by_old += int((dcs[k] > cutoff_px).sum())
        del dx, dy, dcs, k
    pairs = pairs[keep]
    del keep

    logging.info(f"[{tag}] nuclear radius um: median "
                 f"{np.median(rv) * pinst_mpp:.2f}, p{radius_pct} "
                 f"{r_cap * pinst_mpp:.2f}, max {rv.max() * pinst_mpp:.2f} "
                 f"({n_over} nuclei above the cap) | search "
                 f"{search_px * pinst_mpp:.1f} um")
    logging.info(f"[{tag}] candidate pairs: {len(pairs)} "
                 f"(wide search {n_wide}, {n_missed_by_old} recovered that a "
                 f"centroid-only {cutoff_um:.0f} um cutoff would have lost)  "
                 f"[{time.time()-t0:.0f}s]")
    if len(pairs) == 0:
        return None

    # ---- which pairs actually need a boundary distance? ----
    ci, cj = cls0[pairs[:, 0]], cls0[pairs[:, 1]]
    is_le = ((ci == LYM) & (cj == EPI)) | ((ci == EPI) & (cj == LYM))

    if save_graphs:
        # every edge needs a distance, so compute the lot
        d_um = boundary_distance_chunked(pts, pairs, pinst_mpp, device=device,
                                         chunk=chunk, show_progress=show_progress,
                                         tag=tag)
        d_le = d_um[is_le]
        d_sample = d_um
    else:
        # statistics only need lymphocyte-epithelial pairs (~10% of the total),
        # plus a random sample of all pairs for the sanity median.
        le_pairs = pairs[is_le]
        logging.info(f"[{tag}] lymph-epi pairs: {len(le_pairs)} "
                     f"({100*is_le.mean():.1f}% of candidates)")
        d_le = boundary_distance_chunked(pts, le_pairs, pinst_mpp, device=device,
                                         chunk=chunk, show_progress=show_progress,
                                         tag=f"{tag} LE")
        rs = np.random.default_rng(0)
        k = min(200_000, len(pairs))
        sidx = rs.choice(len(pairs), size=k, replace=False)
        d_sample = boundary_distance_chunked(pts, pairs[sidx], pinst_mpp,
                                             device=device, chunk=chunk,
                                             show_progress=False, tag=tag)
        d_um = None

    summary = {
        "slide_id": slide_id, "msi_status": msi,
        "n_cells": n, "n_epi": n_epi, "n_lym": n_lym,
        "frac_epi": n_epi / max(n, 1), "frac_lym": n_lym / max(n, 1),
        "n_pairs": int(len(pairs)), "n_lymphepi_pairs": int(is_le.sum()),
        "median_boundary_um_all": float(np.median(d_sample)),
        "median_boundary_um_lymphepi": float(np.median(d_le)) if len(d_le) else np.nan,
        # nuclear size, reported so that a differential effect between the
        # groups can be ruled in or out rather than assumed away
        "nuc_radius_um_median": float(np.median(rv) * pinst_mpp),
        "nuc_radius_um_p99": float(np.percentile(rv, 99) * pinst_mpp),
        "nuc_radius_um_max": float(rv.max() * pinst_mpp),
        "nuc_radius_cap_um": float(r_cap * pinst_mpp),
        "n_nuclei_over_cap": n_over,
        "nuc_radius_um_epi": float(np.median(radius_px[valid & is_epi_arr])
                                   * pinst_mpp) if (valid & is_epi_arr).any() else np.nan,
        "nuc_radius_um_lym": float(np.median(radius_px[valid & is_lym_arr])
                                   * pinst_mpp) if (valid & is_lym_arr).any() else np.nan,
        "search_radius_um": float(search_px * pinst_mpp),
        "n_pairs_recovered_by_fix": n_missed_by_old,
        "frac_pairs_recovered_by_fix": (n_missed_by_old / max(len(pairs), 1)),
    }
    # distance-resolved: lymph-epi pairs per epithelial nucleus, by bin
    for lo, hi in zip(DIST_BINS_UM[:-1], DIST_BINS_UM[1:]):
        c = int(((d_le >= lo) & (d_le < hi)).sum())
        summary[f"le_per_epi_{int(lo)}_{int(hi)}um"] = c / max(n_epi, 1)
    # Gaussian-weighted contact per epithelial nucleus
    for s in SIGMAS_UM:
        w = np.exp(-(d_le.astype(np.float64) ** 2) / (2 * s * s))
        summary[f"le_gauss_sigma{int(s)}"] = float(w.sum()) / max(n_epi, 1)

    # ---- observed / expected enrichment (composition-normalised) ----
    # Under a label-permutation null that keeps every cell where it is and only
    # reshuffles class labels, the expected fraction of pairs that are
    # lymphocyte-epithelial is 2 * p_lym * p_epi, independent of distance.
    # So E[N_LE(bin)] = N_all(bin) * 2 * p_lym * p_epi, and the ratio
    # O/E > 1 means lymphocytes sit closer to epithelium than chance,
    # O/E < 1 means they are spatially separated from it.
    p_le_null = 2.0 * (n_lym / max(n, 1)) * (n_epi / max(n, 1))
    summary["p_le_null"] = p_le_null
    summary["le_oe_global"] = (float(is_le.sum()) /
                               max(len(pairs) * p_le_null, 1e-12))
    hist_all, _ = np.histogram(d_sample, bins=DIST_BINS_UM)
    n_samp = max(len(d_sample), 1)
    for (lo, hi), h in zip(zip(DIST_BINS_UM[:-1], DIST_BINS_UM[1:]), hist_all):
        n_all_bin = (h / n_samp) * len(pairs)          # est. all-pairs in bin
        n_le_bin = float(((d_le >= lo) & (d_le < hi)).sum())
        exp_bin = n_all_bin * p_le_null
        summary[f"le_oe_{int(lo)}_{int(hi)}um"] = (
            n_le_bin / exp_bin if exp_bin > 0 else np.nan)

    # ---- optional graph save ----
    if save_graphs:
        import torch
        m = d_um <= graph_cutoff_um
        g = {
            "slide_id": slide_id, "msi_status": msi,
            "x_class": torch.from_numpy(cls0.astype(np.int64)),
            "xy_um": torch.from_numpy((xy_pinst * pinst_mpp).astype(np.float32)),
            "edge_index": torch.from_numpy(pairs[m].T.astype(np.int32)),
            "edge_d_um": torch.from_numpy(d_um[m].astype(np.float16)),
            "sigmas_um": SIGMAS_UM, "pinst_mpp": pinst_mpp,
            "graph_cutoff_um": graph_cutoff_um,
        }
        gdir = out_root / "graphs"; gdir.mkdir(parents=True, exist_ok=True)
        torch.save(g, gdir / f"{slide_id}.pt")
        summary["n_edges_saved"] = int(m.sum())

    summary["seconds"] = round(time.time() - t0, 1)
    logging.info(
        f"[{tag}] DONE {summary['seconds']}s | med_LE={summary['median_boundary_um_lymphepi']:.2f}um | "
        + " ".join(f"s{int(s)}={summary[f'le_gauss_sigma{int(s)}']:.4f}" for s in SIGMAS_UM))
    return summary


# ---------------------------------------------------------------------------
def _worker(task):
    (slide_id, msi, out_root, cutoff_um, device, save_graphs,
     graph_cutoff_um, wid, gpu_per_worker, n_gpu, chunk, contour_cache,
     radius_pct) = task
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = device
    if gpu_per_worker and device.startswith("cuda") and n_gpu > 0:
        dev = f"cuda:{wid % n_gpu}"
    try:
        return build_for_slide(slide_id, msi, Path(out_root), cutoff_um, dev,
                               save_graphs, graph_cutoff_um,
                               show_progress=(wid == 0), chunk=chunk,
                               contour_cache=contour_cache,
                               radius_pct=radius_pct)
    except Exception:
        logging.error(f"[{slide_id}] FAILED:\n{traceback.format_exc()}")
        return None


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return (gt - lt) / (len(x) * len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels-csv", default=LABELS_CSV)
    ap.add_argument("--out-root", default="analysis/distance_graph")
    ap.add_argument("--cutoff-um", type=float, default=DEFAULT_CUTOFF_UM)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu",
                    help="'cpu', 'cuda:0', or 'cuda' with --gpu-per-worker")
    ap.add_argument("--gpu-per-worker", action="store_true",
                    help="assign worker i to cuda:(i %% n_gpu)")
    ap.add_argument("--save-graphs", action="store_true",
                    help="also write per-slide .pt graphs (LARGE; see docstring)")
    ap.add_argument("--graph-cutoff-um", type=float, default=30.0,
                    help="only edges within this boundary distance are saved")
    ap.add_argument("--radius-pct", type=float, default=99.9,
                    help="percentile of nuclear radius used to widen the "
                         "candidate search. Using the maximum lets one merged "
                         "segmentation blob set the radius for the whole slide.")
    ap.add_argument("--contour-cache", default="data/contour_cache",
                    help="reuse extracted contours across runs; '' disables. "
                         "Contours depend only on the segmentation, so the "
                         "cache written by other v16 scripts is valid here.")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help="pairs per distance batch; lower it if the GPU is "
                         "shared (50k ~ 80 MB, 10k ~ 16 MB)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for s in SIGMAS_UM:
        if 3 * s > args.cutoff_um:
            logging.warning(f"sigma={s}um: 3*sigma ({3*s}) > cutoff "
                            f"({args.cutoff_um}um) -> that scale is truncated")

    labels = pd.read_csv(args.labels_csv)
    lab = dict(zip(labels["slide_id"], labels["msi_status"]))

    if args.all:
        targets = [(s, lab[s]) for s in labels["slide_id"]]
    elif args.slides:
        targets = []
        for s in args.slides:
            if s not in lab:
                hit = [k for k in lab if short_id(s) in k]
                if not hit:
                    logging.warning(f"{s} not in labels csv; skipping")
                    continue
                s = hit[0]
            targets.append((s, lab[s]))
    else:
        sys.exit("give --slides <ids...> (probe) or --all (full cohort)")

    n_gpu = 0
    if args.device.startswith("cuda"):
        try:
            import torch
            n_gpu = torch.cuda.device_count()
        except Exception:
            n_gpu = 0
        logging.info(f"CUDA devices visible: {n_gpu}")

    logging.info(f"processing {len(targets)} slides | workers={args.workers} "
                 f"| device={args.device} | save_graphs={args.save_graphs}")
    out_root = Path(args.out_root)
    tasks = [(s, m, str(out_root), args.cutoff_um, args.device,
              args.save_graphs, args.graph_cutoff_um, i,
              args.gpu_per_worker, n_gpu, args.chunk, args.contour_cache,
              args.radius_pct)
             for i, (s, m) in enumerate(targets)]

    results = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn") if args.device.startswith("cuda") else mp
        with ctx.Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, tasks),
                          total=len(tasks), desc="slides", unit="slide"):
                if r:
                    results.append(r)
    else:
        for t in tqdm(tasks, desc="slides", unit="slide"):
            r = _worker(t)
            if r:
                results.append(r)

    if not results:
        sys.exit("no slides succeeded")

    df = pd.DataFrame(results)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "contact_summary.csv", index=False)
    logging.info(f"\nwrote {out_root/'contact_summary.csv'}  ({len(df)} slides)")

    mh = df[df.msi_status == "MSI-H"]; ms = df[df.msi_status == "MSS"]
    logging.info(f"MSI-H n={len(mh)} | MSS n={len(ms)}")

    # ---- was the old, too-narrow candidate cutoff differentially biased? ----
    if "nuc_radius_um_median" in df.columns and len(mh) >= 2 and len(ms) >= 2:
        logging.info("")
        logging.info("=" * 78)
        logging.info("CANDIDATE-CUTOFF FIX: was the old truncation biased "
                     "between the groups?")
        logging.info("=" * 78)
        logging.info("  The previous version searched for candidates at the "
                     "boundary cutoff itself.")
        logging.info("  Boundary distance is smaller than centroid distance by "
                     "the two nuclear radii,")
        logging.info("  so pairs near the cutoff were lost, and the outermost "
                     "band lost the most. If")
        logging.info("  nuclear size differs between the groups, so does the "
                     "loss.")
        logging.info("")
        logging.info(f"{'measure':<36}{'MSI-H':>10}{'MSS':>10}{'delta':>9}")
        logging.info("-" * 78)
        for f, lbl in (("nuc_radius_um_median", "nuclear radius um (median)"),
                       ("nuc_radius_um_epi", "  epithelial nuclei"),
                       ("nuc_radius_um_lym", "  lymphocyte nuclei"),
                       ("nuc_radius_um_p99", "  99th percentile"),
                       ("frac_pairs_recovered_by_fix",
                        "fraction of pairs the fix recovered")):
            if f not in df.columns:
                continue
            d = cliffs_delta(mh[f].values, ms[f].values)
            logging.info(f"{lbl:<36}{mh[f].median():>10.3f}"
                         f"{ms[f].median():>10.3f}{d:>+9.3f}")
        dd = cliffs_delta(mh["frac_pairs_recovered_by_fix"].values,
                          ms["frac_pairs_recovered_by_fix"].values)
        logging.info("")
        if abs(dd) > 0.15:
            logging.info(f"  The loss WAS differential (delta {dd:+.3f}). Every "
                         f"distance-band result from\n  the previous version is "
                         f"biased between the groups and must not be reused.")
        else:
            logging.info(f"  The loss was similar in both groups (delta "
                         f"{dd:+.3f}), so the old bias was mainly\n  a loss of "
                         f"power rather than a differential distortion. The "
                         f"corrected numbers\n  below are still the ones to "
                         f"report.")
    if len(mh) >= 2 and len(ms) >= 2:
        rows = []
        for lo, hi in zip(DIST_BINS_UM[:-1], DIST_BINS_UM[1:]):
            col = f"le_per_epi_{int(lo)}_{int(hi)}um"
            rows.append({"scale": f"raw {lo}-{hi}um", "x": (lo + hi) / 2,
                         "kind": "raw_bin",
                         "cliffs_delta": cliffs_delta(mh[col], ms[col])})
        for lo, hi in zip(DIST_BINS_UM[:-1], DIST_BINS_UM[1:]):
            col = f"le_oe_{int(lo)}_{int(hi)}um"
            rows.append({"scale": f"O/E {lo}-{hi}um", "x": (lo + hi) / 2,
                         "kind": "oe_bin",
                         "cliffs_delta": cliffs_delta(mh[col], ms[col])})
        for s in SIGMAS_UM:
            col = f"le_gauss_sigma{int(s)}"
            rows.append({"scale": f"gauss s={s}um", "x": s, "kind": "gauss",
                         "cliffs_delta": cliffs_delta(mh[col], ms[col])})
        rows.append({"scale": "O/E global", "x": np.nan, "kind": "oe_global",
                     "cliffs_delta": cliffs_delta(mh["le_oe_global"],
                                                  ms["le_oe_global"])})
        dd = pd.DataFrame(rows)
        dd.to_csv(out_root / "delta_by_scale.csv", index=False)
        for _, r in dd.iterrows():
            logging.info(f"  {r['scale']:>16}  delta = {r['cliffs_delta']:+.3f}")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            styles = {"raw_bin": ("o-", "raw count / epi cell"),
                      "oe_bin": ("^-", "O/E enrichment"),
                      "gauss": ("s--", "Gaussian weighted")}
            for kind, (mk, lbl) in styles.items():
                sub = dd[dd["kind"] == kind].dropna(subset=["x"])
                if len(sub):
                    ax.plot(sub["x"], sub["cliffs_delta"], mk, lw=2, label=lbl)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel("distance scale (um)")
            ax.set_ylabel("Cliff's delta (MSI-H vs MSS)")
            ax.set_title("Lymphocyte-epithelial proximity: MSI signal vs distance")
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_root / "delta_by_scale.png", dpi=150)
            logging.info(f"wrote {out_root/'delta_by_scale.png'}")
        except Exception as e:
            logging.warning(f"plot failed: {e}")
    else:
        logging.info("(need >=2 slides per class for the delta curve)")


if __name__ == "__main__":
    main()
