"""Measure lymphocyte–epithelial proximity within the tumour compartment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

TISSUE = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM_I = TISSUE.index("TUM")
NORM_I = TISSUE.index("NORM")
LYMTILE_I = TISSUE.index("LYM")
# signed distance to the tumour boundary; negative inside. The +-500 um margin
# follows the invasive-margin definition in Lakatos/Gunasri et al.
BANDS = [(-1e9, -500, "deep"), (-500, 0, "inner_margin"),
         (0, 500, "outer_margin"), (500, 1e9, "far")]
EPI, LYM = 1, 2                      # 0-based HoVer-NeXt classes
PINST_TO_L0 = 2.0

CONTACT_UM = 5.0                     # what counts as "in contact"
ENCLOSE_RADIUS_UM = 20.0             # neighbourhood for the angular measure
ENCLOSED_DEG = 270.0
ONESIDED_DEG = 180.0
COVERAGE_BINS = [(0, 60), (60, 120), (120, 180),
                 (180, 240), (240, 300), (300, 360.01)]
CUTOFF_UM = 60.0                     # candidate-pair cutoff, as in step B


# ---------------------------------------------------------------------------
def load_bdg(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[fatal] cannot find {p}; pass --bdg-py")
    spec = importlib.util.spec_from_file_location("bdg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short_id(name: str) -> str:
    return name.split(".")[0]


def resolve_cells_dir(slide_id: str, cells_root: Path) -> Optional[Path]:
    sid = short_id(slide_id)
    for j in cells_root.rglob("class_inst.json"):
        if sid in j.parent.name:
            return j.parent
    return None


# ---------------------------------------------------------------------------
def angular_coverage(dx: np.ndarray, dy: np.ndarray) -> float:
    """
    Degrees of the circle spanned by neighbours seen from the centre point,
    defined as 360 minus the largest angular gap. One neighbour gives 0;
    two opposite neighbours give 180; neighbours all around give close to 360.
    """
    n = len(dx)
    if n < 2:
        return 0.0
    ang = np.sort(np.arctan2(dy, dx))
    gaps = np.diff(ang)
    wrap = (ang[0] + 2 * np.pi) - ang[-1]
    largest = max(float(gaps.max()) if len(gaps) else 0.0, float(wrap))
    return float(np.degrees(2 * np.pi - largest))


def signed_distance_map(coords_l0: np.ndarray, pred: np.ndarray,
                        patch_l0: int, mpp: float, close_tiles: int = 3
                        ) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Signed distance (um) to the boundary of the TUMOUR MASS, on the tile grid.
    Negative inside, positive outside. Returns
        (signed, tum_closed, gx0, gy0, W, H)

    A raw TUM mask is not the right thing to take a boundary of. In a glandular
    tumour the stroma between neighbouring glands is not TUM, so every gland
    ends up a few tiles from an "edge" and there is essentially no interior --
    which is not what a pathologist means by the invasive margin. Morphological
    closing bridges those inter-glandular gaps, and hole filling removes
    enclosed stromal islands, leaving one solid tumour mass whose boundary is
    the advancing edge.

    close_tiles is the structuring-element radius in tiles; at 112 um per tile
    a radius of 3 bridges gaps up to roughly 300-600 um, which covers
    inter-glandular stroma without swallowing genuine tumour boundary.
    """
    from scipy.ndimage import (distance_transform_edt, binary_closing,
                               binary_fill_holes)
    gx = (coords_l0[:, 0] // patch_l0).astype(np.int64)
    gy = (coords_l0[:, 1] // patch_l0).astype(np.int64)
    gx0, gy0 = int(gx.min()), int(gy.min())
    W = int(gx.max()) - gx0 + 1
    H = int(gy.max()) - gy0 + 1
    tum = np.zeros((H, W), dtype=bool)
    m = pred == TUM_I
    tum[gy[m] - gy0, gx[m] - gx0] = True
    if not tum.any():
        return (np.full((H, W), np.inf, dtype=np.float32), tum, gx0, gy0, W, H)

    if close_tiles > 0:
        r = close_tiles
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        tum_c = binary_closing(tum, structure=disk)
        tum_c = binary_fill_holes(tum_c)
    else:
        tum_c = tum

    d_out = distance_transform_edt(~tum_c)     # 0 inside the mass
    d_in = distance_transform_edt(tum_c)       # 0 outside the mass
    signed = (d_out - d_in).astype(np.float32) * (patch_l0 * mpp)
    return signed, tum_c, gx0, gy0, W, H


def lymphoid_aggregates(pred: np.ndarray, coords_l0: np.ndarray,
                        patch_l0: int, mpp: float, gx0: int, gy0: int,
                        W: int, H: int, sd_grid: np.ndarray) -> dict:
    """
    Lymphoid aggregates as connected components of LYM tiles, with size and
    position relative to the tumour boundary.

    A Crohn's-like reaction is not "many lymphocytes"; it is discrete,
    organised lymphoid aggregates at the advancing edge. Counting LYM tiles
    alone conflates one large follicle with several scattered small ones, so
    components are labelled and their areas measured. Aggregates are then
    located by the signed distance at their centroid.
    """
    from scipy.ndimage import label
    grid = np.zeros((H, W), dtype=bool)
    gx = (coords_l0[:, 0] // patch_l0).astype(np.int64) - gx0
    gy = (coords_l0[:, 1] // patch_l0).astype(np.int64) - gy0
    m = pred == LYMTILE_I
    grid[np.clip(gy[m], 0, H - 1), np.clip(gx[m], 0, W - 1)] = True

    tile_area = (patch_l0 * mpp / 1000.0) ** 2
    lab, n = label(grid, structure=np.ones((3, 3), dtype=int))
    out = {"n_lymphagg": 0, "n_lymphagg_large": 0,
           "max_lymphagg_mm2": 0.0, "total_lymphagg_mm2": 0.0,
           "n_lymphagg_outer_margin": 0}
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())[1:]                 # tiles per component
    areas = sizes * tile_area
    out["n_lymphagg"] = int(n)
    # >=3 tiles is about 200 um across, the scale at which an aggregate is a
    # follicle rather than a few adjacent lymphocyte-rich tiles
    out["n_lymphagg_large"] = int((sizes >= 3).sum())
    out["max_lymphagg_mm2"] = float(areas.max())
    out["total_lymphagg_mm2"] = float(areas.sum())
    # position: signed distance at each component's centroid
    ys, xs = np.nonzero(grid)
    comp = lab[ys, xs]
    n_outer = 0
    for c in range(1, n + 1):
        sel = comp == c
        cy = int(round(ys[sel].mean())); cx = int(round(xs[sel].mean()))
        sd = sd_grid[np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)]
        if 0 <= sd < 500:
            n_outer += 1
    out["n_lymphagg_outer_margin"] = n_outer
    return out


def tumour_geometry(tum_c: np.ndarray, patch_l0: int, mpp: float) -> dict:
    """Area of the tumour mass and the length of its advancing edge."""
    tile_mm = patch_l0 * mpp / 1000.0
    area = float(tum_c.sum()) * tile_mm ** 2
    if not tum_c.any():
        return {"tum_area_mm2": 0.0, "tum_front_mm": 0.0}
    pad = np.pad(tum_c, 1)
    nb = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    edge = tum_c & ~nb                       # mass tiles touching non-mass
    return {"tum_area_mm2": area, "tum_front_mm": float(edge.sum()) * tile_mm}


def assign_tiles(xy_l0: np.ndarray, coords_l0: np.ndarray,
                 patch_l0: int) -> np.ndarray:
    """Index of the tile containing each point, or -1."""
    gx = (coords_l0[:, 0] // patch_l0).astype(np.int64)
    gy = (coords_l0[:, 1] // patch_l0).astype(np.int64)
    span = int(gx.max()) + 2 if len(gx) else 1
    key = gy * span + gx
    order = np.argsort(key)
    ks = key[order]
    cgx = (xy_l0[:, 0] // patch_l0).astype(np.int64)
    cgy = (xy_l0[:, 1] // patch_l0).astype(np.int64)
    ck = cgy * span + cgx
    pos = np.clip(np.searchsorted(ks, ck), 0, len(ks) - 1)
    hit = ks[pos] == ck
    out = np.full(len(ck), -1, dtype=np.int64)
    out[hit] = order[pos[hit]]
    return out


# ---------------------------------------------------------------------------
def process_slide(slide_id: str, msi: str, args, bdg) -> Optional[dict]:
    t0 = time.time()
    tag = short_id(slide_id).replace("TCGA-", "")

    mp = Path(args.maps_dir) / f"{slide_id}.npz"
    if not mp.exists():
        return None
    cells_dir = resolve_cells_dir(slide_id, Path(args.cells_root))
    if cells_dir is None:
        return None
    wsi = bdg.resolve_wsi(slide_id, Path(args.wsi_root))
    if wsi is None:
        return None

    mpp_l0 = bdg.read_slide_mpp(wsi)
    pinst_mpp = mpp_l0 * PINST_TO_L0

    # ---- cells (pinst coords) ----
    inst_ids, xy_pinst, cls0 = bdg.load_centroids(cells_dir)
    n = len(inst_ids)
    xy_l0 = xy_pinst * PINST_TO_L0
    xy_um = xy_pinst * pinst_mpp

    # ---- tissue compartment per cell ----
    z = np.load(mp, allow_pickle=True)
    coords = z["coords_l0"]; pred = z["pred"]; patch_l0 = int(z["patch_l0"])
    tile_idx = assign_tiles(xy_l0, coords, patch_l0)
    comp = np.full(n, -1, dtype=np.int64)
    inside = tile_idx >= 0
    comp[inside] = pred[tile_idx[inside]]

    is_epi = cls0 == EPI
    is_lym = cls0 == LYM
    epi_tum = is_epi & (comp == TUM_I)
    n_epi_all = int(is_epi.sum())
    n_epi_tum = int(epi_tum.sum())
    if n_epi_tum < 100:
        logging.warning(f"[{tag}] only {n_epi_tum} tumour epithelial nuclei; skip")
        return None

    # ---- signed distance to the boundary of the tumour MASS ----
    sd_grid, tum_c, gx0, gy0, GW, GH = signed_distance_map(
        coords, pred, patch_l0, mpp_l0, close_tiles=args.close_tiles)
    cgx = np.clip((xy_l0[:, 0] // patch_l0).astype(np.int64) - gx0, 0, GW - 1)
    cgy = np.clip((xy_l0[:, 1] // patch_l0).astype(np.int64) - gy0, 0, GH - 1)
    sd_um = sd_grid[cgy, cgx]

    # ---- boundary distance for candidate pairs (same code as step B) ----
    from scipy.spatial import cKDTree
    cutoff_px = CUTOFF_UM / pinst_mpp

    # Contour extraction dominates the runtime (~60 s per slide), and the
    # contours depend only on the segmentation, so cache them.
    cache = Path(args.contour_cache) / f"{slide_id}.npz" if args.contour_cache else None
    pts = valid = None
    if cache is not None and cache.exists():
        try:
            zc = np.load(cache)
            if int(zc["n"]) == n:
                pts, valid = zc["pts"], zc["valid"]
        except Exception:
            pts = valid = None
    if pts is None:
        arr, store = bdg.open_pinst(cells_dir)
        pts, valid = bdg.extract_contours(arr, inst_ids, xy_pinst,
                                          show_progress=False, tag=tag)
        try:
            store.close()
        except Exception:
            pass
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, pts=pts.astype(np.float32), valid=valid,
                     n=np.int64(n))

    tree = cKDTree(xy_pinst)
    pairs = tree.query_pairs(r=cutoff_px, output_type="ndarray")
    keep = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    pairs = pairs[keep]
    ci, cj = cls0[pairs[:, 0]], cls0[pairs[:, 1]]
    is_le = ((ci == LYM) & (cj == EPI)) | ((ci == EPI) & (cj == LYM))
    le_pairs = pairs[is_le]
    if len(le_pairs) == 0:
        return None
    d_le = bdg.boundary_distance_chunked(pts, le_pairs, pinst_mpp,
                                         device=args.device, chunk=args.chunk,
                                         show_progress=False, tag=tag)

    # orient each pair as (lymphocyte, epithelial)
    a_is_lym = cls0[le_pairs[:, 0]] == LYM
    lym_idx = np.where(a_is_lym, le_pairs[:, 0], le_pairs[:, 1])
    epi_idx = np.where(a_is_lym, le_pairs[:, 1], le_pairs[:, 0])

    contact = d_le <= CONTACT_UM
    epi_in_tum = comp[epi_idx] == TUM_I
    epi_in_norm = comp[epi_idx] == NORM_I

    # ---- enclosure, for lymphocytes that actually contact tumour epithelium ---
    lym_contact_tum = np.unique(lym_idx[contact & epi_in_tum])
    epi_xy_um = xy_um[is_epi]
    epi_tree = cKDTree(epi_xy_um)
    cov = np.full(len(lym_contact_tum), np.nan)
    if len(lym_contact_tum):
        nbrs = epi_tree.query_ball_point(xy_um[lym_contact_tum],
                                         r=ENCLOSE_RADIUS_UM)
        for k, nb in enumerate(nbrs):
            if len(nb) < 2:
                cov[k] = 0.0
                continue
            d = epi_xy_um[nb] - xy_um[lym_contact_tum[k]]
            cov[k] = angular_coverage(d[:, 0], d[:, 1])
    enclosed = cov >= ENCLOSED_DEG
    onesided = cov < ONESIDED_DEG

    per100 = 100.0 / max(n_epi_tum, 1)
    rec = {
        "slide_id": slide_id, "msi_status": msi,
        "n_cells": n, "n_epi_all": n_epi_all, "n_epi_tum": n_epi_tum,
        "frac_epi_in_tum": n_epi_tum / max(n_epi_all, 1),
        "frac_epi_in_norm": float((is_epi & (comp == NORM_I)).sum()) / max(n_epi_all, 1),
        "n_lym": int(is_lym.sum()),

        # the whole-slide feature, reproduced for direct comparison
        "le_per_epi_0_5um_allslide":
            float((d_le <= CONTACT_UM).sum()) / max(n_epi_all, 1),
        # restricted to tumour epithelium
        "le_per_epi_0_5um_TUM":
            float((contact & epi_in_tum).sum()) / max(n_epi_tum, 1),
        # what was being counted in normal mucosa
        "le_per_epi_0_5um_NORM":
            float((contact & epi_in_norm).sum()) /
            max(int((is_epi & (comp == NORM_I)).sum()), 1),

        # pathology-convention counts, per 100 tumour epithelial nuclei
        "iel_per100epi_all": len(lym_contact_tum) * per100,
        "iel_per100epi_enclosed": int(np.nansum(enclosed)) * per100,
        "iel_per100epi_onesided": int(np.nansum(onesided)) * per100,
        "enclosed_fraction": float(np.nanmean(enclosed)) if len(cov) else np.nan,
        "median_coverage_deg": float(np.nanmedian(cov)) if len(cov) else np.nan,

        # signed distance context (tumour interior is negative)
        "frac_lym_inside_tum": float((sd_um[is_lym] < 0).mean()),
        "median_sd_lym_um": float(np.median(sd_um[is_lym])),
    }
    # Coverage-response profile. A fixed "enclosed" cutoff bakes in an
    # assumption about geometry: in single-layer columnar colonic glands a
    # genuine intraepithelial lymphocyte has epithelium on either side along
    # the layer, lumen on one side and basement membrane on the other, so its
    # coverage is nearer 180 deg than 360. Rather than pick a threshold,
    # report the whole profile and let the delta-versus-coverage curve show
    # which geometry carries the MSI signal.
    for lo, hi in COVERAGE_BINS:
        m_ = (cov >= lo) & (cov < hi)
        rec[f"iel_per100epi_cov{int(lo)}_{int(hi)}"] = int(np.nansum(m_)) * per100
    # ---- bands relative to the tumour boundary, as DENSITIES ----
    # Proportions would only say how much of each compartment this section
    # happened to contain; density (per mm2) is the immunological quantity.
    # Band areas come from the tile grid, so they are exact given the tiling.
    tgx = np.clip((coords[:, 0] // patch_l0).astype(np.int64) - gx0, 0, GW - 1)
    tgy = np.clip((coords[:, 1] // patch_l0).astype(np.int64) - gy0, 0, GH - 1)
    sd_tile = sd_grid[tgy, tgx]
    tile_area_mm2 = (patch_l0 * mpp_l0 / 1000.0) ** 2
    is_lymtile = pred == LYMTILE_I

    # ---- tumour mass geometry and lymphoid aggregates ----
    rec.update(tumour_geometry(tum_c, patch_l0, mpp_l0))
    agg = lymphoid_aggregates(pred, coords, patch_l0, mpp_l0,
                              gx0, gy0, GW, GH, sd_grid)
    rec.update(agg)
    front = rec["tum_front_mm"]
    # the pathology convention for a Crohn's-like reaction is aggregates AT the
    # advancing edge, so normalise by the length of that edge
    rec["lymphagg_per_mm_front"] = (agg["n_lymphagg_outer_margin"] / front
                                    if front > 0 else np.nan)
    rec["lymphagg_large_per_mm_front"] = (agg["n_lymphagg_large"] / front
                                          if front > 0 else np.nan)

    for lo, hi, nm in BANDS:
        tm = (sd_tile >= lo) & (sd_tile < hi)
        area = float(tm.sum()) * tile_area_mm2
        cm = (sd_um >= lo) & (sd_um < hi)
        rec[f"area_mm2_{nm}"] = area
        rec[f"lym_density_{nm}"] = (float((is_lym & cm).sum()) / area
                                    if area > 0 else np.nan)
        rec[f"epi_density_{nm}"] = (float((is_epi & cm).sum()) / area
                                    if area > 0 else np.nan)
        # lymphoid-aggregate tiles per mm2: the Crohn's-like precursor, which
        # is defined pathologically at the advancing edge, not in the centre
        rec[f"lymtile_per_mm2_{nm}"] = (float((tm & is_lymtile).sum()) / area
                                        if area > 0 else np.nan)
        rec[f"lym_frac_{nm}"] = float((is_lym & cm).mean())

    rec["seconds"] = round(time.time() - t0, 1)
    logging.info(
        f"[{tag}] {msi} | epi_tum={n_epi_tum:,} ({100*rec['frac_epi_in_tum']:.0f}% "
        f"of epi) | LE/epi all={rec['le_per_epi_0_5um_allslide']:.4f} "
        f"TUM={rec['le_per_epi_0_5um_TUM']:.4f} | IEL/100epi "
        f"encl={rec['iel_per100epi_enclosed']:.2f} "
        f"one={rec['iel_per100epi_onesided']:.2f} | {rec['seconds']}s")
    return rec


# ---------------------------------------------------------------------------
def _worker(task):
    slide_id, msi, args_d = task
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argparse.Namespace(**args_d)
    try:
        bdg = load_bdg(args.bdg_py)
        return process_slide(slide_id, msi, args, bdg)
    except Exception:
        logging.error(f"[{slide_id}] FAILED:\n{traceback.format_exc()}")
        return None


def cliffs_delta(x, y) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    ys = np.sort(y)
    lt = np.searchsorted(ys, x, side="left").sum()
    le = np.searchsorted(ys, x, side="right").sum()
    return (lt - (len(x) * len(y) - le)) / (len(x) * len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels-csv", default="data/labels/msi_tcga.csv")
    ap.add_argument("--meta-csv", default="data/labels/metadata_tcga_crc_dx.csv")
    ap.add_argument("--maps-dir", default="data/tissue/maps")
    ap.add_argument("--cells-root", default="data/cells")
    ap.add_argument("--wsi-root", default="data/wsi")
    ap.add_argument("--bdg-py", default=str(Path(__file__).with_name("07_build_distance_graph.py")))
    ap.add_argument("--out", default="data/tissue/compartment_iel.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--contour-cache", default="data/contour_cache",
                    help="reuse extracted contours across runs; '' disables")
    ap.add_argument("--close-tiles", type=int, default=3,
                    help="morphological closing radius in tiles when forming "
                         "the tumour mass (3 tiles ~ 336 um at 112 um/tile)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    labels = pd.read_csv(args.labels_csv)
    lab = dict(zip(labels["slide_id"], labels["msi_status"]))

    if args.all:
        targets = [(s, lab[s]) for s in labels["slide_id"]]
    elif args.slides:
        targets = [(s, lab.get(s, "?")) for s in args.slides]
    else:
        sys.exit("give --slides <ids...> or --all")

    logging.info(f"{len(targets)} slides | workers={args.workers}")
    logging.info(f"contact<={CONTACT_UM}um | enclosure radius={ENCLOSE_RADIUS_UM}um "
                 f"| enclosed>={ENCLOSED_DEG}deg | one-sided<{ONESIDED_DEG}deg")

    ad = vars(args)
    tasks = [(s, m, ad) for s, m in targets]
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, tasks),
                          total=len(tasks), desc="slides", unit="slide"):
                if r:
                    rows.append(r)
    else:
        for t in tqdm(tasks, desc="slides", unit="slide"):
            r = _worker(t)
            if r:
                rows.append(r)

    if not rows:
        sys.exit("no slides succeeded")
    df = pd.DataFrame(rows)

    meta = Path(args.meta_csv)
    if meta.exists():
        md = pd.read_csv(meta)
        if {"slide_id", "project_id"}.issubset(md.columns):
            df = df.merge(md[["slide_id", "project_id"]].drop_duplicates("slide_id"),
                          on="slide_id", how="left")
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logging.info(f"\nwrote {p}  ({len(df)} slides)")

    # ---------------- analysis ----------------
    if "project_id" in df.columns and df["project_id"].notna().any():
        coad = df[df["project_id"].astype(str).str.contains("COAD", na=False)]
    else:
        coad = df
    mh = coad[coad.msi_status == "MSI-H"]
    ms = coad[coad.msi_status == "MSS"]
    logging.info(f"\nCOAD: MSI-H={len(mh)}  MSS={len(ms)}")

    logging.info("")
    logging.info("=" * 92)
    logging.info("1. WHAT RESTRICTING TO TUMOUR EPITHELIUM DOES")
    logging.info("=" * 92)
    logging.info(f"  epithelium that is tumour : "
                 f"{100*coad['frac_epi_in_tum'].mean():.1f}% (mean per slide)")
    logging.info(f"  epithelium in normal mucosa: "
                 f"{100*coad['frac_epi_in_norm'].mean():.1f}%")
    logging.info("")
    logging.info(f"{'feature':<34}{'MSI-H':>10}{'MSS':>10}{'delta':>9}{'AUROC':>8}")
    logging.info("-" * 92)
    for f, lbl in (("le_per_epi_0_5um_allslide", "whole slide (as before)"),
                   ("le_per_epi_0_5um_TUM", "tumour epithelium only"),
                   ("le_per_epi_0_5um_NORM", "normal mucosa only")):
        if f not in coad:
            continue
        d = cliffs_delta(mh[f], ms[f])
        logging.info(f"{lbl:<34}{mh[f].median():>10.4f}{ms[f].median():>10.4f}"
                     f"{d:>+9.3f}{(1+d)/2:>8.3f}")

    logging.info("")
    logging.info("=" * 92)
    logging.info("2. WHICH GEOMETRY CARRIES THE SIGNAL?")
    logging.info("=" * 92)
    logging.info(f"  median angular coverage of epithelium around a contacting "
                 f"lymphocyte: {coad['median_coverage_deg'].median():.0f} deg")
    logging.info("")
    logging.info("  In a single-layer columnar gland a true intraepithelial "
                 "lymphocyte has epithelium")
    logging.info("  along the layer on either side, lumen on one side and "
                 "basement membrane on the")
    logging.info("  other, so it sits nearer 180 deg than 360. The profile below "
                 "shows where the MSI")
    logging.info("  signal actually lives, without committing to a cutoff.")
    logging.info("")
    logging.info(f"{'coverage band':<22}{'MSI-H':>10}{'MSS':>10}{'delta':>9}"
                 f"{'AUROC':>8}   (per 100 tumour epi nuclei)")
    logging.info("-" * 92)
    curve = []
    for lo, hi in COVERAGE_BINS:
        f = f"iel_per100epi_cov{int(lo)}_{int(hi)}"
        if f not in coad:
            continue
        d = cliffs_delta(mh[f], ms[f])
        curve.append(((lo + hi) / 2, d))
        logging.info(f"{f'{int(lo)}-{int(hi)} deg':<22}{mh[f].median():>10.3f}"
                     f"{ms[f].median():>10.3f}{d:>+9.3f}{(1+d)/2:>8.3f}")
    f = "iel_per100epi_all"
    d_all = cliffs_delta(mh[f], ms[f])
    logging.info(f"{'ALL contacting':<22}{mh[f].median():>10.3f}"
                 f"{ms[f].median():>10.3f}{d_all:>+9.3f}{(1+d_all)/2:>8.3f}")

    if len(curve) >= 3 and len(mh) >= 2 and len(ms) >= 2:
        xs = np.array([c[0] for c in curve])
        ds = np.array([c[1] for c in curve])
        ok = np.isfinite(ds)
        if ok.sum() >= 3:
            slope = np.polyfit(xs[ok], ds[ok], 1)[0] * 100  # delta per 100 deg
            hi_band = ds[ok][xs[ok] >= 180].mean() if (xs[ok] >= 180).any() else np.nan
            lo_band = ds[ok][xs[ok] < 180].mean() if (xs[ok] < 180).any() else np.nan
            logging.info("")
            logging.info(f"  trend: {slope:+.3f} delta per 100 deg of coverage")
            logging.info(f"  mean delta at >=180 deg: {hi_band:+.3f}   "
                         f"at <180 deg: {lo_band:+.3f}")
            if np.isfinite(hi_band) and np.isfinite(lo_band):
                if hi_band - lo_band > 0.05:
                    logging.info("  -> the signal strengthens as epithelium wraps "
                                 "further around the lymphocyte,\n     which is "
                                 "what intraepithelial lymphocytosis predicts.")
                elif lo_band - hi_band > 0.05:
                    logging.info("  -> the signal sits with lymphocytes that have "
                                 "epithelium on one side only,\n     i.e. against "
                                 "the outside of glands: peritumoural rather than "
                                 "intraepithelial.")
                else:
                    logging.info("  -> the signal does not depend on this geometry. "
                                 "The feature should NOT be\n     described as "
                                 "measuring intraepithelial lymphocytes.")
    else:
        logging.info("\n  (need >=2 slides per class for the coverage curve; "
                     "with one per class every\n   delta is +-1 by construction "
                     "and means nothing)")

    logging.info("")
    logging.info("=" * 92)
    logging.info("3. DENSITIES RELATIVE TO THE TUMOUR BOUNDARY")
    logging.info("=" * 92)
    logging.info("  Bands follow the invasive-margin definition in "
                 "Lakatos/Gunasri et al. (tumour-normal")
    logging.info("  interface expanded by 500 um either side). Tile resolution "
                 "is 112 um, so band")
    logging.info("  edges are accurate to about one tile. Densities, not "
                 "proportions: a proportion")
    logging.info("  would only reflect how much of each compartment this section "
                 "happened to contain.")
    logging.info("")
    logging.info(f"{'band':<20}{'area mm2':>10}{'MSI-H':>12}{'MSS':>12}"
                 f"{'delta':>9}{'AUROC':>8}")
    logging.info("-" * 92)
    for _, _, nm in BANDS:
        f = f"lym_density_{nm}"
        if f not in coad:
            continue
        d = cliffs_delta(mh[f], ms[f])
        logging.info(f"{'lymphocytes/mm2 ' + nm[:4]:<20}"
                     f"{coad[f'area_mm2_{nm}'].median():>10.1f}"
                     f"{mh[f].median():>12.1f}{ms[f].median():>12.1f}"
                     f"{d:>+9.3f}{(1+d)/2:>8.3f}")
    logging.info("")
    logging.info(f"{'lymphoid aggregates (connected LYM tiles)':<44}{'MSI-H':>12}"
                 f"{'MSS':>12}{'delta':>9}")
    logging.info("-" * 92)
    for f, lbl in (("n_lymphagg", "  count per slide"),
                   ("n_lymphagg_large", "  count, >=3 tiles (~200 um across)"),
                   ("max_lymphagg_mm2", "  largest aggregate, mm2"),
                   ("total_lymphagg_mm2", "  total aggregate area, mm2"),
                   ("n_lymphagg_outer_margin", "  at the advancing edge (0-500 um)"),
                   ("lymphagg_per_mm_front", "  per mm of advancing edge"),
                   ("lymphagg_large_per_mm_front", "  large, per mm of edge")):
        if f not in coad:
            continue
        d = cliffs_delta(mh[f], ms[f])
        logging.info(f"{lbl:<44}{mh[f].median():>12.3f}{ms[f].median():>12.3f}"
                     f"{d:>+9.3f}")
    logging.info("")
    logging.info(f"  tumour mass: {coad['tum_area_mm2'].median():.0f} mm2 median, "
                 f"advancing edge {coad['tum_front_mm'].median():.0f} mm median")
    logging.info("")
    logging.info("  A Crohn's-like reaction is discrete organised lymphoid "
                 "aggregates at the advancing")
    logging.info("  edge, so aggregates are labelled as connected components "
                 "and sized, rather than")
    logging.info("  counting LYM tiles. At 112 um per tile this still cannot "
                 "resolve a germinal centre,")
    logging.info("  and cannot separate a true follicle from dense diffuse "
                 "infiltrate: it is a proxy.")


if __name__ == "__main__":
    main()
