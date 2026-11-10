"""Build explicit cellular graphs from tissue maps and HoVer-NeXt outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

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
N_CELL_CLASS = 7
EPI, LYM = 1, 2
PINST_TO_L0 = 2.0

# graph geometry
WINDOW_UM = 400.0        # side of a window; ~0.16 mm2, a few hundred nuclei
EDGE_CUTOFF_UM = 40.0    # no edge beyond this boundary distance
K_NEIGHBOURS = 12        # cap on out-degree, keeps memory linear in nodes
MAX_CELLS_PER_WINDOW = 1200
MIN_CELLS_PER_WINDOW = 40
MAX_WINDOWS_PER_SLIDE = 400
RBF_CENTRES_UM = np.array([0, 3, 6, 10, 15, 22, 30, 40], dtype=np.float32)
RBF_WIDTH_UM = 4.0


def load_bdg(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[fatal] cannot find {p}")
    spec = importlib.util.spec_from_file_location("bdg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short_id(name: str) -> str:
    return name.split(".")[0]


def resolve_cells_dir(slide_id: str, root: Path) -> Optional[Path]:
    sid = short_id(slide_id)
    for j in root.rglob("class_inst.json"):
        if sid in j.parent.name:
            return j.parent
    return None


def rbf(d_um: np.ndarray) -> np.ndarray:
    """Distance as a smooth basis rather than a single number or a hard cut."""
    z = (d_um[:, None] - RBF_CENTRES_UM[None, :]) / RBF_WIDTH_UM
    return np.exp(-0.5 * z * z).astype(np.float32)


# ---------------------------------------------------------------------------
def build_slide(slide_id: str, msi: str, args, bdg) -> Optional[dict]:
    import torch
    from scipy.spatial import cKDTree
    from scipy.ndimage import distance_transform_edt, binary_closing, binary_fill_holes

    t0 = time.time()
    tag = short_id(slide_id).replace("TCGA-", "")
    out_path = Path(args.out_dir) / f"{slide_id}.pt"
    if out_path.exists() and not args.overwrite:
        return {"slide_id": slide_id, "status": "cached"}

    mp = Path(args.maps_dir) / f"{slide_id}.npz"
    cells_dir = resolve_cells_dir(slide_id, Path(args.cells_root))
    wsi = bdg.resolve_wsi(slide_id, Path(args.wsi_root))
    if not mp.exists() or cells_dir is None or wsi is None:
        return None

    mpp_l0 = bdg.read_slide_mpp(wsi)
    pinst_mpp = mpp_l0 * PINST_TO_L0

    # ---- tissue map -> compartment per tile, and the tumour mass ----
    z = np.load(mp, allow_pickle=True)
    coords = z["coords_l0"]; pred = z["pred"]; patch_l0 = int(z["patch_l0"])
    gx = (coords[:, 0] // patch_l0).astype(np.int64)
    gy = (coords[:, 1] // patch_l0).astype(np.int64)
    gx0, gy0 = int(gx.min()), int(gy.min())
    GW = int(gx.max()) - gx0 + 1
    GH = int(gy.max()) - gy0 + 1
    comp_grid = np.full((GH, GW), -1, dtype=np.int8)
    comp_grid[gy - gy0, gx - gx0] = pred

    tum = comp_grid == TUM_I
    if tum.sum() < 20:
        return None
    r = 3
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r ** 2
    mass = binary_fill_holes(binary_closing(tum, structure=disk))
    sd_grid = ((distance_transform_edt(~mass) - distance_transform_edt(mass))
               .astype(np.float32) * (patch_l0 * mpp_l0))

    # ---- cells, restricted to the tumour compartment ----
    inst_ids, xy_pinst, cls0 = bdg.load_centroids(cells_dir)
    n_all = len(inst_ids)
    xy_l0 = xy_pinst * PINST_TO_L0
    cgx = np.clip((xy_l0[:, 0] // patch_l0).astype(np.int64) - gx0, 0, GW - 1)
    cgy = np.clip((xy_l0[:, 1] // patch_l0).astype(np.int64) - gy0, 0, GH - 1)
    cell_comp = comp_grid[cgy, cgx]
    cell_sd = sd_grid[cgy, cgx]
    inside = mass[cgy, cgx]
    if inside.sum() < MIN_CELLS_PER_WINDOW * 4:
        return None

    # ---- contours -> nuclear radius (cheap morphology, and needed for the
    #      boundary distance the edges are defined by) ----
    cache = Path(args.contour_cache) / f"{slide_id}.npz" if args.contour_cache else None
    pts = valid = None
    if cache is not None and cache.exists():
        try:
            zc = np.load(cache)
            if int(zc["n"]) == n_all:
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
                     n=np.int64(n_all))
    radius_px = np.sqrt(((pts - xy_pinst[:, None, :]) ** 2).sum(-1).max(1))
    radius_um = (radius_px * pinst_mpp).astype(np.float32)

    sel = inside & valid
    idx_all = np.where(sel)[0]
    xy_um = (xy_pinst[idx_all] * pinst_mpp).astype(np.float32)
    cls_s = cls0[idx_all]
    comp_s = cell_comp[idx_all]
    sd_s = cell_sd[idx_all]
    rad_s = radius_um[idx_all]
    pts_s = pts[idx_all]

    # ---- local neighbourhood composition, a cheap context feature ----
    tree_all = cKDTree(xy_um)
    nb = tree_all.query_ball_point(xy_um, r=args.context_um)
    local_frac = np.zeros((len(xy_um), N_CELL_CLASS), dtype=np.float32)
    local_n = np.zeros(len(xy_um), dtype=np.float32)
    for i, lst in enumerate(nb):
        if not lst:
            continue
        c = np.bincount(cls_s[lst], minlength=N_CELL_CLASS)
        local_frac[i] = c / c.sum()
        local_n[i] = len(lst)
    del nb

    # ---- windows ----
    win = args.window_um
    wx = (xy_um[:, 0] // win).astype(np.int64)
    wy = (xy_um[:, 1] // win).astype(np.int64)
    span = int(wx.max()) + 2
    wkey = wy * span + wx
    uw, counts = np.unique(wkey, return_counts=True)
    uw = uw[counts >= MIN_CELLS_PER_WINDOW]
    if len(uw) == 0:
        return None
    if len(uw) > MAX_WINDOWS_PER_SLIDE:
        # keep the most populated windows; they carry the most structure
        cnt = np.array([int((wkey == k).sum()) for k in uw])
        uw = uw[np.argsort(-cnt)[:MAX_WINDOWS_PER_SLIDE]]

    rng = np.random.default_rng(0)
    windows = []
    for k in uw:
        loc = np.where(wkey == k)[0]
        if len(loc) > MAX_CELLS_PER_WINDOW:
            loc = rng.choice(loc, MAX_CELLS_PER_WINDOW, replace=False)
        nloc = len(loc)
        p = xy_um[loc]

        # kNN by centroid, then exact boundary distance, then cut
        tree = cKDTree(p)
        kk = min(K_NEIGHBOURS + 1, nloc)
        dist, nbr = tree.query(p, k=kk)
        src = np.repeat(np.arange(nloc), kk - 1)
        dst = nbr[:, 1:].reshape(-1)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            continue
        # boundary distance for the candidate edges
        a = pts_s[loc][src] * pinst_mpp
        b = pts_s[loc][dst] * pinst_mpp
        d2 = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)
        d_um = np.sqrt(d2.min(axis=(1, 2))).astype(np.float32)
        m = d_um <= args.edge_cutoff_um
        src, dst, d_um = src[m], dst[m], d_um[m]
        if len(src) < 2:
            continue

        cl = cls_s[loc]
        cm = comp_s[loc]
        # node features: class, morphology, context, compartment, position
        x = np.concatenate([
            np.eye(N_CELL_CLASS, dtype=np.float32)[cl],           # 7
            rad_s[loc][:, None],                                   # 1
            np.log1p(local_n[loc])[:, None],                       # 1
            local_frac[loc],                                       # 7
            np.eye(len(TISSUE), dtype=np.float32)[np.clip(cm, 0, len(TISSUE)-1)],  # 9
            (sd_s[loc] / 1000.0)[:, None],                         # 1, mm
        ], axis=1).astype(np.float32)

        # Only the primitive edge quantities are stored. The RBF expansion and
        # the cross-compartment flag are recomputed at load time: the expansion
        # is eight floats per edge, which would dominate disk for no gain,
        # and the flag is a function of the two endpoints' node features.
        rel = (cl[src] * N_CELL_CLASS + cl[dst]).astype(np.int16)  # 49 relations

        windows.append({
            "x": torch.from_numpy(x.astype(np.float16)),
            "edge_index": torch.from_numpy(np.stack([src, dst]).astype(np.int32)),
            "edge_d_um": torch.from_numpy(d_um.astype(np.float16)),
            "rel": torch.from_numpy(rel),
            "n_nodes": nloc,
            "centre_um": torch.tensor([float(p[:, 0].mean()),
                                       float(p[:, 1].mean())],
                                      dtype=torch.float32),
            "frac_epi": float((cl == EPI).mean()),
            "frac_lym": float((cl == LYM).mean()),
        })

    if len(windows) < 4:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"slide_id": slide_id, "msi_status": msi,
                "windows": windows,
                "node_dim": int(windows[0]["x"].shape[1]),
                "n_rbf": len(RBF_CENTRES_UM),
                "rbf_centres_um": RBF_CENTRES_UM,
                "rbf_width_um": RBF_WIDTH_UM,
                "edge_cutoff_um": float(args.edge_cutoff_um),
                "compartment_slice": (16, 16 + len(TISSUE)),
                "window_um": win, "mpp_l0": mpp_l0}, out_path)

    n_nodes = sum(w["n_nodes"] for w in windows)
    n_edges = sum(w["edge_index"].shape[1] for w in windows)
    dt = time.time() - t0
    mb = out_path.stat().st_size / 1e6
    logging.info(f"[{tag}] {msi} | {len(windows)} windows, {n_nodes:,} nodes, "
                 f"{n_edges:,} edges | node_dim={windows[0]['x'].shape[1]} "
                 f"| {mb:.0f} MB | {dt:.0f}s")
    return {"slide_id": slide_id, "msi_status": msi, "status": "ok",
            "n_windows": len(windows), "n_nodes": n_nodes, "n_edges": n_edges,
            "seconds": round(dt, 1)}


def _worker(task):
    slide_id, msi, ad = task
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argparse.Namespace(**ad)
    try:
        return build_slide(slide_id, msi, args, load_bdg(args.bdg_py))
    except Exception:
        logging.error(f"[{slide_id}] FAILED:\n{traceback.format_exc()}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels-csv", default="data/labels/msi_tcga.csv")
    ap.add_argument("--maps-dir", default="data/tissue/maps")
    ap.add_argument("--cells-root", default="data/cells")
    ap.add_argument("--wsi-root", default="data/wsi")
    ap.add_argument("--bdg-py", default=str(Path(__file__).with_name("07_build_distance_graph.py")))
    ap.add_argument("--contour-cache", default="data/contour_cache")
    ap.add_argument("--out-dir", default="data/graphs")
    ap.add_argument("--window-um", type=float, default=WINDOW_UM)
    ap.add_argument("--edge-cutoff-um", type=float, default=EDGE_CUTOFF_UM)
    ap.add_argument("--context-um", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    labels = pd.read_csv(args.labels_csv)
    lab = dict(zip(labels["slide_id"], labels["msi_status"]))
    targets = ([(s, lab[s]) for s in labels["slide_id"]] if args.all
               else [(s, lab.get(s, "?")) for s in (args.slides or [])])
    if not targets:
        sys.exit("give --slides or --all")

    logging.info(f"{len(targets)} slides | window {args.window_um:.0f} um | "
                 f"edge cutoff {args.edge_cutoff_um:.0f} um | "
                 f"k={K_NEIGHBOURS} | workers={args.workers}")

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
        sys.exit("nothing built")
    df = pd.DataFrame(rows)
    mf = Path(args.out_dir) / "_manifest.csv"
    df.to_csv(mf, index=False)
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    logging.info(f"\nwrote {mf}")
    if len(ok) and "n_windows" in ok:
        logging.info(f"  slides built     : {len(ok)}")
        logging.info(f"  windows per slide: median {ok.n_windows.median():.0f} "
                     f"(min {ok.n_windows.min()}, max {ok.n_windows.max()})")
        logging.info(f"  nodes per slide  : median {ok.n_nodes.median():,.0f}")
        logging.info(f"  edges per slide  : median {ok.n_edges.median():,.0f}")
        if "mb" in ok:
            logging.info(f"  disk per slide   : median {ok.mb.median():.0f} MB "
                         f"-> {ok.mb.sum()/1000:.1f} GB total")
        if "msi_status" in ok:
            logging.info(f"  MSI-H {int((ok.msi_status=='MSI-H').sum())} | "
                         f"MSS {int((ok.msi_status=='MSS').sum())}")


if __name__ == "__main__":
    main()
