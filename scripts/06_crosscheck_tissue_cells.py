"""Count nuclei by tissue class for the mucin phenotype target."""
from __future__ import annotations

import argparse
import json
import logging
import sys
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
CELL = ["neutrophil", "epithelial", "lymphocyte", "plasma",
        "eosinophil", "connective", "mitosis"]
EPI, LYM_C, CONN = 1, 2, 5          # 0-based indices into CELL
PINST_TO_L0 = 2.0                   # exact, forensic-verified

# what each tissue class should look like in terms of cells
EXPECT = {
    "TUM":  ("epithelial-dominant",   "epithelial", 0.40, None),
    "NORM": ("epithelial-dominant",   "epithelial", 0.40, None),
    "STR":  ("connective-dominant",   "connective", 0.40, None),
    "MUS":  ("connective-dominant",   "connective", 0.40, None),
    "LYM":  ("lymphocyte-dominant",   "lymphocyte", 0.30, None),
    "ADI":  ("near-acellular",        None,         None, 800.0),
    "BACK": ("near-acellular",        None,         None, 800.0),
}


def short_id(name: str) -> str:
    return name.split(".")[0]


def resolve_cells_dir(slide_id: str, cells_root: Path) -> Optional[Path]:
    sid = short_id(slide_id)
    for j in cells_root.rglob("class_inst.json"):
        if sid in j.parent.name:
            return j.parent
    return None


def load_cells(cells_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xy_l0 [N,2], cls0 [N]) — centroids converted to level-0 pixels."""
    with open(cells_dir / "class_inst.json") as f:
        d = json.load(f)
    n = len(d)
    xy = np.empty((n, 2), dtype=np.float64)
    cls = np.empty(n, dtype=np.int64)
    for i, v in enumerate(d.values()):
        cls[i] = v[0]
        xy[i, 1] = v[1][0] * PINST_TO_L0     # y
        xy[i, 0] = v[1][1] * PINST_TO_L0     # x
    return xy, np.clip(cls - 1, 0, 6)


def assign_cells_to_tiles(xy_l0: np.ndarray, coords_l0: np.ndarray,
                          patch_l0: int) -> np.ndarray:
    """
    Map each cell to the index of the tile containing it, or -1 if that tile
    was not classified (i.e. it failed the tissue mask and was skipped).

    Tiles lie on a regular grid anchored at 0 with stride patch_l0, so the grid
    cell is a pure integer division; the lookup is a sorted search over the
    tiles that actually exist.
    """
    gx = (coords_l0[:, 0] // patch_l0).astype(np.int64)
    gy = (coords_l0[:, 1] // patch_l0).astype(np.int64)
    span = gx.max() + 2 if len(gx) else 1
    tile_key = gy * span + gx

    order = np.argsort(tile_key)
    keys_sorted = tile_key[order]

    cgx = (xy_l0[:, 0] // patch_l0).astype(np.int64)
    cgy = (xy_l0[:, 1] // patch_l0).astype(np.int64)
    cell_key = cgy * span + cgx

    pos = np.searchsorted(keys_sorted, cell_key)
    pos_clipped = np.clip(pos, 0, len(keys_sorted) - 1)
    hit = keys_sorted[pos_clipped] == cell_key
    out = np.full(len(cell_key), -1, dtype=np.int64)
    out[hit] = order[pos_clipped[hit]]
    return out


def crosscheck_slide(slide_id: str, maps_dir: Path,
                     cells_root: Path) -> Optional[dict]:
    mp = maps_dir / f"{slide_id}.npz"
    if not mp.exists():
        logging.warning(f"[{slide_id}] no tissue map at {mp}")
        return None
    cells_dir = resolve_cells_dir(slide_id, cells_root)
    if cells_dir is None:
        logging.warning(f"[{slide_id}] no cells dir")
        return None

    z = np.load(mp, allow_pickle=True)
    coords = z["coords_l0"]
    pred = z["pred"]
    patch_l0 = int(z["patch_l0"])
    mpp = float(z["mpp"])
    tile_area_mm2 = (patch_l0 * mpp / 1000.0) ** 2

    xy, cls0 = load_cells(cells_dir)
    tile_idx = assign_cells_to_tiles(xy, coords, patch_l0)
    inside = tile_idx >= 0

    # counts[tissue, cell]
    counts = np.zeros((len(TISSUE), len(CELL)), dtype=np.int64)
    tissue_of_cell = np.full(len(cls0), -1, dtype=np.int64)
    tissue_of_cell[inside] = pred[tile_idx[inside]]
    for t in range(len(TISSUE)):
        m = tissue_of_cell == t
        if m.any():
            counts[t] = np.bincount(cls0[m], minlength=len(CELL))

    n_tiles = np.bincount(pred, minlength=len(TISSUE))
    return {"slide_id": slide_id, "counts": counts, "n_tiles": n_tiles,
            "tile_area_mm2": tile_area_mm2, "n_cells": len(cls0),
            "n_inside": int(inside.sum()), "patch_l0": patch_l0, "mpp": mpp}


def report(agg_counts: np.ndarray, agg_tiles: np.ndarray, tile_area_mm2: float,
           n_cells_total: int, n_inside_total: int, n_slides: int):
    logging.info("")
    logging.info("=" * 104)
    logging.info(f"CELL COMPOSITION BY TISSUE CLASS   ({n_slides} slides, "
                 f"{n_inside_total:,} of {n_cells_total:,} nuclei inside "
                 f"classified tiles = {100*n_inside_total/max(n_cells_total,1):.1f}%)")
    logging.info("=" * 104)
    hdr = f"{'tissue':<7}{'tiles':>9}{'nuclei':>11}{'/mm2':>9}  " + \
          "".join(f"{c[:5]:>8}" for c in CELL)
    logging.info(hdr)
    logging.info("-" * 104)
    fracs = np.zeros((len(TISSUE), len(CELL)))
    dens = np.zeros(len(TISSUE))
    for t, name in enumerate(TISSUE):
        n = agg_counts[t].sum()
        area = agg_tiles[t] * tile_area_mm2
        dens[t] = n / area if area > 0 else np.nan
        if n > 0:
            fracs[t] = agg_counts[t] / n
        logging.info(f"{name:<7}{agg_tiles[t]:>9,}{n:>11,}{dens[t]:>9,.0f}  " +
                     "".join(f"{100*fracs[t, c]:>7.1f}%" for c in range(len(CELL))))

    logging.info("")
    logging.info("=" * 104)
    logging.info("DO THE TWO MODELS AGREE?  (each row is a prediction the tissue "
                 "class makes about cells)")
    logging.info("=" * 104)
    passes = 0
    tested = 0
    for name, (desc, cell_name, min_frac, max_dens) in EXPECT.items():
        t = TISSUE.index(name)
        if agg_counts[t].sum() < 500 and agg_tiles[t] < 20:
            logging.info(f"  {name:<5} {desc:<20} -> too few tiles to judge")
            continue
        tested += 1
        if cell_name is not None:
            c = CELL.index(cell_name)
            got = fracs[t, c]
            ok = got >= min_frac
            logging.info(f"  {name:<5} {desc:<20} -> {cell_name} "
                         f"{100*got:5.1f}%  (expect >={100*min_frac:.0f}%)   "
                         f"{'PASS' if ok else 'FAIL'}")
        else:
            got = dens[t]
            ok = got <= max_dens
            logging.info(f"  {name:<5} {desc:<20} -> {got:,.0f} nuclei/mm2  "
                         f"(expect <={max_dens:,.0f})   {'PASS' if ok else 'FAIL'}")
        passes += int(ok)
    logging.info("")
    logging.info(f"  {passes}/{tested} predictions upheld")
    if tested and passes == tested:
        logging.info("  Two independently trained models, neither validated on "
                     "TCGA, agree on all counts.")
    elif tested:
        logging.info("  Any FAIL above marks a tissue class where one of the two "
                     "models is unreliable on TCGA; treat it with care downstream.")

    # the number that matters for the next step
    tum = TISSUE.index("TUM")
    norm = TISSUE.index("NORM")
    n_tum_epi = agg_counts[tum, EPI]
    n_all_epi = agg_counts[:, EPI].sum()
    logging.info("")
    logging.info("=" * 104)
    logging.info("IMPACT ON THE DOWNSTREAM FEATURE")
    logging.info("=" * 104)
    logging.info(f"  epithelial nuclei in TUM tiles     : {n_tum_epi:,}")
    logging.info(f"  epithelial nuclei anywhere         : {n_all_epi:,}")
    logging.info(f"  fraction of 'epithelium' that is tumour: "
                 f"{100*n_tum_epi/max(n_all_epi,1):.1f}%")
    logging.info(f"  epithelial nuclei in NORM tiles    : "
                 f"{agg_counts[norm, EPI]:,} "
                 f"({100*agg_counts[norm, EPI]/max(n_all_epi,1):.1f}% of all "
                 f"epithelium)")
    logging.info("")
    logging.info("  le_per_epi normalises by epithelial count, so restricting to "
                 "TUM changes both\n  numerator and denominator. The line above is "
                 "how much of the denominator was\n  never tumour to begin with.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--all-maps", action="store_true",
                    help="use every tissue map present in --maps-dir")
    ap.add_argument("--maps-dir", default="data/tissue/maps")
    ap.add_argument("--cells-root", default="data/cells")
    ap.add_argument("--out", default="data/tissue/crosscheck.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    maps_dir = Path(args.maps_dir)

    if args.all_maps:
        targets = sorted(p.stem for p in maps_dir.glob("*.npz")
                         if not p.name.startswith("_"))
    elif args.slides:
        targets = args.slides
    else:
        sys.exit("give --slides <ids...> or --all-maps")
    logging.info(f"{len(targets)} slides")

    agg_counts = np.zeros((len(TISSUE), len(CELL)), dtype=np.int64)
    agg_tiles = np.zeros(len(TISSUE), dtype=np.int64)
    n_cells_total = n_inside_total = 0
    tile_areas = []
    rows = []

    for s in tqdm(targets, desc="slides", unit="slide"):
        r = crosscheck_slide(s, maps_dir, Path(args.cells_root))
        if r is None:
            continue
        agg_counts += r["counts"]
        agg_tiles += r["n_tiles"]
        n_cells_total += r["n_cells"]
        n_inside_total += r["n_inside"]
        tile_areas.append(r["tile_area_mm2"])
        c = r["counts"]
        row = {"slide_id": s, "n_cells": r["n_cells"],
               "pct_inside": 100 * r["n_inside"] / max(r["n_cells"], 1)}
        for t, name in enumerate(TISSUE):
            tot = c[t].sum()
            row[f"n_{name}"] = int(tot)
            row[f"{name}_epi_frac"] = float(c[t, EPI] / tot) if tot else np.nan
            row[f"{name}_conn_frac"] = float(c[t, CONN] / tot) if tot else np.nan
            row[f"{name}_lym_frac"] = float(c[t, LYM_C] / tot) if tot else np.nan
        rows.append(row)

    if not rows:
        sys.exit("no slides processed")

    report(agg_counts, agg_tiles, float(np.mean(tile_areas)),
           n_cells_total, n_inside_total, len(rows))

    df = pd.DataFrame(rows)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    # full cross-tab for the record
    ct = pd.DataFrame(agg_counts, index=TISSUE, columns=CELL)
    ct.to_csv(p.with_name(p.stem + "_crosstab.csv"))
    logging.info(f"\nwrote {p}")
    logging.info(f"wrote {p.with_name(p.stem + '_crosstab.csv')}")


if __name__ == "__main__":
    main()
