"""Extract NCT-CRC patch features, then fit and validate the tissue probe."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    class _NoBar:
        def __init__(self, it=None, **kw): self._it = it
        def __iter__(self): return iter(self._it if self._it is not None else [])
        def update(self, n=1): pass
        def close(self): pass
    def tqdm(it=None, **kw): return _NoBar(it, **kw)

# NCT-CRC-HE-100K class folders
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_LONG = {
    "ADI": "adipose", "BACK": "background", "DEB": "debris",
    "LYM": "lymphocytes", "MUC": "mucus", "MUS": "smooth muscle",
    "NORM": "normal colon mucosa", "STR": "cancer-associated stroma",
    "TUM": "adenocarcinoma epithelium",
}
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# reuse the project's own UNI2-h builder (module name starts with a digit)
# ---------------------------------------------------------------------------
def load_project_extractor(path: str = str(Path(__file__).with_name("03_extract_features.py"))):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[fatal] cannot find {p}; pass --extractor-py")
    spec = importlib.util.spec_from_file_location("proj_extract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_transform(mod):
    """Exactly the transform used for the project's FM features."""
    from torchvision import transforms
    mean = mod.SlidePatchDataset.NORM_MEAN
    std = mod.SlidePatchDataset.NORM_STD
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]), mean, std


# ---------------------------------------------------------------------------
def list_patches(root: Path) -> Tuple[List[Path], np.ndarray]:
    """Expect root/<CLASS>/*.tif . Returns (paths, y)."""
    paths, ys = [], []
    for ci, c in enumerate(CLASSES):
        d = root / c
        if not d.is_dir():
            logging.warning(f"class dir missing: {d}")
            continue
        fs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        paths += fs
        ys += [ci] * len(fs)
        logging.info(f"  {c:<5} ({CLASS_LONG[c]:<26}) {len(fs):>7} patches")
    if not paths:
        # some releases nest one level deeper
        sub = [d for d in root.iterdir() if d.is_dir()]
        sys.exit(f"[fatal] no class folders under {root}. Found: "
                 f"{[d.name for d in sub][:12]}")
    return paths, np.array(ys, dtype=np.int64)


class PatchList:
    """Minimal dataset over a list of image files."""
    def __init__(self, paths: List[Path], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


@np.errstate(all="ignore")
def extract_features(root: Path, model, tf, device, batch_size: int,
                     num_workers: int, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    logging.info(f"[{tag}] scanning {root}")
    paths, y = list_patches(root)
    logging.info(f"[{tag}] total {len(paths)} patches")

    ds = PatchList(paths, tf)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, shuffle=False)
    feats = np.empty((len(ds), 1536), dtype=np.float16)
    cur = 0
    t0 = time.time()
    with torch.inference_mode():
        for batch in tqdm(dl, desc=f"{tag} UNI2", unit="batch"):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(batch)
            out = out.float().cpu().numpy().astype(np.float16)
            feats[cur:cur + len(out)] = out
            cur += len(out)
    dt = time.time() - t0
    logging.info(f"[{tag}] {cur} patches in {dt:.0f}s ({cur/max(dt,1e-9):.0f}/s)")
    return feats, y


# ---------------------------------------------------------------------------
def stage_features(args):
    import torch
    mod = load_project_extractor(args.extractor_py)
    tf, mean, std = build_transform(mod)
    logging.info(f"transform normalise mean={mean} std={std}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        sys.exit("[fatal] CUDA unavailable; UNI2-h on CPU is impractical")
    logging.info(f"device={device} ({torch.cuda.get_device_name(0)})")
    model = mod.build_uni2h(args.uni_dir, device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, root in (("train", args.nct_root), ("val", args.val_root)):
        if root is None:
            continue
        rp = Path(root)
        if not rp.exists():
            logging.warning(f"[{tag}] {rp} not found, skipping")
            continue
        f, y = extract_features(rp, model, tf, device,
                                args.batch_size, args.num_workers, tag)
        np.savez_compressed(out_dir / f"nct_feats_{tag}.npz",
                            features=f, labels=y, classes=np.array(CLASSES))
        logging.info(f"[{tag}] wrote {out_dir / f'nct_feats_{tag}.npz'}")


def stage_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, classification_report)
    import joblib

    out_dir = Path(args.out_dir)
    tr = np.load(out_dir / "nct_feats_train.npz")
    Xtr, ytr = tr["features"].astype(np.float32), tr["labels"]
    logging.info(f"train: {Xtr.shape}")

    val_path = out_dir / "nct_feats_val.npz"
    if val_path.exists():
        va = np.load(val_path)
        Xva, yva = va["features"].astype(np.float32), va["labels"]
        val_name = "CRC-VAL-HE-7K (different patients)"
    else:
        logging.warning("no val features; holding out 10% of train instead "
                        "(NOTE: same patients, so this OVERSTATES accuracy)")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(Xtr))
        k = int(0.1 * len(idx))
        Xva, yva = Xtr[idx[:k]], ytr[idx[:k]]
        Xtr, ytr = Xtr[idx[k:]], ytr[idx[k:]]
        val_name = "10% of train (same patients)"
    logging.info(f"val  : {Xva.shape}   [{val_name}]")

    # L2-normalise: standard for linear probes on frozen features
    def l2(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-8)
    Xtr, Xva = l2(Xtr), l2(Xva)

    clf = LogisticRegression(max_iter=args.max_iter, C=args.C,
                             multi_class="multinomial", n_jobs=-1)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    logging.info(f"fitted in {time.time()-t0:.0f}s")

    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred)
    bacc = balanced_accuracy_score(yva, pred)
    logging.info("")
    logging.info("=" * 78)
    logging.info(f"LINEAR PROBE on frozen UNI2-h — evaluated on {val_name}")
    logging.info("=" * 78)
    logging.info(f"accuracy          : {acc:.4f}")
    logging.info(f"balanced accuracy : {bacc:.4f}")
    logging.info("")
    logging.info(classification_report(yva, pred, target_names=CLASSES,
                                       digits=3, zero_division=0))

    cm = confusion_matrix(yva, pred, labels=range(len(CLASSES)))
    logging.info("confusion matrix (rows = truth, cols = predicted)")
    logging.info("        " + "".join(f"{c:>7}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        logging.info(f"{c:<8}" + "".join(f"{v:>7d}" for v in cm[i]))

    # the pair the downstream analysis actually depends on
    ti, ni = CLASSES.index("TUM"), CLASSES.index("NORM")
    tum_as_norm = int(cm[ti, ni]); norm_as_tum = int(cm[ni, ti])
    logging.info("")
    logging.info("-" * 78)
    logging.info("TUM vs NORM — the distinction the whole compartment step rests on")
    logging.info("-" * 78)
    logging.info(f"  TUM recall        : {cm[ti,ti]/max(cm[ti].sum(),1):.4f}  "
                 f"({cm[ti,ti]}/{cm[ti].sum()})")
    logging.info(f"  NORM recall       : {cm[ni,ni]/max(cm[ni].sum(),1):.4f}  "
                 f"({cm[ni,ni]}/{cm[ni].sum()})")
    logging.info(f"  TUM  called NORM  : {tum_as_norm}")
    logging.info(f"  NORM called TUM   : {norm_as_tum}   <-- this one contaminates "
                 f"the tumour compartment")

    joblib.dump({"clf": clf, "classes": CLASSES, "l2_normalised": True},
                out_dir / "probe.joblib")
    report = {"accuracy": float(acc), "balanced_accuracy": float(bacc),
              "val_set": val_name, "classes": CLASSES,
              "confusion_matrix": cm.tolist(),
              "tum_recall": float(cm[ti, ti] / max(cm[ti].sum(), 1)),
              "norm_recall": float(cm[ni, ni] / max(cm[ni].sum(), 1)),
              "norm_called_tum": norm_as_tum, "tum_called_norm": tum_as_norm}
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    logging.info(f"\nwrote {out_dir/'probe.joblib'} and {out_dir/'probe_report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "probe"], required=True)
    ap.add_argument("--nct-root", default="data/nct_crc/NCT-CRC-HE-100K-NONORM")
    ap.add_argument("--val-root", default="data/nct_crc/CRC-VAL-HE-7K")
    ap.add_argument("--uni-dir", default=None,
                    help="directory holding UNI2-h pytorch_model.bin")
    ap.add_argument("--extractor-py", default=str(Path(__file__).with_name("03_extract_features.py")))
    ap.add_argument("--out-dir", default="data/tissue")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.stage == "features":
        if not args.uni_dir:
            sys.exit("[fatal] --uni-dir is required for the features stage")
        stage_features(args)
    else:
        stage_probe(args)


if __name__ == "__main__":
    main()
