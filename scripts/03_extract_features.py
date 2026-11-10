"""Define the UNI2-h encoder and preprocessing shared by tissue-probe scripts."""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import openslide
import timm
from timm.layers import SwiGLUPacked


def build_uni2h(uni_dir: str, device: torch.device) -> torch.nn.Module:
    """Build UNI2-h with the EXACT MahmoodLab spec, then load local weights."""
    timm_kwargs = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, **timm_kwargs
    )

    weights_path = Path(uni_dir) / "pytorch_model.bin"
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[uni2h] loaded with strict=False")
    print(f"[uni2h] missing keys     ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print(f"[uni2h] unexpected keys  ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    if len(missing) > 20:
        print(f"[uni2h] WARNING: many missing keys — architecture may still be wrong",
              file=sys.stderr)

    # quick forward check
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
    assert out.shape == (1, 1536), f"expected (1,1536), got {tuple(out.shape)}"
    print(f"[uni2h] forward sanity check OK, output dim = {out.shape[-1]}")

    model.to(device)
    return model


class SlidePatchDataset(Dataset):
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD  = (0.229, 0.224, 0.225)

    def __init__(self, svs_path: str, coords: np.ndarray, patch_l0: int):
        self.svs_path = svs_path
        self.coords = coords
        self.patch_l0 = patch_l0
        self.tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.NORM_MEAN, self.NORM_STD),
        ])
        self._slide = None

    def _slide_handle(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.svs_path)
        return self._slide

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        img = self._slide_handle().read_region(
            (int(x), int(y)), 0, (self.patch_l0, self.patch_l0)
        ).convert("RGB")
        return self.tf(img)


@torch.inference_mode()
def extract_one_slide(slide_id, patches_h5, out_h5, model, device,
                      batch_size, num_workers):
    if out_h5.exists():
        return {"slide_id": slide_id, "status": "skipped", "n": -1, "sec": 0.0}

    with h5py.File(patches_h5, "r") as f:
        coords = f["coords"][:]
        patch_l0 = int(f.attrs["patch_size_level0"])
        svs_path = f.attrs["svs_path"]
        if isinstance(svs_path, bytes):
            svs_path = svs_path.decode()

    if len(coords) == 0:
        return {"slide_id": slide_id, "status": "no_coords", "n": 0, "sec": 0.0}

    ds = SlidePatchDataset(svs_path, coords, patch_l0)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)

    feats = np.empty((len(ds), 1536), dtype=np.float16)
    t0 = time.time()
    cur = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(batch)
        feats[cur:cur + out.shape[0]] = out.float().cpu().numpy().astype(np.float16)
        cur += out.shape[0]
    sec = time.time() - t0

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("features", data=feats, compression="gzip", compression_opts=4)
        f.create_dataset("coords",   data=coords, compression="gzip")
        f.attrs["slide_id"] = slide_id
        f.attrs["dim"] = 1536
    return {"slide_id": slide_id, "status": "ok", "n": len(ds), "sec": sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-dir",  required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--uni-dir",      required=True)
    ap.add_argument("--batch-size",   type=int, default=256)
    ap.add_argument("--num-workers",  type=int, default=8)
    ap.add_argument("--shard-idx",    type=int, default=0)
    ap.add_argument("--shard-total",  type=int, default=1)
    ap.add_argument("--max-slides",   type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: CUDA not available, refusing to run on CPU "
              "(would take days). Fix torch/CUDA first.", file=sys.stderr)
        sys.exit(2)

    print(f"[extract] device={device}  GPU={torch.cuda.get_device_name(0)}")
    print(f"[extract] loading UNI2-h from {args.uni_dir} ...", flush=True)
    model = build_uni2h(args.uni_dir, device)
    print(f"[extract] model ready", flush=True)

    patches_dir  = Path(args.patches_dir)
    features_dir = Path(args.features_dir)
    h5s = sorted(p for p in patches_dir.glob("*.h5") if not p.name.startswith("_"))
    h5s = [p for i, p in enumerate(h5s) if i % args.shard_total == args.shard_idx]
    if args.max_slides > 0:
        h5s = h5s[:args.max_slides]

    print(f"[extract] shard {args.shard_idx}/{args.shard_total}: {len(h5s)} slides",
          flush=True)

    log = []
    for i, p in enumerate(h5s, 1):
        slide_id = p.stem
        out = features_dir / f"{slide_id}.h5"
        res = extract_one_slide(slide_id, p, out, model, device,
                                args.batch_size, args.num_workers)
        log.append(res)
        rate = res["n"] / max(res["sec"], 1e-6) if res["n"] > 0 else 0
        print(f"  [{i}/{len(h5s)}] {slide_id}  status={res['status']} "
              f"n={res['n']}  {res['sec']:.1f}s  ({rate:.0f} patch/s)", flush=True)

    import pandas as pd
    log_path = features_dir / f"_extract_log_shard{args.shard_idx}.csv"
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"[done] log -> {log_path}")


if __name__ == "__main__":
    main()	
