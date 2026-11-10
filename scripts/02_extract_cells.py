"""Run the HoVer-NeXt inference CLI on whole-slide images."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HOVERNEXT_REPO = "third_party/hover_next_inference"
WEIGHTS_NAME   = "lizard_convnextv2_large"


def find_wsi(sid, manifest_df, wsi_dir):
    """Locate the WSI file path for a given slide_id."""
    row = manifest_df[manifest_df["slide_id"] == sid]
    if not row.empty:
        wsi_path = row.iloc[0].get("wsi_path", "")
        if wsi_path and Path(str(wsi_path)).exists():
            return str(wsi_path)
    # fallback: search wsi_dir for matching .svs
    for p in Path(wsi_dir).rglob(f"*{sid}*.svs"):
        return str(p)
    return None


def slide_already_done(out_root, slide_id, wsi_path):
    """HoVer-NeXt names the output dir after the WSI basename (no .svs)."""
    # the output dir uses the basename of the WSI minus extension
    wsi_stem = Path(wsi_path).stem
    out_dir = Path(out_root) / wsi_stem
    return (out_dir / "class_inst.json").exists()


def run_one(wsi_path, out_root, gpu, tta=4, inf_workers=8, pp_workers=8,
            pp_tiling=10, hovernext_dir=HOVERNEXT_REPO):
    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "main.py",
        "--input", str(Path(wsi_path).resolve()),
        "--output_root", str(Path(out_root).resolve()),
        "--cp", WEIGHTS_NAME,
        "--tta", str(tta),
        "--inf_workers", str(inf_workers),
        "--pp_workers", str(pp_workers),
        "--pp_tiling", str(pp_tiling),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=hovernext_dir, env=env,
                          capture_output=True, text=True, check=False)
    dur = time.time() - t0
    return proc.returncode, dur, proc.stdout[-500:], proc.stderr[-500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-dir",
                    default="data/patches")
    ap.add_argument("--wsi-dir",
                    default="data/wsi/tcga_crc_dx")
    ap.add_argument("--manifest",
                    default="data/labels/metadata_tcga_crc_dx.csv")
    ap.add_argument("--out-root",
                    default="data/cells")
    ap.add_argument("--hovernext-dir", default=HOVERNEXT_REPO)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0,
                    help="logical GPU id within the shard (usually 0 since "
                         "CUDA_VISIBLE_DEVICES is set by the shell launcher)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--slide-list", default=None,
                    help="optional .txt with one slide_id per line "
                         "(for pilot: hardcore + controls only)")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--inf-workers", type=int, default=8)
    ap.add_argument("--pp-workers", type=int, default=8)
    ap.add_argument("--pp-tiling", type=int, default=10)
    args = ap.parse_args()

    if not (Path(args.hovernext_dir) / "main.py").is_file():
        ap.error("--hovernext-dir must contain the HoVer-NeXt main.py")
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # decide slide list
    if args.slide_list:
        with open(args.slide_list) as f:
            all_slides = [l.strip() for l in f if l.strip()]
    else:
        patch_files = sorted(Path(args.patches_dir).glob("*.h5"))
        all_slides = [f.stem for f in patch_files]

    # shard
    my_slides = [s for i, s in enumerate(all_slides)
                 if i % args.num_shards == args.shard]
    if args.limit > 0:
        my_slides = my_slides[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(my_slides)} slides "
          f"on GPU {args.gpu}", flush=True)

    manifest = pd.read_csv(args.manifest) if Path(args.manifest).exists() \
        else pd.DataFrame(columns=["slide_id", "wsi_path"])

    n_done = n_skip = n_fail = n_nowsi = 0
    t_total = time.time()

    for i, sid in enumerate(my_slides, 1):
        wsi = find_wsi(sid, manifest, args.wsi_dir)
        if wsi is None:
            print(f"  [{i}/{len(my_slides)}] {sid}: no WSI found, skip",
                  flush=True)
            n_nowsi += 1
            continue
        if slide_already_done(out_root, sid, wsi):
            n_skip += 1
            continue

        rc, dur, stdout, stderr = run_one(
            wsi, out_root, args.gpu, args.tta,
            args.inf_workers, args.pp_workers, args.pp_tiling, args.hovernext_dir,
        )
        if rc == 0:
            n_done += 1
            elapsed = (time.time() - t_total) / 60
            eta = (elapsed / max(n_done, 1)) * (len(my_slides) - i - n_skip - n_nowsi)
            print(f"  [{i}/{len(my_slides)}] {sid}: OK ({dur/60:.1f} min)  "
                  f"running {elapsed:.0f}m, ETA ~{eta:.0f}m", flush=True)
        else:
            n_fail += 1
            print(f"  [{i}/{len(my_slides)}] {sid}: FAIL rc={rc}\n"
                  f"      stderr tail: {stderr.strip()[-300:]}", flush=True)

    print(f"\n[shard {args.shard}] done: {n_done} ok, {n_skip} skipped, "
          f"{n_fail} failed, {n_nowsi} no-wsi  total {(time.time()-t_total)/60:.0f}m",
          flush=True)


if __name__ == "__main__":
    main()
