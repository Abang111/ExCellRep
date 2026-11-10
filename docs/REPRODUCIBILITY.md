# Reproduction guide

The commands use Bash from the repository root. A Linux environment with
CUDA is needed for the HoVer-NeXt and UNI2-h stages. Use a Python environment
compatible with the HoVer-NeXt inference checkout. Install OpenSlide's native
library (for example, `libopenslide0` on Ubuntu) and a matching CUDA build of
PyTorch and torchvision, then install the Python dependencies.

```bash
python -m pip install -r requirements.txt
```

Install the upstream HoVer-NeXt inference project and its dependencies
according to its own setup instructions. Point `--hovernext-dir` to the
checkout containing the inference `main.py` and configure the upstream
`lizard_convnextv2_large` checkpoint there. The UNI2-h scripts read
`weights/UNI2-h/pytorch_model.bin`. Use the original checkpoints and input
manifests to reproduce the study.

## Inputs

| Path used below | Expected contents |
| --- | --- |
| `data/wsi/tcga_crc_dx/` | GDC UUID subdirectories containing `.svs` files whose basenames start with the corresponding slide ID |
| `data/labels/msi_tcga.csv` | One row per eligible input slide, with `slide_id,msi_status`. The trainer treats `MSI-H` as positive and groups MSI-L with MSS. |
| `data/labels/metadata_tcga_crc_dx.csv` | `slide_id,project_id` for COAD filtering. HoVer-NeXt can also use a `wsi_path` column to locate files. |
| `data/labels/folds_tcga.csv` | The original `slide_id,fold` assignments for the five TCGA folds. A previous FM prediction CSV with these columns also works. |
| `data/nct_crc/NCT-CRC-HE-100K-NONORM/` | Patch images grouped into the nine tissue class directories below |
| `data/nct_crc/CRC-VAL-HE-7K/` | Independent validation patches in the same layout |

The tissue class order is `ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`.
Only load trusted serialized graph files, checkpoints and probe files.
Check the stage logs and manifests for skipped slides. For the paper's
primary TCGA MSI run, the intersection of COAD labels, folds and valid graphs
contains 409 slides, including 66 MSI-H.

## 1. Tile WSIs and extract nuclei

```bash
python scripts/01_tile_wsi.py --labels data/labels/msi_tcga.csv --wsi-root data/wsi/tcga_crc_dx --out-dir data/patches --workers 8
CUDA_VISIBLE_DEVICES=0 python scripts/02_extract_cells.py --hovernext-dir third_party/hover_next_inference --patches-dir data/patches --wsi-dir data/wsi/tcga_crc_dx --manifest data/labels/metadata_tcga_crc_dx.csv --out-root data/cells
```

The `*.h5` files store tile coordinates and their metadata. Script 02 uses
the HDF5 filenames as its slide list and runs HoVer-NeXt on the whole slide.
The downstream scripts use `class_inst.json` and `pinst_pp.zip` from its
slide output directory. Set `CUDA_VISIBLE_DEVICES` to choose the physical GPU.
For multiple processes, use one per GPU with distinct `--shard` values from
0 through `--num-shards - 1`.

## 2. Train and apply the tissue probe

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/04_train_tissue_probe.py --stage features --nct-root data/nct_crc/NCT-CRC-HE-100K-NONORM --val-root data/nct_crc/CRC-VAL-HE-7K --uni-dir weights/UNI2-h --out-dir data/tissue
python scripts/04_train_tissue_probe.py --stage probe --out-dir data/tissue
CUDA_VISIBLE_DEVICES=0 python scripts/05_apply_tissue_probe.py --all --labels-csv data/labels/msi_tcga.csv --wsi-root data/wsi/tcga_crc_dx --probe data/tissue/probe.joblib --uni-dir weights/UNI2-h --out-dir data/tissue/maps
```

Script 03 supplies UNI2-h construction and preprocessing to both commands that
use the encoder. The features stage writes `nct_feats_train.npz` and
`nct_feats_val.npz`. The probe stage writes `probe.joblib` and
`probe_report.json`. Script 05 tiles each WSI independently of stage 1 and
writes `data/tissue/maps/<slide_id>.npz` with level-0 tile coordinates, the
predicted tissue classes and the level-0 tile size.

## 3. Generate phenotype targets

```bash
python scripts/06_crosscheck_tissue_cells.py --all-maps --maps-dir data/tissue/maps --cells-root data/cells --out data/tissue/crosscheck.csv
python scripts/08_compute_compartment_iel.py --all --labels-csv data/labels/msi_tcga.csv --meta-csv data/labels/metadata_tcga_crc_dx.csv --wsi-root data/wsi/tcga_crc_dx --maps-dir data/tissue/maps --cells-root data/cells --contour-cache data/contour_cache --out data/tissue/compartment_iel.csv
```

Script 10 derives `frac_MUC` as `n_MUC / sum(n_<tissue>)` over the nine
counts in `crosscheck.csv`. It reads `le_per_epi_0_5um_TUM` directly from
`compartment_iel.csv`. Script 08 imports script 07 for contour processing,
as does the graph builder. The contour cache can be reused between scripts
08 and 09 for the same nuclei and slide assets. Use distinct caches for
different cohorts or segmentation runs.

## 4. Build the graphs

```bash
python scripts/09_build_graphs.py --all --labels-csv data/labels/msi_tcga.csv --wsi-root data/wsi/tcga_crc_dx --maps-dir data/tissue/maps --cells-root data/cells --contour-cache data/contour_cache --out-dir data/graphs
```

The builder uses the WSI to read pixel spacing, the tissue map to select
tumour-guided regions, and the HoVer-NeXt outputs to retrieve the nuclear
classes, centroids and contours. Default window size is 400 µm, the nuclear
composition radius is 25 µm, and edges join up to 12 nearest-centroid
candidates subject to a 40 µm contour-distance cutoff. Per-slide `.pt`
files contain 26-dimensional node features and directed edges. Check
`data/graphs/_manifest.csv` before training.

## 5. Train on fixed folds

```bash
python scripts/11_run_training.py --gpus 0 --seeds 0 1 2 --tag L1_concept --out-root runs/msi --extra \
  --graph-dir data/graphs --labels-csv data/labels/msi_tcga.csv \
  --meta-csv data/labels/metadata_tcga_crc_dx.csv --fm-csv data/labels/folds_tcga.csv \
  --compartment-csv data/tissue/compartment_iel.csv --crosscheck-csv data/tissue/crosscheck.csv
```

Put `--dry-run` before `--extra` to inspect the launch. Use
`--gpus 0 1` to run two seeds concurrently where available. The launcher
calls script 10 and gives each seed its own output directory. Rerunning the
same command skips completed seeds and restarts incomplete seeds. Use a
separate output root or tag for a different recipe.

The trainer defaults are 48 sampled windows, 30 epochs, batch size 4,
learning rate `3e-4`, hidden dimension 96, three layers, and auxiliary loss
weight 0.3. Leave the optional UNI2 region embedding and FM logit fusion
flags off for the explicit-graph predictor. The fold CSV sets the held-out
splits and checkpoint selection uses the highest AUROC on each scored fold.
The trainer fits node and phenotype normalization on the training fold.
Preserve the exact study inputs and run configuration when comparing with
the manuscript.

```text
runs/msi/seed0/L1_concept/fold_records.csv
runs/msi/seed0/L1_concept/oof_predictions.csv
runs/msi/seed0/L1_concept/epoch_history.csv
runs/msi/seed0/L1_concept_seed0/model_fold0.pt
runs/log_L1_concept_seed0.log
```

## Optional. Evaluate CPTAC-COAD

Prepare CPTAC graphs and phenotype tables using the same stages in separate
directories. Supply the original 72-slide eligible MSI manifest, containing
19 MSI-H slides, for the primary CPTAC comparison. The supplied evaluator
intersects the labelled cohort with available graphs and both phenotype
tables, so verify the printed slide counts before interpreting scores.

```bash
python scripts/12_infer_cptac.py --inspect --run-root runs/msi --tag L1_concept --seeds 0
python scripts/12_infer_cptac.py --run-root runs/msi --tag L1_concept --seeds 0 \
  --graphs data/cptac/graphs --labels data/cptac/labels.csv \
  --cptac-comp data/cptac/tissue/compartment_iel.csv \
  --cptac-tissue data/cptac/tissue/crosscheck.csv --out evaluation/cptac --gpu 0
```

The evaluator uses frozen TCGA checkpoints and stored training-fold
statistics. `per_ckpt_w48.csv` reports individual AUROCs, and the saved
NumPy arrays follow `slide_order.csv`. For the manuscript's external summary,
use the mean and sample SD across the five per-checkpoint AUROCs for the
corresponding seed. The legacy console summary also reports an ensemble
AUROC and a between-seed SD, which summarize different quantities.
