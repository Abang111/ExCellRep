# ExCellRep

**CellGraph-MSI: Explicit Cellular Representations for Microsatellite Instability Prediction from Histology**

ExCellRep implements the core CellGraph-MSI pipeline for slide-level MSI
prediction in colorectal H&E histology. It combines nuclear instances and a
tissue-compartment map into spatial graphs. Every node feature and edge
attribute corresponds to a defined measurement, such as nuclear type,
neighbourhood composition, tissue compartment, or contour separation.

[Method](#method) · [Reported results](#reported-results) ·
[Getting started](#getting-started) · [Pipeline](#pipeline) ·
[Reproducibility](#reproducibility)

## Method

**Measure.** HoVer-NeXt provides nuclear classes and instance contours. A
nine-class linear probe trained on frozen UNI2-h patch features assigns tissue
compartments, including tumour epithelium, stroma, normal mucosa, and mucin.
These upstream outputs define the inputs to the graph. The main MSI predictor
operates on the explicit graph features.

**Build.** The processed tumour region defines a set of 400 × 400 µm windows.
Each nuclear node stores 26 features. The tissue compartment and distance to
the processed tumour-region boundary describe where the nucleus lies. Its
type, size, and local neighbourhood describe the cellular context at that
location.

| Node feature group | Dimensions | Measurement |
| --- | ---: | --- |
| Nuclear type | 7 | One-hot HoVer-NeXt class |
| Nuclear size | 1 | Maximum centroid-to-sampled-contour distance in µm |
| Local nuclear count | 1 | `log(1 + count)` within 25 µm, including the centre nucleus |
| Local composition | 7 | Proportions of the seven nuclear classes within the same neighbourhood |
| Tissue compartment | 9 | One-hot tissue-probe class |
| Tumour-boundary distance | 1 | Signed distance to the processed mask boundary in mm |

Each nucleus proposes up to 12 directed neighbours by centroid distance within
its window. The graph keeps edges with sampled contour separation at most
40 µm. The predictor encodes separation with radial basis functions and a
normalized scalar, adds a cross-compartment indicator, and learns an embedding
for the ordered nuclear-type pair.

**Predict.** The default model uses three edge-aware message-passing layers
and gated aggregation across depths. Attention pools nuclei into windows and
then windows into a slide representation. An auxiliary head estimates
lymphocyte–epithelial proximity and mucin-associated tissue content. The MSI
classifier receives the slide representation and these predicted phenotypes.
The corresponding measured phenotypes supervise training.

## Reported results

The values below are for **CellGraph-MSI** in manuscript Table 1. MSI-H is the
positive class. MSI-L is grouped with MSS.

| Cohort | Eligible slides | MSI-H | Evaluation | AUROC, mean ± SD |
| --- | ---: | ---: | --- | ---: |
| TCGA-COAD | 409 | 66 | Internal, five fixed folds | 0.9061 ± 0.0509 |
| CPTAC-COAD | 72 | 19 | External, five frozen TCGA checkpoints | 0.8238 ± 0.0258 |
| SurGen | 513 | 82 | External, five frozen TCGA checkpoints | 0.8495 ± 0.0188 |

For TCGA-COAD, the paper also reports **AUPRC 0.7460 ± 0.1301**,
**balanced accuracy 0.7810 ± 0.0844**, and **F1 0.6264 ± 0.1394**.
Balanced accuracy and F1 use a fixed threshold of 0.5.

The scored TCGA folds also select their checkpoints by AUROC, so the internal
figures are best-epoch estimates. The external figures average the five
individual checkpoint scores for the primary run. Their standard deviations
describe checkpoint variation. The CPTAC script additionally prints ensemble
AUROC, which evaluates averaged slide probabilities.

## Getting started

Run the examples from the repository root. For processing raw WSIs, use a
Linux environment with CUDA, a compatible PyTorch and torchvision pair,
and the native OpenSlide library. Install the remaining Python dependencies.

```bash
python -m pip install -r requirements.txt
```

The [reproduction guide](docs/REPRODUCIBILITY.md) lists the exact directory
layout, input fields, external software, and commands for each stage.
Choose your starting point based on the assets you have.

| Starting point | What you need | First action |
| --- | --- | --- |
| Raw WSIs | Cohort WSIs, labels and fold manifests, tissue-probe training patches or weights, HoVer-NeXt and UNI2-h assets | Follow stages 1–5 in the [reproduction guide](docs/REPRODUCIBILITY.md#1-tile-wsis-and-extract-nuclei) |
| Precomputed graphs | Per-slide graphs, MSI labels, original fold assignments, and both phenotype CSVs | Preview and launch training as shown below |
| Trained checkpoints | Five TCGA checkpoints for a seed, CPTAC graphs, the eligible MSI labels, and phenotype CSVs | Inspect the checkpoints, then follow the [CPTAC command](docs/REPRODUCIBILITY.md#optional-evaluate-cptac-coad) |

### Train from precomputed graphs

Place the graphs and CSVs at the default paths shown below. The launcher runs
one five-fold job per seed and restarts incomplete seeds from the beginning.

```text
data/
├── graphs/<slide_id>.pt
├── labels/msi_tcga.csv
├── labels/metadata_tcga_crc_dx.csv
├── labels/folds_tcga.csv
├── tissue/compartment_iel.csv
└── tissue/crosscheck.csv
```

```bash
# Preview the trainer commands.
python scripts/11_run_training.py --gpus 0 --seeds 0 1 2 --dry-run

# Run the three seeds. Set --gpus to the available CUDA device IDs.
python scripts/11_run_training.py --gpus 0 --seeds 0 1 2

# Inspect the five checkpoints for seed 0.
python scripts/12_infer_cptac.py --inspect --run-root runs/msi --tag L1_concept --seeds 0
```

The trainer defaults to 48 sampled windows per slide, 30 epochs, batch size
4, hidden width 96, and auxiliary loss weight 0.3. The reproduction guide shows
how to pass custom input paths through `--extra`.

## Pipeline

| Stage | Script | Original filename | Output or role |
| --- | --- | --- | --- |
| 1. Nuclear measurements | [01_tile_wsi.py](scripts/01_tile_wsi.py) | `01_tile_wsi.py` | Coordinate-only HDF5 files defining the slide list |
| 1. Nuclear measurements | [02_extract_cells.py](scripts/02_extract_cells.py) | `10_extract_cells.py` | HoVer-NeXt nuclear instances and class labels |
| 2. Tissue measurements | [03_extract_features.py](scripts/03_extract_features.py) | `02_extract_features.py` | Shared UNI2-h model and preprocessing, loaded by scripts 04 and 05 |
| 2. Tissue measurements | [04_train_tissue_probe.py](scripts/04_train_tissue_probe.py) | `tissue_probe_train.py` | NCT-CRC feature extraction and nine-class probe training |
| 2. Tissue measurements | [05_apply_tissue_probe.py](scripts/05_apply_tissue_probe.py) | `tissue_apply.py` | Slide-level tissue maps on a separate tile grid |
| 3. Phenotype targets | [06_crosscheck_tissue_cells.py](scripts/06_crosscheck_tissue_cells.py) | `tissue_cell_crosscheck.py` | Per-compartment nuclear counts used to derive `frac_MUC` |
| 3. Phenotype targets | [07_build_distance_graph.py](scripts/07_build_distance_graph.py) | `build_distance_graph.py` | Shared contour and distance routines, loaded by scripts 08 and 09 |
| 3. Phenotype targets | [08_compute_compartment_iel.py](scripts/08_compute_compartment_iel.py) | `compartment_iel.py` | The `le_per_epi_0_5um_TUM` proximity target |
| 4. Graph construction | [09_build_graphs.py](scripts/09_build_graphs.py) | `01_build_graphs.py` | Per-slide graph files and the build manifest |
| 5. Model training | [10_train_cellgraph.py](scripts/10_train_cellgraph.py) | `_gen_train_v17.py` | Fold metrics, out-of-fold predictions, and checkpoints |
| 5. Model training | [11_run_training.py](scripts/11_run_training.py) | `01_run.py` | Three-seed launcher with independent run directories and resume |
| 6. External evaluation | [12_infer_cptac.py](scripts/12_infer_cptac.py) | `02_infer_cptac.py` | Per-checkpoint CPTAC scores and slide predictions |

## Reproducibility

### Main outputs

| Output | Use |
| --- | --- |
| `data/graphs/_manifest.csv` | Check graph eligibility, counts, and sizes before training |
| `runs/msi/seed0/L1_concept/fold_records.csv` | Compare metrics from the five selected folds |
| `runs/msi/seed0/L1_concept/oof_predictions.csv` | Analyze held-out predictions by slide |
| `runs/msi/seed0/L1_concept_seed0/model_fold*.pt` | Transfer frozen models and their training-fold normalization statistics |
| `evaluation/cptac/per_ckpt_w48.csv` | Compute the external mean and sample SD across checkpoints for the chosen seed |
| `evaluation/cptac/probs_w48.npy` and `slide_order.csv` | Map saved CPTAC probabilities back to slides |

### Release scope and validation

- **Code included.** WSI tiling, HoVer-NeXt orchestration, UNI2-h tissue-probe
  training and application, phenotype measurements, graph construction,
  TCGA MSI training, and CPTAC-COAD evaluation.
- **Assets supplied separately.** WSIs, cohort labels, exact fold assignments,
  external inference code, upstream model weights, and study checkpoints.
  Exact cohort results depend on the original manifests and run configuration.
- **Other manuscript experiments.** SurGen, mutation benchmarks, baseline
  comparisons, ablations, and spatial interventions require their own
  experiment scripts or assets.
- **Release checks.** On a synthetic fixture, the renamed graph builder
  matched the source builder's outputs. CPU five-fold training, seed resume,
  and the CPTAC evaluation entry point ran on synthetic inputs. The table
  above reports manuscript results. Full-cohort GPU runs require the study
  assets listed above.

## License

The repository code is distributed under the [Apache 2.0 license](LICENSE).
Datasets, external software, and pretrained weights are subject to their
respective terms.
