# s1lulc — Sentinel-1-only land cover mapping and monthly change analysis

Semantic segmentation of land use / land cover from Sentinel-1 C-band SAR (VV + VH) only, and a
prototype that, for any latitude/longitude and start month, produces 12 monthly maps, class
percentages, month-to-month and total change (in percentage points), error-adjusted estimates
with confidence intervals, change-significance flags, uncertainty maps and an independent check
against Dynamic World.

> Status: complete pipeline implemented and tested (export, dataset, training, evaluation,
> prototype with interactive HTML report).

## Pipeline

```
Earth Engine ──export──> 41 region GeoTIFFs (Drive) ──build-dataset──> region .npz + norm stats
     │                                                                     │
     │                                                       train (variants, Kaggle GPU)
     │                                                                     │
     │                                                evaluate ──> metrics, figures, bundle
     │                                                                     │
     └──────────── same recipe ──────────> predict (lat, lon, start month) ┘
```

| Step | Where | Command / notebook |
|---|---|---|
| 1. Export training regions | Colab / any Python | `s1lulc export --project ID` |
| 2. Build dataset + upload to Kaggle | Colab (CPU) | `notebooks/01_build_dataset_colab.ipynb` |
| 3. Train + evaluate model variants | Kaggle (GPU) | `notebooks/02_train_evaluate_kaggle.ipynb` |
| 4. Run the prototype | Colab / laptop | `notebooks/03_prototype_colab.ipynb` or `s1lulc predict` |

## Results (released model)
Final model `v4_r34_scratch_t3_32reg`: U-Net with ResNet-34, three-month input, trained on 32
European regions (2021). Reference = ESA WorldCover 2021; whole held-out regions, all 12 months.

| Split | Regions | OA | mIoU |
|---|---|---|---|
| Validation | 4 | 0.703 | 0.488 |
| Test (in-distribution) | 3 | 0.828 | 0.467 |
| Test out-of-distribution (Cairo, Sao Paulo) | 2 | 0.825 | 0.353 |

Water, tree cover and built-up are mapped reliably (test IoU 0.90 / 0.82 / 0.63); cropland, low
vegetation, bare land and wetland are often confused. Details, per-class scores and limitations:
[MODEL_CARD.md](MODEL_CARD.md) and the project report.

## Documentation
- [MODEL_CARD.md](MODEL_CARD.md): intended use, performance, limitations of the released model
- [DATA_CARD.md](DATA_CARD.md): dataset sources, processing, splits, class distribution
- [CHANGELOG.md](CHANGELOG.md): release history

## Design principles
- **One recipe, two consumers.** `s1lulc/gee/recipe.py` builds the monthly composite for both
  the training data and the prototype; `s1lulc/preprocessing.normalise_db` is the single
  function that scales model input. `configs/recipe.yaml` is versioned, and model bundles
  refuse to run on a different recipe version.
- **Model bundles, not loose weights.** A bundle holds the weights plus architecture, classes,
  normalisation statistics, validation confusion matrix and noise floor.
- **Region-level splits.** Every region belongs wholly to train, val, test or test_ood, which
  avoids spatial-autocorrelation leakage. Model selection uses validation only.
- **Reproducible.** Seeded, config-driven, resumable training; dataset QC and manifest.
- **Tested.** Calculations are checked against hand-computed values; an end-to-end test runs
  build, train, evaluate and reload on synthetic GeoTIFFs. CI runs lint and tests on every push.

## Layout
```
configs/          classes.yaml, recipe.yaml, regions.yaml, train.yaml
s1lulc/
  config.py       typed configuration, WorldCover / Dynamic World class mapping
  geo.py          UTM zone + snapped 10 m pixel grid (radar/label alignment)
  preprocessing.py dB conversion, normalisation with saved training statistics
  gee/            recipe.py (monthly composite), export.py (idempotent dataset export)
  data/           build.py (GeoTIFF -> dataset, QC, reports), dataset.py (patch sampling)
  models/         U-Net factory: random / ImageNet / SSL4EO-S12 Sentinel-1 initialisation
  inference/      predictor.py (overlapping tiles, blended), bundle.py (versioned bundles)
  analysis/       stats, area (Olofsson error-adjusted), significance (noise floor),
                  temporal (smoothing, entropy), metrics (OA, IoU, F1)
  train.py        training loop      evaluate.py   evaluation and variant comparison
  plots.py        report figures     cli.py        `s1lulc` command
notebooks/        01 build dataset, 02 train + evaluate, 03 prototype (saved with outputs)
tests/            unit tests + synthetic end-to-end pipeline test
```

## Setup
Python 3.10+.
```bash
pip install -e ".[dev]"                 # core + tests
pip install -e ".[gee,geo]"             # Earth Engine + GeoTIFF support
pip install -e ".[train]"               # PyTorch, segmentation-models-pytorch, matplotlib
pytest -q
```

### Accounts
- **Google Earth Engine:** a Google Cloud project registered for noncommercial use
  (https://code.earthengine.google.com/register). The Community tier is sufficient: one
  10 x 10 km region export used about 240-320 EECU-seconds in our measurements.
- **Kaggle** (training only): phone-verified account for GPU and internet access, and an API
  token to upload the dataset. Notebook 01 asks for the token at runtime (`getpass`); never
  paste it into a notebook cell.
- No credentials are stored in the repository. Authenticate Earth Engine with
  `earthengine authenticate` or `ee.Authenticate()`.

## Usage
```bash
s1lulc export --project YOUR-EE-PROJECT --dry-run        # plan; re-running skips existing tasks
s1lulc export --project YOUR-EE-PROJECT
s1lulc build-dataset --src /path/to/lulc_s1_data --out dataset
s1lulc train --data dataset/data --variant all            # resumes automatically
s1lulc evaluate --bundle outputs/train/v4_r34_scratch_t3/bundle --data dataset/data \
                --out outputs/eval/v4_r34_scratch_t3
```

## Prototype

```bash
s1lulc predict --lat 48.14 --lon 11.42 --start 2025-10 \
               --bundle path/to/final_bundle --project YOUR-EE-PROJECT --out outputs/munich
```

| Output | Content |
|---|---|
| `report.html` | interactive report: month slider, raw / smoothed / Dynamic World / radar layers, basemap, tables, charts |
| `lulc_monthly.tif`, `lulc_monthly_smoothed.tif` | 12-band GeoTIFF, one band per month (class ids, 255 = no data) |
| `uncertainty_mean.tif`, `label_switches.tif` | mean normalised entropy; label switches per pixel |
| `class_shares.csv` | share per class and month: raw, smoothed, error-adjusted (± 95% CI), Dynamic World |
| `change_month_to_month.csv`, `change_total.csv` | changes in percentage points with significance flags, gross gain/loss |
| `transitions_first_to_last.csv` | from-class x to-class matrix (% of area) |
| `results.json` | everything above plus acquisitions, platforms, warnings, runtime |

**Calculations.** Shares use the area valid in every month (common mask), so missing data never
appears as change. Month-to-month change = P(m) - P(m-1); total change = P(last) - P(first), in
percentage points. A change is *significant* when it exceeds the model's noise floor (95th
percentile of change on stable evaluation regions). Error-adjusted shares follow Olofsson et
al. (2014) using the validation confusion matrix.

**Robustness.** Input validation (coordinates, complete 12-month window with the latest valid
start month suggested), retries with backoff for Earth Engine, on-disk cache, months without
radar data excluded and reported, warnings for months with fewer than 3 acquisitions and for
high model uncertainty, model bundle refuses a mismatched recipe version.

## Data sources and licences
- Sentinel-1 GRD, `COPERNICUS/S1_GRD_FLOAT` (contains modified Copernicus Sentinel data 2021,
  ESA), via Google Earth Engine.
- ESA WorldCover 10 m 2021 v200 (labels), CC-BY 4.0. © ESA WorldCover project 2021.
- Dynamic World V1 (independent comparison only; never used for training), CC-BY 4.0.
- SSL4EO-S12 pretrained weights (Wang et al., 2023), CC-BY 4.0.

## Docker (experimental)
A `Dockerfile` for CPU inference is included as a starting point. It has **not been tested**;
the supported ways to run the prototype are notebook 03 on Colab or a local install (Setup).
