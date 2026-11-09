# Model card: s1lulc v4_r34_scratch_t3_32reg

## Model
U-Net with a ResNet-34 encoder (random initialisation, 24.4 M parameters), predicting seven
land-cover classes at 10 m from Sentinel-1 VV/VH backscatter. The map for month t is computed
from the monthly composites of months t-1, t and t+1 (six input channels); at the edges of a
12-month window the first/last month is repeated. Trained on 32 European regions of 2021 with
ESA WorldCover 2021 labels. Selected among five round-2 variants by validation mIoU
(best epoch 31 of 43). Bundle: `model.pt` + `bundle.json` (classes, normalisation statistics,
recipe version 1.0, temporal context, validation confusion matrix, noise floor).

## Intended use
- Monthly land-cover maps and class-share statistics for 10 x 10 km areas, especially where
  optical imagery is cloud-limited; detection of large, persistent changes in water and
  built-up area.
- Education and research prototypes. **Not** validated for official statistics, legal or
  financial decisions, or for areas far from the training domain without local validation.

## Performance (all months of 2021, reference = WorldCover, region-level held-out splits)
| Split | Regions | OA | mIoU | mIoU (smoothed) |
|---|---|---|---|---|
| Validation | 4 | 0.703 | 0.488 | 0.497 |
| Test (in-distribution) | 3 | 0.828 | 0.467 | 0.479 |
| Test OOD (Cairo, Sao Paulo) | 2 | 0.825 | 0.353 | 0.357 |

Per-class IoU, test: tree 0.82, low vegetation 0.45, cropland 0.47, built-up 0.63, bare 0.00
(3k pixels only), water 0.90, wetland 0.01 (7k pixels only). Validation: wetland 0.60,
cropland 0.61. Expected calibration error on test: 0.009.

Comparison: the best round-1 model (single month, 16 regions) reached validation mIoU 0.423,
test OA 0.783 and OOD mIoU 0.293.

## Limitations
- Reference labels have ~23 % error; scores measure agreement with WorldCover.
- Cropland / low vegetation / bare land and wetland / grassland are confused (C-band physics).
- Seasonal effects (crop growth, snow, frozen ground) and sensor changes (Sentinel-1A to
  Sentinel-1C/1D in 2026) cause apparent change. Month-to-month noise floors are 0.9-1.2 pp
  for built-up and water but 12-21 pp for tree cover, low vegetation, cropland and wetland.
- Trained on 2021 data from Sentinel-1A/1B only; radiometry of later satellites is untested.
- Lower accuracy outside Europe (OOD mIoU 0.353); fewer acquisitions there (3-8 per month).
- No radiometric terrain flattening: steep slopes are less reliable.

## Ethical and environmental notes
Uses open Copernicus data; no personal data. Training used about 2 GPU-hours (T4).
