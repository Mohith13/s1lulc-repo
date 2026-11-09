# Data card: s1lulc dataset (Sentinel-1 monthly composites + WorldCover 2021)

## Summary
41 regions of 10 x 10 km (1000 x 1000 pixels at 10 m) in Europe, North Africa and South
America. For each region: twelve monthly Sentinel-1 VV/VH composites of 2021, the number of
acquisitions per pixel and month, and the ESA WorldCover 2021 label.

| Split | Regions | Purpose |
|---|---|---|
| train | 32 (16 in round 1) | model fitting, normalisation statistics |
| val | 4 | model selection, early stopping, area-adjustment confusion matrix |
| test | 3 | in-distribution evaluation (used once per model) |
| test_ood | 2 (Cairo, Sao Paulo) | out-of-distribution evaluation |

Regions are assigned whole to one split (no spatial leakage). Coordinates and splits:
`configs/regions.yaml`.

## Sources
- Sentinel-1 GRD, IW mode, Google Earth Engine `COPERNICUS/S1_GRD_FLOAT` (Copernicus Sentinel
  data 2021, ESA). Earth Engine preprocessing: orbit file, border and thermal noise removal,
  radiometric calibration to sigma0, terrain correction with SRTM 30 m.
- ESA WorldCover 10 m 2021 v200 (Zanaga et al. 2022), CC-BY 4.0; global accuracy 76.7 %.

## Processing (recipe version 1.0, `configs/recipe.yaml`)
1. Scenes with VV and VH, ascending and descending, VV <= -30 dB masked.
2. Calendar-month mean in linear power (speckle reduction by temporal averaging).
3. Export with the label as one image on a snapped UTM 10 m grid (exact pixel alignment).
4. `s1lulc build-dataset`: dB conversion, WorldCover -> 7 classes (`configs/classes.yaml`),
   quality checks, normalisation statistics from training regions only.

## Files (Kaggle dataset)
`<region>.npz`: `db` (12, 2, H, W) float16 dB, NaN = no data; `count` (12, H, W) uint8 scenes per
pixel; `label` (H, W) uint8 target class (255 = ignore); `label_raw` raw WorldCover code;
`transform` affine transform. `meta.json`: regions, splits, CRS, class counts, class weights, QC.
`norm.json`: normalisation statistics.

## Class distribution (percent of labelled pixels)
| Class | train (32) | val | test | test OOD |
|---|---|---|---|---|
| Tree cover | 27.3 | 13.3 | 62.4 | 6.2 |
| Low vegetation | 20.7 | 11.5 | 8.9 | 1.0 |
| Cropland | 12.1 | 31.3 | 5.4 | 0.6 |
| Built-up | 24.0 | 18.5 | 16.6 | 85.6 |
| Bare / sparse | 2.6 | 1.4 | 0.1 | 3.8 |
| Water | 10.2 | 7.0 | 6.4 | 2.8 |
| Wetland | 3.0 | 17.0 | 0.2 | 0.0 |

## Known limitations
- Labels are model output with ~23 % error; label noise is part of every reported metric.
- Bare land and wetland are rare in the test split; their test scores have low support.
- Acquisition density varies from 3-5 (Sao Paulo) to >20 scenes per month (northern Europe).
- One year (2021) only; two viewing geometries averaged; no radiometric terrain flattening.
- Mostly European training regions: performance elsewhere is lower (see MODEL_CARD.md).
