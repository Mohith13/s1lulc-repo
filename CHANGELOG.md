# Changelog

## 0.2.2 (2026-11-09)
- Prototype fails early with a clear message when Sentinel-1 has no data for the area
  (open sea, polar regions); CLI reports input, bundle and Earth Engine errors without a
  traceback.
- Docs: README updated to the final project (41 regions, results table, notebook 03, executed
  notebooks, Kaggle token handling); Dockerfile marked as experimental and untested.

## 0.2.1 (2026-11-09)
- HTML report basemap: Esri satellite imagery and CARTO street map with a layer switch.
  OpenStreetMap's tile servers refuse tile requests without a Referer header, which is what a
  report opened from disk sends.
- Leaflet 1.9.4 is embedded in the report (no CDN dependency); BSD-2 licence included.

## 0.2.0 (2026-11-08)
- Three-month input models (`temporal_context: 1`): map for month t from months t-1, t, t+1.
- SSL4EO-S12 first convolution replicated for multi-month input.
- 16 additional training regions (32 in total) for the data-scaling experiment.
- Validation on all 12 months for stable model selection.
- Prototype: acquisition provenance (scenes, passes, satellites per month) and a warning when
  the Sentinel-1 satellite combination changes within the analysis window.
- Notebooks: Kaggle dataset update instead of re-creation; selectable variants and run tag.

## 0.1.0 (2026-11-07)
- Shared monthly composite recipe, idempotent Earth Engine export with manifest.
- Dataset builder with quality checks and dataset reports.
- U-Net training (random / ImageNet / SSL4EO-S12 initialisation), resumable, AMP.
- Evaluation: per-split, per-month, calibration, noise floor, share error, variant comparison.
- Prototype `s1lulc predict`: 12 monthly maps, shares, changes with significance flags,
  error-adjusted shares, uncertainty, gross vs net change, Dynamic World comparison,
  GeoTIFF/CSV/JSON outputs and an interactive HTML report.
