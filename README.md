# ClimateConsciousImli

Machine learning experiments for NO2 prediction from satellite and weather data. This repo collects the data exploration notebooks, preprocessing work, baseline experiments, and training scripts used for CNN, ConvLSTM, UNet, and weather-aware models.

## Quick Navigation

Start here if you want the shortest path through the project:

1. `data_preprocess.ipynb` for the main preprocessing flow.
2. `NotebooksForDataExploration/` for data checks, access tests, and exploratory analysis.
3. `training/` for the model training scripts.
4. `model_comparison.py` for evaluation and side-by-side checkpoint comparison.
5. `persistence_baseline.ipynb` for a simple baseline reference.

## Project Map

### Root Files
- `data_preprocess.ipynb` is the main preprocessing notebook.
- `model_comparison.py` loads trained checkpoints and compares model performance.
- `persistence_baseline.ipynb` contains the persistence baseline experiment.

### Exploratory Notebooks
- `NotebooksForDataExploration/testing_copernicus_access.ipynb` tests access to Sentinel-5P data through Copernicus.
- `NotebooksForDataExploration/testing_gee_batch_access.ipynb` explores Google Earth Engine batch access.
- `NotebooksForDataExploration/testing_gee_batch_access copy.ipynb` is a duplicate/working copy of the GEE access notebook.
- `NotebooksForDataExploration/missingdata_analysis.ipynb` studies missing-value patterns in the pollutant grids.
- `NotebooksForDataExploration/no2_CO_TOGETHER_LINEAR.ipynb` explores linear modeling ideas for pollutant prediction.
- `NotebooksForDataExploration/tensor_creation_plan.ipynb` outlines tensor creation and dataset shaping.
- `NotebooksForDataExploration/testingWindandTemp.ipynb` checks weather feature handling.

### Training Scripts
- `training/cnn_training.py` trains the CNN-based model.
- `training/convlstm_training.py` trains the ConvLSTM model.
- `training/unet_training.py` trains the UNet model.
- `training/weather_aware_training.py` trains the weather-aware fusion model.
- `training/xgboost.ipynb` is an XGBoost-based experiment.

## How the Pieces Fit Together

The workflow is roughly:

1. Explore the source data and access methods in `NotebooksForDataExploration/`.
2. Build or validate the processed dataset in `data_preprocess.ipynb`.
3. Train models from the scripts in `training/`.
4. Compare saved checkpoints with `model_comparison.py`.

## Notes

- Several scripts currently use hardcoded local dataset or checkpoint paths under `/storage/...` or similar machine-specific locations. Update those paths before running outside the original environment.
- The repo focuses on NO2, but the exploration notebooks also touch related pollutants and weather inputs.

## Summary

- Satellite source: Sentinel-5P / Copernicus / Google Earth Engine.
- Target: gridded NO2 prediction.
- Model family: CNN, ConvLSTM, UNet, weather-aware fusion, plus baseline and exploratory experiments.
