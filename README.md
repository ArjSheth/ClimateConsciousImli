# ClimateConsciousImli

---
This is a machine learning project, a joint work with @anwesha-ghosh7 as part of an introductory course in machine learning at Ashoka University

### Project :
- Google Earth Engine's Sentinel 5P dataset is used to collect data on concentrations of atmospheric pollutants (SO2, NO2, CO).
- Geographical scope is restricted to a bounding box
- Data is transformed so that it is coarser than before, for easier handling and testing.
- Ultimately want to test CNNs, Autoregressive models, and possibly others.

### Repo Layout
- `testing_copernicus_access.ipynb` contains our initial analysis of Sentinel-5P's NO2 dataset, taken from the [Copernicus browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&themeId=DEFAULT-THEME&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE)
- `testing_gee_batch_access.ipynb` contains our analysis of the NO2 dataset from Google Earth Engine, which is itself a derivation of the above.
- `smooth_pipeline.py` is a streamlined generalization of the previous file.
- `missingdata_analysis.ipynb` explores missing-value patterns in the pollutant data, and plots the same for pattern inference.

### Future plans
- `preprocess.py` : Call the GEE API, obtain dataframes from `smooth_pipeline.py`, and process them into tensors.
- `fill_in_the_blanks.py` : Enable testing spatial -vs- temporal interpolation on the data as measures to fix missing data. May add Diffusion-driven approach later!
- `CNN` : Train a CNN on "good" data to produce a prediction on NO2 levels given data with some lead time.
- `Linear` : Train one (or multiple, locality-specific) linear models to predict NO2 levels given data with some lead time.
