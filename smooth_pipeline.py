import ee
import pandas as pd


ee.Authenticate()
ee.Initialize(project="climateconsciousimli")

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTIONS = {
    "NO2": {
        "asset": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
    },
    # "SO2": { "asset": "...", "band": "..." },
    # "WIND_U": { "asset": "...", "band": "..." },
}

def make_micropixel(lat_center, lon_center, step=0.25):
    return ee.Geometry.Rectangle([
        lon_center - step, lat_center - step,
        lon_center + step, lat_center + step
    ])

def make_coarse(image, scale=1100):
    return (image
        .reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True)
        .reproject(scale=scale, crs='EPSG:4326')
        .copyProperties(image, ['system:time_start'])
    )

def attach_date(image):
    t = image.get('system:time_start')
    return image.set({
        'date': ee.Date(t).format('YYYY-MM-dd'),
        'datetime': ee.Date(t).format('YYYY-MM-dd HH:mm:ss')
    })

# ── Core fetcher ──────────────────────────────────────────────────────────────
def fetch_df(lat, lon, start_date, end_date, step=0.25, scale=1100):
    """
    Returns a tidy DataFrame with columns:
        lat, lon, date, datetime, <band_name>, ...
    for all configured collections, merged on (lat, lon, date).
    """
    region = make_micropixel(lat, lon, step)
    dfs = []

    for name, cfg in COLLECTIONS.items():
        collection = (
            ee.ImageCollection(cfg["asset"])
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select(cfg["band"])
            .map( lambda p: make_coarse(p, scale=scale) )
            #.map(make_coarse, scale=scale)
            .map(attach_date)
        )

        def sample_image(image):
            samples = image.sample(region=region, geometries=True, dropNulls=True)
            return samples.map(lambda fool: fool.set({
                'date': image.get('date'),
                'datetime': image.get('datetime')
            }))

        fc = collection.map(sample_image).flatten()
        features = fc.getInfo()['features']

        rows = []
        for f in features:
            props = f['properties']
            coords = f['geometry']['coordinates']
            if cfg["band"] in props:
                rows.append({
                    'lon': coords[0], 'lat': coords[1],
                    'date': props['date'], 'datetime': props['datetime'],
                    name: props[cfg["band"]]
                })
        dfs.append(pd.DataFrame(rows))

    # Merge all variable dataframes on spatial-temporal key
    if len(dfs) == 1:
        return dfs[0]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=['lat', 'lon', 'date', 'datetime'], how='outer')
    return merged

print("hiii")

# import smooth_pipeline as pp
# df = pp.fetch_df(lat=28.695, lon=77.65, start_date='2024-01-01', end_date='2024-01-03', scale=2000)
# print(df)