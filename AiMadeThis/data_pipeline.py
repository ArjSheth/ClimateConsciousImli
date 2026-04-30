import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


ee.Authenticate()
ee.Initialize(project="climateconsciousimli")


COLLECTIONS = {
    "NO2": {
        "asset": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
    },
    "CO": {
        "asset": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
    },
    "TEMP": {
        "asset": "ECMWF/ERA5/HOURLY",
        "band": "temperature_2m",
    },
    "WIND_U": {
        "asset": "ECMWF/ERA5/HOURLY",
        "band": "u_component_of_wind_10m",
    },
    "WIND_V": {
        "asset": "ECMWF/ERA5/HOURLY",
        "band": "v_component_of_wind_10m",
    },
}

def make_micropixel(lat_center, lon_center, step=0.25):
    return ee.Geometry.Rectangle([
        lon_center - step, lat_center - step,
        lon_center + step, lat_center + step
    ])


def make_coarse(image, scale=1100):
    return (image
        .reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True)
        .reproject(crs='EPSG:4326', scale=scale)
        .copyProperties(image, ['system:time_start'])
    )


def attach_date(image):
    t = image.get('system:time_start')
    return image.set({
        'date': ee.Date(t).format('YYYY-MM-dd'),
        'datetime': ee.Date(t).format('YYYY-MM-dd HH:mm:ss')
    })


def get_collection_config(var_name):
    return COLLECTIONS[var_name]


def fetch_single_variable(lat, lon, var_name, start_date, end_date, step=0.25, scale=11000):
    cfg = get_collection_config(var_name)
    region = make_micropixel(lat, lon, step)
    
    collection = (
        ee.ImageCollection(cfg["asset"])
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select(cfg["band"])
    )
    
    is_era5 = "ERA5" in cfg["asset"]
    is_satellite = not is_era5
    
    try:
        count = collection.size().getInfo()
        print(f"  Collection has {count} images")
    except Exception as e:
        print(f"  Could not get collection count: {e}")
        return pd.DataFrame()
    
    if is_satellite and scale < 11132:
        collection = collection.map(lambda img: make_coarse(img, scale=scale))
    
    collection = collection.map(attach_date)
    
    def sample_image(image):
        samples = image.sample(region=region, scale=scale, geometries=True, dropNulls=False)
        return samples.map(lambda f: f.set({
            'date': image.get('date'),
            'datetime': image.get('datetime')
        }))
    
    fc = collection.map(sample_image).flatten()
    
    try:
        features = fc.getInfo()['features']
    except Exception as e:
        print(f"  Error getting features: {e}")
        return pd.DataFrame()
    
    rows = []
    for f in features:
        props = f['properties']
        coords = f['geometry']['coordinates']
        band_val = cfg["band"]
        if band_val in props:
            val = props[band_val]
            if var_name in ["TEMP"]:
                val = val - 273.15
            rows.append({
                'lon': coords[0],
                'lat': coords[1],
                'date': props['date'],
                'datetime': props['datetime'],
                var_name: val
            })
    
    return pd.DataFrame(rows)


def fetch_all_variables(lat, lon, start_date, end_date, step=0.25, scale=11000):
    dfs = {}
    
    variables = ["NO2", "CO", "TEMP", "WIND_U", "WIND_V"]
    
    for name in variables:
        print(f"Fetching {name}...")
        try:
            dfs[name] = fetch_single_variable(lat, lon, name, start_date, end_date, step, scale)
            if dfs[name].empty:
                print(f"  {name}: No data returned")
            else:
                print(f"  {name}: Got {len(dfs[name])} rows")
        except Exception as e:
            print(f"  Error fetching {name}: {e}")
            dfs[name] = pd.DataFrame()
    
    return dfs


def merge_variables(dfs, on_cols=['lat', 'lon', 'date', 'datetime']):
    result = None
    
    for name, df in dfs.items():
        if df.empty:
            continue
        if result is None:
            result = df
        else:
            result = result.merge(df, on=on_cols, how='outer')
    
    return result if result is not None else pd.DataFrame()


def add_holes(df, start_date, end_date, lat_center, lon_center, step=0.25):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if df.empty:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['datetime'] = df['datetime'].fillna(df['date'] + ' 00:00:00')
    
    unique_locs = df[['lat', 'lon']].drop_duplicates().values.tolist()
    
    if len(unique_locs) == 0:
        unique_locs = [[lat_center - step, lon_center - step], 
                     [lat_center + step, lon_center - step], 
                     [lat_center - step, lon_center + step], 
                     [lat_center + step, lon_center + step]]
    
    print(f"  add_holes: {len(unique_locs)} unique locations, {len(date_range)} dates")
    print(f"  DataFrame input rows: {len(df)}")
    
    holes = []
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        for loc in unique_locs:
            holes.append({
                'lat': loc[0],
                'lon': loc[1],
                'date': date_str,
            })
    
    holes_df = pd.DataFrame(holes)
    
    result = holes_df.merge(df, on=['lat', 'lon', 'date'], how='left')
    result['datetime'] = result['date'] + ' 00:00:00'
    
    return result


def temporal_interpolate(df, variable, temporal_radius=3):
    if df.empty or variable not in df.columns:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['lat', 'lon', 'date'])
    
    if df[['lat', 'lon', 'date']].drop_duplicates().empty:
        return df
    
    lat_lon_groups = df.groupby(['lat', 'lon'])
    
    interpolated_frames = []
    
    for (lat, lon), group in lat_lon_groups:
        group = group.copy()
        group = group.set_index('date')
        
        try:
            full_date_range = pd.date_range(
                start=group.index.min(),
                end=group.index.max(),
                freq='D'
            )
            group = group.reindex(full_date_range)
        except:
            interpolated_frames.append(group.reset_index())
            continue
        
        group[variable] = group[variable].interpolate(
            method='time',
            limit_area='inside',
            limit_direction='both',
            limit=temporal_radius
        )
        
        group['date'] = group.index
        group['lat'] = lat
        group['lon'] = lon
        group = group.reset_index(drop=True)
        
        interpolated_frames.append(group)
    
    if not interpolated_frames:
        return df
    
    result = pd.concat(interpolated_frames, ignore_index=True)
    
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result['datetime'] = result['date'] + ' 00:00:00'
    
    return result


def temporal_interpolate_all(df, temporal_radii=None, variables=None):
    if df.empty:
        return df
    
    if temporal_radii is None:
        temporal_radii = {
            "NO2": 3,
            "CO": 3,
            "TEMP": 3,
            "WIND_U": 3,
            "WIND_V": 3,
        }
    
    if variables is None:
        variables = ["NO2", "CO", "TEMP", "WIND_U", "WIND_V"]
    
    result = df.copy()
    for var in variables:
        if var in result.columns and var in temporal_radii:
            result = temporal_interpolate(result, var, temporal_radii[var])
    
    return result


def fetch_and_process(lat, lon, start_date, end_date, 
                    step=0.25, scale=11000, 
                    add_holes_flag=True,
                    temporal_radius=3):
    dfs = fetch_all_variables(lat, lon, start_date, end_date, step, scale)
    
    merged_df = merge_variables(dfs)
    
    if add_holes_flag:
        merged_df = add_holes(merged_df, start_date, end_date, lat, lon, step)
    
    result = temporal_interpolate_all(merged_df, 
                                 temporal_radii={
                                     "NO2": temporal_radius,
                                     "CO": temporal_radius,
                                     "TEMP": temporal_radius,
                                     "WIND_U": temporal_radius,
                                     "WIND_V": temporal_radius,
                                 })
    
    return result


def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")


def load_from_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded from {filepath}")
    return df


def fetch_and_save(lat, lon, start_date, end_date, output_path,
                   step=0.25, scale=11000,
                   add_holes_flag=True,
                   temporal_radius=3):
    df = fetch_and_process(
        lat=lat, lon=lon,
        start_date=start_date, end_date=end_date,
        step=step, scale=scale,
        add_holes_flag=add_holes_flag,
        temporal_radius=temporal_radius
    )
    save_to_csv(df, output_path)
    return df


if __name__ == "__main__":
    df = fetch_and_process(
        lat=28.695, 
        lon=77.65, 
        start_date='2024-01-01', 
        end_date='2024-01-10',
        step=0.25,
        scale=2000,
        add_holes_flag=True,
        temporal_radius=3
    )
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")