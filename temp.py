import smooth_pipeline as pp
df = pp.fetch_df(lat=28.695, lon=77.65, start_date='2024-01-01', end_date='2024-01-03', scale=2000)
print(df)
