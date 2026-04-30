#!/usr/bin/env python3
"""
Demo script to fetch NO2, CO, Wind (U, V), and Temperature data from GEE
and save it as a CSV file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Ai_Made_This.data_pipeline import (
    fetch_and_process,
    save_to_csv
)

def main():
    print("=" * 60)
    print("GEE Data Fetching Demo")
    print("=" * 60)
    
    LAT = 28.695
    LON = 77.65
    START_DATE = '2024-01-01'
    END_DATE = '2024-01-31'
    OUTPUT_FILE = 'demo_data.csv'
    CUSTOM_SCALE = 20000 # 3000 works for NO2, CO, but not Temp, Wind.
    STEP = 0.25
    
    print(f"\nLocation: ({LAT}, {LON})")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("\nFetching data from GEE...")
    
    df = fetch_and_process(
        lat=LAT,
        lon=LON,
        start_date=START_DATE,
        end_date=END_DATE,
        step=STEP,
        scale=CUSTOM_SCALE,
        add_holes_flag=True,
        temporal_radius=3
    )
    
    print(f"\n{'=' * 60}")
    print("Data Summary")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    print(f"\n{'=' * 60}")
    print("First 20 rows:")
    print(f"{'=' * 60}")
    print(df.head(20).to_string())
    
    print(f"\n{'=' * 60}")
    print("Saving to CSV...")
    print(f"{'=' * 60}")
    save_to_csv(df, OUTPUT_FILE)
    
    print(f"\nDemo complete! Data saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Variables: NO2, CO, TEMP (C), WIND_U (m/s), WIND_V (m/s)")
    
    return df


# def main():
#     print("=" * 60)
#     print("GEE Data Fetching Demo")
#     print("=" * 60)
    
#     LAT = 28.695
#     LON = 77.65
#     START_DATE = '2024-01-01'
#     END_DATE = '2024-01-06'
#     OUTPUT_FILE = 'demo_data.csv'
    
#     print(f"\nLocation: ({LAT}, {LON})")
#     print(f"Date Range: {START_DATE} to {END_DATE}")
#     print("\nFetching data from GEE...")
    
#     df = fetch_and_process(
#         lat=LAT,
#         lon=LON,
#         start_date=START_DATE,
#         end_date=END_DATE,
#         step=0.25,
#         scale=11000,
#         add_holes_flag=True,
#         temporal_radius=3
#     )
    
#     print(f"\n{'=' * 60}")
#     print("Data Summary")
#     print(f"{'=' * 60}")
#     print(f"Shape: {df.shape}")
#     print(f"Columns: {df.columns.tolist()}")
#     print(f"\nMissing values per column:")
#     print(df.isnull().sum())
    
#     print(f"\n{'=' * 60}")
#     print("First 20 rows:")
#     print(f"{'=' * 60}")
#     print(df.head(20).to_string())
    
#     print(f"\n{'=' * 60}")
#     print("Saving to CSV...")
#     print(f"{'=' * 60}")
#     save_to_csv(df, OUTPUT_FILE)
    
#     print(f"\nDemo complete! Data saved to {OUTPUT_FILE}")
#     print(f"Total rows: {len(df)}")
#     print(f"Variables available: NO2, CO")
#     print(f"Note: TEMP, WIND_U, WIND_V require ERA5 access")
    
#     return df


if __name__ == "__main__":
    main()