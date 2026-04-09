# Data Cleaning Pipeline — Local Version
#
# loads boston_listings_with_census.csv, cleans it, remaps
# image paths, and saves to boston_cleaned.csv
#
# run:
#   python data_cleaning.py
# ============================================================

import os
import ast
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# paths
BASE = Path.home() / 'mmai' / 'mmai_midterm_report'
CSV_PATH = BASE / 'boston_listings_with_census.csv'
OUT_PATH = BASE / 'boston_cleaned.csv'
SAT_DIR = BASE / 'satellite'
GSV_DIR = BASE / 'street_view'

# load
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} properties.")

# parse photo urls
def safe_parse_list(s):
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

df['photo_list'] = df['photo_urls'].apply(safe_parse_list)
df['n_photos'] = df['photo_list'].apply(len)

# force numeric types
for col in ['price', 'price_listed', 'days_on_market', 'area_sqft',
            'beds', 'baths', 'zestimate']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# price gap:
# positive = sold below listing (seller cut price)
# negative = sold above listing (bidding war / underpricing)
mask = df['price_listed'].notna() & (df['price_listed'] > 0)
df['price_gap'] = np.where(
    mask,
    (df['price_listed'] - df['price']) / df['price_listed'],
    np.nan
)

# flag outliers (deviations > 50% almost always reflect data)
# quality issues (re-listings, auctions, scraping errors)
df['price_gap_outlier'] = df['price_gap'].abs() > 0.5

n_outlier = df['price_gap_outlier'].sum()
n_valid   = mask.sum() - n_outlier
print(f"\nPrice gap: {mask.sum()} properties with both prices.")
print(f"Flagged {n_outlier} as outliers (|gap| > 50%).")
print(f"{n_valid} usable for regression.")
print(f"Mean gap (excl. outliers): "
      f"{df.loc[~df['price_gap_outlier'], 'price_gap'].mean():.4f}")
print(f"Std gap (excl. outliers):  "
      f"{df.loc[~df['price_gap_outlier'], 'price_gap'].std():.4f}")

# remap image paths to local disk bc used to be on google drive
def remap_path(original, directory):
    if pd.isna(original):
        return None
    fname    = os.path.basename(original)
    new_path = directory / fname
    return str(new_path) if new_path.exists() else None

# construct paths directly from property ID 
def get_sat_file(pid):
    p = SAT_DIR / f"{pid}_sat.jpg"
    return str(p) if p.exists() else None

def get_gsv_file(pid, heading):
    p = GSV_DIR / f"{pid}_h{heading}.jpg"
    return str(p) if p.exists() else None

df['sat_file'] = df['id'].apply(get_sat_file)
for h in [0, 90, 180, 270]:
    df[f'gsv_file_{h}'] = df['id'].apply(lambda pid: get_gsv_file(pid, h))

# image availablibity
sat_ok = df['sat_file'].notna().sum()
gsv_ok = df['gsv_file_0'].notna().sum()
print(f"\nSatellite images found: {sat_ok}/{len(df)}")
print(f"Street view images found (h=0):  {gsv_ok}/{len(df)}")

# save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned CSV to {OUT_PATH}")
