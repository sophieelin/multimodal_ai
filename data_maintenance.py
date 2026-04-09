# Data Maintenance
#
# - gets missing census info
# - re-download missing GSV/satellite
# - downloads lisitng photos from Zillow URLs
# - get property details by scrape listing descriptions + price history
# using hasData Zillow APIs
#
# dependencies:
# python -m pip install requests pandas python-dotenv
#
# run:
# python data_maintenance.py
#

import os
import ast
import time
import requests
import urllib.parse
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# API keys
HASDATA_KEY  = os.getenv('HASDATA_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_KEY = os.getenv('CENSUS_KEY')

for name, val in [('HASDATA_KEY', HASDATA_KEY), ('GOOGLE_API_KEY', GOOGLE_API_KEY), ('CENSUS_KEY', CENSUS_KEY)]:
    if not val:
        raise RuntimeError(f"Missing API key: {name}. Add it to your .env file.")

# paths
BASE = Path.home() / 'mmai' / 'mmai_midterm_report'
MASTER_CSV = BASE / 'boston_listings_with_census.csv'
GSV_DIR = BASE / 'street_view'
SAT_DIR = BASE / 'satellite'
PHOTOS_DIR = BASE / 'interior_photos'

GSV_DIR.mkdir(parents=True, exist_ok=True)
SAT_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# helper functions

def census_get(url, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("retried alrdy, skipping.")
                return None
    return None


def get_census_tract(lat, lon):
    url = (
        f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
        f"?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Census2020_Current"
        f"&layers=Census+Tracts&format=json"
    )
    r = census_get(url)
    if r is None or r.status_code != 200:
        print("  Tract lookup failed")
        return None
    try:
        tract = r.json()['result']['geographies']['Census Tracts'][0]
        return {'state': tract['STATE'], 'county': tract['COUNTY'], 'tract': tract['TRACT']}
    except (KeyError, IndexError) as e:
        print(f"  Tract parse error: {e}")
        return None


def get_census_demographics(state, county, tract):
    variables = ','.join([
        'B19013_001E', 'B25077_001E', 'B25064_001E',
        'B15003_022E', 'B15003_001E', 'B01003_001E',
        'B02001_002E', 'B02001_003E', 'B02001_005E', 'B03003_003E',
    ])
    url = (
        f"https://api.census.gov/data/2023/acs/acs5"
        f"?get={variables}"
        f"&for=tract:{tract}&in=state:{state}+county:{county}"
        f"&key={CENSUS_KEY}"
    )
    r = census_get(url)
    if r is None or r.status_code != 200:
        print("  Demographics failed")
        return None
    data   = r.json()
    result = dict(zip(data[0], data[1]))

    total_pop  = int(result.get('B01003_001E') or 0)
    pop_25plus = int(result.get('B15003_001E') or 1)

    return {
        'census_median_income': result.get('B19013_001E'),
        'census_median_home_value': result.get('B25077_001E'),
        'census_median_rent': result.get('B25064_001E'),
        'census_pct_educated': round(int(result.get('B15003_022E') or 0) / pop_25plus, 4),
        'census_total_population': total_pop,
        'census_pct_white': round(int(result.get('B02001_002E') or 0) / max(total_pop, 1), 4),
        'census_pct_black': round(int(result.get('B02001_003E') or 0) / max(total_pop, 1), 4),
        'census_pct_asian': round(int(result.get('B02001_005E') or 0) / max(total_pop, 1), 4),
        'census_pct_hispanic': round(int(result.get('B03003_003E') or 0) / max(total_pop, 1), 4),
    }


def fetch_street_view(lat, lon, prop_id, heading=0, size='640x640', fov=90):
    path = GSV_DIR / f"{prop_id}_h{heading}.jpg"
    if path.exists():
        return str(path)
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lon}"
        f"&heading={heading}&fov={fov}&pitch=0&source=outdoor&key={GOOGLE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
        with open(path, 'wb') as f:
            f.write(r.content)
        return str(path)
    return None


def fetch_satellite(lat, lon, prop_id, zoom=18, size='640x640'):
    path = SAT_DIR / f"{prop_id}_sat.jpg"
    if path.exists():
        return str(path)
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}"
        f"&size={size}&maptype=satellite&key={GOOGLE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
        with open(path, 'wb') as f:
            f.write(r.content)
        return str(path)
    return None


# gets missing census data

def fill_missing_census(csv_path=MASTER_CSV):
    df = pd.read_csv(csv_path, dtype={'id': str})

    if 'census_unavailable' not in df.columns:
        df['census_unavailable'] = False

    def is_missing(series):
        return series.isnull() | (series.astype(str).str.strip() == '')

    bad_mask = (
        is_missing(df['census_median_income']) |
        is_missing(df['census_median_home_value']) |
        is_missing(df['census_median_rent'])
    ) & (df['census_unavailable'] != True)

    missing = df[bad_mask].copy()
    print(f"Found {len(missing)} properties with missing census data")

    if missing.empty:
        print("Nothing to fill.")
        return df

    filled_count = 0
    for idx, row in missing.iterrows():
        print(f"\nRetrying: {row['street']}")
        time.sleep(2.0)

        tract_info = get_census_tract(row['lat'], row['lon'])
        if not tract_info:
            print("Tract lookup failed, marking unavailable")
            df.at[idx, 'census_unavailable'] = True
            df.to_csv(csv_path, index=False)
            continue

        demo = get_census_demographics(tract_info['state'], tract_info['county'], tract_info['tract'])
        if not demo:
            print("Demographics failed, marking unavailable")
            df.at[idx, 'census_unavailable'] = True
            df.to_csv(csv_path, index=False)
            continue

        for col, val in {**tract_info, **demo}.items():
            if col in df.columns:
                try:
                    val = df[col].dtype.type(val)
                except (ValueError, TypeError):
                    val = str(val)
            df.at[idx, col] = val

        df.at[idx, 'census_unavailable'] = False
        filled_count += 1
        print("Filled successfully")
        df.to_csv(csv_path, index=False)

    print(f"\nDone. Filled {filled_count} of {len(missing)} properties")
    print(f"{int(df['census_unavailable'].sum())} marked as unavailable")
    return df


# fix missing satellite + gsv pics

def remap_image_paths(df):
    """
    update path columns based on what's actually on disk
    """
    df['sat_path'] = df['id'].apply(
        lambda pid: str(SAT_DIR / f"{pid}_sat.jpg")
        if (SAT_DIR / f"{pid}_sat.jpg").exists() else None
    )
    for heading in [0, 90, 180, 270]:
        df[f'gsv_path_{heading}'] = df['id'].apply(
            lambda pid: str(GSV_DIR / f"{pid}_h{heading}.jpg")
            if (GSV_DIR / f"{pid}_h{heading}.jpg").exists() else None
        )
    return df


def fix_missing_images(csv_path=MASTER_CSV):
    df = pd.read_csv(csv_path, dtype={'id': str})
    df = remap_image_paths(df)

    missing_sat = df[df['sat_path'].isna()]
    missing_gsv = df[df['gsv_path_0'].isna()]
    print(f"Missing satellite: {len(missing_sat)}")
    print(f"Missing street view: {len(missing_gsv)}")

    if not missing_sat.empty:
        print("\nDownloading missing satellite images...")
        for _, row in missing_sat.iterrows():
            print(f"  {row['street']}")
            p = fetch_satellite(row['lat'], row['lon'], row['id'])
            if p:
                df.loc[df['id'] == row['id'], 'sat_path'] = p
            time.sleep(0.1)

    if not missing_gsv.empty:
        print("\nDownloading missing street view images...")
        for _, row in missing_gsv.iterrows():
            print(f"  {row['street']}")
            for heading in [0, 90, 180, 270]:
                col = f'gsv_path_{heading}'
                if pd.isna(df.loc[df['id'] == row['id'], col].values[0]):
                    p = fetch_street_view(row['lat'], row['lon'], row['id'], heading=heading)
                    if p:
                        df.loc[df['id'] == row['id'], col] = p
                time.sleep(0.1)

    df = remap_image_paths(df)
    df.to_csv(csv_path, index=False)

    sat_found = df['sat_path'].notna().sum()
    gsv_found = df['gsv_path_0'].notna().sum()
    print(f"\nDone.")
    print(f"Satellite: {sat_found} / {len(df)}")
    print(f"Street view: {gsv_found} / {len(df)}")

    still_missing_sat = df[df['sat_path'].isna()][['id', 'street', 'city']]
    still_missing_gsv = df[df['gsv_path_0'].isna()][['id', 'street', 'city']]
    if not still_missing_sat.empty:
        print(f"\nStill missing satellite ({len(still_missing_sat)}):")
        print(still_missing_sat.to_string(index=False))
    if not still_missing_gsv.empty:
        print(f"\nStill missing street view ({len(still_missing_gsv)}):")
        print(still_missing_gsv.to_string(index=False))

    return df


# download listing photos

def download_single_photo(args):
    url, path = args
    if path.exists():
        return False
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.headers.get('content-type', '').startswith('image'):
            with open(path, 'wb') as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def download_photos_for_row(row, max_workers=8):
    prop_id = row['id']
    try:
        urls = ast.literal_eval(row['photo_urls']) if pd.notna(row['photo_urls']) else []
    except (ValueError, SyntaxError):
        return 0
    if not urls:
        return 0

    prop_dir = PHOTOS_DIR / prop_id
    prop_dir.mkdir(exist_ok=True)

    tasks = [
        (url, prop_dir / f"{prop_id}_interior_{i}.jpg")
        for i, url in enumerate(urls)
    ]
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_photo, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                downloaded += 1
    return downloaded


def download_all_photos(csv_path=MASTER_CSV):
    df = pd.read_csv(csv_path, dtype={'id': str})
    total = 0
    for i, (_, row) in enumerate(df.iterrows()):
        count = download_photos_for_row(row)
        total += count
        if count > 0:
            print(f"{row['street']}: {count} photos")
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(df)}] {total} photos downloaded so far")
    print(f"\nDone. {total} photos total across {len(df)} properties")


# fetch listing desc and price history

def get_property_detail_hasdata(zillow_url):
    encoded_url = urllib.parse.quote(zillow_url, safe='')
    r = requests.get(
        f'https://api.hasdata.com/scrape/zillow/property?url={encoded_url}',
        headers={'Content-Type': 'application/json', 'x-api-key': HASDATA_KEY},
        timeout=30
    )
    if r.status_code == 200:
        return r.json().get('property', None)
    print(f"Failed: {r.status_code}")
    return None


def parse_property_detail(prop):
    if not prop:
        return {}

    price_history = prop.get('priceHistory', [])
    price_listed = None
    date_listed = None
    price_sold = None
    date_sold = None
    days_on_market = None

    sold_index = None
    for i, event in enumerate(price_history):
        if event.get('event') == 'sold':
            price_sold = event.get('price')
            date_sold = event.get('date')
            sold_index = i
            break

    if sold_index is not None:
        for event in price_history[sold_index + 1:]:
            if event.get('event') == 'listedForSale':
                price_listed = event.get('price')
                date_listed  = event.get('date')
                break

    if date_listed and date_sold:
        try:
            days_on_market = (
                datetime.strptime(date_sold, '%Y-%m-%d') -
                datetime.strptime(date_listed, '%Y-%m-%d')
            ).days
        except Exception:
            pass

    return {
        'listing_description': prop.get('description', None),
        'price_listed': price_listed,
        'date_listed': date_listed,
        'price_sold': price_sold,
        'date_sold': date_sold,
        'days_on_market': days_on_market,
        'price_history_raw': str(price_history),
    }


def fetch_property_details(csv_path=MASTER_CSV, limit=None):
    df = pd.read_csv(csv_path, dtype={'id': str})

    for col in ['listing_description', 'price_listed', 'date_listed',
                'price_sold', 'date_sold', 'days_on_market', 'price_history_raw']:
        if col not in df.columns:
            df[col] = None

    missing = df[df['listing_description'].isnull() | (df['listing_description'] == '')]
    print(f"Missing descriptions: {len(missing)}")
    print(f"Credits needed: {len(missing) * 5}")

    to_scrape = missing.head(limit) if limit else missing

    fetched = 0
    for idx, row in to_scrape.iterrows():
        print(f"\n[{fetched + 1}/{len(to_scrape)}] {row['street']}")
        prop = get_property_detail_hasdata(row['url'])
        result = parse_property_detail(prop)

        for col, val in result.items():
            if val is not None:
                df.at[idx, col] = val

        df.to_csv(csv_path, index=False)  # always save, even if no data returned

        if not result:
            print("No data returned, saved to avoid re-fetching")
        else:
            print(f"desc={bool(result.get('listing_description'))} "
                  f"listed=${result.get('price_listed')} "
                  f"DOM={result.get('days_on_market')}")
        fetched += 1
        time.sleep(2.0)

    print(f"\nDone. Fetched {fetched}/{len(to_scrape)} properties")
    return df


# main block

if __name__ == '__main__':
    print("1. Getting missing data:")
    fill_missing_census()

    print("\n2. Getting missing satellite/gsv images:")
    fix_missing_images()

    print("\n3. Downloading interior photos:")
    download_all_photos()

    print("\n4.Fetching Listing Descriptions:")
    fetch_property_details(limit=None)
