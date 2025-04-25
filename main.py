#!/usr/bin/env python3
"""
DSA210 Coffee-Shop Traffic Analysis
-----------------------------------
Usage:
    python main.py \
        --traffic data/cleaned_coffee_shop_data.xlsx \
        --weather data/weather_df.csv \
        --out-dir results
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# REQUIREMENTS (add these to requirements.txt)
#
# pandas
# openpyxl       # for reading .xlsx
# numpy
# matplotlib
# seaborn
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def load_cleaned_data(path: Path) -> pd.DataFrame:
    """
    Load pre‐cleaned coffee‐shop traffic data.
    Supports .csv or .xlsx (requires openpyxl).
    Must contain at least columns ['hour','check_count'].
    """
    if not path.exists():
        logging.error(f"Traffic data file not found: {path}")
        sys.exit(1)

    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in {'.xls', '.xlsx'}:
        try:
            df = pd.read_excel(path, engine='openpyxl')
        except ImportError:
            logging.error("Missing openpyxl. Install it with `pip install openpyxl`.")
            sys.exit(1)
    else:
        logging.error(f"Unsupported file type for traffic data: {path.suffix}")
        sys.exit(1)

    # enforce dtypes
    df['hour'] = df['hour'].astype(int)
    df['check_count'] = df['check_count'].astype(int)
    return df

def load_weather_data(path: Path) -> pd.DataFrame:
    """
    Load weather DataFrame from CSV (must have 'date','hour','temp','precip').
    """
    if not path.exists():
        logging.error(f"Weather data file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=['date'])
    df['hour'] = df['hour'].astype(int)
    return df

def expand_to_full_grid(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create date×hour grid for the given dates (0–23)."""
    dates_df = pd.DataFrame({'date': dates})
    hours_df = pd.DataFrame({'hour': range(24)})
    dates_df['key'] = 1
    hours_df['key'] = 1
    grid = dates_df.merge(hours_df, on='key').drop(columns='key')
    return grid

def enrich_with_traffic(grid: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Left‐join raw traffic onto the full date×hour grid, filling missing checks with 0."""
    df = grid.merge(traffic, on='hour', how='left')
    df['check_count'] = df['check_count'].fillna(0).astype(int)
    # Rebuild time_slot if your downstream code uses it
    df['time_slot'] = df.get('time_slot', df['hour'].astype(str).str.zfill(2) +
                              ':00-' + (df['hour']+1).astype(str).str.zfill(2) + ':00')
    return df

def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge in temperature & precipitation, filling missing with mean/zero."""
    df['date_str']    = df['date'].dt.date.astype(str)
    weather['date_str']= weather['date'].dt.date.astype(str)

    merged = pd.merge(
        df,
        weather[['date_str','hour','temp','precip']],
        on=['date_str','hour'],
        how='left'
    )
    merged['temp']   = merged['temp'].fillna(merged['temp'].mean())
    merged['precip']= merged['precip'].fillna(0)
    return merged

def apply_weather_effects(df: pd.DataFrame, weight: float=0.2) -> pd.DataFrame:
    """
    Adds:
      - 'check_orig': original check_count
      - 'check_adj' : weather‐adjusted simulated checks
    """
    df = df.copy()
    df['check_orig'] = df['check_count']

    df['temp_eff'] = np.select(
        [df['temp'] < -5, df['temp'] < 5, df['temp'] > 20],
        [1.3,        1.15,       0.85],
        default=1.0
    )
    df['precip_eff'] = np.where(df['precip'] > 0, 1.1, 1.0)

    df['check_adj'] = (
        df['check_orig']
        * ((1-weight) + weight * df['temp_eff'] * df['precip_eff'])
    ).round().astype(int)
    return df

def plot_weekday_hour_heatmap(df: pd.DataFrame, out_path: Path):
    """Save a seaborn heatmap of average checks by weekday and hour."""
    df['weekday'] = df['date'].dt.day_name()
    pivot = df.pivot_table(
        index='weekday', columns='hour', values='check_count', aggfunc='mean'
    ).reindex([
        'Monday','Tuesday','Wednesday','Thursday',
        'Friday','Saturday','Sunday'
    ])
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap='viridis', cbar_kws={'label':'Avg Checks'})
    plt.title('Avg Coffee-Shop Checks by Weekday & Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Saved heatmap to {out_path}")

def plot_correlations(df: pd.DataFrame, out_temp: Path, out_prec: Path):
    """Save bar plots of temperature/precipitation vs check_count correlations by hour."""
    hours = range(24)
    temp_corrs = [df[df.hour==h]['temp'].corr(df[df.hour==h]['check_count']) for h in hours]
    precip_corrs = [df[df.hour==h]['precip'].corr(df[df.hour==h]['check_count']) for h in hours]

    for corrs, title, out in [
        (temp_corrs, 'Temp vs Traffic Correlation', out_temp),
        (precip_corrs,'Precip vs Traffic Correlation', out_prec)
    ]:
        plt.figure(figsize=(10,5))
        plt.bar(hours, corrs)
        plt.axhline(0, color='red', alpha=0.3)
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Correlation')
        plt.xticks(hours, rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        logging.info(f"Saved correlation plot to {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--traffic', default='data/cleaned_coffee_shop_data.xlsx',
                   help='Path to cleaned traffic data (.csv or .xlsx)')
    p.add_argument('--weather', default='data/weather_df.csv',
                   help='Path to weather data CSV')
    p.add_argument('--out-dir', default='results',
                   help='Directory to save output figures')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load inputs
    traffic_df = load_cleaned_data(Path(args.traffic))
    weather_df = load_weather_data(Path(args.weather))

    # Build full date×hour grid for January 2025
    date_index = pd.date_range('2025-01-01','2025-01-31', freq='D')
    grid       = expand_to_full_grid(date_index)

    # Merge & enrich
    df_full     = enrich_with_traffic(grid, traffic_df)
    df_weather  = merge_weather(df_full, weather_df)
    df_features = apply_weather_effects(df_weather, weight=0.2)

    # Plot & save
    plot_weekday_hour_heatmap(
        df_features, out_dir / 'weekday_hour_heatmap.png'
    )
    plot_correlations(
        df_features,
        out_dir / 'temp_correlation.png',
        out_dir / 'precip_correlation.png'
    )

    logging.info("All done!")

if __name__ == '__main__':
    main()
