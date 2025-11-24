# src/data_prep.py

from typing import Tuple, Optional, List

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # go up from src/ to repo root
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"



# Map station codes → (City, State) so we can join with US_Accidents
STATION_TO_CITY_STATE = {
    "KCLT": ("Charlotte", "NC"),
    "KCQT": ("Los Angeles", "CA"),
    "KHOU": ("Houston", "TX"),
    "KIND": ("Indianapolis", "IN"),
    "KJAX": ("Jacksonville", "FL"),
    "KMDW": ("Chicago", "IL"),
    "KNYC": ("New York", "NY"),
    "KPHL": ("Philadelphia", "PA"),
    "KPHX": ("Phoenix", "AZ"),
    "KSEA": ("Seattle", "WA"),
}



# Adjust based on your final scope


def _ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)




CITIES_FILTER = [c for c, _ in STATION_TO_CITY_STATE.values()]
STATES_FILTER = [s for _, s in STATION_TO_CITY_STATE.values()]
ANALYSIS_YEARS = (2016, 2021)
ACCIDENTS_SUBSET = DATA_RAW / "US_Accidents_10cities_2016_2021.parquet"



def build_accidents_subset(accidents_csv_path: Path) -> pd.DataFrame:
    """
    Stream US_Accidents_March23.csv in chunks, keep only:
      - 2016–2021
      - the 10 cities we have weather for
      - a few needed columns
    Save as a small Parquet file and return it.
    """
    if ACCIDENTS_SUBSET.exists():
        print(f"[accidents] Using cached subset: {ACCIDENTS_SUBSET}")
        return pd.read_parquet(ACCIDENTS_SUBSET)

    print(f"[accidents] Building subset from {accidents_csv_path} …")

    usecols = ["ID", "Start_Time", "City", "State", "Severity"]
    chunks = pd.read_csv(
        accidents_csv_path,
        usecols=usecols,
        chunksize=250_000,          # adjust up/down if needed
        low_memory=False,
    )

    kept_chunks = []
    year_strings = {str(y) for y in ANALYSIS_YEARS}

    for i, chunk in enumerate(chunks, start=1):
        # quick string-based year filter (much cheaper than full datetime on all rows)
        years = chunk["Start_Time"].str.slice(0, 4)
        mask_year = years.isin(year_strings)

        mask_city = chunk["City"].isin(CITIES_FILTER)
        mask_state = chunk["State"].isin(STATES_FILTER)

        mask = mask_year & mask_city & mask_state
        sub = chunk.loc[mask].copy()

        if sub.empty:
            continue

        # now pay the cost of datetime only for kept rows
        sub["Start_Time"] = pd.to_datetime(sub["Start_Time"], errors="coerce")
        sub = sub.dropna(subset=["Start_Time"])

        sub["Date"] = sub["Start_Time"].dt.normalize()

        kept_chunks.append(sub)

        print(f"[accidents] processed chunk {i}, kept {len(sub)} rows")

    if not kept_chunks:
        raise RuntimeError("No rows matched filters (years + cities). Check filters.")

    df_sub = pd.concat(kept_chunks, ignore_index=True)
    df_sub.to_parquet(ACCIDENTS_SUBSET, index=False)

    print(f"[accidents] Subset saved to {ACCIDENTS_SUBSET} with shape {df_sub.shape}")
    return df_sub


def load_accidents(path: Path) -> pd.DataFrame:
    # Only load the columns we care about
    usecols = [
        "ID", "Start_Time", "City", "State", "Severity"
    ]

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time", "City", "State", "Severity"])

    # Filter by year window early
    df = df[
        (df["Start_Time"].dt.year >= ANALYSIS_YEARS[0])
        & (df["Start_Time"].dt.year <= ANALYSIS_YEARS[1])
    ]

    # Filter only cities/states we have weather for
    df = df[
        (df["City"].isin(CITIES_FILTER))
        & (df["State"].isin(STATES_FILTER))
    ]

    df["Date"] = df["Start_Time"].dt.normalize()
    return df

ACCIDENTS_PARQUET = DATA_PROCESSED / "accidents_filtered.parquet"

def load_accidents_cached(path: Path) -> pd.DataFrame:
    if ACCIDENTS_PARQUET.exists():
        return pd.read_parquet(ACCIDENTS_PARQUET)

    df = load_accidents(path)
    df.to_parquet(ACCIDENTS_PARQUET, index=False)
    return df



def load_weather_us_weather_history(weather_dir: Path) -> pd.DataFrame:
    """
    Ingests the 'US Weather History' style directory:
    data/raw/2/KCLT.csv, KNYC.csv, ... and returns a single DataFrame with
    standardized columns: City, State, Date, Temperature, Precipitation.
    """
    all_frames = []

    for csv_path in weather_dir.glob("*.csv"):
        station = csv_path.stem  # e.g., "KNYC"
        df = pd.read_csv(csv_path)

        # Standardize date column
        # (in this dataset it's called 'date')
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")

        # Attach station, city, state
        df["Station"] = station
        city, state = STATION_TO_CITY_STATE.get(station, (station, None))
        df["City"] = city
        df["State"] = state

        # Normalize key weather columns:
        #   actual_mean_temp          -> Temperature
        #   actual_precipitation     -> Precipitation
        rename_map = {}
        if "actual_mean_temp" in df.columns:
            rename_map["actual_mean_temp"] = "Temperature"
        if "actual_precipitation" in df.columns:
            rename_map["actual_precipitation"] = "Precipitation"

        df = df.rename(columns=rename_map)

        # Keep only the fields we care about right now
        keep_cols = ["City", "State", "Date"]
        if "Temperature" in df.columns:
            keep_cols.append("Temperature")
        if "Precipitation" in df.columns:
            keep_cols.append("Precipitation")

        all_frames.append(df[keep_cols])

    weather = pd.concat(all_frames, ignore_index=True)

    # Optional: drop rows with missing Date just to keep the dataset clean
    weather = weather.dropna(subset=["Date"])

    return weather
def load_holidays(path: Path, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Normalize schema
    # Expect: Date, Holiday_Name / Holiday, Is_Holiday (0/1 or boolean)
    if "Is_Holiday" not in df.columns:
        df["Is_Holiday"] = 1  # if every row is a holiday date

    # De-duplicate by date if necessary
    df = df.drop_duplicates(subset=[date_col])

    return df[[date_col, "Is_Holiday"]].rename(columns={date_col: "Date"})


def load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # You may need to harmonize city naming here
    # e.g. strip spaces, upper/lower case, etc.
    # df["City"] = df["City"].str.strip()

    return df


def aggregate_accidents_city_day(df_acc: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_acc
        .groupby(["City", "State", "Date"], as_index=False)
        .agg(
            AccidentCount=("ID", "count") if "ID" in df_acc.columns
            else ("Severity", "count"),
            AverageSeverity=("Severity", "mean"),
            MaxSeverity=("Severity", "max"),
        )
    )

    # HighSeverityFlag = 1 if any accident with Severity >= 3 on that day
    grouped["HighSeverityFlag"] = (grouped["MaxSeverity"] >= 3).astype(int)
    return grouped


def engineer_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # Monday=0
    df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Season mapping (Northern Hemisphere)
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["Season"] = df["Month"].map(month_to_season)

    return df


def engineer_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Adjust column names depending on your weather dataset
    precip_col = "Precipitation" if "Precipitation" in df.columns else None
    snow_col = "Snow" if "Snow" in df.columns else None
    vis_col = "Visibility" if "Visibility" in df.columns else None

    if precip_col:
        df["is_rainy"] = (df[precip_col] > 0).astype(int)
    if snow_col:
        df["is_snowy"] = (df[snow_col] > 0).astype(int)
    if vis_col:
        df["low_visibility"] = (df[vis_col] < 5).astype(int)  # threshold tunable

    return df


def add_demographic_features(df_city_day: pd.DataFrame,
                             df_demo: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_demo is None:
        return df_city_day

    # Expect demographics to have City-level population, etc.
    merged = df_city_day.merge(df_demo, on="City", how="left")

    if "Population" in merged.columns:
        merged["AccidentsPer100k"] = (
            merged["AccidentCount"] / merged["Population"] * 100000
        )

    return merged


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort for lag/rolling
    df = df.sort_values(["City", "Date"])

    # 1-day lag
    df["AccidentCount_lag1"] = (
        df.groupby("City")["AccidentCount"].shift(1)
    )

    # 7-day moving average
    df["AccidentCount_7d_ma"] = (
        df.groupby("City")["AccidentCount"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    return df


def build_city_day_dataset(
    accidents_path: Path,
    holidays_path: Path,
    demographics_path: Optional[Path] = None,
) -> pd.DataFrame:
    # Load
    df_acc = load_accidents(accidents_path)
    weather_dir = DATA_RAW / "2"  # folder that contains KCLT.csv, KNYC.csv, ...
    df_weather = load_weather_us_weather_history(weather_dir)
    df_holidays = load_holidays(holidays_path)
    df_demo = load_demographics(demographics_path) if demographics_path is not None else None

    # Aggregate accidents
    df_city_day = aggregate_accidents_city_day(df_acc)

    # Merge weather: you may need to tune join keys depending on dataset
    # If weather is city-level, expect City + Date
    weather_key_cols = ["City", "Date"] if "City" in df_weather.columns else ["Date"]
    df_weather_renamed = df_weather.rename(columns={df_weather.columns[0]: "Date"}) \
        if "Date" not in df_weather.columns else df_weather

    df_merged = df_city_day.merge(
        df_weather,
        on=["City", "State", "Date"],
        how="left",
        suffixes=("", "_weather"),
    )

    # Merge holidays (Date-level)
    df_merged = df_merged.merge(df_holidays, on="Date", how="left")
    df_merged["Is_Holiday"] = df_merged["Is_Holiday"].fillna(0).astype(int)

    # Calendar & weather feature engineering
    df_merged = engineer_calendar_features(df_merged)
    df_merged = engineer_weather_flags(df_merged)

    # Demographics / normalization
    df_merged = add_demographic_features(df_merged, df_demo)

    # Lag features
    df_merged = create_lag_features(df_merged)

    return df_merged


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    accidents_path = DATA_RAW / "US_Accidents_March23.csv"
    holidays_path = DATA_RAW / "US Holiday Dates (2004-2021).csv"
    weather_dir = DATA_RAW / "2"

    #  one-time heavy step (cached after first run)
    df_acc = build_accidents_subset(accidents_path)

    # weather + holidays as before
    df_weather = load_weather_us_weather_history(weather_dir)
    df_holidays = load_holidays(holidays_path)

    df_city_day = aggregate_accidents_city_day(df_acc)

    df_merged = df_city_day.merge(
        df_weather,
        on=["City", "State", "Date"],
        how="left",
        suffixes=("", "_weather"),
    )

    df_merged = df_merged.merge(df_holidays, on="Date", how="left")
    df_merged["Is_Holiday"] = df_merged["Is_Holiday"].fillna(0).astype(int)

    df_merged = engineer_calendar_features(df_merged)
    df_merged = engineer_weather_flags(df_merged)
    df_merged = create_lag_features(df_merged)

    out_path = DATA_PROCESSED / "city_day_merged.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"✅ Saved merged dataset to {out_path} with shape {df_merged.shape}")



if __name__ == "__main__":
    main()
