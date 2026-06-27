"""
02_clean.py - Per-source cleaning: spatial QC, date parsing,
              species name normalisation, count validation.
"""
import pandas as pd
import numpy as np
from src.config_00 import (
    AUS_LAT, AUS_LON, AUTHOR_YEAR_PATTERNS, PROCESSED, water_year
)

PROCESSED.mkdir(parents=True, exist_ok=True)

# -- Helpers -----------------------------------------------------------------

def _spatial_qc(df: pd.DataFrame, lat_col: str, lon_col: str,
                source_id: str, log: list) -> pd.DataFrame:
    """Remove rows with coordinates outside Australia."""
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    mask = (
        df[lat_col].between(*AUS_LAT) &
        df[lon_col].between(*AUS_LON)
    )
    bad = (~mask).sum()
    if bad:
        log.append(
            f"  {source_id} spatial: removed {bad} invalid coords "
            f"(lat range seen: {df[lat_col].min():.2f} to {df[lat_col].max():.2f})"
        )
    return df[mask].copy()


def _parse_dates(series: pd.Series) -> pd.Series:
    # format="mixed" is the pandas 2.x replacement for infer_datetime_format
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _add_time_cols(df: pd.DataFrame, date_col: str = "survey_date") -> pd.DataFrame:
    df["year"]       = df[date_col].dt.year.astype("Int64")
    df["month"]      = df[date_col].dt.month.astype("Int64")
    df["water_year"] = df.apply(
        lambda r: water_year(int(r["year"]), int(r["month"]))
        if pd.notna(r["year"]) and pd.notna(r["month"]) else pd.NA,
        axis=1
    ).astype("Int64")
    return df


def _normalise_species(series: pd.Series) -> pd.Series:
    """Strip author + year annotations from scientific names."""
    def clean(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        for pattern in AUTHOR_YEAR_PATTERNS:
            name = pattern.sub("", name)
        parts = name.strip().split()
        if len(parts) >= 2:
            name = f"{parts[0].capitalize()} {parts[1].lower()}"
        return name.strip()
    return series.apply(clean)


def _count_qc(df: pd.DataFrame, col: str, log: list, sid: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    neg = (df[col] < 0).sum()
    if neg:
        log.append(f"  {sid} {col}: {neg} negative values set to NaN")
        df.loc[df[col] < 0, col] = np.nan
    return df


# -- Per-source cleaners -----------------------------------------------------

def clean_s1(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Flow-MER Waterbird Diversity"""
    sid = "S1"
    df = _spatial_qc(df, "Latitude", "Longitude", sid, log)
    df["survey_date"] = _parse_dates(df["SampleDate"])
    df = _add_time_cols(df)
    df["speciesName"] = _normalise_species(df["speciesName"])
    df = _count_qc(df, "abundance", log, sid)
    df = df.rename(columns={"propSurveyed(number)": "prop_surveyed"})
    log.append(f"  {sid} clean: {len(df):,} rows retained")
    return df


def clean_s2(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Flow-MER Waterbird Breeding Colonies"""
    sid = "S2"
    df = _spatial_qc(df, "Latitude", "Longitude", sid, log)
    df["survey_date"] = _parse_dates(df["SampleDate"])
    df = _add_time_cols(df)
    df["speciesName"] = _normalise_species(df["speciesName"])
    for col in ["adultCountColonyTotal", "nestCountColonyTotal"]:
        df = _count_qc(df, col, log, sid)
    log.append(f"  {sid} clean: {len(df):,} rows retained")
    return df


def clean_s3(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """MDBA 38-site AWS"""
    sid = "S3"
    df = _spatial_qc(df, "LatitudeDec", "LongitudeDec", sid, log)
    df["survey_date"] = _parse_dates(df["Date"])
    df["year"]  = pd.to_numeric(df["SurvYear"], errors="coerce").astype("Int64")
    df["month"] = df["survey_date"].dt.month.astype("Int64")
    df["water_year"] = df.apply(
        lambda r: water_year(int(r["year"]), int(r["month"]))
        if pd.notna(r["year"]) and pd.notna(r["month"]) else pd.NA,
        axis=1
    ).astype("Int64")
    df["ScientificName"] = _normalise_species(df["ScientificName"])
    for col in ["SumOfCount", "SumOfNests", "SumOfBroods"]:
        df = _count_qc(df, col, log, sid)
    df["PercentFilled"] = pd.to_numeric(df["PercentFilled"], errors="coerce").clip(0, 100)
    df["SurveyArea"]  = pd.to_numeric(df.get("SurveyArea"),  errors="coerce")
    df["WetlandArea"] = pd.to_numeric(df.get("WetlandArea"), errors="coerce")
    df["prop_surveyed"] = (df["SurveyArea"] / df["WetlandArea"]).clip(0, 1)
    log.append(f"  {sid} clean: {len(df):,} rows retained")
    return df


def clean_s4(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """UNSW Aerial Survey 1983-2019"""
    sid = "S4"
    df = _spatial_qc(df, "Latitude", "Longitude", sid, log)
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["survey_date"] = pd.to_datetime(
        df["year"].astype(str) + "-01-01", errors="coerce"
    )
    df["month"]      = pd.NA
    df["water_year"] = (
        df["year"].apply(lambda y: y + 1 if pd.notna(y) else pd.NA)
        .astype("Int64")
    )
    df["Scientific_name"] = _normalise_species(df["Scientific_name"])
    for col in ["Sum_of_count", "Sum_of_nests", "Sum_of_broods"]:
        df = _count_qc(df, col, log, sid)
    df["Percent_full"] = pd.to_numeric(df["Percent_full"], errors="coerce").clip(0, 100)
    df["Survey_Wetland_Area"] = pd.to_numeric(df.get("Survey_Wetland_Area"), errors="coerce")
    df["Wetland_Area"]        = pd.to_numeric(df.get("Wetland_Area"),        errors="coerce")
    df["prop_surveyed"] = (df["Survey_Wetland_Area"] / df["Wetland_Area"]).clip(0, 1)
    log.append(f"  {sid} clean: {len(df):,} rows retained")
    return df


def clean_all(dfs: dict, verbose: bool = True) -> tuple:
    log = ["", "=== STAGE 02: CLEAN ==="]
    cleaners = {"S1": clean_s1, "S2": clean_s2, "S3": clean_s3, "S4": clean_s4}
    cleaned = {}
    for sid, fn in cleaners.items():
        cleaned[sid] = fn(dfs[sid].copy(), log)
        out = PROCESSED / f"{sid}_clean.csv"
        cleaned[sid].to_csv(out, index=False, encoding="utf-8")
    if verbose:
        print("\n".join(log))
    return cleaned, log

if __name__ == "__main__":
    from src.ingest_01 import load_all
    dfs = load_all(verbose=False)
    clean_all(dfs)
