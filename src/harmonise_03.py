"""
03_harmonise.py - Map each cleaned source to the unified 21-column schema.

ROOT CAUSE FIX: The original _base() created an empty DataFrame and assigned
scalar values to it. When columns were then assigned from the source df using
'out["col"] = df["col"]', pandas had no shared index between the empty 'out'
and the source 'df', resulting in all-NaN columns and an empty source_id.

Fix: build 'out' directly from the source DataFrame's index so all column
assignments align correctly. source_id and survey_type are broadcast scalars
AFTER the frame is sized correctly.
"""
import pandas as pd
import numpy as np
from src.config_00 import SCHEMA_COLS


def _make_out(df: pd.DataFrame, sid: str, survey_type: str) -> pd.DataFrame:
    """Create output frame sized to match df, with metadata columns pre-filled."""
    out = pd.DataFrame(index=df.index)
    out["source_id"]   = sid
    out["survey_type"] = survey_type
    return out


def harmonise_s1(df: pd.DataFrame) -> pd.DataFrame:
    out = _make_out(df, "S1", "ground")
    out["program"]          = df["Program"].values
    out["site_name"]        = df["SamplePoint"].str.strip().values
    out["latitude"]         = df["Latitude"].values
    out["longitude"]        = df["Longitude"].values
    out["survey_date"]      = df["survey_date"].values
    out["year"]             = df["year"].values
    out["month"]            = df["month"].values
    out["water_year"]       = df["water_year"].values
    out["scientific_name"]  = df["speciesName"].values
    out["common_name"]      = np.nan
    out["functional_group"] = np.nan
    out["species_code"]     = df["speciesCode"].astype(str).values if "speciesCode" in df else np.nan
    out["abundance"]        = df["abundance"].values
    out["nest_count"]       = np.nan
    out["brood_count"]      = np.nan
    out["prop_surveyed"]    = df["prop_surveyed"].values if "prop_surveyed" in df else np.nan
    out["percent_full"]     = np.nan
    out["breeding_evidence"] = df["breedEvidence"].astype(bool).values if "breedEvidence" in df else False
    out["veg_community"]    = df["vegCommunity"].values if "vegCommunity" in df else np.nan
    return out[SCHEMA_COLS].reset_index(drop=True)


def harmonise_s2(df: pd.DataFrame) -> pd.DataFrame:
    out = _make_out(df, "S2", "colony")
    out["program"]          = df["Program"].values
    out["site_name"]        = df["SamplePoint"].str.strip().values
    out["latitude"]         = df["Latitude"].values
    out["longitude"]        = df["Longitude"].values
    out["survey_date"]      = df["survey_date"].values
    out["year"]             = df["year"].values
    out["month"]            = df["month"].values
    out["water_year"]       = df["water_year"].values
    out["scientific_name"]  = df["speciesName"].values
    out["common_name"]      = np.nan
    out["functional_group"] = np.nan
    out["species_code"]     = df["speciesCode"].astype(str).values if "speciesCode" in df else np.nan
    out["abundance"]        = df["adultCountColonyTotal"].values
    out["nest_count"]       = df["nestCountColonyTotal"].values
    out["brood_count"]      = np.nan
    out["prop_surveyed"]    = np.nan
    out["percent_full"]     = np.nan
    out["breeding_evidence"] = True
    out["veg_community"]    = np.nan
    return out[SCHEMA_COLS].reset_index(drop=True)


def harmonise_s3(df: pd.DataFrame) -> pd.DataFrame:
    out = _make_out(df, "S3", "aerial")
    out["program"]          = df["SurveyProgram"].values
    site_col = "Subwetland" if "Subwetland" in df else "Wetland"
    out["site_name"]        = df[site_col].str.strip().values
    out["latitude"]         = df["LatitudeDec"].values
    out["longitude"]        = df["LongitudeDec"].values
    out["survey_date"]      = df["survey_date"].values
    out["year"]             = df["year"].values
    out["month"]            = df["month"].values
    out["water_year"]       = df["water_year"].values
    out["scientific_name"]  = df["ScientificName"].values
    out["common_name"]      = df["CommonName"].values
    out["functional_group"] = df["Fx_Group"].values
    out["species_code"]     = df["SppCode"].astype(str).values
    out["abundance"]        = df["SumOfCount"].values
    out["nest_count"]       = df["SumOfNests"].values
    out["brood_count"]      = df["SumOfBroods"].values
    out["prop_surveyed"]    = df["prop_surveyed"].values
    out["percent_full"]     = df["PercentFilled"].values
    out["breeding_evidence"] = (df["SumOfNests"].fillna(0) > 0).values
    out["veg_community"]    = np.nan
    return out[SCHEMA_COLS].reset_index(drop=True)


def harmonise_s4(df: pd.DataFrame) -> pd.DataFrame:
    out = _make_out(df, "S4", "aerial")
    out["program"]          = df["Program"].values
    out["site_name"]        = df["Name"].str.strip().values
    out["latitude"]         = df["Latitude"].values
    out["longitude"]        = df["Longitude"].values
    out["survey_date"]      = df["survey_date"].values
    out["year"]             = df["year"].values
    out["month"]            = df["month"].values
    out["water_year"]       = df["water_year"].values
    out["scientific_name"]  = df["Scientific_name"].values
    out["common_name"]      = df["Common_name"].values
    out["functional_group"] = df["Functional_group"].values
    out["species_code"]     = df["Species_code"].astype(str).values
    out["abundance"]        = df["Sum_of_count"].values
    out["nest_count"]       = df["Sum_of_nests"].values
    out["brood_count"]      = df["Sum_of_broods"].values
    out["prop_surveyed"]    = df["prop_surveyed"].values
    out["percent_full"]     = df["Percent_full"].values
    out["breeding_evidence"] = (df["Sum_of_nests"].fillna(0) > 0).values
    out["veg_community"]    = np.nan
    return out[SCHEMA_COLS].reset_index(drop=True)


HARMONISERS = {
    "S1": harmonise_s1,
    "S2": harmonise_s2,
    "S3": harmonise_s3,
    "S4": harmonise_s4,
}


def harmonise_all(cleaned: dict, verbose: bool = True) -> tuple:
    log = ["", "=== STAGE 03: HARMONISE ==="]
    harmonised = {}
    for sid, fn in HARMONISERS.items():
        h = fn(cleaned[sid])
        # Verify source_id populated correctly
        assert h["source_id"].notna().all(), f"{sid}: source_id has nulls after harmonise"
        assert (h["source_id"] == sid).all(), f"{sid}: source_id values wrong after harmonise"
        harmonised[sid] = h
        log.append(f"  {sid}: {len(h):,} rows -> unified schema")
    if verbose:
        print("\n".join(log))
    return harmonised, log
