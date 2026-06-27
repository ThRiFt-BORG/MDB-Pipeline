"""
00_config.py — Central configuration for MDB Waterbird Pipeline
All paths, constants, bounding boxes, and schema definitions live here.
"""
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Raw input files (drop your CSVs here) ───────────────────────────────────
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
HARMONISED = ROOT / "data" / "harmonised"
OUTPUTS = ROOT / "outputs"

RAW_FILES = {
    "S1": RAW / "flow_mer_diversity.csv",
    "S2": RAW / "flow_mer_breeding.csv",
    "S3": RAW / "mdba_aws.csv",
    "S4": RAW / "unsw_aerial.csv",
}

# ── Output files ─────────────────────────────────────────────────────────────
MASTER_PARQUET = HARMONISED / "mdb_waterbirds_master.parquet"
MASTER_CSV     = HARMONISED / "mdb_waterbirds_master.csv"
MACQUARIE_CSV  = HARMONISED / "macquarie_marshes.csv"
GWYDIR_CSV     = HARMONISED / "gwydir_wetlands.csv"
MDB_CSV        = HARMONISED / "mdb_all_sites.csv"
QC_LOG         = OUTPUTS / "qc_report.txt"

# ── Australia bounding box (coordinate validation) ───────────────────────────
AUS_LAT = (-44.0, -10.0)
AUS_LON = (112.0, 154.0)

# ── Study-site bounding boxes [lat_min, lat_max, lon_min, lon_max] ───────────
SITES = {
    "macquarie_marshes": {
        "lat": (-31.8, -30.2),
        "lon": (147.0, 148.5),
        "name_keywords": ["macquarie", "mac marsh"],
    },
    "gwydir_wetlands": {
        "lat": (-29.8, -28.2),
        "lon": (149.0, 151.0),
        "name_keywords": ["gwydir"],
        "program_keywords": ["gwydir"],
    },
}

# ── MDB bounding box (coarse; use MDBA polygon shapefile for precise filter) ─
MDB_BBOX = {"lat": (-37.5, -23.0), "lon": (138.0, 153.0)}

# ── Unified schema column order ───────────────────────────────────────────────
SCHEMA_COLS = [
    "source_id", "program", "site_name", "latitude", "longitude",
    "survey_date", "year", "month", "water_year", "survey_type",
    "scientific_name", "common_name", "functional_group", "species_code",
    "abundance", "nest_count", "brood_count",
    "prop_surveyed", "percent_full",
    "breeding_evidence", "veg_community",
]

# ── Water year helper: Jul–Jun (e.g. Jul 2022–Jun 2023 = WY2023) ─────────────
def water_year(year: int, month: int) -> int:
    return year + 1 if month >= 7 else year

# ── Species name normalisation regex patterns ─────────────────────────────────
import re
AUTHOR_YEAR_PATTERNS = [
    re.compile(r'\s*\([^)]*\d{4}[^)]*\)'),   # (Temminck, 1824)
    re.compile(r',\s*\d{4}\s*$'),              # trailing , 1824
    re.compile(r'\s+[A-Z][a-z]+,\s*\d{4}'),   # Temminck, 1824 without parens
]
