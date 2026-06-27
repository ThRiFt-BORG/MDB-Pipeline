"""
05_site_filter.py - Extract site-specific subsets from the master dataset.

Filter strategy:
  Primary sites: bounding box OR name keyword (union - more inclusive).
  Full MDB: spatial bbox clip AND mdb_scope=True (excludes 2008 S4
            continent-wide survey records and any other out-of-basin records
            flagged in Stage 04). Falls back to bbox-only if mdb_scope column
            absent (e.g. running against an older master).

Shapefile support:
  If MDB_SHAPEFILE exists, uses the real polygon boundary for the MDB clip
  instead of the coarse bounding box. Falls back to bbox if geopandas/file
  unavailable.
"""
import pandas as pd
from src.config_00 import (
    SITES, MDB_BBOX, MDB_SHAPEFILE,
    MACQUARIE_CSV, GWYDIR_CSV, MDB_CSV
)


def _bbox_mask(df: pd.DataFrame, lat_range: tuple, lon_range: tuple) -> pd.Series:
    return (
        df["latitude"].between(*lat_range) &
        df["longitude"].between(*lon_range)
    )


def _name_mask(df: pd.DataFrame, keywords: list) -> pd.Series:
    if not keywords:
        return pd.Series(False, index=df.index)
    pattern = "|".join(keywords)
    return (
        df["site_name"].str.lower().str.contains(pattern, na=False) |
        df["program"].str.lower().str.contains(pattern, na=False)
    )


def _shapefile_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for records inside the real MDB polygon.
    Falls back to bbox mask if geopandas unavailable or shapefile missing.
    """
    if not MDB_SHAPEFILE.exists():
        return _bbox_mask(df, MDB_BBOX["lat"], MDB_BBOX["lon"])
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import os
        # Suppress PROJ conflict warning from PostGIS
        os.environ.setdefault("PROJ_DATA", "")
        mdb_gdf = gpd.read_file(MDB_SHAPEFILE)
        mdb_union = mdb_gdf.union_all() if hasattr(mdb_gdf, "union_all") else mdb_gdf.unary_union
        coords = df[["longitude", "latitude"]].dropna()
        inside = coords.apply(
            lambda r: mdb_union.contains(Point(r["longitude"], r["latitude"])), axis=1
        )
        mask = pd.Series(False, index=df.index)
        mask.loc[inside.index] = inside
        return mask
    except Exception:
        # Any geopandas/PROJ error: fall back to bbox
        return _bbox_mask(df, MDB_BBOX["lat"], MDB_BBOX["lon"])


def filter_site(df: pd.DataFrame, site_key: str) -> pd.DataFrame:
    cfg = SITES[site_key]
    bbox = _bbox_mask(df, cfg["lat"], cfg["lon"])
    name = _name_mask(
        df, cfg.get("name_keywords", []) + cfg.get("program_keywords", [])
    )
    return df[bbox | name].copy()


def filter_mdb(df: pd.DataFrame, use_shapefile: bool = True) -> pd.DataFrame:
    """
    Clip to MDB using real shapefile polygon (if available) then apply
    the mdb_scope flag from Stage 04 to remove out-of-basin records.
    """
    if use_shapefile:
        spatial_mask = _shapefile_mask(df)
    else:
        spatial_mask = _bbox_mask(df, MDB_BBOX["lat"], MDB_BBOX["lon"])

    if "mdb_scope" in df.columns:
        combined = spatial_mask & df["mdb_scope"]
    else:
        combined = spatial_mask

    return df[combined].copy()


def filter_all(master: pd.DataFrame, verbose: bool = True) -> tuple:
    log = ["", "=== STAGE 05: SITE FILTER ==="]
    subsets = {}

    # -- Macquarie Marshes ----------------------------------------------------
    macquarie = filter_site(master, "macquarie_marshes")
    subsets["macquarie"] = macquarie
    macquarie.to_csv(MACQUARIE_CSV, index=False, encoding="utf-8")
    log.append(
        f"  Macquarie Marshes: {len(macquarie):,} records, "
        f"{int(macquarie['year'].min())}-{int(macquarie['year'].max())}, "
        f"{macquarie['scientific_name'].nunique()} species"
    )

    # -- Gwydir Wetlands ------------------------------------------------------
    gwydir = filter_site(master, "gwydir_wetlands")
    subsets["gwydir"] = gwydir
    gwydir.to_csv(GWYDIR_CSV, index=False, encoding="utf-8")
    log.append(
        f"  Gwydir Wetlands:   {len(gwydir):,} records, "
        f"{int(gwydir['year'].min())}-{int(gwydir['year'].max())}, "
        f"{gwydir['scientific_name'].nunique()} species"
    )

    # -- Full MDB (shapefile clip + mdb_scope flag) ---------------------------
    shp_available = MDB_SHAPEFILE.exists()
    mdb = filter_mdb(master, use_shapefile=shp_available)
    subsets["mdb"] = mdb
    mdb.to_csv(MDB_CSV, index=False, encoding="utf-8")

    method = "shapefile polygon" if shp_available else "bounding box (shapefile not found)"
    s4_2008_excl = 0
    if "mdb_scope" in master.columns:
        s4_2008_excl = int((
            (master["source_id"] == "S4") &
            (master["year"] == 2008) &
            (~master["mdb_scope"])
        ).sum())
    log.append(
        f"  Full MDB:          {len(mdb):,} records, "
        f"{int(mdb['year'].min())}-{int(mdb['year'].max())}, "
        f"{mdb['scientific_name'].nunique()} species "
        f"[clip: {method}, {s4_2008_excl:,} S4-2008 out-of-basin excluded]"
    )

    if verbose:
        print("\n".join(log))
    return subsets, log
