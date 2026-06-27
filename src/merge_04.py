"""
04_merge.py - Concatenate all harmonised sources, deduplicate, and save.

Deduplication — order matters:

  Pass 1 (first) - Within-S4 dual-pass dedup:
           Must run BEFORE cross-source dedup. Same exact lat/lon + year +
           species appearing twice in S4 = two aerial survey passes over the
           same wetland in the same year. Keep MAX abundance (full count).
           This fixes the inflated 2010-2011 La Nina spike at sites like
           Booligal (-33.75861, 144.8986): 63,030 and 313,602 for the same
           species — keep 313,602, drop 63,030.

  Pass 2 (second) - Cross-source coord-based dedup (S3 vs S4):
           S3 (MDBA) and S4 (UNSW) share the same aerial survey program
           2007-2019. Uses lat+lon rounded to 0.01 deg + year + species.
           Do NOT use site_name: 36,801 S4 rows have null site_name and
           drop_duplicates treats NaN==NaN, causing massive false removal.
           Keep S3 (richer metadata: CommonName, Fx_Group, brood counts).

MDB scope flag:
  Boolean column mdb_scope=True if record inside MDB bbox.
  Flags 2008 S4 continent-wide survey records and any out-of-basin records.
"""
import pandas as pd
import numpy as np
from src.config_00 import MASTER_PARQUET, MASTER_CSV, HARMONISED, MDB_BBOX

HARMONISED.mkdir(parents=True, exist_ok=True)


def merge_all(harmonised: dict, verbose: bool = True) -> tuple:
    log = ["", "=== STAGE 04: MERGE ==="]

    # -- Concat ---------------------------------------------------------------
    chunks = []
    for sid, df in harmonised.items():
        df = df.copy()
        df["source_id"] = str(sid)
        chunks.append(df)

    master = pd.concat(chunks, ignore_index=True)
    log.append(f"  Pre-dedup total: {len(master):,} rows")

    # -- Pass 1: Within-S4 dual-pass dedup (MUST run before cross-source) -----
    # Booligal 2010 example: same lat/lon/year/species with abundances 63,030
    # and 313,602 — two survey passes. Keep the larger (full count).
    s4     = master[master["source_id"] == "S4"].copy()
    non_s4 = master[master["source_id"] != "S4"].copy()

    s4 = s4.sort_values("abundance", ascending=False, na_position="last")
    s4_deduped = s4.drop_duplicates(
        subset=["latitude", "longitude", "year", "scientific_name"],
        keep="first"
    )
    removed_p1 = len(s4) - len(s4_deduped)
    log.append(
        f"  Pass 1 - within-S4 dual-pass dedup: removed {removed_p1:,} rows "
        f"(kept max abundance per exact lat/lon/year/species)"
    )

    master = pd.concat([non_s4, s4_deduped], ignore_index=True)

    # -- Pass 2: Cross-source coord-based dedup (S3 vs S4) --------------------
    aerial  = master[master["survey_type"] == "aerial"].copy()
    other   = master[master["survey_type"] != "aerial"].copy()

    aerial["_lat_r"] = aerial["latitude"].round(2)
    aerial["_lon_r"] = aerial["longitude"].round(2)
    aerial = aerial.sort_values("source_id")   # S3 sorts before S4 alphabetically
    aerial_p2 = aerial.drop_duplicates(
        subset=["_lat_r", "_lon_r", "year", "scientific_name"],
        keep="first"
    ).drop(columns=["_lat_r", "_lon_r"])

    removed_p2 = len(aerial) - len(aerial_p2)
    log.append(
        f"  Pass 2 - cross-source coord dedup (S3/S4): removed {removed_p2:,} rows"
    )

    master = pd.concat([other, aerial_p2], ignore_index=True)
    log.append(f"  Post-dedup total: {len(master):,} rows")

    # -- MDB scope flag -------------------------------------------------------
    in_bbox = (
        master["latitude"].between(*MDB_BBOX["lat"]) &
        master["longitude"].between(*MDB_BBOX["lon"])
    )
    master["mdb_scope"] = in_bbox
    outside     = (~in_bbox).sum()
    s4_2008_out = int((
        (master["source_id"] == "S4") &
        (master["year"] == 2008) &
        (~in_bbox)
    ).sum())
    log.append(
        f"  MDB scope: {outside:,} records outside MDB bbox "
        f"flagged mdb_scope=False "
        f"(incl. {s4_2008_out:,} S4-2008 continent-wide records)"
    )

    # -- Summary stats (MDB-scoped) -------------------------------------------
    mdb = master[master["mdb_scope"]]
    log.append(f"  Year range (MDB):  {int(mdb['year'].min())} - "
               f"{int(mdb['year'].max())}")
    log.append(f"  Species (MDB):     {mdb['scientific_name'].nunique():,} unique")
    log.append(f"  Sites (MDB):       {mdb['site_name'].nunique():,} unique")

    per_src     = master.groupby("source_id", observed=True).size().to_dict()
    per_src_mdb = mdb.groupby("source_id",   observed=True).size().to_dict()
    log.append(f"  Per source (all):  {per_src}")
    log.append(f"  Per source (MDB):  {per_src_mdb}")

    # -- Spike check (MDB-scoped) ---------------------------------------------
    s4_ann = mdb[mdb["source_id"] == "S4"].groupby("year")["abundance"].sum()
    log.append("  S4 MDB annual totals 2007-2012 (spike check):")
    for yr in range(2007, 2013):
        if yr in s4_ann.index:
            flag = "  <- real La Nina flood" if yr in [2010, 2011, 2012] else ""
            log.append(f"    {yr}: {int(s4_ann[yr]):,}{flag}")

    # -- Save -----------------------------------------------------------------
    master.to_parquet(MASTER_PARQUET, index=False)
    master.to_csv(MASTER_CSV, index=False, encoding="utf-8")
    log.append(f"  Saved -> {MASTER_PARQUET}")
    log.append(f"  Saved -> {MASTER_CSV}")

    if verbose:
        print("\n".join(log))
    return master, log
