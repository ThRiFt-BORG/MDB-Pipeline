"""
06_eda_plots.py - EDA visualisations clipped strictly to the MDB polygon.

ALL plots operate on MDB-scoped data only:
  - mdb_scope=True (set in Stage 04) removes out-of-basin records incl 2008 S4
  - Shapefile polygon clip (via geopandas) removes anything outside the real
    basin boundary even if it passed the bbox filter
  - Plot 4 draws the real MDB boundary from the shapefile, no study-site boxes,
    zoomed tightly to the basin extent
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from typing import Optional
from pathlib import Path
from src.config_00 import OUTPUTS, MASTER_PARQUET, MDB_SHAPEFILE, MDB_BBOX

OUTPUTS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "S1": "#3bb8e0", "S2": "#c05fa3",
    "S3": "#e8a230", "S4": "#2d9b6f",
}
SOURCE_LABELS = {
    "S1": "Flow-MER Diversity",
    "S2": "Flow-MER Breeding",
    "S3": "MDBA 38-site AWS",
    "S4": "UNSW Aerial 1983-2019",
}

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
})


def _load(path=MASTER_PARQUET) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Run the pipeline first to generate {path}")


def _clip_to_mdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MDB scope in two steps:
      1. mdb_scope flag (set in Stage 04) - drops 2008 continent-wide records
         and anything outside the coarse MDB bbox
      2. Precise shapefile polygon clip via geopandas - drops anything inside
         the bbox but outside the real basin boundary
    Falls back to bbox-only if geopandas/shapefile unavailable.
    """
    # Step 1: mdb_scope flag
    if "mdb_scope" in df.columns:
        df = df[df["mdb_scope"]].copy()
    else:
        # Fallback: apply bbox manually
        df = df[
            df["latitude"].between(*MDB_BBOX["lat"]) &
            df["longitude"].between(*MDB_BBOX["lon"])
        ].copy()

    # Step 2: precise polygon clip
    if MDB_SHAPEFILE.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf   = gpd.read_file(MDB_SHAPEFILE)
            mdb_union = (mdb_gdf.union_all()
                         if hasattr(mdb_gdf, "union_all")
                         else mdb_gdf.unary_union)
            pts = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326",
            )
            df = pts[pts.within(mdb_union)].drop(columns="geometry").copy()
        except Exception:
            pass  # geopandas/PROJ conflict - bbox clip already applied above

    return df


# -- Plot 1: Abundance over time (MDB only) -----------------------------------
def plot_abundance_over_time(df: pd.DataFrame):
    mdb = _clip_to_mdb(df)
    annual = (
        mdb.dropna(subset=["year", "abundance"])
           .groupby(["year", "source_id"])["abundance"]
           .sum()
           .reset_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for sid, grp in annual.groupby("source_id"):
        ax.plot(
            grp["year"].astype(int), grp["abundance"] / 1e6,
            color=PALETTE.get(str(sid), "grey"), linewidth=1.8,
            label=SOURCE_LABELS.get(str(sid), str(sid)),
            marker="o", markersize=3,
        )
    ax.set_title(
        "Total Waterbird Abundance Over Time by Source  —  MDB Region",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Count (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    out = OUTPUTS / "plot1_abundance_over_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# -- Plot 2: Species richness (MDB only) --------------------------------------
def plot_species_richness(df: pd.DataFrame):
    mdb = _clip_to_mdb(df)
    richness = (
        mdb.dropna(subset=["water_year", "scientific_name"])
           .groupby(["water_year", "source_id"])["scientific_name"]
           .nunique()
           .reset_index(name="species_richness")
    )
    if richness.empty:
        print("  Plot 2: No data after MDB clip - skipping")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    n_sources = richness["source_id"].nunique()
    width = 0.8 / max(n_sources, 1)
    keys  = list(PALETTE.keys())
    for sid, grp in richness.groupby("source_id"):
        offset = keys.index(str(sid)) * width
        ax.bar(
            grp["water_year"].astype(int) + offset,
            grp["species_richness"],
            width=width,
            color=PALETTE.get(str(sid), "grey"),
            label=SOURCE_LABELS.get(str(sid), str(sid)),
            alpha=0.85,
        )
    ax.set_title(
        "Waterbird Species Richness per Water Year  —  MDB Region",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Water Year (Jul-Jun)")
    ax.set_ylabel("Unique Species")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")
    fig.tight_layout()
    out = OUTPUTS / "plot2_species_richness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# -- Plot 3: Inundation vs abundance (MDB only) --------------------------------
def plot_inundation_vs_abundance(df: pd.DataFrame):
    mdb = _clip_to_mdb(df)
    sub = mdb.dropna(subset=["percent_full", "abundance"]).copy()
    sub = sub[(sub["abundance"] > 0) & (sub["percent_full"] > 0)]
    if sub.empty:
        print("  Plot 3: No inundation data after MDB clip - skipping")
        return
    sub["log_abundance"] = np.log1p(sub["abundance"])

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter_data = sub.sample(min(len(sub), 8000), random_state=42)
    for sid, grp in scatter_data.groupby("source_id"):
        ax.scatter(
            grp["percent_full"], grp["log_abundance"],
            color=PALETTE.get(str(sid), "grey"), alpha=0.3, s=8,
            label=SOURCE_LABELS.get(str(sid), str(sid)),
        )
    from numpy.polynomial.polynomial import polyfit
    x = scatter_data["percent_full"].values.astype(float)
    y = scatter_data["log_abundance"].values.astype(float)
    b, m = polyfit(x, y, 1)
    xs = np.linspace(0, 100, 100)
    ax.plot(xs, m * xs + b, color="white", linewidth=1.5,
            linestyle="--", label="Trend (all sources)")
    ax.set_title(
        "Inundation Level vs Waterbird Abundance  —  MDB Region",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Wetland Percent Full (%)")
    ax.set_ylabel("log(Abundance + 1)")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True)
    fig.tight_layout()
    out = OUTPUTS / "plot3_inundation_vs_abundance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# -- Plot 4: Spatial map — MDB polygon boundary, bubbles, no study-site boxes --
def plot_spatial_map(df: pd.DataFrame):
    # Clip data to MDB polygon first
    mdb = _clip_to_mdb(df)
    sub = mdb.dropna(subset=["latitude", "longitude"])

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")   # type: ignore[union-attr]

    # Draw MDB shapefile boundary
    bounds = None
    if MDB_SHAPEFILE.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf = gpd.read_file(MDB_SHAPEFILE)
            bounds  = mdb_gdf.total_bounds   # minx miny maxx maxy
            region_fills = ["#1a2332","#1c2838","#1e2d3d","#1a2636","#192030"]
            for i, row in mdb_gdf.iterrows():
                polys = (list(row.geometry.geoms)
                         if row.geometry.geom_type == "MultiPolygon"
                         else [row.geometry])
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=region_fills[int(i) % len(region_fills)],
                            alpha=1.0, zorder=1)
                    ax.plot(x, y, color="#2d4a6b", linewidth=0.6,
                            alpha=0.7, zorder=2)
            # Outer bold boundary line
            for geom in mdb_gdf.geometry:
                polys = (list(geom.geoms)
                         if geom.geom_type == "MultiPolygon"
                         else [geom])
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color="#4a9eca", linewidth=1.4,
                            alpha=0.9, zorder=3)
        except Exception:
            bounds = None

    # Fallback axis limits if shapefile unavailable
    if bounds is None:
        bounds = (
            MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
            MDB_BBOX["lon"][1], MDB_BBOX["lat"][1],
        )

    # Bubble plot — site-level aggregation, sqrt scaling per source
    legend_handles = []
    for sid in ["S4", "S3", "S1", "S2"]:
        grp = sub[sub["source_id"] == sid].copy()
        if grp.empty:
            continue
        site_agg = (
            grp.groupby(["latitude", "longitude"])
               .agg(total_abundance=("abundance", "sum"))
               .reset_index()
        )
        cap   = site_agg["total_abundance"].quantile(0.98)
        sizes = np.sqrt(site_agg["total_abundance"].clip(0, cap) / max(cap, 1)) * 180
        sizes = sizes.clip(3, 180)

        ax.scatter(
            site_agg["longitude"], site_agg["latitude"],
            s=sizes,
            color=PALETTE[sid],
            alpha=0.75, zorder=4,
            edgecolors="none",
            rasterized=True,
        )
        legend_handles.append(
            plt.scatter([], [], s=40, color=PALETTE[sid], alpha=0.8,
                        label=f"{SOURCE_LABELS[sid]}  ({len(site_agg):,} sites)")
        )

    pad = 0.5
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.set_title(
        "Waterbird Survey Records — Murray-Darling Basin  (all sources)\n"
        "Bubble size = total abundance per site (sqrt-scaled per source)"
        "  |  Clipped to MDB polygon",
        fontsize=11, pad=12, loc="left",
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              markerscale=1, framealpha=0.4,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax.text(0.01, 0.01,
            f"{len(sub):,} records inside MDB polygon",
            transform=ax.transAxes, fontsize=8, color="#8b949e", va="bottom")
    ax.grid(True, zorder=0)
    fig.tight_layout()
    out = OUTPUTS / "plot4_spatial_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_all(df: Optional[pd.DataFrame] = None, verbose: bool = True):
    log = ["", "=== STAGE 06: EDA PLOTS ==="]
    if df is None:
        df = _load()
    plot_abundance_over_time(df)
    plot_species_richness(df)
    plot_inundation_vs_abundance(df)
    plot_spatial_map(df)
    log.append("  All plots saved to outputs/")
    if verbose:
        print("\n".join(log))
    return log


if __name__ == "__main__":
    plot_all()
