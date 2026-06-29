"""
07_source_maps.py — Four separate spatial maps, one per source, filtered to 2010+.
Colour intensity encodes log(abundance+1) within each source.
Saved to outputs/plot_source_maps_2010plus.png

Clipping: identical two-step approach to eda_plots_06.py
  1. mdb_scope flag (set in Stage 04)
  2. Precise MDB polygon clip via geopandas + MDB_SHAPEFILE

Visual style: publication-quality light-background aesthetic consistent with
eda_plots_06.py and species_analysis_08.py.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from src.config_00 import MASTER_PARQUET, OUTPUTS, MDB_BBOX, MDB_SHAPEFILE

OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source palette & labels  (unchanged)
# ---------------------------------------------------------------------------
PALETTE = {
    "S1": "#3bb8e0",
    "S2": "#c05fa3",
    "S3": "#e8a230",
    "S4": "#2d9b6f",
}
SOURCE_LABELS = {
    "S1": "Flow-MER Diversity",
    "S2": "Flow-MER Breeding Colonies",
    "S3": "MDBA 38-site AWS",
    "S4": "UNSW Aerial 1983–2019",
}
SITE_BOXES = [
    ("Macquarie\nMarshes", (147.0, 148.5), (-31.8, -30.2)),
    ("Gwydir\nWetlands",   (149.0, 151.0), (-29.8, -28.2)),
]

# ---------------------------------------------------------------------------
# Global style — publication-quality, light-grey background
# Matches eda_plots_06.py / species_analysis_08.py design language.
# ---------------------------------------------------------------------------
LIGHT_GREY  = "#F2F2F0"
DARK_GREY   = "#3A3A3A"
MID_GREY    = "#888888"
GRID_GREY   = "#DDDDDA"
SPINE_GREY  = "#CCCCCA"
MAP_FILL    = "#DDEAF0"   # pale ice-blue basin fill — same as eda_plots_06.py
MAP_WATER   = "#EEF3F8"   # background water / outside-basin colour

mpl.rcParams.update({
    "figure.facecolor":  LIGHT_GREY,
    "axes.facecolor":    MAP_WATER,
    "axes.edgecolor":    SPINE_GREY,
    "axes.labelcolor":   DARK_GREY,
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "axes.titlepad":     7,
    "xtick.color":       MID_GREY,
    "ytick.color":       MID_GREY,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "text.color":        DARK_GREY,
    "grid.color":        GRID_GREY,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.4,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  SPINE_GREY,
    "legend.facecolor":  LIGHT_GREY,
    "legend.fontsize":   8,
    "font.family":       "DejaVu Sans",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": LIGHT_GREY,
})

# ---------------------------------------------------------------------------
# Shared styling helpers  (mirrors eda_plots_06.py)
# ---------------------------------------------------------------------------

def add_panel_label(ax, label: str, x: float = -0.06, y: float = 1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=DARK_GREY,
            va="top", ha="right")


def _draw_mdb_polygon(ax, mdb_gdf) -> None:
    """Draw MDB sub-polygons (light fill) and bold outer boundary."""
    for _, row in mdb_gdf.iterrows():
        polys = list(getattr(row.geometry, "geoms", [row.geometry]))
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=MAP_FILL, alpha=1.0, zorder=1)
            ax.plot(x, y, color=SPINE_GREY, linewidth=0.35, alpha=0.7, zorder=2)
    # Bold outer boundary
    for geom in mdb_gdf.geometry:
        polys = list(getattr(geom, "geoms", [geom]))
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#4A7FA5", linewidth=1.1, alpha=0.85, zorder=3)


def _draw_study_boxes(ax) -> None:
    """Draw primary study-site annotation boxes."""
    for label, lon_rng, lat_rng in SITE_BOXES:
        ax.add_patch(mpatches.Rectangle(
            (lon_rng[0], lat_rng[0]),
            lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0],
            fill=False, edgecolor="#C0392B",
            linewidth=1.0, linestyle="--", zorder=5))
        ax.text(lon_rng[0] + 0.05, lat_rng[1] + 0.08, label,
                color="#C0392B", fontsize=6.5, fontweight="bold",
                va="bottom", zorder=6)


# ---------------------------------------------------------------------------
# MDB clipping — identical two-step logic to eda_plots_06.py
# ---------------------------------------------------------------------------

def _clip_to_mdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-step MDB clip matching eda_plots_06.py exactly:
      1. mdb_scope flag (Stage 04) — drops 2008 continent-wide S4 records
      2. Precise shapefile polygon clip via geopandas
    Falls back to bbox-only if geopandas/shapefile unavailable.
    """
    # Step 1: mdb_scope flag
    if "mdb_scope" in df.columns:
        df = df[df["mdb_scope"]].copy()
    else:
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
            pass  # geopandas/PROJ conflict — bbox clip already applied

    return df


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_source_maps(df: Optional[pd.DataFrame] = None, year_from: int = 2010):
    # ---- Data loading (unchanged) ----
    if df is None:
        if not MASTER_PARQUET.exists():
            raise FileNotFoundError(f"Run the pipeline first: {MASTER_PARQUET}")
        df = pd.read_parquet(MASTER_PARQUET)

    # ---- MDB clip (now uses polygon, not bbox) ----
    df = _clip_to_mdb(df)

    # ---- Year filter (unchanged) ----
    df = df[df["year"] >= year_from].copy()
    df = df.dropna(subset=["latitude", "longitude"])

    # ---- Load MDB shapefile once for all panels ----
    mdb_gdf = None
    bounds  = None
    if MDB_SHAPEFILE.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf = gpd.read_file(MDB_SHAPEFILE)
            bounds  = mdb_gdf.total_bounds   # minx miny maxx maxy
        except Exception:
            pass

    if bounds is None:
        bounds = (
            MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
            MDB_BBOX["lon"][1], MDB_BBOX["lat"][1],
        )

    pad = 0.5
    xlim = (bounds[0] - pad, bounds[2] + pad)
    ylim = (bounds[1] - pad, bounds[3] + pad)

    # ---- Figure layout ----
    sources = ["S1", "S2", "S3", "S4"]
    panel_labels = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(
        2, 2, figsize=(17, 13),
        facecolor=LIGHT_GREY,
        gridspec_kw={"hspace": 0.18, "wspace": 0.10},
    )
    axes = axes.flatten()

    for ax, sid, panel in zip(axes, sources, panel_labels):
        # Restore all four spines for map frame
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_GREY)
            spine.set_linewidth(0.7)
            spine.set_visible(True)

        ax.set_facecolor(MAP_WATER)

        # 1. MDB polygon base layer
        if mdb_gdf is not None:
            _draw_mdb_polygon(ax, mdb_gdf)

        # 2. Ghost context — all-source points, very faint
        ax.scatter(df["longitude"], df["latitude"],
                   s=1, alpha=0.05, color=MID_GREY,
                   zorder=2, rasterized=True)

        sub = df[df["source_id"] == sid].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No records ≥ {year_from}",
                    transform=ax.transAxes, ha="center", va="center",
                    color=MID_GREY, fontsize=11)
        else:
            # log-abundance colour encoding (unchanged)
            sub["log_abund"] = np.log1p(sub["abundance"].fillna(0))
            vmin = sub["log_abund"].quantile(0.05)
            vmax = sub["log_abund"].quantile(0.95)
            if vmax <= vmin:
                vmax = vmin + 1

            # Colourmap: white → source colour (light bg reads correctly)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                f"cmap_{sid}",
                ["#E8E8E8", PALETTE[sid], DARK_GREY],
                N=256,
            )
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(
                sub["longitude"], sub["latitude"],
                c=sub["log_abund"], cmap=cmap, norm=norm,
                s=9, alpha=0.72, zorder=4,
                edgecolors="none", rasterized=True,
            )

            cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.025)
            cbar.set_label("log(abundance + 1)", color=MID_GREY, fontsize=7)
            cbar.ax.yaxis.set_tick_params(color=MID_GREY, labelsize=6.5)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MID_GREY)
            for spine in cbar.ax.spines.values():
                spine.set_edgecolor(SPINE_GREY)

        # 3. Study-site annotation boxes
        _draw_study_boxes(ax)

        # 4. Axes limits — tight to MDB extent, not full Australia
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel("Longitude (°E)", fontsize=8, color=DARK_GREY)
        ax.set_ylabel("Latitude (°N)",  fontsize=8, color=DARK_GREY)
        ax.tick_params(labelsize=7, length=3, colors=MID_GREY)
        ax.grid(visible=True, zorder=0, color=GRID_GREY,
                linestyle="--", linewidth=0.35, alpha=0.6)

        # Compact panel title
        n = len(sub)
        yr_range = (f"{int(sub['year'].min())}–{int(sub['year'].max())}"
                    if n > 0 else "")
        ax.set_title(
            f"{SOURCE_LABELS[sid]}\n"
            f"{n:,} records · {yr_range}",
            fontsize=9.5, fontweight="bold",
            color=PALETTE[sid], pad=6,
        )

        add_panel_label(ax, panel)

    # ---- Figure-level title ----
    fig.suptitle(
        f"MDB waterbird survey records by source · {year_from} onwards\n"
        "Colour intensity = log(abundance + 1) within each source",
        fontsize=13, fontweight="bold", color=DARK_GREY, y=1.01,
    )

    # ---- Shared legend ----
    legend_handles = [
        mpatches.Patch(facecolor=MAP_FILL, edgecolor="#4A7FA5",
                       linewidth=0.8, label="MDB polygon (basin boundary)"),
        mpatches.Rectangle((0, 0), 1, 1,
                            fill=False, edgecolor="#C0392B",
                            linestyle="--", linewidth=1.0,
                            label="Primary study sites"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MID_GREY, markersize=5, alpha=0.5,
               label="All-source context (ghost)"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=3, fontsize=8,
               framealpha=0.92, facecolor=LIGHT_GREY,
               edgecolor=SPINE_GREY, labelcolor=DARK_GREY,
               bbox_to_anchor=(0.5, -0.025))

    # ---- Save ----
    out = OUTPUTS / f"plot_source_maps_{year_from}plus.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


if __name__ == "__main__":
    plot_source_maps()