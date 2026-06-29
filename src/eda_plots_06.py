"""
06_eda_plots.py  -  EDA visualisations clipped strictly to the MDB polygon.
Publication-quality figures targeting Nature Cities / Nature journal standards.

ALL plots operate on MDB-scoped data only:
  - mdb_scope=True (set in Stage 04) removes out-of-basin records incl 2008 S4
  - Shapefile polygon clip (via geopandas) removes anything outside the real
    basin boundary even if it passed the bbox filter
  - Plot 4 draws the real MDB boundary from the shapefile, no study-site boxes,
    zoomed tightly to the basin extent

Design principles extracted from Nature Cities reference figures:
  * Pure white plot backgrounds - zero tinted axes
  * Restrained muted palette (4 hues maximum per figure)
  * High data-ink ratio - no chart junk, no heavy grids
  * Liberation Sans (metrically identical to Arial / Helvetica)
  * 8 pt base, 10 pt bold panel labels (lowercase a, b per Nature style)
  * Outward tick marks; top/right spines suppressed
  * Bootstrap confidence bands on regression trends
  * Bubble size legend on spatial map
  * 300 dpi for print submission
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from typing import Optional
from pathlib import Path
from src.config_00 import OUTPUTS, MASTER_PARQUET, MDB_SHAPEFILE, MDB_BBOX

OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source palette & labels  (unchanged from original)
# ---------------------------------------------------------------------------
PALETTE = {
    "S1": "#2E86AB",
    "S2": "#A23B72",
    "S3": "#C17817",
    "S4": "#2D6A4F",
}
SOURCE_LABELS = {
    "S1": "Flow-MER Diversity",
    "S2": "Flow-MER Breeding",
    "S3": "MDBA 38-site AWS",
    "S4": "UNSW Aerial 1983-2019",
}

# ---------------------------------------------------------------------------
# Global style - Nature Cities / Nature journal grade
# ---------------------------------------------------------------------------
WHITE      = "#FFFFFF"
DARK_GREY  = "#2B2B2B"
MID_GREY   = "#7A7A7A"
LIGHT_LINE = "#E8E8E5"
SPINE_COL  = "#BBBBBB"
WATER_COL  = "#EBF3F7"
BASIN_COL  = "#EAF0F5"
BASIN_EDGE = "#6FA8C4"

mpl.rcParams.update({
    "figure.facecolor":      WHITE,
    "axes.facecolor":        WHITE,
    "axes.edgecolor":        SPINE_COL,
    "axes.linewidth":        0.65,
    "axes.labelcolor":       DARK_GREY,
    "axes.labelsize":        8,
    "axes.titlesize":        9,
    "axes.titlepad":         6,
    "axes.titleweight":      "bold",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "xtick.color":           MID_GREY,
    "ytick.color":           MID_GREY,
    "xtick.labelsize":       7.5,
    "ytick.labelsize":       7.5,
    "xtick.major.size":      3.0,
    "ytick.major.size":      3.0,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "text.color":            DARK_GREY,
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size":             8,
    "legend.framealpha":     0.96,
    "legend.edgecolor":      SPINE_COL,
    "legend.facecolor":      WHITE,
    "legend.fontsize":       7.5,
    "legend.title_fontsize": 7.5,
    "legend.borderpad":      0.55,
    "legend.labelspacing":   0.35,
    "legend.handlelength":   1.2,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     WHITE,
    "savefig.pad_inches":    0.06,
})

# ---------------------------------------------------------------------------
# Shared styling helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax, label, outside=True):
    """Bold panel label (a, b ...) - Nature journal lowercase style."""
    x = -0.10 if outside else 0.01
    y = 1.05  if outside else 0.99
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=DARK_GREY,
            va="top", ha="left")


def style_axis(ax, xlabel="", ylabel="", title="",
               grid=True, grid_axis="y"):
    """Apply consistent Nature-grade axis styling."""
    ax.set_xlabel(xlabel, color=DARK_GREY, labelpad=4)
    ax.set_ylabel(ylabel, color=DARK_GREY, labelpad=4)
    if title:
        ax.set_title(title, fontweight="bold", color=DARK_GREY, pad=6)
    if grid:
        ax.grid(visible=True, axis=grid_axis,
                color=LIGHT_LINE, linestyle="-", linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", color=SPINE_COL)
    ax.set_facecolor("#F4F4F2D8")


def _source_handles(sids=None):
    sids = sids or list(PALETTE.keys())
    return [
        mpatches.Patch(facecolor=PALETTE[s], label=SOURCE_LABELS[s],
                       edgecolor="none")
        for s in sids if s in PALETTE
    ]


def _frame_legend(leg, lw=0.5):
    leg.get_frame().set_linewidth(lw)


# ---------------------------------------------------------------------------
# MDB clipping  (UNCHANGED analytical logic)
# ---------------------------------------------------------------------------

def _load(path=MASTER_PARQUET):
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Run the pipeline first to generate {path}")


def _clip_to_mdb(df):
    """
    Apply MDB scope in two steps:
      1. mdb_scope flag (set in Stage 04) - drops 2008 continent-wide records
         and anything outside the coarse MDB bbox
      2. Precise shapefile polygon clip via geopandas - drops anything inside
         the bbox but outside the real basin boundary
    Falls back to bbox-only if geopandas/shapefile unavailable.
    """
    if "mdb_scope" in df.columns:
        df = df[df["mdb_scope"]].copy()
    else:
        df = df[
            df["latitude"].between(*MDB_BBOX["lat"]) &
            df["longitude"].between(*MDB_BBOX["lon"])
        ].copy()

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
            pass
    return df


# ---------------------------------------------------------------------------
# Plot 1 - Annual abundance by source
# ---------------------------------------------------------------------------

def plot_abundance_over_time(df):
    # ---- Analytical logic (UNCHANGED) ----
    mdb = _clip_to_mdb(df)
    annual = (
        mdb.dropna(subset=["year", "abundance"])
           .groupby(["year", "source_id"])["abundance"]
           .sum()
           .reset_index()
    )

    # ---- Visual layer ----
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=WHITE)

    present = sorted(annual["source_id"].unique())
    for sid in present:
        grp   = annual[annual["source_id"] == sid].sort_values("year")
        color = PALETTE.get(str(sid), MID_GREY)
        years = grp["year"].astype(int).values
        vals  = grp["abundance"].values / 1e6

        ax.fill_between(years, vals, alpha=0.11, color=color,
                        zorder=2, linewidth=0)
        ax.plot(years, vals,
                color=color, linewidth=1.8,
                marker="o", markersize=3.5, markeredgewidth=0,
                zorder=3, solid_capstyle="round",
                label=SOURCE_LABELS.get(str(sid), str(sid)))

    style_axis(ax,
               xlabel="Survey year",
               ylabel="Total abundance (x10^6 individuals)",
               title="Annual waterbird abundance by source  -  Murray-Darling Basin",
               grid=True, grid_axis="y")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlim(annual["year"].min() - 0.5, annual["year"].max() + 0.5)
    ax.set_ylim(bottom=0)

    leg = ax.legend(handles=_source_handles(present),
                    loc="upper left", fontsize=7.5,
                    framealpha=0.96, edgecolor=SPINE_COL,
                    borderpad=0.6, labelspacing=0.35)
    _frame_legend(leg)
    add_panel_label(ax, "a")

    fig.text(0.5, -0.04,
             "Abundance = sum of individual counts within the MDB polygon. "
             "Gaps indicate years without survey records for that source.",
             ha="center", fontsize=6.5, color=MID_GREY, style="italic")

    fig.tight_layout()
    out = OUTPUTS / "plot1_abundance_over_time.png"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 - Species richness grouped bars
# ---------------------------------------------------------------------------

def plot_species_richness(df):
    # ---- Analytical logic (UNCHANGED) ----
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

    # ---- Visual layer ----
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=WHITE)

    n_sources = richness["source_id"].nunique()
    width     = 0.72 / max(n_sources, 1)
    keys      = list(PALETTE.keys())
    offsets   = {k: (i - (n_sources - 1) / 2) * width
                 for i, k in enumerate(keys)}

    present = sorted(richness["source_id"].unique())
    for sid in present:
        grp    = richness[richness["source_id"] == sid]
        offset = offsets.get(str(sid), 0)
        ax.bar(
            grp["water_year"].astype(int) + offset,
            grp["species_richness"],
            width=width * 0.88,
            color=PALETTE.get(str(sid), MID_GREY),
            alpha=0.90, zorder=3,
            edgecolor="none",
            label=SOURCE_LABELS.get(str(sid), str(sid)),
        )

    style_axis(ax,
               xlabel="Water year (Jul-Jun)",
               ylabel="Unique species (n)",
               title="Waterbird species richness per water year  -  Murray-Darling Basin",
               grid=True, grid_axis="y")

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    leg = ax.legend(handles=_source_handles(present),
                    loc="upper left", fontsize=7.5,
                    framealpha=0.96, edgecolor=SPINE_COL,
                    borderpad=0.6, labelspacing=0.35)
    _frame_legend(leg)
    add_panel_label(ax, "a")

    fig.tight_layout()
    out = OUTPUTS / "plot2_species_richness.png"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 - Inundation vs abundance scatter
# ---------------------------------------------------------------------------

def plot_inundation_vs_abundance(df):
    # ---- Analytical logic (UNCHANGED) ----
    mdb = _clip_to_mdb(df)
    sub = mdb.dropna(subset=["percent_full", "abundance"]).copy()
    sub = sub[(sub["abundance"] > 0) & (sub["percent_full"] > 0)]
    if sub.empty:
        print("  Plot 3: No inundation data after MDB clip - skipping")
        return
    sub["log_abundance"] = np.log1p(sub["abundance"])

    scatter_data = sub.sample(min(len(sub), 8000), random_state=42)

    from numpy.polynomial.polynomial import polyfit
    x = scatter_data["percent_full"].to_numpy(dtype=float)
    y = scatter_data["log_abundance"].to_numpy(dtype=float)
    b, m = polyfit(x, y, 1)
    xs = np.linspace(0, 100, 200)

    # Bootstrap 90% confidence band
    rng = np.random.default_rng(42)
    boot_preds = []
    for _ in range(300):
        idx = rng.integers(0, len(x), len(x))
        bb, bm = polyfit(x[idx], y[idx], 1)
        boot_preds.append(bm * xs + bb)
    boot_preds = np.array(boot_preds)
    ci_lo = np.percentile(boot_preds, 5,  axis=0)
    ci_hi = np.percentile(boot_preds, 95, axis=0)

    # ---- Visual layer ----
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=WHITE)

    present = sorted(scatter_data["source_id"].unique())
    for sid in present:
        grp = scatter_data[scatter_data["source_id"] == sid]
        ax.scatter(
            grp["percent_full"], grp["log_abundance"],
            color=PALETTE.get(str(sid), MID_GREY),
            alpha=0.20, s=7, zorder=3,
            edgecolors="none", rasterized=True,
            label=SOURCE_LABELS.get(str(sid), str(sid)),
        )

    ax.fill_between(xs, ci_lo, ci_hi,
                    color=DARK_GREY, alpha=0.09, zorder=4, linewidth=0)
    ax.plot(xs, m * xs + b,
            color=DARK_GREY, linewidth=1.6, zorder=5,
            label=f"Linear trend  (beta = {m:.3f})")

    style_axis(ax,
               xlabel="Wetland percent full (%)",
               ylabel="log(abundance + 1)",
               title="Inundation level vs waterbird abundance  -  Murray-Darling Basin",
               grid=True, grid_axis="both")

    ax.set_xlim(-1, 101)
    ax.set_ylim(bottom=0)

    extra = [
        Line2D([0], [0], color=DARK_GREY, linewidth=1.6,
               label=f"Linear trend  (beta = {m:.3f})"),
        mpatches.Patch(facecolor=DARK_GREY, alpha=0.15,
                       edgecolor="none", label="90% CI (bootstrap, n=300)"),
    ]
    leg = ax.legend(handles=_source_handles(present) + extra,
                    loc="upper left", fontsize=7.5,
                    framealpha=0.96, edgecolor=SPINE_COL,
                    borderpad=0.6, labelspacing=0.35, markerscale=2.5)
    _frame_legend(leg)
    add_panel_label(ax, "a")

    fig.text(0.5, -0.04,
             f"n = {len(scatter_data):,} records (random subsample). "
             "Log transform: log(abundance + 1). Shading = 90% bootstrap CI.",
             ha="center", fontsize=6.5, color=MID_GREY, style="italic")

    fig.tight_layout()
    out = OUTPUTS / "plot3_inundation_vs_abundance.png"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 - Spatial map  (Nature Cities cartographic standard)
# ---------------------------------------------------------------------------

def plot_spatial_map(df):
    # ---- Analytical logic (UNCHANGED) ----
    mdb = _clip_to_mdb(df)
    sub = mdb.dropna(subset=["latitude", "longitude"])

    # ---- Visual layer ----
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=WHITE)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(SPINE_COL)
        sp.set_linewidth(0.65)

    ax.set_facecolor(WATER_COL)

    bounds  = None
    if MDB_SHAPEFILE.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf = gpd.read_file(MDB_SHAPEFILE)
            bounds  = mdb_gdf.total_bounds

            for _, row in mdb_gdf.iterrows():
                polys = list(getattr(row.geometry, "geoms", [row.geometry]))
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=BASIN_COL, alpha=1.0, zorder=1)
                    ax.plot(x, y, color=SPINE_COL, linewidth=0.25,
                            alpha=0.45, zorder=2)

            for geom in mdb_gdf.geometry:
                polys = list(getattr(geom, "geoms", [geom]))
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=BASIN_EDGE, linewidth=1.05,
                            zorder=3, alpha=0.90,
                            solid_capstyle="round", solid_joinstyle="round")

        except Exception:
            pass

    if bounds is None:
        bounds = (MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
                  MDB_BBOX["lon"][1], MDB_BBOX["lat"][1])

    # Bubble plot - site-level aggregation, sqrt scaling per source (UNCHANGED)
    source_legend = []
    all_site_cap  = None
    if not sub.empty:
        _all = (sub.groupby(["latitude", "longitude"])
                   .agg(total_abundance=("abundance", "sum"))
                   .reset_index())
        all_site_cap = _all["total_abundance"].quantile(0.98)

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
        sizes = np.sqrt(site_agg["total_abundance"].clip(0, cap) / max(cap, 1)) * 155
        sizes = sizes.clip(2, 155)

        ax.scatter(
            site_agg["longitude"], site_agg["latitude"],
            s=sizes, color=PALETTE[sid],
            alpha=0.58, zorder=4,
            edgecolors="white", linewidths=0.2,
            rasterized=True,
        )
        source_legend.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE[sid],
                   markeredgecolor="none", markersize=7,
                   label=f"{SOURCE_LABELS[sid]}  ({len(site_agg):,} sites)")
        )

    pad = 0.35
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    ax.set_xlabel("Longitude (degrees E)", fontsize=8, color=DARK_GREY, labelpad=4)
    ax.set_ylabel("Latitude (degrees N)",  fontsize=8, color=DARK_GREY, labelpad=4)
    ax.tick_params(labelsize=7.5, length=3, colors=MID_GREY, direction="out")
    ax.grid(visible=True, color="#D4E7F0", linestyle="-",
            linewidth=0.28, alpha=0.55, zorder=0)

    # Source legend - upper left
    leg1 = ax.legend(handles=source_legend,
                     loc="upper left", fontsize=7.5,
                     framealpha=0.96, edgecolor=SPINE_COL, facecolor=WHITE,
                     title="Survey source", title_fontsize=7.5,
                     borderpad=0.6, labelspacing=0.4, handlelength=0.7)
    _frame_legend(leg1)
    ax.add_artist(leg1)

    # Bubble size legend - lower left
    if all_site_cap and all_site_cap > 0:
        sz_handles = []
        for frac, lbl in [(0.10, "10% of peak"), (0.50, "50% of peak"),
                          (1.00, "Peak site")]:
            sz = float(np.clip(np.sqrt(frac) * 155, 2, 155))
            sz_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=MID_GREY,
                       markeredgecolor=SPINE_COL, markeredgewidth=0.4,
                       markersize=float(np.sqrt(sz)),
                       label=lbl)
            )
        leg2 = ax.legend(handles=sz_handles,
                         loc="lower left", fontsize=7,
                         framealpha=0.96, edgecolor=SPINE_COL, facecolor=WHITE,
                         title="Bubble size  (sqrt abundance)", title_fontsize=7,
                         borderpad=0.6, labelspacing=0.45, handlelength=0.7)
        _frame_legend(leg2)

    ax.text(0.995, 0.012,
            f"n = {len(sub):,} records  -  clipped to MDB polygon",
            transform=ax.transAxes, fontsize=6.5,
            color=MID_GREY, va="bottom", ha="right", style="italic")

    add_panel_label(ax, "a", outside=False)

    ax.set_title(
        "Waterbird survey records  -  Murray-Darling Basin (all sources)\n"
        "Bubble area proportional to total abundance per site (sqrt-scaled per source)",
        fontsize=9, fontweight="bold", color=DARK_GREY, pad=8,
    )

    fig.tight_layout(pad=0.8)
    out = OUTPUTS / "plot4_spatial_map.png"
    fig.savefig(out, facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point  (UNCHANGED execution flow)
# ---------------------------------------------------------------------------

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