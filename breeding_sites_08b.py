"""
breeding_sites_08b.py
=====================
Phase 1 — Breeding site characterisation for the MDB Waterbird Pipeline.

Reads:  data/harmonised/mdb_waterbirds_master.parquet
Filter: source_id == "S3"  |  year 2016–2021  |  breeding_evidence == True

Produces:
  outputs/fig_breeding_map.png        — Panel a: MDB map with bubble sizes
  outputs/fig_breeding_heatmap.png    — Panel b: Species × site heatmap
  outputs/fig_breeding_boxplot.png    — Panel c: Inundation boxplot
  outputs/breeding_sites_summary.csv  — Per-site × per-species summary

Usage:
  python breeding_sites_08b.py
  python breeding_sites_08b.py --parquet path/to/master.parquet
  python breeding_sites_08b.py --shapefile path/to/MDB_Basin.gpkg
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DEFAULT_PARQUETS = [
    ROOT / "data" / "harmonised" / "mdb_waterbirds_master.parquet",
    Path("/mnt/user-data/uploads/mdb_waterbirds_master.parquet"),
]

DEFAULT_SHAPEFILES = [
    Path(r"E:\mdb_waterbird_pipeline\mdb_pipeline\data\raw\MDB_Basin.gpkg"),
    ROOT / "data" / "raw" / "MDB_Basin.gpkg",
    ROOT / "data" / "spatial" / "mdb_boundary.shp",
    ROOT / "data" / "mdb_boundary.shp",
]

# Pull authoritative path from config if available
try:
    from src.config_00 import MDB_SHAPEFILE as _CFG_SHP
    DEFAULT_SHAPEFILES.insert(0, _CFG_SHP)
except Exception:
    pass

MDB_BBOX = {"lat": (-37.6, -23.0), "lon": (138.0, 153.1)}

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------
SOURCE_FILTER  = "S3"
YEAR_MIN, YEAR_MAX = 2016, 2021
N_SITES   = 5
N_SPECIES = 6

# ---------------------------------------------------------------------------
# Style — consistent with pipeline (Nature Cities grade)
# ---------------------------------------------------------------------------
WHITE      = "#FFFFFF"
LIGHT_GREY = "#F4F4F2"
DARK_GREY  = "#2B2B2B"
MID_GREY   = "#7A7A7A"
SPINE_COL  = "#BBBBBB"
LIGHT_LINE = "#E8E8E5"
WATER_COL  = "#EBF3F7"
BASIN_COL  = "#EAF0F5"
BASIN_EDGE = "#6FA8C4"

# Five distinct site colours — perceptually separated, print-safe
SITE_COLOURS = {
    "Lake Alexandrina":    "#1F78B4",
    "Lake Wyara":          "#33A02C",
    "Coorong South Lagoon":"#E31A1C",
    "Macquarie Marshes":   "#FF7F00",
    "Lake Brewster":       "#6A3D9A",
}

mpl.rcParams.update({
    "figure.facecolor":      LIGHT_GREY,
    "axes.facecolor":        WHITE,
    "axes.edgecolor":        SPINE_COL,
    "axes.linewidth":        0.65,
    "axes.labelcolor":       DARK_GREY,
    "axes.labelsize":        10,
    "axes.titlesize":        11,
    "axes.titlepad":         7,
    "axes.titleweight":      "bold",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "xtick.color":           MID_GREY,
    "ytick.color":           MID_GREY,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "xtick.major.size":      3.0,
    "ytick.major.width":     0.6,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "text.color":            DARK_GREY,
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size":             10,
    "legend.framealpha":     0.96,
    "legend.edgecolor":      SPINE_COL,
    "legend.facecolor":      WHITE,
    "legend.fontsize":       9,
    "legend.borderpad":      0.55,
    "legend.labelspacing":   0.4,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     LIGHT_GREY,
    "savefig.pad_inches":    0.08,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax, label, outside=True):
    x = -0.10 if outside else 0.01
    y = 1.05  if outside else 0.99
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=DARK_GREY,
            va="top", ha="left")


def style_axis(ax, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_xlabel(xlabel, color=DARK_GREY, labelpad=4)
    ax.set_ylabel(ylabel, color=DARK_GREY, labelpad=4)
    if title:
        ax.set_title(title, fontweight="bold", color=DARK_GREY, pad=6)
    ax.grid(visible=True, axis=grid_axis,
            color=LIGHT_LINE, linestyle="-", linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", color=SPINE_COL)
    ax.set_facecolor(WHITE)


def draw_mdb_boundary(ax, shapefile):
    """Draw MDB polygon — identical to eda_plots_06.py / species_analysis_08.py."""
    if shapefile and Path(shapefile).exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf = gpd.read_file(shapefile)
            bounds  = tuple(mdb_gdf.total_bounds)
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
                    ax.plot(x, y, color=BASIN_EDGE, linewidth=1.05, zorder=3,
                            alpha=0.90, solid_capstyle="round",
                            solid_joinstyle="round")
            return bounds
        except Exception as e:
            print(f"    Boundary draw skipped ({e}) — bbox fallback.")
    return (MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
            MDB_BBOX["lon"][1], MDB_BBOX["lat"][1])


# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------

def load_breeding_data(parquet_path):
    print(f"  Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)

    # Filter to S3, 2016-2021, breeding records only
    s3 = df[
        (df["source_id"] == SOURCE_FILTER) &
        (df["year"].between(YEAR_MIN, YEAR_MAX))
    ].copy()

    brd = s3[s3["breeding_evidence"] == True].copy()
    all_s3 = s3.copy()  # keep full S3 for inundation comparison

    print(f"    S3 {YEAR_MIN}-{YEAR_MAX} total records : {len(s3):,}")
    print(f"    Breeding records               : {len(brd):,}")
    return brd, all_s3


def select_top_sites_species(brd):
    """Determine top N_SITES and N_SPECIES from breeding records."""
    exclude_sites = ["Other subwetlands"]
    clean = brd[~brd["site_name"].isin(exclude_sites)].copy()

    # Top sites by total breeding abundance
    site_abund = (clean.groupby("site_name")["abundance"]
                  .sum().sort_values(ascending=False))
    top_sites = site_abund.head(N_SITES).index.tolist()

    # Top species by abundance across those sites
    sp_abund = (clean[clean["site_name"].isin(top_sites)]
                .groupby(["scientific_name", "common_name"])["abundance"]
                .sum().sort_values(ascending=False))
    top_species = sp_abund.head(N_SPECIES).index.get_level_values(0).tolist()
    cname_map   = {sp: cn for sp, cn in sp_abund.head(N_SPECIES).index}

    print(f"\n  Top {N_SITES} breeding sites:")
    for s in top_sites:
        print(f"    {s}: {site_abund[s]:,.0f} individuals")

    print(f"\n  Top {N_SPECIES} breeding species:")
    for sp in top_species:
        print(f"    {cname_map[sp]} ({sp})")

    return top_sites, top_species, cname_map


def build_site_summary(brd, top_sites, top_species, cname_map):
    """Per-site summary: coords, total abundance, nest counts, inundation."""
    sub = brd[brd["site_name"].isin(top_sites)].copy()

    summary = (sub.groupby("site_name")
               .agg(
                   latitude       =("latitude",       "mean"),
                   longitude      =("longitude",      "mean"),
                   total_abundance=("abundance",       "sum"),
                   total_nests    =("nest_count",      "sum"),
                   mean_pct_full  =("percent_full",    "mean"),
                   n_records      =("abundance",       "count"),
                   n_years        =("year",            "nunique"),
                   n_species      =("scientific_name", "nunique"),
               )
               .reset_index())

    # Species × site abundance matrix
    sp_site = (sub[sub["scientific_name"].isin(top_species)]
               .groupby(["site_name", "scientific_name"])["abundance"]
               .sum().unstack(fill_value=0))
    sp_site.columns = [cname_map.get(c, c) for c in sp_site.columns]
    sp_site = sp_site.reindex(top_sites, fill_value=0)

    return summary, sp_site


# ---------------------------------------------------------------------------
# Figure functions (standalone)
# ---------------------------------------------------------------------------

def make_map_figure(summary, brd_all, top_sites, shapefile):
    """Generate standalone map of breeding sites (Panel a)."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=LIGHT_GREY)
    ax.set_facecolor(WATER_COL)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(SPINE_COL)
        sp.set_linewidth(0.65)

    bounds = draw_mdb_boundary(ax, shapefile)
    pad = 0.4
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    # Ghost points
    ghost = brd_all.dropna(subset=["latitude", "longitude"])
    ax.scatter(ghost["longitude"], ghost["latitude"],
               s=3, color=MID_GREY, alpha=0.12, zorder=2, rasterized=True)

    # Top sites bubbles
    size_cap = summary["total_abundance"].max()
    for _, row in summary.iterrows():
        site = row["site_name"]
        color = SITE_COLOURS.get(site, MID_GREY)
        bsize = float(np.sqrt(float(row["total_abundance"]) / size_cap) * 320 + 20)
        ax.scatter(row["longitude"], row["latitude"],
                   s=bsize, color=color, alpha=0.82,
                   zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(site, xy=(row["longitude"], row["latitude"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color=DARK_GREY, fontweight="bold", zorder=6)

    ax.set_xlabel("Longitude (°E)", fontsize=10, color=DARK_GREY, labelpad=4)
    ax.set_ylabel("Latitude (°N)", fontsize=10, color=DARK_GREY, labelpad=4)
    ax.tick_params(labelsize=9, color=SPINE_COL, direction="out")
    ax.grid(visible=True, color="#D4E7F0", linestyle="-",
            linewidth=0.28, alpha=0.55, zorder=0)

    ax.set_title("Waterbird breeding sites · MDB · 2016–2021 (S3)\n"
                 "Bubble size ∝ total breeding abundance",
                 fontweight="bold", color=DARK_GREY, pad=7)

    # Legend
    leg_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=SITE_COLOURS.get(s, MID_GREY),
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=9,
               label=f"{s}  ({int(summary.loc[summary['site_name']==s,'total_abundance'].values[0]):,})")
        for s in top_sites
    ]
    leg = ax.legend(handles=leg_handles, loc="upper left", fontsize=8,
                    framealpha=0.96, edgecolor=SPINE_COL,
                    title="Breeding site (total individuals)", title_fontsize=8,
                    borderpad=0.6, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.5)

    ax.text(0.99, 0.015, f"n = {len(brd_all):,} breeding records · S3 source",
            transform=ax.transAxes, fontsize=7.5,
            color=MID_GREY, ha="right", va="bottom", style="italic")

    out = OUTPUTS / "fig_breeding_map.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"    Saved → {out}")


def make_heatmap_figure(sp_site, top_sites):
    """Generate standalone species × site heatmap (Panel b)."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=LIGHT_GREY)
    heat_vals = sp_site.values.astype(float)
    heat_log = np.where(heat_vals > 0, np.log10(heat_vals + 1), 0)

    im = ax.imshow(heat_log, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=heat_log.max())

    ax.set_yticks(range(len(sp_site.index)))
    ax.set_yticklabels(sp_site.index, fontsize=9)
    ax.set_xticks(range(len(sp_site.columns)))
    ax.set_xticklabels(sp_site.columns, rotation=38, ha="right", fontsize=8.5)

    # Annotate cells
    for r in range(heat_log.shape[0]):
        for c in range(heat_log.shape[1]):
            val = int(heat_vals[r, c])
            if val > 0:
                txt_col = "white" if heat_log[r, c] > heat_log.max() * 0.65 else DARK_GREY
                ax.text(c, r, f"{val:,}", ha="center", va="center",
                        fontsize=7.5, color=txt_col, fontweight="bold")
            else:
                ax.text(c, r, "—", ha="center", va="center",
                        fontsize=7.5, color=SPINE_COL)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(SPINE_COL)
        sp.set_linewidth(0.65)
    ax.tick_params(bottom=True, left=True, color=SPINE_COL, length=3)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("log₁₀(abundance + 1)", fontsize=8.5, color=DARK_GREY)
    cbar.ax.tick_params(labelsize=8)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor(SPINE_COL)

    ax.set_title("Breeding abundance by species × site",
                 fontweight="bold", color=DARK_GREY, pad=7)

    out = OUTPUTS / "fig_breeding_heatmap.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"    Saved → {out}")


def make_boxplot_figure(brd_all, all_s3):
    """Generate standalone inundation boxplot (Panel c)."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=LIGHT_GREY)
    brd_pf = brd_all[brd_all["breeding_evidence"] == True]["percent_full"].dropna()
    nbrd_pf = all_s3[all_s3["breeding_evidence"] == False]["percent_full"].dropna()

    bp = ax.boxplot([brd_pf.values, nbrd_pf.values],
                    positions=[1, 2], widths=0.45,
                    patch_artist=True, notch=False,
                    medianprops=dict(color=DARK_GREY, linewidth=2.0),
                    whiskerprops=dict(color=MID_GREY, linewidth=0.9),
                    capprops=dict(color=MID_GREY, linewidth=0.9),
                    flierprops=dict(marker=".", color=MID_GREY, alpha=0.3, markersize=3))
    bp["boxes"][0].set_facecolor("#FF7F00")
    bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor("#1F78B4")
    bp["boxes"][1].set_alpha(0.38)
    for box in bp["boxes"]:
        box.set_edgecolor(SPINE_COL)

    # Jitter
    for data, pos, color in [(brd_pf, 1, "#FF7F00"),
                              (nbrd_pf.sample(min(300, len(nbrd_pf)), random_state=42), 2, "#1F78B4")]:
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
        ax.scatter(np.full(len(data), pos) + jitter, data,
                   s=5, color=color, alpha=0.35, zorder=3, rasterized=True)

    # Median labels
    for data, pos in [(brd_pf, 1), (nbrd_pf, 2)]:
        med = data.median()
        ax.text(pos + 0.27, med, f"{med:.0f}%",
                va="center", fontsize=9, color=DARK_GREY, fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Breeding\nrecords", "Non-breeding\nrecords"], fontsize=9.5)
    ax.set_xlim(0.4, 2.9)
    ax.set_ylim(-5, 108)
    style_axis(ax, ylabel="Wetland percent full (%)",
               title="Inundation level at\nbreeding vs non-breeding records",
               grid_axis="y")

    # P-value
    try:
        from scipy.stats import mannwhitneyu
        result = mannwhitneyu(brd_pf, nbrd_pf, alternative="greater")
        pval = getattr(result, "pvalue", None)
        if pval is None:
            pval = result[1]
        pval = float(pval)  # type: ignore[arg-type]
        ptext = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
        ax.text(0.5, 0.93, ptext, transform=ax.transAxes, ha="center",
                fontsize=9, color=DARK_GREY, style="italic")
        ax.text(0.5, 0.87, "(Mann–Whitney, one-sided)",
                transform=ax.transAxes, ha="center", fontsize=7.5,
                color=MID_GREY, style="italic")
    except ImportError:
        pass

    out = OUTPUTS / "fig_breeding_boxplot.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"    Saved → {out}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_summary(brd, top_sites, top_species, cname_map, sp_site):
    rows = []
    for site in top_sites:
        sub = brd[brd["site_name"] == site]
        for sp in top_species:
            sp_sub = sub[sub["scientific_name"] == sp]
            rows.append({
                "site_name":          site,
                "scientific_name":    sp,
                "common_name":        cname_map.get(sp, sp),
                "total_abundance":    int(sp_sub["abundance"].sum()),
                "total_nests":        int(sp_sub["nest_count"].sum()),
                "n_breeding_records": len(sp_sub),
                "n_years":            sp_sub["year"].nunique(),
                "mean_percent_full":  round(sub["percent_full"].mean(), 1)
                                      if sub["percent_full"].notna().any() else None,
            })
    out_df = pd.DataFrame(rows)
    out_path = OUTPUTS / "breeding_sites_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"    Saved → {out_path}")
    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MDB breeding site characterisation — Stage 08b")
    parser.add_argument("--parquet",   type=str, default=None)
    parser.add_argument("--shapefile", type=str, default=None)
    args = parser.parse_args()

    parquet_path = (Path(args.parquet) if args.parquet
                    else next((p for p in DEFAULT_PARQUETS if p.exists()), None))
    if parquet_path is None:
        sys.exit("ERROR: master parquet not found. Pass --parquet <path>.")

    shapefile = (Path(args.shapefile) if args.shapefile
                 else next((p for p in DEFAULT_SHAPEFILES if p.exists()), None))
    if shapefile and shapefile.exists():
        print(f"  Shapefile: {shapefile}")
    else:
        print("  Shapefile not found — MDB bbox fallback.")
        shapefile = None

    print("\n" + "=" * 60)
    print("  MDB WATERBIRD PIPELINE — breeding_sites_08b.py")
    print("=" * 60)

    # Load
    brd, all_s3 = load_breeding_data(parquet_path)

    # Select top sites / species
    top_sites, top_species, cname_map = select_top_sites_species(brd)

    # Build matrices
    summary, sp_site = build_site_summary(brd, top_sites, top_species, cname_map)

    # Generate three separate figures
    brd_all = brd.copy()
    print("\n  Generating figures …")
    make_map_figure(summary, brd_all, top_sites, shapefile)
    make_heatmap_figure(sp_site, top_sites)
    make_boxplot_figure(brd_all, all_s3)

    # Export CSV
    print("\n  Exporting summary CSV …")
    export_summary(brd, top_sites, top_species, cname_map, sp_site)

    print("\n" + "=" * 60)
    print("  Outputs written to: outputs/")
    print("  fig_breeding_map.png")
    print("  fig_breeding_heatmap.png")
    print("  fig_breeding_boxplot.png")
    print("  breeding_sites_summary.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()