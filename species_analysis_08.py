"""
species_analysis_08.py
======================
Publication-quality species analysis for the MDB Waterbird Pipeline.

Produces:
  outputs/fig1_top10_temporal.png          — temporal lines + per-species stats panel
  outputs/fig2_spatiotemporal_2016_17.png  — spatial maps: 2016 & 2017
  outputs/fig2_spatiotemporal_2018_19.png  — spatial maps: 2018 & 2019
  outputs/fig2_spatiotemporal_2020_21.png  — spatial maps: 2020 & 2021
  outputs/fig2_spatiotemporal_2022_23.png  — spatial maps: 2022 & 2023
  outputs/fig2_spatiotemporal_2024_25.png  — spatial maps: 2024 & 2025
  outputs/top10_species_summary.csv        — species metadata table

Usage:
  python species_analysis_08.py
  python species_analysis_08.py --parquet path/to/file.parquet
  python species_analysis_08.py --shapefile path/to/mdb_boundary.shp
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
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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
    ROOT / "data" / "spatial" / "mdb_boundary.shp",
    ROOT / "data" / "mdb_boundary.shp",
    ROOT / "src" / "mdb_boundary.shp",
]

# Pull the authoritative path from config (same file used by eda_plots_06.py)
try:
    from src.config_00 import MDB_SHAPEFILE as _CFG_SHP
    DEFAULT_SHAPEFILES.insert(0, _CFG_SHP)
except Exception:
    pass

# MDB bounding box fallback (lat/lon)
MDB_BBOX = {"lat": (-37.6, -23.0), "lon": (138.0, 153.1)}

# ---------------------------------------------------------------------------
# Style — matches 06_eda_plots.py Nature Cities grade
# ---------------------------------------------------------------------------
WHITE      = "#FFFFFF"
LIGHT_GREY = "#F4F4F2"   # figure face
DARK_GREY  = "#2B2B2B"
MID_GREY   = "#7A7A7A"
SPINE_COL  = "#BBBBBB"
LIGHT_LINE = "#E8E8E5"
WATER_COL  = "#EBF3F7"
BASIN_COL  = "#EAF0F5"
BASIN_EDGE = "#6FA8C4"

mpl.rcParams.update({
    "figure.facecolor":      LIGHT_GREY,
    "axes.facecolor":        WHITE,
    "axes.edgecolor":        SPINE_COL,
    "axes.linewidth":        0.65,
    "axes.labelcolor":       DARK_GREY,
    "axes.labelsize":        11,
    "axes.titlesize":        13,
    "axes.titlepad":         8,
    "axes.titleweight":      "bold",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             False,
    "xtick.color":           MID_GREY,
    "ytick.color":           MID_GREY,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "xtick.major.size":      3.0,
    "ytick.major.size":      3.0,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "text.color":            DARK_GREY,
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size":             11,
    "legend.framealpha":     0.96,
    "legend.edgecolor":      SPINE_COL,
    "legend.facecolor":      WHITE,
    "legend.fontsize":       10,
    "legend.title_fontsize": 10,
    "legend.borderpad":      0.55,
    "legend.labelspacing":   0.35,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     LIGHT_GREY,
    "savefig.pad_inches":    0.06,
})

# 10-colour palette — perceptually distinct, print-safe
PALETTE_10 = [
    "#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A",
    "#B15928", "#A6CEE3", "#B2DF8A", "#FB9A99", "#CAB2D6",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(parquet_path: Path) -> pd.DataFrame:
    print(f"  Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    df["scientific_name"] = df["scientific_name"].str.strip()
    df = df.dropna(subset=["scientific_name", "year"])
    df["abundance"] = df["abundance"].fillna(0)
    return df


def clip_to_mdb(df: pd.DataFrame, shapefile: Path | None) -> pd.DataFrame:
    """
    Two-step clip mirroring 06_eda_plots.py:
      1. mdb_scope flag (removes out-of-basin records flagged at Stage 04)
      2. Shapefile polygon clip via geopandas (precise boundary)
    Falls back to bbox-only when shapefile unavailable.
    """
    if "mdb_scope" in df.columns:
        df = df[df["mdb_scope"]].copy()
    else:
        df = df[
            df["latitude"].between(*MDB_BBOX["lat"]) &
            df["longitude"].between(*MDB_BBOX["lon"])
        ].copy()

    if shapefile and shapefile.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf   = gpd.read_file(shapefile)
            mdb_union = (mdb_gdf.union_all()
                         if hasattr(mdb_gdf, "union_all")
                         else mdb_gdf.unary_union)
            pts = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326",
            )
            df = pts[pts.within(mdb_union)].drop(columns="geometry").copy()
            print("    Shapefile clip applied.")
        except Exception as e:
            print(f"    Shapefile clip skipped ({e}); using bbox only.")
    return df


def draw_mdb_boundary(ax, shapefile: Path | None):
    """
    Draw MDB polygon on a map axis, mirroring 06_eda_plots.py plot_spatial_map.
    Returns (bounds, drew_polygon) — bounds is (minx, miny, maxx, maxy).
    """
    if shapefile and shapefile.exists():
        try:
            import geopandas as gpd
            os.environ.setdefault("PROJ_DATA", "")
            mdb_gdf = gpd.read_file(shapefile)
            bounds  = tuple(mdb_gdf.total_bounds)  # (minx, miny, maxx, maxy)

            for _, row in mdb_gdf.iterrows():
                polys = list(getattr(row.geometry, "geoms", [row.geometry]))
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=BASIN_COL, alpha=1.0, zorder=1)
                    ax.plot(x, y, color=SPINE_COL, linewidth=0.25, alpha=0.45, zorder=2)

            for geom in mdb_gdf.geometry:
                polys = list(getattr(geom, "geoms", [geom]))
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=BASIN_EDGE, linewidth=1.05, zorder=3,
                            alpha=0.90, solid_capstyle="round", solid_joinstyle="round")

            return bounds, True
        except Exception as e:
            print(f"    Boundary draw skipped ({e}); bbox fallback.")

    # bbox fallback
    bounds = (MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
              MDB_BBOX["lon"][1], MDB_BBOX["lat"][1])
    return bounds, False


def top10_species(df: pd.DataFrame) -> list[str]:
    ranked = (
        df.groupby("scientific_name")["abundance"]
        .sum()
        .sort_values(ascending=False)
    )
    exclude = ["sp.", "spp.", "wader", "duck", "bird", "unknown"]
    ranked = ranked[
        ~ranked.index.str.lower().str.contains("|".join(exclude), na=False)
    ]
    return ranked.head(10).index.tolist()


def common_name_map(df: pd.DataFrame, species: list[str]) -> dict[str, str]:
    out = {}
    for sp in species:
        sub = df[df["scientific_name"] == sp]["common_name"].dropna()
        out[sp] = sub.mode().iloc[0] if not sub.empty else sp
    return out


def add_panel_label(ax, label: str, x: float = -0.10, y: float = 1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=DARK_GREY,
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


# ---------------------------------------------------------------------------
# Figure 1 — Temporal line + per-species stats panel
# ---------------------------------------------------------------------------

def fig1_temporal(df: pd.DataFrame, species: list[str], cnames: dict):
    print("  Generating fig1_top10_temporal.png …")

    df16 = df[df["year"].between(2016, 2025) & df["scientific_name"].isin(species)].copy()
    annual = (
        df16.groupby(["year", "scientific_name"])["abundance"]
        .sum()
        .reset_index()
    )
    pivot = (
        annual.pivot(index="year", columns="scientific_name", values="abundance")
        .fillna(0)
    )
    col_order = [s for s in species if s in pivot.columns]
    pivot = pivot[col_order]
    years = pivot.index.astype(int).tolist()

    # --- compute per-species stats ---
    stat_rows = []
    for sp in col_order:
        series = pivot[sp].astype(float)
        mean_ = float(series.mean())
        max_  = float(series.max())
        cv_   = (float(series.std()) / mean_ * 100) if mean_ > 0 else 0
        peak_ = int(pivot.index[int(np.argmax(series.to_numpy(dtype=float)))])
        stat_rows.append({
            "sp": sp,
            "label": cnames.get(sp, sp),
            "mean": mean_,
            "max": max_,
            "cv": cv_,
            "peak": peak_,
        })

    # layout: tall line panel + compact stats panel
    fig, (ax_line, ax_stats) = plt.subplots(
        2, 1, figsize=(15, 12), facecolor=LIGHT_GREY,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.42}
    )

    # ── Line plot ──────────────────────────────────────────────────────────
    for i, sp in enumerate(col_order):
        vals  = pivot[sp].values
        color = PALETTE_10[i]
        ax_line.plot(years, vals, color=color, linewidth=1.8,
                     marker="o", markersize=3.5, markeredgewidth=0,
                     zorder=3, solid_capstyle="round",
                     label=cnames.get(sp, sp))
        ax_line.fill_between(years, vals, alpha=0.07, color=color, zorder=2)

    style_axis(ax_line,
               xlabel="Survey year",
               ylabel="Total abundance (individuals)",
               title="Annual abundance of top-10 waterbird species  ·  Murray–Darling Basin (2016–2025)")
    ax_line.set_xticks(years)
    ax_line.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f} M" if x >= 1e6
                      else f"{x/1e3:.0f} K" if x >= 1e3
                      else f"{x:.0f}"))
    ax_line.legend(loc="upper left", ncol=2, framealpha=0.96,
                   edgecolor=SPINE_COL, fontsize=10)
    add_panel_label(ax_line, "a")

    ax_line.text(
        0.5, 0.5,
        "Data: Flow-MER (S1/S2), MDBA AWS (S3), UNSW Aerial Survey (S4).  "
        "Abundance = sum of individual counts within MDB scope.",
        transform=ax_line.transAxes, ha="center", fontsize=9,
        color=MID_GREY, style="italic"
    )

    # ── Stats panel ────────────────────────────────────────────────────────
    # Horizontal grouped bar: mean (solid) + max (hatched) per species
    n   = len(col_order)
    idx = np.arange(n)

    means = np.array([r["mean"] for r in stat_rows])
    maxes = np.array([r["max"]  for r in stat_rows])
    cvs   = np.array([r["cv"]   for r in stat_rows])
    peaks = [r["peak"] for r in stat_rows]
    colors = [PALETTE_10[i] for i in range(n)]

    bar_h = 0.32
    ax_stats.barh(idx + bar_h / 2, means, height=bar_h,
                  color=colors, alpha=0.85, zorder=3, label="Mean (2016–2025)")
    ax_stats.barh(idx - bar_h / 2, maxes, height=bar_h,
                  color=colors, alpha=0.38, hatch="///",
                  edgecolor=colors, linewidth=0.4, zorder=3, label="Annual maximum")

    ax_stats.set_yticks(idx)
    ax_stats.set_yticklabels(
        [cnames.get(sp, sp) for sp in col_order],
        fontsize=10
    )
    ax_stats.invert_yaxis()
    ax_stats.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f} M" if x >= 1e6
                      else f"{x/1e3:.0f} K" if x >= 1e3
                      else f"{x:.0f}"))
    style_axis(ax_stats,
               xlabel="Abundance (individuals)",
               title="Per-species summary statistics  ·  2016–2025",
               grid_axis="x")
    ax_stats.spines["left"].set_visible(False)
    ax_stats.tick_params(left=False)

    # annotate C.V and peak year to the right of each max bar
    for i, r in enumerate(stat_rows):
        ax_stats.text(
            maxes[i] * 1.02, i - bar_h / 2,
            f"C.V: {r['cv']:.0f}%   Peak year: {r['peak']}",
            va="center", ha="left", fontsize=9, color=MID_GREY
        )

    ax_stats.text(
        0.98, 0.50,
        "C.V = coefficient of variation",
        transform=ax_stats.transAxes,
        ha="right", va="center",
        fontsize=9, color=MID_GREY, style="italic"
    )

    leg = ax_stats.legend(loc="lower right", fontsize=10,
                          framealpha=0.96, edgecolor=SPINE_COL)
    leg.get_frame().set_linewidth(0.5)
    add_panel_label(ax_stats, "b")

    out = OUTPUTS / "fig1_top10_temporal.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"    Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Spatiotemporal maps: 5 separate PNGs, 2 years per file
# ---------------------------------------------------------------------------

def _draw_year_panel(ax, df_year: pd.DataFrame, df_all: pd.DataFrame,
                     year: int, species: list[str], cnames: dict,
                     bounds: tuple, drew_polygon: bool):
    """Render one year's scatter onto ax, already has boundary drawn."""

    ax.set_facecolor(WATER_COL)

    pad = 0.35
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    # ghost — all MDB records for context
    ghost = df_all.dropna(subset=["latitude", "longitude"])
    ax.scatter(ghost["longitude"], ghost["latitude"],
               s=1, alpha=0.05, color="#AAAAAA", zorder=1, rasterized=True)

    # coloured species points
    for i, sp in enumerate(species):
        sub = df_year[df_year["scientific_name"] == sp].dropna(
            subset=["latitude", "longitude"])
        if sub.empty:
            continue
        sizes = np.clip(np.log1p(sub["abundance"].fillna(0)) * 10, 4, 140)
        ax.scatter(sub["longitude"], sub["latitude"],
                   s=sizes, color=PALETTE_10[i], alpha=0.82,
                   zorder=4 + i, rasterized=True,
                   edgecolors="none")

    ax.set_aspect("equal")
    ax.set_title(str(year), fontsize=13, fontweight="bold",
                 color=DARK_GREY, pad=5)
    ax.set_xlabel("Longitude (°E)", fontsize=10, color=DARK_GREY, labelpad=3)
    ax.set_ylabel("Latitude (°N)",  fontsize=10, color=DARK_GREY, labelpad=3)
    ax.tick_params(labelsize=10, color=SPINE_COL, direction="out")
    ax.grid(visible=True, color="#D4E7F0", linestyle="-",
            linewidth=0.28, alpha=0.55, zorder=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
        sp_.set_edgecolor(SPINE_COL)
        sp_.set_linewidth(0.65)

    n_rec = len(df_year)
    ax.text(0.99, 0.015, f"n = {n_rec:,} records",
            transform=ax.transAxes, fontsize=9, color=MID_GREY,
            ha="right", va="bottom", style="italic")


def fig2_spatiotemporal(df: pd.DataFrame, species: list[str],
                        cnames: dict, shapefile: Path | None):
    print("  Generating fig2 spatiotemporal panels …")

    df16 = df[df["year"].between(2016, 2025) &
              df["scientific_name"].isin(species)].copy()
    df16 = df16.dropna(subset=["latitude", "longitude"])

    # legend handles — built once, reused on every page
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE_10[i], markeredgecolor="none",
               markersize=6, label=cnames.get(sp, sp))
        for i, sp in enumerate(species)
    ]

    year_pairs = [(2016, 2017), (2018, 2019), (2020, 2021),
                  (2022, 2023), (2024, 2025)]

    outputs = []
    for yr_a, yr_b in year_pairs:
        fig, axes = plt.subplots(
            1, 2, figsize=(18, 10), facecolor=LIGHT_GREY,
            gridspec_kw={"wspace": 0.18}
        )

        for ax, yr in zip(axes, [yr_a, yr_b]):
            # draw boundary first so scatter sits on top
            bounds, drew = draw_mdb_boundary(ax, shapefile)
            df_yr = df16[df16["year"] == yr]
            _draw_year_panel(ax, df_yr, df16, yr, species, cnames,
                             bounds, drew)

        # panel labels
        add_panel_label(axes[0], "a")
        add_panel_label(axes[1], "b")

        fig.suptitle(
            f"Spatial distribution of top-10 waterbird species  ·  "
            f"Murray–Darling Basin  ·  {yr_a}–{yr_b}\n"
            "Dot size ∝ log(abundance + 1).  Each colour = one species.",
            fontsize=13, fontweight="bold", color=DARK_GREY, y=1.01
        )

        fig.legend(
            handles=legend_handles, loc="lower center", ncol=5,
            fontsize=10, framealpha=0.96,
            bbox_to_anchor=(0.5, -0.04),
            facecolor=LIGHT_GREY, edgecolor=SPINE_COL
        )

        fname = f"fig2_spatiotemporal_{yr_a}_{yr_b}.png"
        out   = OUTPUTS / fname
        fig.savefig(out, facecolor=LIGHT_GREY)
        plt.close(fig)
        print(f"    Saved → {out}")
        outputs.append(out)

    return outputs


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_species_table(df: pd.DataFrame, species: list[str], cnames: dict):
    rows = []
    for sp in species:
        sub = df[df["scientific_name"] == sp]
        rows.append({
            "scientific_name": sp,
            "common_name":     cnames.get(sp, ""),
            "total_abundance": int(sub["abundance"].sum()),
            "n_records":       len(sub),
            "year_min":        int(sub["year"].min()) if not sub["year"].isna().all() else "",
            "year_max":        int(sub["year"].max()) if not sub["year"].isna().all() else "",
            "n_sites":         sub["site_name"].nunique(),
            "sources":         ", ".join(sorted(sub["source_id"].dropna().unique())),
        })
    out = OUTPUTS / "top10_species_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"    Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MDB species analysis — fig1 + fig2 generator")
    parser.add_argument("--parquet",   type=str, default=None)
    parser.add_argument("--shapefile", type=str, default=None,
                        help="Path to MDB boundary .shp (optional but recommended)")
    args = parser.parse_args()

    # resolve parquet
    parquet_path = (Path(args.parquet) if args.parquet
                    else next((p for p in DEFAULT_PARQUETS if p.exists()), None))
    if parquet_path is None:
        sys.exit("ERROR: master parquet not found. Pass --parquet <path>.")

    # resolve shapefile
    shapefile = (Path(args.shapefile) if args.shapefile
                 else next((p for p in DEFAULT_SHAPEFILES if p.exists()), None))
    if shapefile and shapefile.exists():
        print(f"  Shapefile: {shapefile}")
    else:
        print("  Shapefile not found — bbox boundary fallback will be used.")
        shapefile = None

    print("\n" + "=" * 60)
    print("  MDB WATERBIRD PIPELINE — species_analysis_08.py")
    print("=" * 60)

    df      = load_data(parquet_path)
    df      = clip_to_mdb(df, shapefile)          # two-step MDB clip
    species = top10_species(df)
    cnames  = common_name_map(df, species)

    print(f"\n  Top-10 species:")
    for i, sp in enumerate(species, 1):
        print(f"    {i:>2}. {sp:42s} ({cnames.get(sp, '')})")

    print("\n  Generating figures …")
    fig1_temporal(df, species, cnames)
    fig2_spatiotemporal(df, species, cnames, shapefile)
    save_species_table(df, species, cnames)

    print("\n" + "=" * 60)
    print("  Outputs written to: outputs/")
    print("  fig1_top10_temporal.png")
    print("  fig2_spatiotemporal_2016_2017.png")
    print("  fig2_spatiotemporal_2018_2019.png")
    print("  fig2_spatiotemporal_2020_2021.png")
    print("  fig2_spatiotemporal_2022_2023.png")
    print("  fig2_spatiotemporal_2024_2025.png")
    print("  top10_species_summary.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()