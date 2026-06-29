"""
habitat_cluster_09.py
=====================
Phase 2 — Spatial clustering of waterbird survey records to identify
candidate habitat zones across the Murray–Darling Basin.

Pipeline:
  1. Load & clip to MDB scope (mdb_scope flag + optional shapefile polygon)
  2. Aggregate to unique coordinate pairs with multi-source evidence weight
  3. DBSCAN clustering in haversine space (epsilon = 5 km default)
  4. Compute per-cluster summary statistics
  5. Flag clusters with multi-source overlap (strongest habitat candidates)
  6. Export:
       outputs/clusters_points.geojson   — clustered survey points
       outputs/clusters_hulls.geojson    — convex hull per cluster (upload to GEE)
       outputs/clusters_summary.csv      — per-cluster stats table
       outputs/fig_clusters_map.png      — publication-quality cluster map

Usage:
  python habitat_cluster_09.py
  python habitat_cluster_09.py --parquet path/to/file.parquet
  python habitat_cluster_09.py --shapefile path/to/mdb.shp
  python habitat_cluster_09.py --epsilon 3.0 --min_samples 5
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

from sklearn.cluster import DBSCAN

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
    ROOT / "data" / "raw" / "MDB_Basin.gpkg",
    ROOT / "data" / "spatial" / "mdb_boundary.shp",
    ROOT / "data" / "mdb_boundary.shp",
    ROOT / "src"  / "mdb_boundary.shp",
]
try:
    from src.config_00 import MDB_SHAPEFILE as _CFG_SHP
    DEFAULT_SHAPEFILES.insert(0, _CFG_SHP)
except Exception:
    pass

# MDB bounding box fallback
MDB_BBOX = {"lat": (-37.6, -23.0), "lon": (138.0, 153.1)}

# DBSCAN defaults — ecologically motivated:
#   epsilon = 5 km  → groups wetland complexes without over-splitting isolated sites
#   min_samples = 3 → at least 3 unique coordinate pairs to form a cluster
EPSILON_KM  = 5.0
MIN_SAMPLES = 3

# ---------------------------------------------------------------------------
# Style — matches species_analysis_08 / 06_eda_plots.py
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

# Cluster tier colours
COL_MULTI  = "#E31A1C"   # multi-source clusters — strongest candidates
COL_SINGLE = "#1F78B4"   # single-source clusters
COL_NOISE  = "#BBBBBB"   # DBSCAN noise points

mpl.rcParams.update({
    "figure.facecolor":      LIGHT_GREY,
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
    "legend.borderpad":      0.55,
    "legend.labelspacing":   0.35,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     LIGHT_GREY,
    "savefig.pad_inches":    0.06,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_panel_label(ax, label, x=-0.10, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=DARK_GREY,
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
# Step 1 — Load & clip
# ---------------------------------------------------------------------------

def load_and_clip(parquet_path: Path, shapefile: Path | None) -> pd.DataFrame:
    print(f"  Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    df["abundance"] = df["abundance"].fillna(0)

    # mdb_scope flag
    if "mdb_scope" in df.columns:
        df = df[df["mdb_scope"]].copy()
    else:
        df = df[
            df["latitude"].between(*MDB_BBOX["lat"]) &
            df["longitude"].between(*MDB_BBOX["lon"])
        ].copy()

    df = df.dropna(subset=["latitude", "longitude"])

    # coordinate sanity
    df = df[
        df["latitude"].between(*MDB_BBOX["lat"]) &
        df["longitude"].between(*MDB_BBOX["lon"])
    ]

    # shapefile polygon clip
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
            print(f"    Shapefile clip applied → {len(df):,} records retained.")
        except Exception as e:
            print(f"    Shapefile clip skipped ({e}); bbox only.")
    else:
        print(f"    No shapefile — bbox clip only → {len(df):,} records.")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Aggregate to unique coordinate pairs
# ---------------------------------------------------------------------------

def aggregate_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual records to unique (lat, lon) pairs.
    Preserve:
      - n_records       : total survey records at this location
      - n_sources       : number of distinct source datasets (S1–S4)
      - sources         : comma-joined source list
      - total_abundance : sum of abundance across all records
      - n_species       : unique species observed
      - n_years         : number of distinct survey years
      - has_breeding    : any breeding evidence recorded
      - multi_source    : bool — observed by 2+ independent sources
    """
    agg = (
        df.groupby(["latitude", "longitude"])
        .agg(
            n_records       =("abundance",          "count"),
            total_abundance =("abundance",          "sum"),
            n_sources       =("source_id",          "nunique"),
            sources         =("source_id",          lambda x: ",".join(sorted(x.dropna().unique()))),
            n_species       =("scientific_name",    "nunique"),
            n_years         =("year",               "nunique"),
            has_breeding    =("breeding_evidence",  lambda x: bool(x.any())),
        )
        .reset_index()
    )
    agg["multi_source"] = agg["n_sources"] >= 2
    print(f"  Unique coordinate pairs : {len(agg):,}")
    print(f"  Multi-source pairs      : {agg['multi_source'].sum():,}")
    return agg


# ---------------------------------------------------------------------------
# Step 3 — DBSCAN clustering
# ---------------------------------------------------------------------------

def run_dbscan(coords_df: pd.DataFrame,
               epsilon_km: float,
               min_samples: int) -> pd.DataFrame:
    """
    DBSCAN on haversine distance.
    epsilon_km / 6371 converts km → radians for sklearn's haversine metric.
    """
    coords_rad = np.radians(coords_df[["latitude", "longitude"]].values)
    eps_rad    = epsilon_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree",
                metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)

    coords_df = coords_df.copy()
    coords_df["cluster_id"] = labels  # -1 = noise

    n_clusters = labels.max() + 1 if (labels >= 0).any() else 0
    n_noise    = (labels == -1).sum()
    print(f"  DBSCAN (ε={epsilon_km} km, min_samples={min_samples})")
    print(f"    Clusters formed : {n_clusters}")
    print(f"    Noise points    : {n_noise:,}  ({n_noise/len(labels)*100:.1f}%)")
    return coords_df


# ---------------------------------------------------------------------------
# Step 4 — Per-cluster summary
# ---------------------------------------------------------------------------

def summarise_clusters(pts: pd.DataFrame) -> pd.DataFrame:
    """
    One row per cluster_id (excluding noise = -1).
    Columns ready for GEE upload and publication table.
    """
    clustered = pts[pts["cluster_id"] >= 0].copy()

    summary = (
        clustered.groupby("cluster_id")
        .agg(
            centroid_lat    =("latitude",        "mean"),
            centroid_lon    =("longitude",        "mean"),
            n_points        =("latitude",         "count"),
            n_sources       =("n_sources",        "max"),
            sources         =("sources",          lambda x: ",".join(
                                sorted(set(",".join(x).split(","))))),
            multi_source    =("multi_source",     "any"),
            total_abundance =("total_abundance",  "sum"),
            n_species       =("n_species",        "max"),
            n_years         =("n_years",          "max"),
            has_breeding    =("has_breeding",     "any"),
            lat_min         =("latitude",         "min"),
            lat_max         =("latitude",         "max"),
            lon_min         =("longitude",        "min"),
            lon_max         =("longitude",        "max"),
        )
        .reset_index()
    )

    # approximate cluster extent in km (bounding box diagonal)
    summary["extent_km"] = np.sqrt(
        ((summary["lat_max"] - summary["lat_min"]) * 111) ** 2 +
        ((summary["lon_max"] - summary["lon_min"]) *
         111 * np.cos(np.radians(summary["centroid_lat"]))) ** 2
    ).round(2)

    # habitat tier
    summary["habitat_tier"] = np.where(
        summary["multi_source"], "Priority",    # 2+ sources
        np.where(summary["has_breeding"], "Breeding confirmed", "Candidate")
    )

    summary = summary.sort_values("total_abundance", ascending=False).reset_index(drop=True)

    print(f"\n  Cluster summary:")
    print(f"    Total clusters      : {len(summary)}")
    print(f"    Priority (2+ src)   : {summary['multi_source'].sum()}")
    print(f"    Breeding confirmed  : {summary['has_breeding'].sum()}")
    print(f"    Median extent       : {summary['extent_km'].median():.1f} km")
    return summary


# ---------------------------------------------------------------------------
# Step 5 — Export GeoJSON (points + convex hulls)
# ---------------------------------------------------------------------------

def export_geojson(pts: pd.DataFrame, summary: pd.DataFrame):
    try:
        import geopandas as gpd
        from shapely.geometry import Point, MultiPoint
        os.environ.setdefault("PROJ_DATA", "")

        # --- clustered points ---
        pts_out = pts[pts["cluster_id"] >= 0].copy()
        gdf_pts = gpd.GeoDataFrame(
            pts_out,
            geometry=gpd.points_from_xy(pts_out["longitude"], pts_out["latitude"]),
            crs="EPSG:4326",
        )
        pts_path = OUTPUTS / "clusters_points.geojson"
        gdf_pts.to_file(pts_path, driver="GeoJSON")
        print(f"    Saved → {pts_path}")

        # --- convex hulls per cluster ---
        hulls = []
        for _, row in summary.iterrows():
            cid   = row["cluster_id"]
            cpts  = pts[pts["cluster_id"] == cid][["longitude", "latitude"]].values
            if len(cpts) >= 3:
                geom = MultiPoint(cpts).convex_hull
            elif len(cpts) == 2:
                geom = MultiPoint(cpts).convex_hull   # LineString — GEE handles it
            else:
                geom = Point(cpts[0])
            hulls.append({**row.to_dict(), "geometry": geom})

        gdf_hulls = gpd.GeoDataFrame(hulls, crs="EPSG:4326")
        hulls_path = OUTPUTS / "clusters_hulls.geojson"
        gdf_hulls.to_file(hulls_path, driver="GeoJSON")
        print(f"    Saved → {hulls_path}")

    except ImportError:
        print("    geopandas not available — GeoJSON export skipped.")
        pts_path   = None
        hulls_path = None

    return pts_path, hulls_path


# ---------------------------------------------------------------------------
# Step 6 — Publication-quality cluster map
# ---------------------------------------------------------------------------

def draw_mdb_boundary(ax, shapefile: Path | None):
    if shapefile and shapefile.exists():
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
                    ax.plot(x, y, color=SPINE_COL, linewidth=0.25, alpha=0.45, zorder=2)

            for geom in mdb_gdf.geometry:
                polys = list(getattr(geom, "geoms", [geom]))
                for poly in polys:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=BASIN_EDGE, linewidth=1.05, zorder=3,
                            alpha=0.90, solid_capstyle="round", solid_joinstyle="round")
            return bounds
        except Exception as e:
            print(f"    Boundary draw skipped ({e}).")

    return (MDB_BBOX["lon"][0], MDB_BBOX["lat"][0],
            MDB_BBOX["lon"][1], MDB_BBOX["lat"][1])


def plot_cluster_map(pts: pd.DataFrame, summary: pd.DataFrame,
                     shapefile: Path | None, epsilon_km: float):
    print("  Generating fig_clusters_map.png …")

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(18, 9), facecolor=LIGHT_GREY,
        gridspec_kw={"wspace": 0.28, "width_ratios": [2, 1]}
    )

    # ── Map ──────────────────────────────────────────────────────────────
    ax_map.set_facecolor(WATER_COL)
    bounds = draw_mdb_boundary(ax_map, shapefile)

    pad = 0.35
    ax_map.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax_map.set_ylim(bounds[1] - pad, bounds[3] + pad)

    # noise points
    noise = pts[pts["cluster_id"] == -1]
    ax_map.scatter(noise["longitude"], noise["latitude"],
                   s=2, color=COL_NOISE, alpha=0.18, zorder=2,
                   rasterized=True, label=f"Noise / isolated  (n={len(noise):,})")

    # single-source clusters
    single = pts[(pts["cluster_id"] >= 0) & (~pts["multi_source"])]
    ax_map.scatter(single["longitude"], single["latitude"],
                   s=6, color=COL_SINGLE, alpha=0.45, zorder=3,
                   rasterized=True, label="Candidate cluster  (1 source)")

    # multi-source clusters — most prominent
    multi = pts[(pts["cluster_id"] >= 0) & (pts["multi_source"])]
    ax_map.scatter(multi["longitude"], multi["latitude"],
                   s=14, color=COL_MULTI, alpha=0.75, zorder=4,
                   rasterized=True, edgecolors="white", linewidths=0.3,
                   label="Priority cluster  (2+ sources)")

    # mark breeding confirmed cluster centroids with a star
    breed_cents = summary[summary["has_breeding"]]
    if not breed_cents.empty:
        ax_map.scatter(breed_cents["centroid_lon"], breed_cents["centroid_lat"],
                       s=60, marker="*", color="#FF7F00", zorder=5,
                       edgecolors="white", linewidths=0.4,
                       label="Breeding confirmed centroid")

    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Longitude (°E)", fontsize=8, color=DARK_GREY, labelpad=4)
    ax_map.set_ylabel("Latitude (°N)",  fontsize=8, color=DARK_GREY, labelpad=4)
    ax_map.tick_params(labelsize=7.5, color=SPINE_COL, direction="out")
    ax_map.grid(visible=True, color="#D4E7F0", linestyle="-",
                linewidth=0.28, alpha=0.55, zorder=0)
    for sp in ax_map.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(SPINE_COL)
        sp.set_linewidth(0.65)

    n_priority = int(summary["multi_source"].sum())
    n_total    = len(summary)
    ax_map.set_title(
        f"Candidate waterbird habitat clusters  ·  Murray–Darling Basin\n"
        f"DBSCAN  ε = {epsilon_km} km  ·  {n_total} clusters  "
        f"({n_priority} priority / multi-source)",
        fontweight="bold", color=DARK_GREY, pad=7
    )
    ax_map.text(0.99, 0.012,
                f"n = {len(pts):,} unique survey locations",
                transform=ax_map.transAxes, fontsize=6, color=MID_GREY,
                ha="right", va="bottom", style="italic")

    leg = ax_map.legend(loc="upper left", fontsize=7.5, framealpha=0.96,
                        edgecolor=SPINE_COL, markerscale=1.8)
    leg.get_frame().set_linewidth(0.5)
    add_panel_label(ax_map, "a", x=-0.04)

    # ── Bar chart: cluster size distribution ─────────────────────────────
    bins  = [0, 5, 10, 25, 50, 100, 500, summary["n_points"].max() + 1]
    labels_b = ["1–5", "6–10", "11–25", "26–50", "51–100", "101–500", "500+"]
    counts = pd.cut(summary["n_points"], bins=bins, labels=labels_b,
                    right=True).value_counts().sort_index()

    bar_colors = [COL_SINGLE] * len(labels_b)
    ax_bar.bar(range(len(labels_b)), counts.values,
               color=bar_colors, alpha=0.80, zorder=3, width=0.65,
               edgecolor="white", linewidth=0.4)

    style_axis(ax_bar,
               xlabel="Points per cluster (n)",
               ylabel="Number of clusters",
               title="Cluster size distribution",
               grid_axis="y")
    ax_bar.set_xticks(range(len(labels_b)))
    ax_bar.set_xticklabels(labels_b, rotation=35, ha="right", fontsize=7)
    add_panel_label(ax_bar, "b")

    # annotate median
    median_pts = summary["n_points"].median()
    ax_bar.axvline(
        pd.cut([median_pts], bins=bins, labels=False)[0],
        color=DARK_GREY, linewidth=1.2, linestyle="--",
        label=f"Median = {median_pts:.0f} pts"
    )
    ax_bar.legend(fontsize=7.5, framealpha=0.96, edgecolor=SPINE_COL)

    fig.suptitle(
        "Phase 2 — Habitat clustering  ·  MDB Waterbird Pipeline",
        fontsize=12, fontweight="bold", color=DARK_GREY, y=1.01
    )

    out = OUTPUTS / "fig_clusters_map.png"
    fig.savefig(out, facecolor=LIGHT_GREY)
    plt.close(fig)
    print(f"    Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MDB habitat clustering — Phase 2")
    parser.add_argument("--parquet",     type=str,   default=None)
    parser.add_argument("--shapefile",   type=str,   default=None)
    parser.add_argument("--epsilon",     type=float, default=EPSILON_KM,
                        help=f"DBSCAN epsilon in km (default {EPSILON_KM})")
    parser.add_argument("--min_samples", type=int,   default=MIN_SAMPLES,
                        help=f"DBSCAN min_samples (default {MIN_SAMPLES})")
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
        print("  Shapefile not found — bbox fallback.")
        shapefile = None

    print("\n" + "=" * 60)
    print("  MDB WATERBIRD PIPELINE — habitat_cluster_09.py")
    print("=" * 60)

    # Step 1 — load & clip
    df = load_and_clip(parquet_path, shapefile)

    # Step 2 — aggregate to unique coordinate pairs
    print("\n  Aggregating to unique coordinate pairs …")
    coords_df = aggregate_coords(df)

    # Step 3 — DBSCAN
    print("\n  Running DBSCAN …")
    pts = run_dbscan(coords_df, args.epsilon, args.min_samples)

    # Step 4 — per-cluster summary
    print("\n  Summarising clusters …")
    summary = summarise_clusters(pts)

    # Step 5 — export
    print("\n  Exporting GeoJSON …")
    export_geojson(pts, summary)

    csv_out = OUTPUTS / "clusters_summary.csv"
    summary.to_csv(csv_out, index=False)
    print(f"    Saved → {csv_out}")

    # Step 6 — map
    print("\n  Plotting …")
    plot_cluster_map(pts, summary, shapefile, args.epsilon)

    print("\n" + "=" * 60)
    print("  Outputs written to: outputs/")
    print("  clusters_points.geojson   ← upload to GEE")
    print("  clusters_hulls.geojson    ← upload to GEE (primary)")
    print("  clusters_summary.csv")
    print("  fig_clusters_map.png")
    print("=" * 60)
    print("\n  Next step → upload clusters_hulls.geojson + MDB boundary to GEE")
    print("              then run habitat_gee_10.js for urban mask + Sentinel-2 indices.\n")


if __name__ == "__main__":
    main()