"""
07_source_maps.py — Four separate spatial maps, one per source, filtered to 2010+.
Colour intensity encodes log(abundance+1) within each source.
Saved to outputs/plot_source_maps_2010plus.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Make `src` importable when script is run directly from any working dir ──
from pathlib import Path
_ROOT = Path(__file__).resolve().parent   # already at mdb_pipeline/ root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from src.config_00 import MASTER_PARQUET, OUTPUTS, MDB_BBOX

OUTPUTS.mkdir(parents=True, exist_ok=True)

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
    "S4": "UNSW Aerial 1983-2019",
}
SITE_BOXES = [
    ("Macquarie\nMarshes", (147.0, 148.5), (-31.8, -30.2)),
    ("Gwydir\nWetlands",   (149.0, 151.0), (-29.8, -28.2)),
]

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
    "grid.linewidth":   0.4,
    "font.family":      "DejaVu Sans",
})


def plot_source_maps(df: Optional[pd.DataFrame] = None, year_from: int = 2010):
    if df is None:
        if not MASTER_PARQUET.exists():
            raise FileNotFoundError(f"Run the pipeline first: {MASTER_PARQUET}")
        df = pd.read_parquet(MASTER_PARQUET)

    # Filter
    df = df[df["year"] >= year_from].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(-44, -10) & df["longitude"].between(112, 154)]

    sources = ["S1", "S2", "S3", "S4"]
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 14),
        facecolor="#0d1117",
        gridspec_kw={"hspace": 0.12, "wspace": 0.06}
    )
    axes = axes.flatten()

    for ax, sid in zip(axes, sources):
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")   # type: ignore[union-attr]

        sub = df[df["source_id"] == sid].copy()

        # ghost context — all sources faintly
        ax.scatter(df["longitude"], df["latitude"],
                   s=1, alpha=0.06, color="#8b949e", zorder=1, rasterized=True)

        if sub.empty:
            ax.text(0.5, 0.5, f"No records\nfrom {year_from}+",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#8b949e", fontsize=13)
        else:
            sub["log_abund"] = np.log1p(sub["abundance"].fillna(0))
            vmin = sub["log_abund"].quantile(0.05)
            vmax = sub["log_abund"].quantile(0.95)
            if vmax <= vmin:
                vmax = vmin + 1

            cmap = mcolors.LinearSegmentedColormap.from_list(
                f"cmap_{sid}", ["#161b22", PALETTE[sid], "#ffffff"], N=256
            )
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(sub["longitude"], sub["latitude"],
                            c=sub["log_abund"], cmap=cmap, norm=norm,
                            s=8, alpha=0.75, zorder=2, rasterized=True)

            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("log(abundance + 1)", color="#8b949e", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=7)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
            if hasattr(cbar.outline, "set_edgecolor"):
                cbar.outline.set_edgecolor("#30363d")  # type: ignore[union-attr]

        # study-site boxes
        for label, lon_rng, lat_rng in SITE_BOXES:
            ax.add_patch(mpatches.Rectangle(
                (lon_rng[0], lat_rng[0]),
                lon_rng[1] - lon_rng[0], lat_rng[1] - lat_rng[0],
                fill=False, edgecolor="#ff6b6b",
                linewidth=1.4, linestyle="--", zorder=3))
            ax.text(lon_rng[0] + 0.05, lat_rng[1] + 0.12, label,
                    color="#ff6b6b", fontsize=7, fontweight="bold",
                    va="bottom", zorder=4)

        # MDB boundary
        ax.add_patch(mpatches.Rectangle(
            (MDB_BBOX["lon"][0], MDB_BBOX["lat"][0]),
            MDB_BBOX["lon"][1] - MDB_BBOX["lon"][0],
            MDB_BBOX["lat"][1] - MDB_BBOX["lat"][0],
            fill=False, edgecolor="#444c56",
            linewidth=1, linestyle=":", zorder=2))

        ax.set_xlim(112, 154)
        ax.set_ylim(-44, -10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, zorder=0)

        n = len(sub)
        yr_range = ""
        if n > 0:
            yr_range = f"{int(sub['year'].min())}-{int(sub['year'].max())}"

        ax.set_title(
            f"{sid}  |  {SOURCE_LABELS[sid]}\n{n:,} records  {yr_range}",
            fontsize=11, fontweight="bold", color=PALETTE[sid], pad=8
        )

    fig.suptitle(
        f"MDB Waterbird Survey Records by Source Dataset  ({year_from} onwards)",
        fontsize=15, fontweight="bold", color="#e6edf3", y=0.995
    )

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#ff6b6b",
                       linestyle="--", linewidth=1.4, label="Primary study sites"),
        mpatches.Patch(facecolor="none", edgecolor="#444c56",
                       linestyle=":", linewidth=1, label="MDB boundary (approx)"),
        plt.scatter([], [], s=6, color="#8b949e", alpha=0.4, label="All-source context"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.2, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="#e6edf3",
               bbox_to_anchor=(0.5, -0.01))

    out = OUTPUTS / f"plot_source_maps_{year_from}plus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


if __name__ == "__main__":
    plot_source_maps()
