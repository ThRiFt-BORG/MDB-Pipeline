"""
run_pipeline.py - Orchestrator: runs all 6 stages in order.
Usage:  python run_pipeline.py
        python run_pipeline.py --skip-plots
"""
import sys
import os
import time
from pathlib import Path
from typing import cast
from pandas import DataFrame

# Force UTF-8 for stdout/stderr on Windows so Unicode in logs doesn't crash
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Make src importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config_00      import QC_LOG, OUTPUTS
from src.ingest_01      import load_all
from src.clean_02       import clean_all
from src.harmonise_03   import harmonise_all
from src.merge_04       import merge_all
from src.site_filter_05 import filter_all
from src.eda_plots_06   import plot_all

OUTPUTS.mkdir(parents=True, exist_ok=True)


def run(skip_plots: bool = False):
    all_logs = []
    t0 = time.time()

    print("\n" + "="*55)
    print("  MDB WATERBIRD PIPELINE -- X+GeoAI PROJECT")
    print("="*55)

    dfs                  = load_all()
    cleaned,    log2     = clean_all(dfs)
    harmonised, log3     = harmonise_all(cleaned)
    master_df,  log4     = merge_all(harmonised)
    master_df            = cast(DataFrame, master_df)
    subsets,    log5     = filter_all(master_df)

    all_logs += log2 + log3 + log4 + log5

    if not skip_plots:
        log6 = plot_all(master_df)
        all_logs += log6
    else:
        print("\n  Skipping plots (--skip-plots flag)")

    with open(QC_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(all_logs))

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  QC log          -> {QC_LOG}")
    print(f"  Master parquet  -> data/harmonised/mdb_waterbirds_master.parquet")
    print(f"  Site subsets    -> data/harmonised/")
    print(f"  Plots           -> outputs/")
    print("="*55 + "\n")
    return master_df, subsets


if __name__ == "__main__":
    skip = "--skip-plots" in sys.argv
    run(skip_plots=skip)
