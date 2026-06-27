"""
01_ingest.py - Load and validate raw CSV files.
Returns a dict of DataFrames; logs shape, nulls, date range per source.
"""
import pandas as pd
from pathlib import Path
from src.config_00 import RAW_FILES, QC_LOG, OUTPUTS

OUTPUTS.mkdir(parents=True, exist_ok=True)

EXPECTED_COLS = {
    "S1": ["Program","SamplePoint","Latitude","Longitude","SampleDate",
           "speciesName","abundance","breedEvidence","propSurveyed(number)"],
    "S2": ["Program","SamplePoint","Latitude","Longitude","SampleDate",
           "speciesName","adultCountColonyTotal","nestCountColonyTotal"],
    "S3": ["SurveyProgram","Subwetland","LatitudeDec","LongitudeDec","Date",
           "SurvYear","ScientificName","CommonName","Fx_Group","SppCode",
           "SumOfCount","SumOfNests","SumOfBroods","PercentFilled"],
    "S4": ["Program","Name","Latitude","Longitude","Year",
           "Scientific_name","Common_name","Functional_group","Species_code",
           "Sum_of_count","Sum_of_nests","Sum_of_broods","Percent_full"],
}

def _load(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")

def _validate(df: pd.DataFrame, sid: str, log_lines: list) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLS[sid] if c not in df.columns]
    if missing:
        log_lines.append(f"  [!] {sid}: Missing expected columns: {missing}")
    log_lines.append(f"  [ok] {sid}: {len(df):,} rows x {len(df.columns)} cols")
    null_pct = (df.isnull().mean() * 100).round(1)
    high_null = null_pct[null_pct > 30].to_dict()
    if high_null:
        log_lines.append(f"       High-null cols (>30%): {high_null}")
    return df

def load_all(verbose: bool = True) -> dict:
    log_lines = ["=== STAGE 01: INGEST ==="]
    dfs = {}
    for sid, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{sid}: File not found at {path}\n"
                f"Copy your raw CSV to data/raw/ and rename as shown in config_00.py"
            )
        df = _load(path)
        df = _validate(df, sid, log_lines)
        dfs[sid] = df

    log_lines.append(f"\nTotal records across all sources: {sum(len(d) for d in dfs.values()):,}")

    # encoding='utf-8' required on Windows to handle any non-ASCII in log
    with open(QC_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    if verbose:
        print("\n".join(log_lines))
    return dfs

if __name__ == "__main__":
    load_all()
