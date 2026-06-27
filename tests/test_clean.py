"""
tests/test_clean.py — Unit tests for critical cleaning functions.
Run with: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.clean_02 import _spatial_qc, _normalise_species, _count_qc, _parse_dates


def test_spatial_qc_removes_invalid():
    df = pd.DataFrame({
        "Latitude":  [-31.5, 145.43, -200.0, -10.5],
        "Longitude": [148.0, 149.0,   148.0,  149.0],
    })
    log = []
    result = _spatial_qc(df, "Latitude", "Longitude", "TEST", log)
    assert len(result) == 2
    assert all(result["Latitude"].between(-44, -10))
    assert "invalid coords" in log[0]


def test_spatial_qc_removes_impossible_longitude():
    df = pd.DataFrame({
        "Latitude":  [-31.5, -149.37],
        "Longitude": [148.0,  148.0],
    })
    log = []
    result = _spatial_qc(df, "Latitude", "Longitude", "TEST", log)
    assert len(result) == 1


def test_normalise_species_strips_author_year():
    cases = {
        "Pelecanus conspicillatus Temminck, 1824": "Pelecanus conspicillatus",
        "Threskiornis spinicollis (Jameson, 1835)": "Threskiornis spinicollis",
        "Botaurus poiciloptilus": "Botaurus poiciloptilus",
        "Phalacrocorax (Phalacrocorax) sulcirostris (Brandt, 1837)": "Phalacrocorax sulcirostris",
    }
    series = pd.Series(list(cases.keys()))
    result = _normalise_species(series)
    for i, (original, expected) in enumerate(cases.items()):
        assert result[i] == expected, f"Failed for: {original!r} → got {result[i]!r}"


def test_normalise_species_handles_nan():
    series = pd.Series([np.nan, "Anas superciliosa"])
    result = _normalise_species(series)
    assert pd.isna(result[0])
    assert result[1] == "Anas superciliosa"


def test_count_qc_removes_negatives():
    df = pd.DataFrame({"count": [10, -5, 0, 200, np.nan]})
    log = []
    result = _count_qc(df, "count", log, "TEST")
    assert result.loc[1, "count"] is np.nan or pd.isna(result.loc[1, "count"])
    assert result.loc[2, "count"] == 0
    assert "negative" in log[0]


def test_parse_dates_handles_formats():
    series = pd.Series(["2014-09-29 14:00", "12/11/2007", "2019-01-01"])
    result = _parse_dates(series)
    assert result.notna().all()
    assert result.dt.year.tolist() == [2014, 2007, 2019]


def test_parse_dates_handles_bad_values():
    series = pd.Series(["not-a-date", "2020-05-01", None])
    result = _parse_dates(series)
    assert pd.isna(result[0])
    assert result[1].year == 2020
    assert pd.isna(result[2])
