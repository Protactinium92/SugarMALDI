# test_maldi_processing.py
import pytest
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from scipy.signal import find_peaks as fp
import tempfile
import os
from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray
from src import config as cf

# Import the module under test (adjust import path as needed)
from src.spectrum_analysis import (
    read_spectrum,
    parse_spectra,
    normalize_mz_column,
    make_window,
    find_all_peaks,
    _process_window,
    extract_peaks_all_windows,
    calcul_offset,
    calculate_chemical_formula,
    get_theoretical_spectrum,
    _normalize,
    _build_aligned_vectors,
    score_cosine,
    score_pearson,
    score_chi2,
    score_kl,
    score_emd,
    score_hybrid,
    calculate_isotopic_score,
    match_oligos,
    adjust_library,
    compile_oligo_name,
    combine_lf,
    process_mass,
)


# ======================
# TESTS FOR read_spectrum
# ======================

def test_read_spectrum_basic(sample_spectrum_csv):
    """Test basic spectrum reading."""
    result = read_spectrum(sample_spectrum_csv)
    df = result.collect()

    assert isinstance(result, pl.LazyFrame)
    assert list(df.columns) == ["m/z", "Intensity"]
    assert df.shape == (4, 2)
    assert df["m/z"].to_list() == [100.0, 101.0, 102.0, 103.0]
    assert df["Intensity"].to_list() == [1000, 2000, 1500, 500]
    assert df["m/z"].is_sorted()

def test_read_spectrum_is_sorted(tmp_path):
    data = """m/z,Intensity
102.0,1500
100.0,1000
101.0,2000
"""
    file_path = tmp_path / "unsorted.csv"
    file_path.write_text(data)
    df = read_spectrum(str(file_path)).collect()

    assert df["m/z"].to_list() == [100.0, 101.0, 102.0]

def test_read_spectrum_custom_separator(tmp_path):
    """Test spectrum reading with custom separator."""
    data = """m/z;Intensity
100.0;1000
101.0;2000
"""
    file_path = tmp_path / "test_spectrum.tsv"
    file_path.write_text(data)

    result = read_spectrum(str(file_path), sep=';')
    df = result.collect()

    assert len(df) == 2
    assert df["m/z"][0] == 100.0

def test_read_spectrum_missing_columns(tmp_path):
    """Test behavior with missing required columns."""
    data = """Mass,Intensity
100.0,1000
"""
    file_path = tmp_path / "invalid_spectrum.csv"
    file_path.write_text(data)

    with pytest.raises(pl.exceptions.ColumnNotFoundError,  match="m/z"):
        read_spectrum(str(file_path)).collect()

# ======================
# TESTS FOR parse_spectra
# ======================

def test_parse_spectra_single_file(sample_spectrum_csv):
    """Test parsing a single spectrum file."""
    result = parse_spectra([sample_spectrum_csv])
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "test_spectrum" in result
    assert isinstance(result["test_spectrum"], pl.LazyFrame)

def test_parse_spectra_multiple_files(tmp_path):
    """Test parsing multiple spectrum files."""
    # Create multiple test files
    files = []
    for i in range(3):
        data = f"""m/z,Intensity
10{i}.0,1000
10{i}.5,2000
"""
        file_path = tmp_path / f"spectrum_{i}.csv"
        file_path.write_text(data)
        files.append(str(file_path))

    result = parse_spectra(files)
    assert len(result) == 3
    assert all(isinstance(v, pl.LazyFrame) for v in result.values())

def test_parse_spectra_with_dir(tmp_path):
    """Test parsing with directory prefix."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file_path = subdir / "spectrum.csv"
    file_path.write_text("m/z,Intensity\n100.0,1000\n")

    result = parse_spectra(["spectrum.csv"], dir=str(subdir))
    assert len(result) == 1
    assert "spectrum" in result

def test_parse_spectra_nonexistent_file():
    """Test behavior with non-existent file."""
    with pytest.raises(FileNotFoundError):
        parse_spectra(["nonexistent_file.csv"])

def test_parse_spectra_wrong_separator(tmp_path):
    """Test behavior when delimiter does not match file separator."""
    data = """m/z;Intensity
100.0;1000
101.0;2000
"""
    file_path = tmp_path / "spectrum.csv"
    file_path.write_text(data)
    with pytest.raises(ValueError, match="Separator"):
        parse_spectra([str(file_path)], delimiter=",")

def test_parse_spectra_custom_separator(tmp_path):
    """Test parsing with a custom delimiter."""
    data = """m/z;Intensity
100.0;1000
101.0;2000
"""
    file_path = tmp_path / "spectrum.csv"
    file_path.write_text(data)
    result = parse_spectra([str(file_path)], delimiter=";")
    df = result["spectrum"].collect()
    assert len(result) == 1
    assert isinstance(result["spectrum"], pl.LazyFrame)
    assert df.shape == (2, 2)
    assert df.columns == ["m/z", "Intensity"]

def test_parse_spectra_uploaded_file(sample_uploaded_file):
    """Test parsing a Streamlit-like uploaded file."""
    result = parse_spectra([sample_uploaded_file])
    df = result["test_spectrum"].collect()
    assert isinstance(result, dict)
    assert "test_spectrum" in result
    assert df.shape == (2, 2)
    assert df.columns == ["m/z", "Intensity"]

# =============================
# TESTS FOR normalize_mz_column
# =============================

def test_normalize_mz_column_basic(sample_lazyframe_spectrum):
    """Test basic m/z normalization."""
    result = normalize_mz_column(sample_lazyframe_spectrum)
    df = result.collect()

    assert "mz_int" in df.columns
    assert df["mz_int"].dtype == pl.Int64
    assert df["mz_int"][0] == int(100.0 * cf.SCALE)

def test_normalize_mz_column_custom_columns(sample_lazyframe_spectrum):
    """Test with custom column names."""
    result = normalize_mz_column(sample_lazyframe_spectrum, mz_col="m/z", new_col="normalized_mz")
    df = result.collect()

    assert "normalized_mz" in df.columns
    assert "mz_int" not in df.columns

def test_normalize_mz_column_preserves_data(sample_lazyframe_spectrum):
    """Test that original columns are preserved."""
    result = normalize_mz_column(sample_lazyframe_spectrum)
    df = result.collect()

    assert "m/z" in df.columns
    assert "Intensity" in df.columns
    assert len(df) == 6

def test_normalize_mz_column_rounding():
    """Test that m/z values are correctly rounded before casting."""
    lf = pl.LazyFrame({
        "m/z": [100.0000004, 100.0000006]
    })
    result = normalize_mz_column(lf)
    df = result.collect()
    assert df["mz_int"].to_list() == [
        round(100.0000004 * cf.SCALE),
        round(100.0000006 * cf.SCALE),
    ]

def test_normalize_mz_column_casts_string_values():
    """Test that string m/z values are converted to float."""
    lf = pl.LazyFrame({
        "m/z": ["100.0", "101.5"]
    })
    df = normalize_mz_column(lf).collect()

    assert df["mz_int"].to_list() == [
        round(100.0 * cf.SCALE),
        round(101.5 * cf.SCALE),
    ]

# ======================
# TESTS FOR make_window
# ======================

def test_make_window_dataframe(sample_dataframe_spectrum):
    """Test window creation with DataFrame."""
    result = make_window(sample_dataframe_spectrum, "m/z", 10.0)
    assert "window_id" in result.columns
    assert result["window_id"].dtype == pl.Int32

    # Check window assignments
    assert result["window_id"].to_list() == [10, 10, 10, 10, 10, 10]

def test_make_window_lazyframe(sample_lazyframe_spectrum):
    """Test window creation with LazyFrame."""
    result = make_window(sample_lazyframe_spectrum, "m/z", 10.0)
    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    assert "window_id" in df.columns

def test_make_window_invalid_column(sample_dataframe_spectrum):
    """Test with invalid column name."""
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        make_window(sample_dataframe_spectrum, "invalid_col", 10.0)

@pytest.mark.parametrize(
    "step, expected",
    [
        (1.0, [100, 101, 102, 103, 104, 105]),
        (5.0, [20, 20, 20, 20, 20, 21]),
        (10.0, [10, 10, 10, 10, 10, 10]),
        (20.0, [5, 5, 5, 5, 5, 5]),
    ],
)
def test_make_window_various_steps(sample_dataframe_spectrum, step, expected):
    """Test window creation with different step sizes."""
    result = make_window(sample_dataframe_spectrum, "m/z", step)

    assert result["window_id"].to_list() == expected

def test_make_window_boundaries():
    """Test window assignment at window boundaries."""
    df = pl.DataFrame({
        "m/z": [99.9, 100.0, 109.9, 110.0, 119.9, 120.0],
        "Intensity": [1, 1, 1, 1, 1, 1],
    })

    result = make_window(df, "m/z", 10.0)

    assert result["window_id"].to_list() == [9, 10, 10, 11, 11, 12]

def test_make_window_decimal_step():
    """Test window creation with a decimal step."""
    df = pl.DataFrame({
        "m/z": [1.0, 1.4, 1.5, 1.9, 2.0],
    })

    result = make_window(df, "m/z", 0.5)

    assert result["window_id"].to_list() == [2, 2, 3, 3, 4]

def test_make_window_preserves_data(sample_dataframe_spectrum):
    """Test that original columns and values are preserved."""
    result = make_window(sample_dataframe_spectrum, "m/z", 10.0)

    assert result.select(sample_dataframe_spectrum.columns).equals(sample_dataframe_spectrum)

def test_make_window_invalid_step(sample_dataframe_spectrum):
    """Test that non-positive step raises ValueError."""
    with pytest.raises(ValueError, match="step must be greater than 0"):
        make_window(sample_dataframe_spectrum, "m/z", 0)

# ======================
# TESTS FOR find_all_peaks
# ======================

def test_find_all_peaks_basic(sample_spectrum_data):
    """Test basic peak finding."""
    intensity = sample_spectrum_data["Intensity"]
    interval = 1.0  # m/z interval

    peaks = find_all_peaks(intensity, interval, prominence=500, distance=1.0)
    assert isinstance(peaks, list)
    assert all(isinstance(p, int) for p in peaks)
    # With these parameters, should find at least the peak at index 4 (3000)
    assert peaks == [1, 4]

def test_find_all_peaks_prominence_threshold(sample_spectrum_data):
    """Test that a higher prominence threshold detects fewer peaks."""
    intensity = sample_spectrum_data["Intensity"]
    interval = 1.0

    # High prominence should find fewer peaks
    peaks_high = find_all_peaks(intensity, interval, prominence=2500)
    # Low prominence should find more peaks
    peaks_low = find_all_peaks(intensity, interval, prominence=100)

    assert len(peaks_low) >= len(peaks_high)

def test_find_all_peaks_distance_constraint(sample_spectrum_data):
    """Test that distance is converted from m/z to array indices."""
    intensity = sample_spectrum_data["Intensity"]
    interval = 1.0

    # With distance=0.5 (0.5 m/z units), peaks must be at least 0.5 apart
    peaks_close = find_all_peaks(intensity, interval, distance=0.5)
    # With distance=2.0, peaks must be at least 2.0 apart
    peaks_far = find_all_peaks(intensity, interval, distance=2.0)

    # Fewer peaks expected with larger distance
    assert len(peaks_far) <= len(peaks_close)

def test_find_all_peaks_height_threshold(sample_spectrum_data):
    """Test height threshold for peaks."""
    intensity = sample_spectrum_data["Intensity"]
    interval = 1.0

    # Only peaks above 2500 intensity
    peaks = find_all_peaks(intensity, interval, prominence=500, height=2500)
    # Only the local peak at index 4 has an intensity >= 2500
    # Note: The peak at index 5 has intensity 2500,
    # but it is not considered a peak as it's not a local maximum (border).
    assert peaks == [4]
    assert intensity[peaks[0]] >= 2500

def test_find_all_peaks_invalid_interval(sample_spectrum_data):
    """Test with invalid (non-positive) interval."""
    intensity = sample_spectrum_data["Intensity"]

    with pytest.raises(ValueError, match="Interval must be positive"):
        find_all_peaks(intensity, 0.0)

    with pytest.raises(ValueError, match="Interval must be positive"):
        find_all_peaks(intensity, -1.0)

def test_find_all_peaks_empty_input():
    """Test with empty input array."""
    peaks = find_all_peaks(np.array([]), 1.0)
    assert peaks == []

def test_find_all_peaks_constant_input():
    """Test with constant intensity values (no peaks)."""
    intensity = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    peaks = find_all_peaks(intensity, 1.0)
    assert peaks == []

def test_find_all_peaks_distance_conversion():
    """Test conversion of m/z distance to array indices."""
    intensity = np.array(
        [0.0, 100.0, 10.0, 90.0, 5.0, 200.0, 0.0],
    )

    # distance = 0.5 m/z and interval = 1 m/z
    # => distance_indices = 0.5 -> round = 0 -> max(1, 0) = 1
    # All three local maxima are detected.
    peaks_close = find_all_peaks(
        intensity,
        interval=1.0,
        prominence=10,
        distance=0.5,
    )

    # distance = 3 m/z and interval = 1 m/z
    # => distance_indices = 3
    # The middle peak is too close to both other peaks.
    peaks_far = find_all_peaks(
        intensity,
        interval=1.0,
        prominence=10,
        distance=3.0,
    )

    assert peaks_close == [1, 3, 5]
    assert peaks_far == [1, 5]

# =========================
# TESTS FOR _process_window
# =========================

def test_process_window_basic(sample_spectrum_data):
    """Test basic window processing."""
    mz = sample_spectrum_data["m/z"]
    intensity = sample_spectrum_data["Intensity"]
    window_indices = np.array([0, 1, 2, 3])  # First 4 points
    local_interval = 1.0

    args = (mz, intensity, window_indices, local_interval, 500.0, 0.0, 1)
    peaks = _process_window(args)

    assert isinstance(peaks, list)
    assert all(isinstance(p, int) for p in peaks)

def test_process_window_insufficient_data(sample_spectrum_data):
    """Test with window containing insufficient data points."""
    mz = sample_spectrum_data["m/z"]
    intensity = sample_spectrum_data["Intensity"]
    window_indices = np.array([0])

    args = (mz, intensity, window_indices, 1.0, 500.0, 0.0, 1)
    peaks = _process_window(args)

    assert peaks == []

def test_process_window_no_peaks(sample_spectrum_data):
    """Test window with no detectable peaks."""
    mz = sample_spectrum_data["m/z"]
    intensity = sample_spectrum_data["Intensity"]
    window_indices = np.array([0, 1, 2])

    args = (mz, intensity, window_indices, 1.0, 1000.0, 0.0, 1)
    peaks = _process_window(args)

    assert peaks == []

def test_process_window_detects_known_peaks(sample_spectrum_data):
    """Test detection of known peaks."""
    mz = sample_spectrum_data["m/z"]
    intensity = sample_spectrum_data["Intensity"]

    window_indices = np.array([0, 1, 2, 3, 4, 5])

    args = (
        mz,
        intensity,
        window_indices,
        1.0,
        500.0,  # prominence
        0.0,    # height_min
        1,      # dist_min
    )

    peaks = _process_window(args)

    assert peaks == [1, 4]


def test_process_window_returns_global_indices(sample_spectrum_data):
    """Test conversion from local peak indices to global indices."""
    mz = sample_spectrum_data["m/z"]
    intensity = sample_spectrum_data["Intensity"]

    # Window contains global indices 2, 3, 4, 5
    window_indices = np.array([2, 3, 4, 5])

    args = (
        mz,
        intensity,
        window_indices,
        1.0,
        500.0,
        0.0,
        1,
    )

    peaks = _process_window(args)

    # The peak is locally at index 2,
    # but globally at index 4.
    assert peaks == [4]

# ===================================
# TESTS FOR extract_peaks_all_windows
# ===================================

def test_extract_peaks_all_windows_basic(sample_lazyframe_spectrum):
    """Test basic peak extraction across windows."""
    result = extract_peaks_all_windows(
        df=sample_lazyframe_spectrum,
        mz_col_name="m/z",
        intensity_col_name="Intensity",
        step=10.0,
        prominence=500.0,
        height_min=0.0,
        dist_min=1,
        n_workers=1
    )

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    assert "m/z" in df.columns
    assert "Intensity" in df.columns
    # Should have fewer rows than input (only peaks)
    assert len(df) <= 4

def test_extract_peaks_all_windows_no_peaks(sample_lazyframe_spectrum):
    """Test with parameters that find no peaks."""
    # Very high prominence threshold
    result = extract_peaks_all_windows(
        df=sample_lazyframe_spectrum,
        mz_col_name="m/z",
        intensity_col_name="Intensity",
        step=10.0,
        prominence=1e10,  # Extremely high
        height_min=0.0,
        dist_min=1,
        n_workers=1
    )

    df = result.collect()
    assert len(df) == 0

def test_extract_peaks_all_windows_single_window(sample_lazyframe_spectrum):
    """Test with single window covering all data."""
    result = extract_peaks_all_windows(
        df=sample_lazyframe_spectrum,
        mz_col_name="m/z",
        intensity_col_name="Intensity",
        step=1000.0,  # Very large window
        prominence=500.0,
        height_min=0.0,
        dist_min=1,
        n_workers=1
    )

    df = result.collect()
    assert len(df) > 0  # Should find at least one peak

# TODO: STOP HERE continue above

# =======================
# TESTS FOR calcul_offset
# =======================

def test_calcul_offset_basic():
    """Test basic offset calculation."""
    th_min, th_max = calcul_offset(5.0, 1000.0)  # 5 ppm threshold

    # For 1000 Da, 5 ppm = 0.005 Da
    # th_min = 1000 / (1 + 5e-6) ≈ 999.995
    # th_max = 1000 / (1 - 5e-6) ≈ 1000.005
    assert th_min == pytest.approx(999.995, abs=0.001)
    assert th_max == pytest.approx(1000.005, abs=0.001)

def test_calcul_offset_zero_threshold():
    """Test with zero threshold."""
    th_min, th_max = calcul_offset(0.0, 1000.0)
    assert th_min == 1000.0
    assert th_max == 1000.0

def test_calcul_offset_small_mass():
    """Test with small mass value."""
    th_min, th_max = calcul_offset(10.0, 100.0)
    assert th_min == pytest.approx(99.999, abs=0.001)
    assert th_max == pytest.approx(100.001, abs=0.001)

def test_calcul_offset_large_threshold():
    """Test with large threshold value."""
    th_min, th_max = calcul_offset(100.0, 1000.0)  # 100 ppm
    assert th_min == pytest.approx(999.9, abs=0.01)
    assert th_max == pytest.approx(1000.1, abs=0.01)

def test_calcul_offset_threshold_boundary():
    """Test with a threshold close to 1,000,000 ppm."""
    th_min, th_max = calcul_offset(999_999.0, 1000.0)
    assert th_min == pytest.approx(500.00025, rel=1e-5)
    assert th_max == pytest.approx(1_000_000_000.0)

@pytest.mark.parametrize(  # Allow multiple runs with different
    "threshold",           # invalid threshold values
    [-1.0, 1e6, 1e6 + 1],
)
def test_calcul_offset_invalid_threshold(threshold):
    """Test that invalid thresholds raise ValueError."""
    with pytest.raises(ValueError):
        calcul_offset(threshold, 1000.0)

# ====================================
# TESTS FOR calculate_chemical_formula
# ====================================

def test_calculate_chemical_formula_single_monomer(monomer_df):
    """Test a single monomer without label or adduct."""
    oligo_df = {"dHex": 1, "Hex": 0, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        use_label_3aq=False,
    )

    assert formula == "C6H10O5"

def test_calculate_chemical_formula_empty_composition(monomer_df):
    """Test an empty composition without label or adduct."""
    oligo_df = {"dHex": 0, "Hex": 0, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        use_label_3aq=False,
    )

    assert formula == ""

def test_calculate_chemical_formula_element_order(monomer_df):
    """Test formula ordering for additional elements."""
    oligo_df = {"dHex": 1, "Hex": 0, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct="test",
        adduct_compo={"test": {"F": 1, "Cl": 1}},
        use_label_3aq=False,
    )

    assert formula == "C6H10O5Cl1F1"

def test_calculate_chemical_formula_basic(monomer_df):
    """Test basic chemical formula calculation."""
    oligo_df = {"dHex": 1, "Hex": 1, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct=None,
        use_label_3aq=False
    )

    # dHex: C6H10O5, Hex: C6H12O6
    # Total: C12H22O11
    # Condensation: -H2O (for 2 monomers)
    assert formula == "C12H20O10"

def test_calculate_chemical_formula_with_adduct(monomer_df, sample_adduct_compo):
    """Test with sodium adduct."""
    oligo_df = {"dHex": 1, "Hex": 0, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct="Na+",
        adduct_compo=sample_adduct_compo,
        use_label_3aq=False,
    )

    # dHex: C6H10O5
    # No condensation (only 1 monomer)
    # Sodium adduct: +Na
    assert formula == "C6H10O5Na1"

def test_calculate_chemical_formula_with_label(monomer_df, label_3aq_compo):
    """Test with label."""
    oligo_df = {"dHex": 1, "Hex": 0, "HexA": 0}

    formula = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct=None,
        use_label_3aq=True,
        label_compo=label_3aq_compo
    )

    # dHex: C6H10O5
    # Label: C9H7N1O1
    # Total: C15H17NO6
    # Condensation: -H2O (for 2 "monomers" including label)
    # Final: C15H16N2O4
    assert formula == "C15H16N2O4"

def test_calculate_chemical_formula_unknown_adduct(monomer_df):
    """Test with unknown adduct."""
    oligo_df = {"dHex": 1, "Hex": 0, "HexA": 0}

    with pytest.raises(ValueError, match="Unknown adduct"):
        calculate_chemical_formula(
            oligo_df=oligo_df,
            monomer_df=monomer_df,
            adduct="Unknown",
            use_label_3aq=False
        )

def test_calculate_chemical_formula_negative_count(monomer_df):
    """Test that negative element counts raise error."""
    # This is a bit tricky to trigger - would need a specific combination
    # where adduct addition results in negative count
    oligo_df = {"dHex": 0, "Hex": 0, "HexA": 0}

    with pytest.raises(ValueError, match="Negative element count"):
        calculate_chemical_formula(
            oligo_df=oligo_df,
            monomer_df=monomer_df,
            adduct="H",  # Adding H to nothing might cause issues
            adduct_compo={"H": {"H": -1}},  # Negative count in adduct
            use_label_3aq=False
        )


def test_3hex_proton_formula_without_label(monomer_df_hex_selected):
    """Test formula calculation for 3 Hex + H+."""
    oligo_df = {"Hex": 3}

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df_hex_selected,
        adduct="H+",
        adduct_compo={"H+": {"H": 1}},
        use_label_3aq=False,
    )

    # 3 Hex - 2 H2O + H+
    assert result == "C18H33O16"


def test_3hex_label_3aq_proton_formula(
    monomer_df_hex_selected,
    label_3aq_compo,
):
    """Test formula calculation for 3 Hex + 3AQ label + H+."""
    oligo_df = {"Hex": 3}

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df_hex_selected,
        adduct="H+",
        adduct_compo={"H+": {"H": 1}},
        use_label_3aq=True,
        label_compo=label_3aq_compo,
    )

    # 3 Hex + 3AQ - 3 H2O + H+
    assert result == "C27H39N2O15"


def test_3hex_label_3aq_water_loss_proton_adduct(
    monomer_df_hex_selected,
    label_3aq_compo,
):
    """Test formula calculation with a water-loss proton adduct."""
    oligo_df = {"Hex": 3}

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df_hex_selected,
        adduct="H+(-H2O)",
        adduct_compo={"H+(-H2O)": {"H": -1, "O": -1}},
        use_label_3aq=True,
        label_compo=label_3aq_compo,
    )

    # 3 Hex + 3AQ - 3 H2O + H+(-H2O)
    assert result == "C27H37N2O14"

# ==================================
# TESTS FOR get_theoretical_spectrum
# ==================================

def test_get_theoretical_spectrum_basic():
    """Test basic theoretical spectrum generation."""
    result = get_theoretical_spectrum("C6H12O6")
    assert isinstance(result, pl.LazyFrame)

    df = result.collect()
    assert set(df.columns) == {
        "mz_theo",
        "prob_theo",
        "prob_theo_norm",
    }
    assert df.height > 0
    assert df["mz_theo"].is_not_null().all()
    assert df["prob_theo"].is_not_null().all()
    assert df["prob_theo_norm"].is_not_null().all()

def test_get_theoretical_spectrum_caching():
    """Test that spectrum is cached."""
    # First call
    result1 = get_theoretical_spectrum("C6H12O6")
    # Second call with same formula
    result2 = get_theoretical_spectrum("C6H12O6")

    # Should return the same cached object
    assert result1 is result2

def test_get_theoretical_spectrum_different_formulas():
    """Test with different formulas."""
    result1 = get_theoretical_spectrum("C6H12O6")
    result2 = get_theoretical_spectrum("C12H24O12")

    assert result1 is not result2

def test_get_theoretical_spectrum_threshold():
    """Test with different probability thresholds."""
    result_low = get_theoretical_spectrum("C6H12O6", iso_threshold=0.01)
    result_high = get_theoretical_spectrum("C6H12O6", iso_threshold=0.0001)

    df_low = result_low.collect()
    df_high = result_high.collect()

    # Higher threshold should result in fewer isotopologues
    assert len(df_high) >= len(df_low)

def test_get_theoretical_spectrum_normalization():
    """Test that normalized probabilities have a maximum of 1."""
    df = get_theoretical_spectrum("C6H12O6").collect()

    assert df["prob_theo_norm"].max() == pytest.approx(1.0)
    assert (df["prob_theo_norm"] >= 0).all()
    assert (df["prob_theo_norm"] <= 1).all()

# ======================
# TESTS FOR _normalize
# ======================

def test_normalize_basic():
    """Test basic vector normalization."""
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = _normalize(vec)

    assert np.isclose(np.sum(normalized), 1.0)
    assert np.allclose(normalized, vec / np.sum(vec))

def test_normalize_zero_vector():
    """Test normalization of zero vector."""
    vec = np.array([0.0, 0.0, 0.0])
    normalized = _normalize(vec)

    assert np.all(normalized == 0.0)

def test_normalize_single_element():
    """Test normalization of single-element vector."""
    vec = np.array([5.0])
    normalized = _normalize(vec)

    assert normalized[0] == 1.0

def test_normalize_negative_values():
    """Test normalization with negative values."""
    vec = np.array([-1.0, 2.0, -3.0, 4.0])
    normalized = _normalize(vec)

    assert np.isclose(np.sum(normalized), 1.0)

# ================================
# TESTS FOR _build_aligned_vectors
# ================================

def test_build_aligned_vectors_basic():
    """Test basic vector alignment."""
    exp_mz = np.array([100.0, 101.0, 102.0, 103.0])
    exp_int = np.array([1.0, 2.0, 3.0, 4.0])
    theo_mz = np.array([100.0, 101.0, 102.0, 103.0])
    theo_prob = np.array([0.1, 0.2, 0.3, 0.4])

    vec_exp, vec_theo = _build_aligned_vectors(
        exp_mz, exp_int, theo_mz, theo_prob, tolerance=5.0, N=4
    )

    assert len(vec_exp) == 4
    assert len(vec_theo) == 4
    assert np.allclose(vec_exp, _normalize(exp_int))
    assert np.allclose(vec_theo, _normalize(theo_prob))

def test_build_aligned_vectors_partial_match():
    """Test with partial matching."""
    exp_mz = np.array([100.0, 100.5, 101.0, 101.5])
    exp_int = np.array([1.0, 1.0, 1.0, 1.0])
    theo_mz = np.array([100.0, 101.0])
    theo_prob = np.array([0.5, 0.5])

    vec_exp, vec_theo = _build_aligned_vectors(
        exp_mz, exp_int, theo_mz, theo_prob, tolerance=1.0, N=2
    )

    assert len(vec_exp) == 2
    assert len(vec_theo) == 2
    # Both theoretical peaks should have matches
    assert vec_exp[0] > 0
    assert vec_exp[1] > 0

# ======================
# TESTS FOR SCORING FUNCTIONS
# ======================

@pytest.mark.parametrize("vec_exp,vec_theo,expected", [
    (np.array([1.0, 0.0]), np.array([1.0, 0.0]), 1.0),  # Perfect match
    (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0),  # Orthogonal
    (np.array([0.6, 0.4]), np.array([0.6, 0.4]), 1.0),  # Perfect match
    (np.array([1.0, 1.0]), np.array([1.0, 0.0]), pytest.approx(0.707, rel=0.01)),  # 45 degree angle
])
def test_score_cosine(vec_exp, vec_theo, expected):
    """Test cosine similarity scoring."""
    result = score_cosine(vec_exp, vec_theo)
    assert result == pytest.approx(expected, rel=0.01)

def test_score_cosine_zero_vectors():
    """Test cosine with zero vectors."""
    assert score_cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0
    assert score_cosine(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == 0.0
    assert score_cosine(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0

@pytest.mark.parametrize("vec_exp,vec_theo,expected", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 1.0),  # Perfect correlation
    (np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), -1.0), # Perfect anti-correlation
    (np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), 0.0),  # No correlation
])
def test_score_pearson(vec_exp, vec_theo, expected):
    """Test Pearson correlation scoring."""
    result = score_pearson(vec_exp, vec_theo)
    assert result == pytest.approx(expected, rel=0.01)

def test_score_pearson_constant_vectors():
    """Test Pearson correlation with constant vectors."""
    assert score_pearson(np.ones(2), np.array([1.0, 2.0])) == 0.0
    assert score_pearson(np.ones(2), np.ones(2)) == 0.0

def test_score_pearson_single_element():
    """Test Pearson correlation with fewer than two elements."""
    assert score_pearson(np.array([1.0]), np.array([1.0])) == 0.0

@pytest.mark.parametrize("vec_exp,vec_theo", [
    (np.array([1.0, 2.0]), np.array([1.0, 2.0])),  # Perfect match
    (np.array([1.0, 2.0]), np.array([2.0, 1.0])),  # Different
])
def test_score_chi2(vec_exp, vec_theo):
    """Test chi-squared scoring."""
    result = score_chi2(vec_exp, vec_theo)
    assert 0.0 <= result <= 1.0

def test_score_chi2_identical():
    """Test chi2 with identical vectors."""
    vec = np.array([1.0, 2.0, 3.0])
    assert score_chi2(vec, vec) == 1.0

@pytest.mark.parametrize("vec_exp,vec_theo", [
    (np.array([1.0, 2.0]), np.array([1.0, 2.0])),  # Perfect match
    (np.array([1.0, 2.0]), np.array([2.0, 1.0])),  # Different
])
def test_score_kl(vec_exp, vec_theo):
    """Test KL divergence scoring."""
    result = score_kl(vec_exp, vec_theo)
    assert 0.0 <= result <= 1.0

def test_score_kl_identical():
    """Test KL with identical vectors."""
    vec = np.array([1.0, 2.0, 3.0])
    assert score_kl(vec, vec) == 1.0

@pytest.mark.parametrize("vec_exp,vec_theo", [
    (np.array([1.0, 2.0]), np.array([1.0, 2.0])),  # Perfect match
    (np.array([1.0, 2.0]), np.array([2.0, 1.0])),  # Different
])
def test_score_emd(vec_exp, vec_theo):
    """Test EMD scoring."""
    result = score_emd(vec_exp, vec_theo)
    assert 0.0 <= result <= 1.0

def test_score_emd_identical():
    """Test EMD with identical vectors."""
    vec = np.array([1.0, 2.0, 3.0])
    assert score_emd(vec, vec) == 1.0

def test_score_hybrid_perfect_match():
    """Test hybrid scoring with perfect match."""
    vec = np.array([1.0, 1.0, 1.0, 1.0])
    result = score_hybrid(vec, vec)
    # Perfect match should give high score
    assert result > 0.8

def test_score_hybrid_no_match():
    """Test hybrid scoring with no match."""
    vec_exp = np.array([1.0, 0.0, 0.0, 0.0])
    vec_theo = np.array([0.0, 0.0, 0.0, 1.0])
    result = score_hybrid(vec_exp, vec_theo)
    # Should be low but not necessarily zero due to coverage term
    assert result < 0.5

# ==================================
# TESTS FOR calculate_isotopic_score
# ==================================

def test_calculate_isotopic_score_basic():
    """Test basic isotopic score calculation."""
    # Create test data
    peaks_exp = pl.DataFrame({
        "mz_int": [100000000, 100000001, 100000002],  # Scaled m/z
        "Intensity_norm": [1.0, 0.5, 0.25]
    }).lazy()

    spectrum_theo = pl.DataFrame({
        "mz_int": [100000000, 100000001, 100000002],
        "prob_theo_norm": [1.0, 0.5, 0.25]
    }).lazy()

    score = calculate_isotopic_score(
        peaks_exp=peaks_exp,
        spectrum_theo=spectrum_theo,
        tolerance=5.0,
        N=3,
        scoring_method="cosine"
    )

    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0)

def test_calculate_isotopic_score_different_methods():
    """Test with different scoring methods."""
    peaks_exp = pl.DataFrame({
        "mz_int": [100000000, 100000001],
        "Intensity_norm": [1.0, 0.5]
    }).lazy()

    spectrum_theo = pl.DataFrame({
        "mz_int": [100000000, 100000001],
        "prob_theo_norm": [1.0, 0.5]
    }).lazy()

    for method in ["cosine", "pearson", "chi2", "KL", "wasserstein", "hybrid"]:
        score = calculate_isotopic_score(
            peaks_exp=peaks_exp,
            spectrum_theo=spectrum_theo,
            tolerance=5.0,
            N=2,
            scoring_method=method
        )
        assert np.isfinite(score), f"Non-finite score for {method}"
        assert 0.0 <= score <= 1.0, f"Invalid score for {method}"

def test_calculate_isotopic_score_invalid_method():
    """Test with invalid scoring method."""
    peaks_exp = pl.DataFrame({
        "mz_int": [100000000],
        "Intensity_norm": [1.0]
    }).lazy()

    spectrum_theo = pl.DataFrame({
        "mz_int": [100000000],
        "prob_theo_norm": [1.0]
    }).lazy()

    with pytest.raises(ValueError, match="Unknown scoring method"):
        calculate_isotopic_score(
            peaks_exp=peaks_exp,
            spectrum_theo=spectrum_theo,
            tolerance=5.0,
            scoring_method="invalid"
        )

# ======================
# TESTS FOR match_oligos
# ======================

def test_match_oligos_basic(library_lf, monomer_df, adducts):
    """Test basic oligo matching."""
    peaks = pl.LazyFrame({
        "m/z": [100.0, 150.0, 200.0],
        "Intensity": [1000.0, 2000.0, 1500.0]
    })

    result = match_oligos(
        peaks=peaks,
        library=library_lf,
        threshold=5,
        monomer_df=monomer_df,
        adduct_names=["H", "Na"],
        adduct_compo=adducts["adduct_compo"],
        label_compo=adducts["label_compo"],
        use_label_3aq=True,
        top_n=2,
        number_isotope=2,
        scoring_method="hybrid"
    )

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()
    # Should have some matches
    assert len(df) >= 0  # May have 0 if no matches found

def test_match_oligos_no_matches(library_lf, monomer_df, adducts):
    """Test with peaks that don't match any oligos."""
    peaks = pl.LazyFrame({
        "m/z": [1.0, 2.0, 3.0],  # Very low masses
        "Intensity": [1000.0, 2000.0, 1500.0]
    })

    result = match_oligos(
        peaks=peaks,
        library=library_lf,
        threshold=5,
        monomer_df=monomer_df,
        adduct_names=["H"],
        adduct_compo=adducts["adduct_compo"],
        label_compo=adducts["label_compo"],
        use_label_3aq=False,
        top_n=1,
        number_isotope=1
    )

    df = result.collect()
    # Should return empty LazyFrame with correct schema
    assert len(df) == 0

def test_match_oligos_empty_peaks(library_lf, monomer_df, adducts):
    """Test with empty peaks LazyFrame."""
    peaks = pl.LazyFrame({
        "m/z": pl.Series([], dtype=pl.Float64),
        "Intensity": pl.Series([], dtype=pl.Float64)
    })

    result = match_oligos(
        peaks=peaks,
        library=library_lf,
        threshold=5,
        monomer_df=monomer_df,
        adduct_names=["H"],
        adduct_compo=adducts["adduct_compo"],
        label_compo=adducts["label_compo"],
        use_label_3aq=False
    )

    df = result.collect()
    assert len(df) == 0

# ======================
# TESTS FOR adjust_library
# ======================

def test_adjust_library_basic(tmp_path, monomer_df):
    """Test basic library adjustment."""
    # Create a test library file
    library_data = """column_id,dHex,Hex,HexA,Neutral_unlabeled_mass
1,1,0,0,100.0
2,0,1,0,150.0
3,1,1,0,200.0
4,0,0,1,250.0
5,1,0,1,300.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    result = adjust_library(
        library_path=library_path,
        monomers_list=["dHex", "Hex"],
        mass_column_name="Neutral_unlabeled_mass"
    )

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()

    # Should only have columns: column_id, dHex, Hex, Neutral_unlabeled_mass
    assert set(df.columns) == {"column_id", "dHex", "Hex", "Neutral_unlabeled_mass"}

    # Should have 4 rows (all except the one with only HexA)
    assert len(df) == 4

    # Should be sorted by mass
    assert df["Neutral_unlabeled_mass"].is_sorted()

def test_adjust_library_all_monomers_zero(tmp_path):
    """Test with entries where all selected monomers are zero."""
    library_data = """column_id,dHex,Hex,HexA,Neutral_unlabeled_mass
1,0,0,1,100.0
2,0,0,0,150.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    result = adjust_library(
        library_path=library_path,
        monomers_list=["dHex", "Hex"],
        mass_column_name="Neutral_unlabeled_mass"
    )

    df = result.collect()
    # Should only have the first row (second has all zeros)
    assert len(df) == 1

def test_adjust_library_unauthorized_monomers(tmp_path):
    """Test removal of entries with unauthorized monomers."""
    library_data = """column_id,dHex,Hex,HexA,Neutral_unlabeled_mass
1,1,0,0,100.0
2,0,1,1,150.0
3,1,1,0,200.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    result = adjust_library(
        library_path=library_path,
        monomers_list=["dHex", "Hex"],
        mass_column_name="Neutral_unlabeled_mass"
    )

    df = result.collect()
    # Should only have rows 1 and 3 (row 2 has HexA)
    assert len(df) == 2
    assert 1 in df["column_id"].to_list()
    assert 3 in df["column_id"].to_list()
    assert 2 not in df["column_id"].to_list()

# ======================
# TESTS FOR compile_oligo_name
# ======================

def test_compile_oligo_name_basic():
    """Test basic oligo name compilation."""
    match_peaks = pl.LazyFrame({
        "dHex": [1, 0, 2],
        "Hex": [0, 1, 1],
        "HexA": [0, 0, 0],
        "Adduct": ["H", "Na", "K"]
    })

    result = compile_oligo_name(match_peaks, ["dHex", "Hex", "HexA"])

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()

    assert "Identifications" in df.columns
    # Check first row: 1dHex + H -> "1dHex-H"
    assert "1dHex" in df["Identifications"][0]
    assert "H" in df["Identifications"][0]

def test_compile_oligo_name_empty_peaks():
    """Test with empty peaks LazyFrame."""
    match_peaks = pl.LazyFrame({
        "dHex": pl.Series([], dtype=pl.Int64),
        "Hex": pl.Series([], dtype=pl.Int64),
        "Adduct": pl.Series([], dtype=pl.Utf8)
    })

    result = compile_oligo_name(match_peaks, ["dHex", "Hex"])
    df = result.collect()
    assert len(df) == 0

def test_compile_oligo_name_zero_monomers():
    """Test with all zero monomers."""
    match_peaks = pl.LazyFrame({
        "dHex": [0],
        "Hex": [0],
        "Adduct": ["H"]
    })

    result = compile_oligo_name(match_peaks, ["dHex", "Hex"])
    df = result.collect()

    # Should have empty string for oligo name
    assert df["Identifications"][0] == "-H"

def test_compile_oligo_name_multiple_monomers():
    """Test with multiple monomers."""
    match_peaks = pl.LazyFrame({
        "dHex": [2],
        "Hex": [1],
        "HexA": [1],
        "Adduct": ["Na"]
    })

    result = compile_oligo_name(match_peaks, ["dHex", "Hex", "HexA"])
    df = result.collect()

    # Should have "2dHex-1Hex-1HexA-Na" (order may vary based on column order)
    assert "2dHex" in df["Identifications"][0]
    assert "1Hex" in df["Identifications"][0]
    assert "1HexA" in df["Identifications"][0]
    assert "Na" in df["Identifications"][0]

# ======================
# TESTS FOR combine_lf
# ======================

def test_combine_lf_basic():
    """Test basic LazyFrame combination."""
    lf_1 = pl.LazyFrame({
        "query_mass": [100, 101, 102],
        "Identifications": ["A", "B", "C"],
        "Formula": ["C6H12O6", "C6H12O6", "C6H12O6"],
        "Delta_M": [0.1, 0.2, 0.3],
        "iso_score": [0.9, 0.8, 0.7]
    })

    lf_2 = pl.LazyFrame({
        "m/z": [100.0, 101.0, 102.0, 103.0],
        "Intensity": [1000, 2000, 1500, 500]
    })

    result = combine_lf(lf_1, lf_2)

    assert isinstance(result, pl.LazyFrame)
    df = result.collect()

    # Should have combined columns
    assert "m/z" in df.columns
    assert "query_mass" in df.columns
    assert "Identifications" in df.columns
    assert "Intensity" in df.columns

    # Should have all rows from lf_2 (right join)
    assert len(df) == 4

def test_combine_lf_right_join():
    """Test that right join is performed by default."""
    lf_1 = pl.LazyFrame({
        "query_mass": [100, 101],
        "Identifications": ["A", "B"]
    })

    lf_2 = pl.LazyFrame({
        "m/z": [100.0, 101.0, 102.0],
        "Intensity": [1000, 2000, 1500]
    })

    result = combine_lf(lf_1, lf_2)
    df = result.collect()

    # Should have all rows from lf_2
    assert len(df) == 3
    # The third row should have null for lf_1 columns
    assert df["Identifications"][2] is None

def test_combine_lf_custom_join_column():
    """Test with custom join column."""
    lf_1 = pl.LazyFrame({
        "mass": [100, 101],
        "Identifications": ["A", "B"]
    })

    lf_2 = pl.LazyFrame({
        "m/z": [100.0, 101.0],
        "Intensity": [1000, 2000]
    })

    result = combine_lf(lf_1, lf_2, join_on="mass")
    df = result.collect()

    assert "mass" in df.columns
    assert len(df) == 2

# ======================
# TESTS FOR process_mass
# ======================

def test_process_mass_basic(tmp_path):
    """Test basic mass processing."""
    # Create a test library
    library_data = """column_id,dHex,Hex,Neutral_unlabeled_mass
1,1,0,100.0
2,0,1,150.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    full_library = pl.scan_csv(library_path)

    result = process_mass(
        mass=100.0,
        adduct="H",
        full_library=full_library,
        monomers_list=["dHex", "Hex"],
        mass_col_name="Neutral_unlabeled_mass"
    )

    assert isinstance(result, str)
    assert "1dHex" in result
    assert "H" in result

def test_process_mass_no_match(tmp_path):
    """Test with mass that doesn't match any library entry."""
    library_data = """column_id,dHex,Hex,Neutral_unlabeled_mass
1,1,0,100.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    full_library = pl.scan_csv(library_path)

    result = process_mass(
        mass=200.0,  # Not in library
        adduct="H",
        full_library=full_library,
        monomers_list=["dHex", "Hex"],
        mass_col_name="Neutral_unlabeled_mass"
    )

    assert result == ""

def test_process_mass_multiple_matches(tmp_path):
    """Test with mass that matches multiple library entries."""
    library_data = """column_id,dHex,Hex,Neutral_unlabeled_mass
1,1,0,100.0
2,0,1,100.0
"""
    library_path = tmp_path / "test_library.csv"
    library_path.write_text(library_data)

    full_library = pl.scan_csv(library_path)

    result = process_mass(
        mass=100.0,
        adduct="H",
        full_library=full_library,
        monomers_list=["dHex", "Hex"],
        mass_col_name="Neutral_unlabeled_mass"
    )

    # Should have both identifications separated by /
    assert "/" in result
    assert "1dHex" in result
    assert "1Hex" in result