"""
MALDI-MS spectra analysis pipeline for oligosaccharide characterization.

This module provides core functionality for:
- Data processing of high-resolution mass spectrometry spectra (~2M data points)
- Memory-efficient workflows using Polars LazyFrames and piped operations
- Peak detection and noise filtering using window-based algorithms
- Library matching against oligosaccharide databases with configurable mass offset

Key Features:
- Input/Output Compatibility:
  * Input: CSV files with mandatory 'm/z' and 'Intensity' columns
  * Output: Standardized DataFrames 
  * Temporary files stored in: {TEMP_DIR} (auto-cleaned via atexit)

- Processing Pipeline:
  1. Spectrum merging on m/z values
  2. Peak extraction with dynamic windowing (default: ±10 m/z units)
  3. Mass comparison against oligosaccharide library (default offset: ±{SCALE} ppm)
  4. Name compilation using IUPAC-like nomenclature (e.g., "2dHex-3Hex")

- Performance Optimizations:
  * Polars-based operations for memory efficiency
  * Garbage collection hooks for cloud deployment

- Scientific Standards:
  * Mass scaling factor: {SCALE} (1e6 for ppm-level precision)
  * Distance metrics: Euclidean mass difference with thresholding
  * Adduct handling: [M+Na]+, [M+K]+, [M+H]+ 

Dependencies:
    polars>=0.19.0     # Core data processing
    scipy>=1.10.0      # Peak detection (find_peaks)
    numpy>=1.24        # Numerical operations
"""

__author__  = "Pierre-Alexandre HO"

import os
import polars as pl
import numpy as np
import shutil
import zipfile
import gc
import logging
import IsoSpecPy as iso
import load_file as load
import config as cf

from io import BytesIO
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, List
from numpy.typing import NDArray
from scipy.signal import find_peaks as fp
from scipy.spatial import distance
from scipy.stats import pearsonr,  wasserstein_distance
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def find_workers():
    n_workers = os.cpu_count()
    # Leave one CPU free for system processes if we have many cores
    if n_workers > 1:
        n_workers -= 1
    return n_workers


def read_csv(file:str, sep: str = ','):
    """
    Read a lambda csv file

    Args:
        file (str): _description_
        sep (str, optional): _description_. Defaults to ','.

    Returns:
        _type_: _description_
    """
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    return pl.scan_csv(file, separator=sep)


def read_spectrum(
    file: Union[str, "StreamlitUploadedFile"],
    sep: str = ','
) -> pl.LazyFrame:
    """
    Lazy-reads a MALDI spectrum CSV file into a Polars LazyFrame.

    This function reads only the 'm/z' and 'Intensity' columns.
    The data is returned as a LazyFrame to enable efficient, out-of-core processing of large files.

    Args:
        file: Path to the spectrum file (str) or a Streamlit UploadedFile object.
            The file must be a CSV with at least 'm/z' and 'Intensity' columns.
        sep: Delimiter used in the CSV file. Defaults to ','.

    Returns:
        A Polars LazyFrame containing only the 'm/z' and 'Intensity' columns, sorted by 'm/z'.
        This allows for deferred execution and optimization of downstream operations.

    """
    return (
        pl.scan_csv(file, separator=sep)
        .select([
            pl.col("m/z"),       # Mass-to-charge ratio column (join key)
            pl.col("Intensity")  # Signal intensity column
        ])
        .sort(by="m/z")
    )


def parse_spectra(
    files: list[Union[str, "StreamlitUploadedFile"]],
    dir: Optional[str] = None,
    delimiter: str = ","
) -> Dict[str, pl.LazyFrame]:
    """
    Parses a list of MALDI spectra files into a dictionary of Polars LazyFrames.

    Each spectrum is read and stored under a key derived from the filename.
    The function handles both local file paths and Streamlit uploaded files.

    Args:
        files: List of spectrum files to parse. Each element can be either a string (local path) or
            a Streamlit UploadedFile object.
        dir: Optional base directory for local file paths. If provided, each string in `files` is joined
            with `dir` to form the full path. Defaults to None.
        delimiter: Delimiter used in the CSV file. Defaults to ','.

    Returns:
        A dictionary where keys are spectrum names (derived from filenames) and values are Polars LazyFrames
        containing the spectrum data (m/z and intensity columns).

    Raises:
        FileNotFoundError: If a local file does not exist.
        ValueError: If a file cannot be parsed as a spectrum.
    """
    spectra: Dict[str, pl.LazyFrame] = {}

    for file in files:
        if isinstance(file, str):
            if dir is not None:
                file = os.path.join(dir, file)
            spectrum_name = os.path.basename(file).replace('.csv', '').replace(' ', '_')
        else:  # Streamlit uploaded file
            spectrum_name = file.name.replace('.csv', '').replace(' ', '_')
        
        spectra[spectrum_name] = read_spectrum(file, sep = delimiter)
    del file
    return spectra


def normalize_mz_column(
    lf: pl.LazyFrame,
    mz_col: str = "m/z",
    new_col: str = "mz_int"
) -> pl.LazyFrame:
    """
    Normalizes the m/z column by scaling and converting to integer values.

    This function is designed for MALDI-MS spectra preprocessing. It scales the m/z values
    by a fixed factor and rounds them to the nearest integer, which is useful for
    binning or reducing floating-point precision issues in downstream analysis.

    Args:
        lf: Input Polars LazyFrame containing the spectrum data.
        mz_col: Name of the column containing m/z values. Defaults to "m/z".
        new_col: Name of the new column to store the normalized (integer) m/z values.
                 Defaults to "mz_int".

    Returns:
        A new Polars LazyFrame with the original columns and the added normalized column.
        The new column contains m/z values scaled by `SCALE` and cast to Int64.
    """
    mz = pl.col(mz_col).cast(pl.Float64)  # Ensure float64 before scaling
    return lf.with_columns(
        (mz * cf.SCALE).round().cast(pl.Int64).alias(new_col)
    )


def make_window(df: Union[pl.DataFrame, pl.LazyFrame],
                column_name: str,
                step: Union[int, float]) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Divides a DataFrame into windows based on a specified column value range.

    This function creates window identifiers by dividing the values in the specified column
    by the given step size and converting to integers. Each unique window_id represents
    a range of values from [n*step, (n+1)*step] in the original column.

    Args:
        df: Input DataFrame or LazyFrame containing the data to be windowed.
            The function preserves the original type (DataFrame or LazyFrame).
        column_name: Name of the column to use for windowing. This column should contain
                     numeric values that will be divided into windows.
        step: Size of each window in m/z units.
              
    Returns:
        Union[pl.DataFrame, pl.LazyFrame]: A new DataFrame/LazyFrame with an additional
                                          'window_id' column containing the window
                                          identifiers. The original data is preserved.

    Raises:
        ValueError: If the specified column doesn't exist in the DataFrame.
        TypeError: If the column contains non-numeric values.

    Note:
        - The window_id is calculated as floor(column_value / step)
        - The function preserves the original DataFrame type (LazyFrame remains lazy)
    """
    return df.with_columns(
        (pl.col(column_name) // step).cast(pl.Int32).alias("window_id")
    )


def find_all_peaks(intensity: NDArray[np.float64],
                   interval: float,
                   prominence: float = 2e5,
                   height: Optional[Union[float, NDArray[np.float64]]] = None,
                   distance: float = 15) -> list[int]:
    """
    Detects peaks in a 1D intensity array using scipy's find_peaks with m/z-aware distance.

    This function identifies local maxima in intensity data that meet the specified
    criteria for prominence, height, and minimum distance between peaks. The distance
    parameter is automatically converted from m/z units to array indices using the
    provided interval value.

    Args:
        intensity: Array of intensity values to search for peaks.
        interval: Average interval between m/z values in the same units as the distance parameter.
                  Used to convert real-world distance (m/z units) to array indices.
        prominence: Required prominence of peaks (minimum height difference between peak
                    and its neighboring samples). Default is 2e5, suitable for MALDI-MS data.
        height: Required height of peaks. Can be a single value or an array of values.
                If None, no height threshold is applied.
        distance: Minimum distance between peaks in m/z units. Converted to array indices
                  using the interval parameter. Minimum value of 1 index is enforced.

    Returns:
        list[int]: Indices of detected peaks in the input intensity array.

    Raises:
        ValueError: If interval is zero or negative.
        RuntimeError: If peak detection fails (typically due to invalid input parameters).

    Note:
        - The distance parameter is converted from m/z units to array indices using
        - Lower prominence values will detect more (smaller) peaks.
    """
    if interval <= 0:
        raise ValueError("Interval must be positive")

    # Convert real distance value (m/z units) to array indices
    # max(1, ...) ensures we never have distance < 1 index
    distance_indices = max(1, int(round(distance / interval)))

    try:
        # Use scipy's find_peaks function to detect peaks
        peaks, _ = fp(intensity,
                     distance=distance_indices,
                     height=height,
                     prominence=prominence)
        return peaks.tolist()
    except Exception as e:
        raise RuntimeError(f"Peak detection failed: {str(e)}") from e


def _process_window(args: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float, float, float, int ]
                    ) -> List[int]:
    """
    Processes a single window of MALDI spectrum data to extract peak indices.

    This helper function handles the peak detection for a specific m/z window in parallel.
    It takes pre-computed window data and applies peak finding algorithms to identify
    significant peaks within the specified minimum distance constraint.

    Args:
        args: Tuple containing all required parameters for window processing:
            - mz: Array of m/z values for the entire spectrum
            - intensity: Array of intensity values for the entire spectrum
            - window_indices: Indices of data points belonging to current window
            - local_interval: Average m/z interval in current window (accounts for irregular spacing)
            - prominence: Minimum height difference between peak and its neighboring samples
            - dist_min: Minimum distance between peaks in m/z units

    Returns:
        List[int]: Global indices of detected peaks in the original spectrum data.
                  Returns empty list if window contains insufficient data points (<2).

    Note:
        The function converts local peak indices (relative to window) to global indices
        (relative to entire spectrum) before returning.
    """
    # Unpack arguments with explicit types for clarity
    mz: NDArray[np.float64] = args[0]
    intensity: NDArray[np.float64] = args[1]
    window_indices: NDArray[np.int64] = args[2]
    local_interval: float = args[3]
    prominence: float = args[4]
    height_min: float = args[5]
    dist_min: int = args[6]

    # Skip processing if window contains insufficient data points
    if len(window_indices) < 2:
        return []

    # Find peaks in current window using the provided peak detection function
    peaks: NDArray[np.int64] = find_all_peaks(
        intensity[window_indices],  # Intensity values for current window
        local_interval,             # Average spacing between m/z values in this window
        prominence=prominence,       # Minimum height difference between a peak and its neighbors
        height=height_min,           # Minimum height from the intensity axis (0)
        distance=dist_min,          # Minimum required distance between peaks
    )

    return window_indices[peaks].tolist()


def extract_peaks_all_windows(
        df: pl.LazyFrame,
        mz_col_name: str,
        intensity_col_name: str,
        step: float,
        prominence: float,
        height_min: float,
        dist_min: int,
        n_workers: int
) -> pl.LazyFrame:
    """
    Extracts peaks from MALDI-MS spectrum data using parallel window processing.

    This function divides the spectrum into windows of specified size (step) and processes
    each window in parallel to identify peaks. It's optimized for large datasets (millions
    of data points) by using process-based parallelism and efficient memory handling.

    Args:
        df: Input LazyFrame containing spectrum data with m/z and intensity columns.
            Should already be filtered to relevant m/z range if needed.
        mz_col_name: Name of column containing m/z (mass-to-charge) values.
        intensity_col_name: Name of column containing intensity values.
        step: Size of m/z windows for spectrum segmentation (in m/z units).
        dist_min: Minimum required distance between detected peaks (in m/z units).
        prominence: Minimum height difference between peak and its neighboring samples.
        n_workers: Number of parallel workers to use

    Returns:
        pl.LazyFrame: A new LazyFrame containing only the peak data (m/z and intensity columns)
                     with original indices preserved.
    Raises:
        RuntimeError: If parallel processing fails (typically due to memory constraints).

    Note:
        - The function first segments the spectrum into windows using make_window().
        - Each window is processed independently, making it ideal for parallel execution.
        - For best performance with large datasets (>1M points), ensure:
          * n_workers ≤ available CPU cores
          * Each window contains >100 data points (adjust step parameter accordingly)
          * System has sufficient RAM (approximately 8 bytes per data point × total points)

    """
  
    # Create windows for spectrum segmentation
    df = make_window(df, mz_col_name, step)

    # Convert Polars columns to NumPy arrays for vectorized processing
    # Using float64 for m/z and intensity to maintain precision
    mz: NDArray[np.float64] = df.select(mz_col_name).collect().to_numpy().flatten()
    intensity: NDArray[np.float64] = df.select(intensity_col_name).collect().to_numpy().flatten()
    windows: NDArray[np.int64] = df.select("window_id").collect().to_numpy().flatten()

    # Prepare arguments for parallel processing
    window_args: List[Tuple] = []

    # Process each unique window to create processing tasks
    for window_id in np.unique(windows):
        # Create boolean mask for current window
        mask: NDArray[np.bool_] = windows == window_id
        # Get indices of all points in this window
        window_indices: NDArray[np.int64] = np.where(mask)[0]

        # Skip windows with insufficient data points
        if len(window_indices) < 2:
            continue

        # Calculate local m/z interval (accounts for irregular spacing)
        # This is used by peak detection algorithm to determine appropriate thresholds
        mz_window = mz[window_indices]
        diffs = np.diff(mz_window)

        if len(diffs) == 0:
            continue

        local_interval = np.mean(diffs)
        if np.isnan(local_interval) or local_interval == 0:
            local_interval = 1.0  # Default or handle as needed

        # Package arguments for parallel processing
        window_args.append((
            mz,                # Full m/z array (shared across all workers)
            intensity,         # Full intensity array (shared across all workers)
            window_indices,    # Indices specific to this window
            local_interval,    # Calculated spacing for this window
            prominence,
            height_min,
            dist_min,          # Minimum peak distance parameter
        ))

    # Execute peak detection in parallel
    # ProcessPoolExecutor creates separate processes to avoid Python's GIL limitations
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_idx: List[int] = []
        # Process each window and collect results
        for result in executor.map(_process_window, window_args):
            all_idx.extend(result)

    # Return a new LazyFrame containing only peak data
    # Using filter() to preserve original indices and enable lazy evaluation
    #  pl.int_range(0, pl.len()) create column of idx
    #  is_in apply a boolean mask
    return df.select([mz_col_name, intensity_col_name]).filter(
        pl.int_range(0, pl.len()).is_in(all_idx))


def extract_and_detect_peaks(
    spectrum_data: pl.LazyFrame,
    window: int,
    prominence: float,
    height_min: float, 
    dist_min: int,
    n_workers: int
) -> pl.LazyFrame:
    """
    Extracts and detects peaks for all intensity columns in the spectra DataFrame.

    Args:
        spectrum: Spectrum LaztFrame with m/z and intensity columns.
        window: Window size for peak extraction.
        dist_min: Minimum distance between peaks.
        prominence: Minimum height difference between peak and its neighboring samples.
        n_workers: Number of parallel workers to use

    Returns:
        pl.LazyFrame: A new LazyFrame containing only the peak data (m/z and intensity columns)
                     with original indices preserved.
    """

    col_names = spectrum_data.collect_schema().names()
    mz_col, intensity_col = col_names[0], col_names[1]
    spectrum_peaks = extract_peaks_all_windows(
        spectrum_data, mz_col, intensity_col, window, prominence, height_min, dist_min,  n_workers
    )

    return spectrum_peaks


def calcul_offset(rm_threshold: float, real_mass: float) -> Tuple[float, float]:
    """
    Calculate the theoretical mass range [th_min, th_max] based on a mass accuracy threshold.

    Args:
        rm_threshold: Mass accuracy threshold in parts per million (ppm).
        real_mass: Measured mass value in Daltons.

    Returns:
        A tuple containing:
            - th_min (float): Minimum acceptable theoretical mass.
            - th_max (float): Maximum acceptable theoretical mass.
    """
    factor: float = rm_threshold / 1e6
    th_min: float = real_mass / (1 + factor)
    th_max: float = real_mass / (1 - factor)
    del factor
    return th_min, th_max


def calculate_brut_formula(
    oligo_df: Dict[str, int],
    monomer_df: pl.DataFrame,
    adduct: Optional[str] = None,
    use_label_3aq: bool = True
) -> str:
    """
    Compute brute formula from monomer composition using vectorized Polars.

    Parameters
    ----------
    oligo_df : pl.DataFrame
        One-row DataFrame containing monomer counts in columns matching monomer_df "Notation"
    monomer_df : pl.DataFrame
        Columns: ["Notation", "C", "H", "O", ...]
    adduct : str, optional
        Adduct type ("H", "Na", "K", "NH4", ...)
    use_label_3aq : bool
        Add 3AQ label if True

    Returns
    -------
    str
        Brute formula (e.g. "C30H52O21Na1")
    """
    
    elements = monomer_df.columns[3:]  # Skip "Notation"
    totals = {elem: 0 for elem in elements}
    total_monomers = 0

    for row in monomer_df.iter_rows():
        mono_name = row[1]
        count = oligo_df[mono_name]  # Monomer repitition
        if count == 0:
            continue
        total_monomers += count
        atom_values = row[3:]
        for elem, val in zip(elements, atom_values):
            totals[elem] += val * count

    # Add the label
    if use_label_3aq:
        for elem, count in cf.LABEL_3AQ.items():
            totals[elem] = totals.get(elem, 0) + count
            total_monomers += 1
    
    # Condensation reaction
    if total_monomers > 1:
        n_loss = total_monomers - 1
        totals["H"] = totals.get("H", 0) - 2 * n_loss
        totals["O"] = totals.get("O", 0) - 1 * n_loss
    
    # Add adduct
    if adduct is not None:
        if adduct not in cf.ADDUCTS:
            raise ValueError(f"Unknown adduct: {adduct}")

        for elem, count in cf.ADDUCTS[adduct].items():
            totals[elem] = totals.get(elem, 0) + count

    # Build Formula
    # Ordre chimique standard
    priority_order = ["C", "H", "N", "O", "S", "P"]
    other_elements = sorted([e for e in totals.keys() if e not in priority_order])

    ordered_elements = priority_order + other_elements

    formula_parts = [
        f"{elem}{int(totals[elem])}"
        for elem in ordered_elements
        if elem in totals and totals[elem] > 0
    ]

    return "".join(formula_parts)


@lru_cache(maxsize=1000)
def get_theoretical_spectrum(
    formula: str,
    iso_threshold: float = 0.0001,
) -> pl.DataFrame:
    """
    Génère le spectre isotopique théorique pour une formule brute + adduct.
    The theoritical spectrum is saved in the cache
    Args:
        formula: Formule brute (ex: "C29H47N3O13").
        iso_threshold: Seuil de probabilité pour filtrer les isotopologues (défaut: 0.01%).
    Returns:
        LazyFrame avec colonnes ["mz_theo", "prob_theo", "prob_theo_norm"].
    """
    sp = iso.IsoThreshold(formula=formula, threshold=iso_threshold)

    probs = np.fromiter(sp.probs, dtype=float)

    return pl.LazyFrame({
        "mz_theo": np.fromiter(sp.masses, dtype=float),
        "prob_theo": probs,
        "prob_theo_norm": probs / max(probs) if probs.size > 0 else np.array([0.0])
    })


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to sum = 1."""
    s = np.sum(vec)
    return vec / s if s > 0 else vec


def _build_aligned_vectors(
    exp_mz: np.ndarray,
    exp_int: np.ndarray,
    theo_mz: np.ndarray,
    theo_prob: np.ndarray,
    tolerance: float,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build aligned isotopic vectors (experimental vs theoretical).
    """
    aligned_exp = []
    aligned_theo = []

    for mz_theo, prob_theo in zip(theo_mz[:N], theo_prob[:N]):
        th_min, th_max = calcul_offset(tolerance, mz_theo)

        mask = (exp_mz >= mz_theo - th_min) & (exp_mz <= mz_theo + th_max)

        if np.any(mask):
            # closest peak (better than max)
            idx = np.argmin(np.abs(exp_mz[mask] - mz_theo))
            matched_int = exp_int[mask][idx]
        else:
            matched_int = 0.0

        aligned_exp.append(matched_int)
        aligned_theo.append(prob_theo)

    vec_exp = np.array(aligned_exp)
    vec_theo = np.array(aligned_theo)

    return _normalize(vec_exp), _normalize(vec_theo)


# ======================
# SCORING FUNCTIONS
# ======================

def score_cosine(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    if np.any(vec_exp) and np.any(vec_theo):
        return 1 - distance.cosine(vec_exp, vec_theo)
    return 0.0


def score_pearson(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    if len(vec_exp) > 1 and np.std(vec_exp) > 0 and np.std(vec_theo) > 0:
        return pearsonr(vec_exp, vec_theo)[0]
    return 0.0


def score_chi2(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    denom = vec_theo + 1e-12
    chi2 = np.sum((vec_exp - vec_theo) ** 2 / denom)
    return 1 / (1 + chi2)


def score_kl(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    kl = np.sum(vec_theo * np.log((vec_theo + 1e-12) / (vec_exp + 1e-12)))
    return 1 / (1 + kl)


def score_emd(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    emd = wasserstein_distance(vec_exp, vec_theo)
    return 1 / (1 + emd)


def score_hybrid(vec_exp: np.ndarray, vec_theo: np.ndarray) -> float:
    """
    Recommended default score for isotopic pattern matching.
    Combines coverage + cosine similarity.
    """
    cosine = score_cosine(vec_exp, vec_theo)

    coverage = np.sum(vec_exp > 0) / len(vec_theo)

    penalty = np.sum(vec_exp[vec_theo < 1e-6])

    return 0.5 * cosine + 0.4 * coverage - 0.3 * penalty


def calculate_isotopic_score(
    peaks_exp,
    spectrum_theo,
    tolerance: float,
    N: int = 4,
    scoring_method: str = "hybrid",
) -> float:
    """
    Compute isotopic pattern score between experimental and theoretical spectra.

    Parameters
    ----------
    peaks_exp : DataFrame or LazyFrame
        Experimental peaks with columns ["mz_int", "Intensity_norm"].
    spectrum_theo : DataFrame or LazyFrame
        Theoretical isotopic spectrum with ["mz_int", "prob_theo_norm"].
    tolerance : float
        m/z tolerance (ppm or Da depending on calcul_offset).
    N : int, optional
        Number of isotopes to consider (default = 4).
    scoring_method : str, optional
        One of ["cosine", "pearson", "chi2", "kl", "wasserstein", "hybrid"].

    Returns
    -------
    float
        Score between 0 and 1 (higher = better match).
    """

    # Convert to numpy
    peaks_np = peaks_exp.select(["mz_int", "Intensity_norm"]).collect().to_numpy()
    exp_mz = peaks_np[:, 0]
    exp_int = peaks_np[:, 1]

    spectrum_theo_np = spectrum_theo.select(["mz_int", "prob_theo_norm"]).collect().to_numpy()
    theo_mz = spectrum_theo_np[:, 0]
    theo_prob = spectrum_theo_np[:, 1]

    # Build aligned vectors
    vec_exp, vec_theo = _build_aligned_vectors(
        exp_mz, exp_int, theo_mz, theo_prob, tolerance, N
    )

    # Select scoring method
    scoring_functions = {
        "cosine": score_cosine,
        "pearson": score_pearson,
        "chi2": score_chi2,
        "KL": score_kl,
        "wasserstein": score_emd,
        "hybrid": score_hybrid,
    }

    if scoring_method not in scoring_functions:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return scoring_functions[scoring_method](vec_exp, vec_theo)


def match_oligos(
    peaks: pl.LazyFrame,
    library: pl.LazyFrame,
    threshold: int,
    monomer_df: pl.DataFrame,
    use_label_3aq: bool = True,
    top_n: int = 4,
    number_isotope: int = 4,
    scoring_method: str = "hybrid"
) -> pl.LazyFrame:
    """
    Match each mass in a spectrum to oligomers in a reference library.
    Returns a LazyFrame with matches, including:
    - All original library columns
    - Added columns: 'query_mass', 'Adduct', 'Delta_M', 'Formula'

    Args:
        peaks: LazyFrame of detected peaks including the m/z of the spectrum
        library: LazyFrame of the reference oligosaccharides
        threshold: Mass accuracy tolerance for matching in ppm
        monomer_df: DataFrame with monomer compositions
        use_label_3aq: Add 3AQ label if True

    Returns:
        pl.LazyFrame:  
    """

    # Transform the mass in ppm and int to optimize the algorithm
    for adduct in ['labeled_mass_H+', 'labeled_mass_Na+', 'labeled_mass_K+']:
        library = normalize_mz_column(
        lf = library,
        mz_col=adduct,
        new_col=f"{adduct}_int"
    )
    
    peaks = normalize_mz_column(
        lf = peaks,
        mz_col="m/z",
        new_col="mz_int"
    ).with_columns(
        (pl.col("Intensity") / pl.col("Intensity").max()).alias("Intensity_norm")
    )

    peaks_np = peaks.select(["mz_int", "Intensity"]).collect().to_numpy()
    mass_list = peaks_np[:, 0]
    intensity_list = peaks_np[:, 1]
    matched_oligos = []

    # Loop: for each mass select the closest reference oligo
    for mass, intensity in zip(mass_list, intensity_list):

        # --- Step 1: Filter candidates ---

        th_min, th_max = calcul_offset(threshold, mass)

        candidates = library.filter(
            (pl.col("labeled_mass_H+_int").is_between(th_min, th_max)) |
            (pl.col("labeled_mass_Na+_int").is_between(th_min, th_max)) |
            (pl.col("labeled_mass_K+_int").is_between(th_min, th_max))
        )

        if candidates.select(pl.len()).collect().item() == 0:
            continue

        # --- Step 2: Compute the mass ratio ---
        candidates_with_diff = candidates.with_columns(
            pl.lit(int(mass)).cast(pl.Int64).alias("query_mass"),
            pl.lit(float(intensity)).alias("Intensity"),
            ((mass - pl.col("labeled_mass_H+_int"))/pl.col("labeled_mass_H+_int")*10**6).abs().alias("diff_H"),
            ((mass - pl.col("labeled_mass_Na+_int"))/pl.col("labeled_mass_Na+_int")*10**6).abs().alias("diff_Na"),
            ((mass  -pl.col("labeled_mass_K+_int"))/pl.col("labeled_mass_K+_int")*10**6).abs().alias("diff_K")
        )

        # --- Step 3: Find best match ---
        # Find the top_n best matches, this strategy allow best compromise between
        # performance and precision. The top_n will be analyze with isotop for a better
        # precison
        best_matches = (
            candidates_with_diff
            # Find the Minimum ratio between the adducts
            .with_columns(pl.min_horizontal("diff_H", "diff_Na", "diff_K").alias("Delta_M"))
            # Select 1 line of the library for the query mass
            .sort("Delta_M").head(top_n)
            # Add the corresponding best adduct in the lazyframe of the mass 
            .with_columns(
                pl.when(pl.col("diff_H") == pl.col("Delta_M")).then(pl.lit("H+"))
                .when(pl.col("diff_Na") == pl.col("Delta_M")).then(pl.lit("Na+"))
                .otherwise(pl.lit("K+")).alias("Adduct")
            )
            # Select relevant columns (exclude temporary ones)
            .select([
                pl.exclude("diff_H", "diff_Na", "diff_K"),  # Keep all original columns
                #"query_mass", "Adduct", "mass_diff"
            ])
        )

        # --- Step 4: Compute iso_score ---
        scored_candidates = []
        # collect is acceptable here because top_n is small       
        for candidate_row in best_matches.collect().iter_rows(named=True):
            
            # 1. Compute formula
            formula = calculate_brut_formula(
                oligo_df=candidate_row,
                monomer_df=monomer_df,
                adduct=candidate_row.get('Adduct', None),
                use_label_3aq=use_label_3aq
            )

            # 2. Theoretical spectrum
            spectrum_theo = get_theoretical_spectrum(formula)

            spectrum_theo = normalize_mz_column(
                lf=spectrum_theo,
                mz_col="mz_theo",
                new_col="mz_int"
            )

            # 3. Compute isotopic score
            iso_score = calculate_isotopic_score(
                peaks_exp=peaks,              
                spectrum_theo=spectrum_theo,      
                tolerance=threshold,
                N=number_isotope,                              
                scoring_method=scoring_method          
            )

            # 4. Store result
            candidate_row["iso_score"] = iso_score
            candidate_row["Formula"] = formula

            scored_candidates.append(candidate_row)


        # --- Step 5: Select best candidate ---
        best_match = max(scored_candidates, key=lambda x: x["iso_score"], default=None)

        if best_match is not None:
            matched_oligos.append(best_match)
    
    # Concatenate all matches for this spectrum
    if matched_oligos:
        return pl.DataFrame(matched_oligos).lazy()
    else:  # If there is no information ==> creation of an empty lazyframe for the following code
        return pl.LazyFrame(schema={
            "Neutral_unlabeled_mass": pl.Int64,
            "query_mass": pl.Int64,
            "Adduct": pl.Utf8,
            "Delta_M": pl.Float64,
            "Formula": pl.Utf8,
            "iso_score": pl.Float64,
            #"Recurrence": pl.Utf8,
            "Identifications":pl.Utf8
        })
    

def adjust_library(
    library_path: Path,
    monomers_list: List[str],
) -> pl.DataFrame:
    
    """
    Adjusts the oligosaccharide library in function of a list of monomers:
    1. Removing entries with unauthorized monomers (>0 in non-selected columns)
    2. Removing entries where ALL authorized monomers = 0
    3. Selecting only relevant columns and sorting by mass

    Args:
        library_path: Path to the CSV library file
        monomers_list: List of authorized monomer column names (e.g., ["dHex", "Hex"])

    Returns:
        pl.LazyFrame: Processed library in lazy evaluation mode

    Note:
        Collect come back to lazy allow optimize the process by allowing the work only on the 
        desired columns and lines
    """
    
    column_selection = ["column_id"] + monomers_list + [
        "Neutral_unlabeled_mass","labeled_mass_H+","labeled_mass_Na+","labeled_mass_K+",#"Recurrence"
    ]

    library_oligo_unic = read_csv(library_path, sep=",")
    all_columns = library_oligo_unic.collect_schema().names()
    # Identify unauthorized columns (all columns not in selection)
    unauthorized_columns = [col for col in all_columns if col not in column_selection]
    
    # Create filter conditions
    # If unauthorized_columns is empty every columns are True and conserved
    unauthorized_filter_condition = (
        pl.all_horizontal(pl.col(m) == 0 for m in unauthorized_columns)
        if unauthorized_columns  
        else pl.lit(True) 
    )

    return (library_oligo_unic
        # Remove lines with at least 1 monomere unauthorized
        .filter(unauthorized_filter_condition)
        .select(column_selection)
        # Remove all lines where every authorized monomers = 0
        .filter(pl.any_horizontal(pl.col(m) != 0 for m in monomers_list))
        .sort(pl.col("Neutral_unlabeled_mass"), descending=False)
        .collect().lazy()
    )

def add_adduct_label(ref_oligo_df):
    
    
    pass


def compile_oligo_name(match_peaks: pl.LazyFrame, monomers: List[str]) -> Dict[str, pl.LazyFrame]:
    """
    Adds the Name of the oligosachharide to peaks of the lazyframe,
    containing the concatenation of non-zero monomers (format: valMono-valMono).
    
    Args:
        match_peaks: LazyFrame containing matching results
        monomers: List of monomer column names (e.g., ["dHex", "Hex", "HexA"])
    
    Returns:
        pl.LazyFrame: LazyFrames with the columns name
    """

    # Create concatenation expressions
    monomer_parts = []
    for col in monomers:
        # Uses pl.lit("") as default value to avoid None
        monomer_parts.append(
            pl.when(pl.col(col) > 0)
              .then(pl.col(col).cast(pl.Utf8) + col)
              .otherwise(pl.lit(""))
        )
    
    # Add non-empty value to prevent concatenation error
    # (a space that will be removed later)
    monomer_parts.append(pl.lit(" "))
    
    if match_peaks.select(pl.len()).collect().item() > 0:
        named_peaks =  match_peaks.with_columns(
            pl.concat_str(monomer_parts, separator="-")
            .str.replace_all(r'--+', '-')   # Remove double -
            .str.strip_chars('-')  # Remove extra -
            .alias("OligoName")
        ).with_columns(
            pl.concat_str(["OligoName", "Adduct"], separator="")
            .str.replace_all(r'\s+', '')
            .alias("Identifications")
        ).drop("OligoName")
    else:
        named_peaks = match_peaks

    return named_peaks


def save_dataframe_to_temp(file_name:str, lf: pl.DataFrame) -> Path:
    """
    Save the dataframe in tempory CSV file
    
    Args:
        file_name: Base name for the output file (will be prefixed with 'peak_')
        lf: Input DataFrame (or LazyFrame) to save

    Returns:
        Path: Path to the created CSV file
    """
    # Convert lf to df for writing
    df = lf.collect() if isinstance(lf, pl.LazyFrame) else lf
    
    # Generate output path and save
    file_name = f'peak_{file_name}.csv'
    file_path = cf.TEMP_DIR / file_name
    df.write_csv(file_path)

    return file_path


def create_zip(file_paths: Dict[str, Path]) -> BytesIO:
    """
    Create an optimized ZIP in memory with all files.

    Args:
        file_paths: Dictionary mapping file names to their Path objects

    Returns:
        BytesIO: ZIP file buffer ready for download
    """

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, path in file_paths.items():
            zip_file.write(path, f'peak_{file_name}.csv')
    return zip_buffer


def process_mass(mass: float, adduct:str, full_library: pl.LazyFrame, monomers_list: List[str], mass_col_name: str):
    """Traite une seule masse et renvoie le nom d'oligo correspondant.
    Use for parralelisation"""
    selected_oligo_lf = (full_library
                         .filter(pl.col(mass_col_name) == mass)
                         .with_columns(pl.lit(adduct).alias("Adduct")))
    selected_oligo_dict = compile_oligo_name({str(mass): selected_oligo_lf}, monomers_list)
    oligo_names = (
        selected_oligo_dict[str(mass)]
        .select(pl.col("Identifications"))
        .collect()
        .to_series()
        .to_list()
    )

    print(f"Masse : {str(mass)} -> Oligo duplicate liste: {'/'.join(oligo_names)}")
    return "/".join(oligo_names)


def check_shared_mass(
    match_dict: Dict[str, pl.LazyFrame],
    complete_library: Path,
    monomers_list: List[str],
    mass_col_name: str = "Neutral_unlabeled_mass",
    shared_col: str = "Recurrence",
    oglido_id_col_name: str = "Identifications",
    n_workers: Optional[int] = None,
) -> Dict[str, pl.LazyFrame]:
    """
    Parallélise le traitement des masses partagées (niveau intra-spectre).
    Chaque spectre est traité séquentiellement, mais les masses sont traitées en parallèle.
    """
    full_library = read_csv(complete_library)  # lazyframe
    new_dict = {}
    for spectrum, lf in match_dict.items():
        unique_lf = lf.filter(pl.col(shared_col) == "Unique")
        non_unique_lf = lf.filter(pl.col(shared_col) == "Shared")

        mass_list = non_unique_lf.select(pl.col(mass_col_name)).collect().to_series().to_list()
        adduct_list = non_unique_lf.select(pl.col("Adduct")).collect().to_series().to_list()

        print(f"{spectrum} start duplicate analysis")
        # ---- Parallélisation sur les masses ----
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            oligo_name_list = list(
                executor.map(
                    lambda m_a: process_mass(*m_a, full_library, monomers_list, mass_col_name),
                    zip(mass_list, adduct_list),
                )
            )
        print(f"{spectrum} End duplicate analysis")

        # Replace the initial column 
        non_unique_lf = (non_unique_lf.drop(oglido_id_col_name)
                         .with_columns(pl.Series(oglido_id_col_name, oligo_name_list)))

        # Fusion avec le LazyFrame initial
        print(f"Unique : {unique_lf.select(pl.col(oglido_id_col_name)).collect().to_series().to_list()}")
        print(f"Non Unique : {non_unique_lf.select(pl.col(oglido_id_col_name)).collect().to_series().to_list()}")

        full_lf = pl.concat([unique_lf, non_unique_lf])
        print(f'Full lf {full_lf.columns}')
        new_dict[spectrum] = full_lf

    return new_dict
        

def combine_lf(
    lf_1: pl.LazyFrame,
    lf_2: pl.LazyFrame,
    join_on: str = "query_mass",
    how: str = "right"
) -> pl.LazyFrame:
    """
    Combine two LazyFrames by joining on the 'query_mass' column

    Args:
        lf_1: LazyFrames with full anotation
        lf_2: LazyFrames with of detected peak
        join_on: Name of the joined column, defaults to 'querry_mass'
        how: Type of join to perform ('left', 'right', 'inner', 'outer'). Defaults to 'right'

    Returns:
        Combined LazyFrames, joined on 'query_mass'.
    """
    
    # Keep only necessary columns for the join
    cols_to_keep_1 = [join_on, "Identifications", "Formula", "Delta_M", "iso_score"]
    cols_to_keep_2 = ["m/z", join_on, "Intensity"]
    
    try:
        lf_2 = normalize_mz_column(lf_2, new_col=join_on)
        lf_1_simple = lf_1.select(cols_to_keep_1).sort(join_on)
        lf_2_simple = lf_2.select(cols_to_keep_2).sort(join_on)
        return lf_1_simple.join(lf_2_simple, on=join_on, how=how)

    except Exception as e:
        print(f"Join error: {str(e)}")
        # Return empty LazyFrame with correct schema
        return pl.LazyFrame(schema={
            "m:z": pl.Float64,
            "query_mass": pl.Int64,
            "Identifications": pl.Utf8,
            "Delta_M": pl.Float64,
            "Intensity": pl.Int64
        })


def execute_process(
    spectra_dict: Dict[str, pl.LazyFrame],
    monomers_list: List[str],
    monomer_df: pl.DataFrame,
    rm_threshold: int = 5,
    window: int = 10,
    prominence: float = 1e5,
    height_min: float = 1e4,
    dist_min: int = 15,
    number_candidate: int = 4,
    number_isotope: int = 4,
    scoring_method: str = "hybrid",
    progress=None
) -> Tuple[BytesIO, Dict[str, pl.LazyFrame]]:
    """
    Process MALDI spectra and generate a ZIP of results.

    Args:
        spectra_dict: Dictionary of LazyFrames (key = sample name)
        monomers_list: List of monomers used for oligo selection
        monomer_df: DataFrame with monomer compositions
        rm_threshold: Mass accuracy threshold (default: 5.0)
        window: Window for peak detection (default: 10)
        dist_min: Minimum distance between peaks (default: 0.5)
        prominence: Minimum height difference between peak and its neighboring samples (default: 2e5).
        number_candidate: Number of candidate for the isotope selection
        number_isotope: Number of theoretical isotopic peaks considered when computing the isotopic similarity score
        scoring_method: Method used to compute the similarity between experimental and theoretical isotopic patterns.        
        progress: Use to communicate with streamlit progress bar
        
    Returns:
        Tuple[BytesIO, Dict[str, pl.DataFrame]]:
            - BytesIO: Buffer containing ZIP of results
            - Dict[str, pl.LazyFrame]: Dictionary peaks

    Raises:
        Exception: If any error occurs during processing
    """
    try:
        n_workers=find_workers()
        logger = logging.getLogger(__name__)

        # Use only the selected monomers
        logger.info("[INFO] Initialization")
        ref_oligo_path = Path(__file__).parents[1]/"data"/"oligo_combinations.csv"
        ref_oligo_df = adjust_library(ref_oligo_path, monomers_list)
        ref_oligo_df = add_adduct_label(ref_oligo_df)

        file_paths, peaks_dict = {}, {}
        for i, (spectrum_name, spectrum_data) in enumerate(spectra_dict.items(), start=1):    
            
            logger.info(f"[INFO] Start analysis {spectrum_name}")
            
            # 1. Peak extraction with memory optimization
            logger.info(f"[INFO] {spectrum_name} -> Extract peaks")
            peaks_lazy = extract_and_detect_peaks(
                spectrum_data,
                window=window,
                prominence=prominence,
                height_min=height_min,
                dist_min=dist_min,
                n_workers=n_workers
            )

            # 2. Library comparison
            logger.info(f"[INFO] {spectrum_name} -> Library reference comparison")
            #match = compare_to_library(peaks_lazy, monomers_list, ratio_max=rm_threshold)
            match = match_oligos(
                peaks=peaks_lazy, 
                library=ref_oligo_df,
                threshold=rm_threshold,
                monomer_df=monomer_df,
                use_label_3aq=True,
                top_n= number_candidate, 
                number_isotope = number_isotope,
                scoring_method = scoring_method)

            # 3. Oligonucleotide name compilation
            logger.info(f"[INFO] {spectrum_name} -> Name atttribution")
            match = compile_oligo_name(match, monomers_list)

            # 4. Add Shared mass names
            # match = check_shared_mass(
            #     match_dict = match,
            #     complete_library =Path(__file__).parents[1]/"data"/"oligo_combinations.csv",
            #     monomers_list= monomers_list,
            #     mass_col_name = "Neutral_unlabeled_mass",
            #     shared_col = "Recurrence",
            #     n_workers= n_workers)

            # 4. Save to temporary file
            logger.info(f"[INFO] {spectrum_name} -> Save temp file")
            file_paths[spectrum_name] = save_dataframe_to_temp(spectrum_name, match)

            # 6. Combine the lfs to have the non identified peak
            peaks_dict[spectrum_name] = combine_lf(match, peaks_lazy, how="right")

            if progress:
                progress.update(i)

        # 7. Create ZIP in memory
        logger.info("[INFO]: Creation of ZIP files")
        zip_buffer = create_zip(file_paths)
        
        # 8. Explicit cleanup
        del peaks_lazy, match
        gc.collect()

        logger.info("[INFO]: End Analysis generation of spectrum")
        return zip_buffer, peaks_dict

    except Exception as e:
        print(f"Error in execute_process: {str(e)}")
        raise


def cleanup() -> None:
    """Remove the temporary directory and all its contents."""
    shutil.rmtree(cf.TEMP_DIR)


if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )