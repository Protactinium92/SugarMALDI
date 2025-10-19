"""
    Analysis of the MALDI spectrum

    THIS FILE IS A 1st TO PARSE THE SPECTRA


"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

import os
import pandas as pd
import polars as pl
import plotly.express as px
import numpy as np

from pathlib import Path
from typing import Dict, Union, Optional
from numpy.typing import NDArray
from scipy.signal import find_peaks as fp
from concurrent.futures import ProcessPoolExecutor



SCALE = 1_000_000  # 1e6 to convert float mass in int


def normalize_mz_column(
    lf: pl.LazyFrame,
    mz_col: str = "m/z",
    new_col: str = "mz_int"
) -> pl.LazyFrame:
    """
    Normalizes the m/z column by scaling and converting to integer values.

    This function is designed for MALDI-TOF spectra preprocessing. It scales the m/z values
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

    Raises:
        pl.exceptions.ComputeError: If the input column cannot be cast to Float64.
        ValueError: If the input column does not exist in the LazyFrame.
    """
    if mz_col not in lf.columns:
        raise ValueError(f"Column '{mz_col}' not found in the input LazyFrame.")

    mz = pl.col(mz_col).cast(pl.Float64)  # Ensure float64 before scaling
    return lf.with_columns(
        (mz * SCALE).round().cast(pl.Int64).alias(new_col)
    )


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
    dir: Optional[str] = None
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
        
        spectra[spectrum_name] = read_spectrum(file)
    del file
    return spectra


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
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

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
                    and its neighboring samples). Default is 2e5, suitable for MALDI-TOF data.
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


def _process_window(args: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float, float]
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
    dist_min: float = args[4]

    # Skip processing if window contains insufficient data points
    if len(window_indices) < 2:
        return []

    # Find peaks in current window using the provided peak detection function
    peaks: NDArray[np.int64] = find_all_peaks(
        intensity[window_indices],  # Intensity values for current window
        local_interval,             # Average spacing between m/z values in this window
        distance=dist_min           # Minimum required distance between peaks
    )

    return window_indices[peaks].tolist()


def extract_peaks_all_windows(
        df: pl.LazyFrame,
        mz_col_name: str,
        intensity_col_name: str,
        step: float,
        dist_min: float,
        n_workers: Optional[int] = None
) -> pl.LazyFrame:
    """
    Extracts peaks from MALDI-TOF spectrum data using parallel window processing.

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
        n_workers: Number of parallel workers to use. If None, defaults to os.cpu_count().

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
    mz: NDArray[np.float64] = df[mz_col_name].to_numpy()
    intensity: NDArray[np.float64] = df[intensity_col_name].to_numpy()
    windows: NDArray[np.int64] = df["window_id"].to_numpy()

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
        local_interval: float = np.mean(np.diff(mz[window_indices]))

        # Package arguments for parallel processing
        window_args.append((
            mz,                # Full m/z array (shared across all workers)
            intensity,         # Full intensity array (shared across all workers)
            window_indices,    # Indices specific to this window
            local_interval,    # Calculated spacing for this window
            dist_min           # Minimum peak distance parameter
        ))

    # Set default number of workers if not specified
    if n_workers is None:
        n_workers = os.cpu_count()
        # Leave one CPU free for system processes if we have many cores
        if n_workers > 1:
            n_workers -= 1

    # Execute peak detection in parallel
    # ProcessPoolExecutor creates separate processes to avoid Python's GIL limitations
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_idx: List[int] = []
        # Process each window and collect results
        for result in executor.map(_process_window, window_args):
            all_idx.extend(result)

    # Return a new LazyFrame containing only peak data
    # Using slice() to preserve original indices and enable lazy evaluation
    return df.select([mz_col_name, intensity_col_name]).slice(all_idx)


# def extract_peaks_all_windows(
#         df: pl.LazyFrame,
#         mz_col_name: str,
#         intensity_col_name: str,
#         step: float,
#         dist_min: float
#         ) -> pl.LazyFrame:
#     """
#     Découpe un spectre en fenêtres et extrait les pics par fenêtre.
#     Optimisé pour de très gros DataFrames (millions de lignes).
#     """
#     # Découper en fenêtres
#     df = make_window(df, mz_col_name, step)

#     # Extraire en NumPy pour vectorisation
#     mz = df[mz_col_name].to_numpy()
#     intensity = df[intensity_col_name].to_numpy()
#     windows = df["window_id"].to_numpy()

#     all_idx = []

#     # Boucle par fenêtre (unique IDs, beaucoup plus rapide que map_groups)
#     for w in np.unique(windows):
#         mask = windows == w
#         sub_idx = np.where(mask)[0]

#         if len(sub_idx) < 2:
#             continue

#         # Intervalle local (car irrégulier)
#         interval = np.mean(np.diff(mz[sub_idx]))

#         # Trouver les pics dans cette fenêtre
#         peaks = find_all_peaks(intensity[sub_idx], interval, distance=dist_min)

#         # Convertir indices locaux → indices globaux
#         all_idx.extend(sub_idx[peaks])

#     # Construire DataFrame final avec seulement les pics
#     return df[[mz_col_name, intensity_col_name]][all_idx,:]





def extract_and_detect_peaks(
    spectra: Dict[str, pl.DataFrame],
    window: float,
    dist_min: float,
) -> Dict[str, pl.DataFrame]:
    """
    Extracts and detects peaks for all intensity columns in the spectra DataFrame.

    Args:
        spectra: Dictionary of all spectra store as pl.Dataframe.
        window: Window size for peak extraction.
        dist_min: Minimum distance between peaks.
        top_k: Number of top peaks to keep per window.

    Returns:
        Dictionary mapping each intensity column name to its detected peaks DataFrame.
    """

    if not spectra:
        raise ValueError("No spectrum found")

    peaks_dict = {}

    for spectrum_name, spectum_data in spectra.items():
    
        spectrum_peaks = extract_peaks_all_windows(
            spectum_data, spectum_data.columns[0], spectum_data.columns[1], window, dist_min
        )
        peaks_dict[spectrum_name] = spectrum_peaks
    del spectrum_name, spectum_data
    return peaks_dict

















if __name__ == "__main__":
    
    dir = "data/Standard_Gums_Calibrated_files"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(files, dir=dir)
    first_spectrum_name = next(iter(spectra.keys()))  # 1st key
    first_spectrum = spectra[first_spectrum_name].collect()
    print(first_spectrum.head(5))
   
