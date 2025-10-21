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
import tempfile
import shutil
import zipfile
import gc

from io import BytesIO
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, List
from numpy.typing import NDArray
from scipy.signal import find_peaks as fp
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
TEMP_DIR = Path(tempfile.mkdtemp())  # Dossier temporaire pour stocker les fichiers
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
    """
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
        print(f'window ID {window_id}')
        print(window_indices)
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
    # Using filter() to preserve original indices and enable lazy evaluation
    #  pl.int_range(0, pl.len()) create column of idx
    #  is_in apply a boolean mask
    return df.select([mz_col_name, intensity_col_name]).filter(
        pl.int_range(0, pl.len()).is_in(all_idx))


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

    Returns:
        Dictionary mapping each intensity column name to its detected peaks DataFrame.
    """

    if not spectra:
        raise ValueError("No spectrum found")
    
    peaks_dict = {}

    for spectrum_name, spectum_data in spectra.items():
        col_names = spectum_data.collect_schema().names()
        mz_col, intensity_col = col_names[0], col_names[1]
        spectrum_peaks = extract_peaks_all_windows(
            spectum_data, mz_col, intensity_col, window, dist_min
        )
        peaks_dict[spectrum_name] = spectrum_peaks

    del spectrum_name, spectum_data
    return peaks_dict


def match_oligos(
    dico_peak: Dict[str, pl.DataFrame],
    library: pl.DataFrame,
    offset: float,
    threshold: int,
) -> Dict[str, pl.DataFrame]:
    """
    Match each mass in a spectrum to oligomers in a library within ± offset.
    Returns a dictionary of DataFrames (one per spectrum) with matches, including:
    - All original library columns (except temporary ones)
    - Added columns: 'query_mass', 'Adduct', 'mass_diff'

    Args:
        dico_peak: Dict of spectra (key: spectrum name, value: spectra LazyFrame).
        library: DataFrame with columns 'Oligo_H+', 'Oligo_Na+', 'Oligo_K+'.
        offset: Mass tolerance window to speedup the matching.
        threshold: Mass accuracy tolerence for matching: abs((theory-real)/theory) in "ppm"

    Returns:
        Dict[str, pl.DataFrame]: {spectrum_name: matches_df}
    """
    results = {}
    for adduct in ['Oligo_H+', 'Oligo_Na+', 'Oligo_K+']:
        library = normalize_mz_column(
        lf = library,
        mz_col=adduct,
        new_col=f"{adduct}_int"
    )
    for spectrum_name, spectrum_df in dico_peak.items():
        
        # Normalyse to ppm and create int
        spectrum_df = normalize_mz_column(
            lf = spectrum_df,
            mz_col="m/z",
            new_col="mz_int"
        )

        mass_list = spectrum_df.select('mz_int').collect().to_numpy()
        intensity_list = spectrum_df.select('Intensity').collect().to_numpy()
        matched_oligos = []

        for mass, intensity in zip(mass_list, intensity_list):

            # --- Step 1: Filter candidates ---
            candidates = library.filter(
                (pl.col("Oligo_H+_int").is_between(mass - offset, mass + offset)) |
                (pl.col("Oligo_Na+_int").is_between(mass - offset, mass + offset)) |
                (pl.col("Oligo_K+_int").is_between(mass - offset, mass + offset))
            )

            if candidates.select(pl.len()).collect().item() == 0:
                continue

            #print(candidates.collect())

            # --- Step 2: Compute the mass ratio ---
            candidates_with_diff = candidates.with_columns(
                pl.lit(int(mass)).cast(pl.Int64).alias("query_mass"),
                pl.lit(float(intensity)).alias("intensity"),
                ((pl.col("Oligo_H+_int") - mass)/pl.col("Oligo_H+_int")).abs().alias("diff_H"),
                ((pl.col("Oligo_Na+_int") - mass)/pl.col("Oligo_Na+_int")).abs().alias("diff_Na"),
                ((pl.col("Oligo_K+_int") - mass)/pl.col("Oligo_K+_int")).abs().alias("diff_K")
            )

            # --- Step 3: Find best match ---
            best_match = (
                candidates_with_diff
                .with_columns(pl.min_horizontal("diff_H", "diff_Na", "diff_K").alias("Delta_M"))
                .filter(pl.col("Delta_M") == pl.col("Delta_M").min())  # Select 1 lign of the library
                .filter(pl.col("Delta_M") <= threshold)
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

            matched_oligos.append(best_match)

        # Concatenate all matches for this spectrum
        if matched_oligos:
            results[spectrum_name] = pl.concat(matched_oligos)
        else:
            results[spectrum_name] = pl.DataFrame(schema={
                "query_mass": pl.Float64,
                "Adduct": pl.Utf8,
                "mass_diff": pl.Float64
            })
    del spectrum_name, spectrum_df, matched_oligos
    return results


def compare_to_library(dico_peak, offset, ratio_max):
   
    library_oligo_unic = read_csv(Path(__file__).parents[1]/"data"/"oligo_combinations_clean.csv", sep=",")
    library_oligo_unic = library_oligo_unic.sort(pl.col("Oligo_mass"), descending=False)

    return match_oligos(dico_peak, library_oligo_unic, offset, ratio_max)


def compile_oligo_name(match_dict: Dict[str, pl.LazyFrame], monomers: List[str]) -> Dict[str, pl.LazyFrame]:
    """
    Ajoute une colonne 'OligoName' à chaque LazyFrame du dictionnaire,
    contenant la concaténation des monomères non nuls (format: valMono-valMono).

    Args:
        match_dict: Dictionnaire {nom_spectre: LazyFrame} contenant les résultats de matching
        monomers: Liste des noms de colonnes des monomères (ex: ["dHex", "Hex", "HexA"])

    Returns:
        Dict[str, pl.LazyFrame]: Nouveau dictionnaire avec les LazyFrames modifiés
    """

    # 3. Création des expressions de concaténation
    monomer_parts = []
    for col in monomers:
        # Utilise pl.lit("") comme valeur par défaut pour éviter les None
        monomer_parts.append(
            pl.when(pl.col(col) > 0)
              .then(pl.col(col).cast(pl.Utf8) + col)
              .otherwise(pl.lit(""))
        )

    # 4. Ajout d'une valeur non vide pour éviter l'erreur de concaténation
    # (un espace qui sera supprimé plus tard)
    monomer_parts.append(pl.lit(" "))

    new_dict = {}
    for spectrum_name, spectrum_df in match_dict.items():
        # Concaténation avec filtre des chaînes vides
        new_df = spectrum_df.with_columns(
            pl.concat_str(monomer_parts, separator="-")
              .str.replace_all(r'--+', '-')   # Supprime les doubles -
              .str.strip_chars('-')  # Supprime les - en trop
              .alias("OligoName")
        ).with_columns(
            pl.concat_str(["OligoName", "Adduct"], separator="")
            .str.replace_all(r'\s+', '')
            .alias("Oligo-Adduct")
        ).drop("OligoName")
        new_dict[spectrum_name] = new_df

    del new_df
    return new_dict




def save_dataframes_to_temp(df_dict: Dict[str, pl.DataFrame]) -> Dict[str, Path]:
    """
    Save the dataframe in tempory file (for cloud)
    Arg:

    Returns:
        Dict[str, Path]: Path of created files.
    """
    file_paths = {}
    for filename, df in df_dict.items():
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        filename = f'peak_{filename}.csv'
        path = TEMP_DIR / filename
        df.write_csv(path)
        file_paths[filename] = path
    return file_paths


def create_zip(file_paths: Dict[str, Path]) -> BytesIO:
    """Crée un ZIP en mémoire contenant tous les fichiers."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename, path in file_paths.items():
            zip_file.write(path, filename)
    return zip_buffer


def cleanup():
    shutil.rmtree(TEMP_DIR)




def execute_process(
    spectra: Dict[str, pl.DataFrame],
    monomers_list: List[str],
    offset: int = 10_000_000,
    rm_threshold: int = 5,
    window: int = 10,
    dist_min: int = 15
) -> BytesIO:
    """
    Process MALDI spectra and generate a ZIP of results.

    Args:
        spectra: Dictionary of DataFrames (key = sample name)
        monomers_list: List of monomers for oligo name compilation
        offset: Mass comparison offset (default: 10,000,000)
        rm_threshold: Mass accuracy threshold (default: 5.0)
        mz_col_name: Name of m/z column (default: "m/z")
        other_cols: Additional columns to keep (default: None)
        window: Window for peak detection (default: 10)
        dist_min: Minimum distance between peaks (default: 0.5)

    Returns:
        BytesIO: Buffer containing ZIP of results

    Raises:
        Exception: If any error occurs during processing
    """
    try:
        # 1. Peak extraction with memory optimization
        peaks_dict = extract_and_detect_peaks(
            spectra,
            window=window,
            dist_min=dist_min
        )

        # 2. Library comparison
        match = compare_to_library(peaks_dict, offset=offset, ratio_max=rm_threshold)

        # 3. Oligonucleotide name compilation
        match = compile_oligo_name(match, monomers_list)

        # 4. Save to temporary directory
        file_paths = save_dataframes_to_temp(match)

        # 5. Create ZIP in memory
        zip_buffer = create_zip(file_paths)

        # 6. Explicit cleanup
        del peaks_dict, match
        gc.collect()

        return zip_buffer

    except Exception as e:
        print(f"Error in execute_process: {str(e)}")
        raise










if __name__ == "__main__":
    
    dir = "data/std_test"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(files, dir=dir)
    first_spectrum_name = next(iter(spectra.keys()))  # 1st key
    first_spectrum = spectra[first_spectrum_name].collect()
    print(first_spectrum.head(5))

    peaks_dict = extract_and_detect_peaks(
        spectra,
        window=500,
        dist_min=15)
    
    print(f'peak dict 1 \n{peaks_dict[first_spectrum_name].collect()}')

    match = compare_to_library(peaks_dict, 500_000_000, 5)
    print(f'Match dict 1 \n{match[first_spectrum_name].collect()}')

    monomers = read_csv(Path(__file__).parents[1]/"data"/"monomers.csv", sep=";")
    monomers_list = monomers.select(pl.col("Notation")).collect().to_series().to_list()
    match = compile_oligo_name(match, monomers_list)
    print(f'Match dict 2S \n{match[first_spectrum_name].collect()}')
