"""
    Analysis of the MALDI spectrum

    THIS FILE IS A SECOND OPTION TO PARSE THE SPECTRA
    everything is store in a dataframe whereaese the second file
    
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

import polars as pl
import os
import numpy as np


from typing import Dict, List
from pathlib import Path
from scipy.signal import find_peaks as fp


SCALE = 1_000_000  # 1e6, adapte si besoin


def read_spectrum(file: str, spectre_name: str, sep: str = ','):
    """
    Lazy reading of the spectrum file....

    Args:
        file (str): _description_
        spectre_name (str): _description_
        sep (str, optional): _description_. Defaults to ','.

    Returns:
        _type_: _description_
    """
    return (
        pl.scan_csv(file, separator=sep)
        .select([
            pl.col("m/z"),          # Colonne des masses (clé de jointure)
            pl.col("Intensity").alias(spectre_name)        # Colonne d'intensité renommée
        ])
        .sort(by='m/z')
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


def normalize_mz_column(lf: pl.LazyFrame, mz_col="m/z", new_col="mz_int"):
    """
    - ndigits : si tu veux arrondir en décimales avant le scaling (ex: 6)
    - crée une colonne entière 'mz_int' et garde la colonne originale
    """
    mz = pl.col(mz_col).cast(pl.Float64)  # assure float64 avant tout
    return lf.with_columns(
        (mz * SCALE).round().cast(pl.Int64).alias(new_col)
    )


def parse_spectra(files, dir: str = None):
    dfs = []
    for file in files:
        if isinstance(file, str):
            file = os.path.join(dir, file)
            spectre_name = file.replace('.csv', '').replace(' ', '_')
        else: # if uploader file from streamlit
            spectre_name = file.name.replace('.csv', '').replace(' ', '_')
        df = read_spectrum(file, spectre_name)
        df = df.sort("m/z")
        dfs.append(df)
    del df
    # Fusionne tous les DataFrames
    df_combined = dfs[0]
    for df in dfs[1:]:
        df_combined = df_combined.join(df, on="m/z", how="full", coalesce=True)

    # Remplace les valeurs manquantes par 0 (après collect)
    return df_combined.collect().fill_null(0)


def make_window(df, column_name: str, step: int | float):
    """
    Divide the dataframe into different window in fonction of the desired column

    Args:
        df (pl.Dataframe | pl.Lazyframe): _description_
        column_name (str): _description_
        step (int | float): _description_

    Returns:
        _type_: _description_
    """
    return df.with_columns(
        (pl.col(column_name) // step).cast(pl.Int32).alias("window_id")
    )


def find_all_peaks(intensity: np.ndarray, interval, prominence=2e5, height=None, distance=15):
    # in fp the distance correspond to the indice of the list
    # Convert real distance value m/z into the distance indice
    # max 1 avoid the risk to have distance < 1 indice
    distance = max(1, int(distance / interval))
    # Renvoie les indices des peaks
    peaks, _ = fp(intensity,
                  distance=distance,
                  height=height,
                  prominence=prominence)
    return peaks.tolist()


def extract_peaks_all_windows(df: pl.DataFrame, mz_col: str, intensity_col: str, step: float, dist_min: float) -> pl.DataFrame:
    """
    Découpe un spectre en fenêtres et extrait les pics par fenêtre.
    Optimisé pour de très gros DataFrames (millions de lignes).
    """
    # Découper en fenêtres
    df = make_window(df, mz_col, step)

    # Extraire en NumPy pour vectorisation
    mz = df[mz_col].to_numpy()
    intensity = df[intensity_col].to_numpy()
    windows = df["window_id"].to_numpy()

    all_idx = []

    # Boucle par fenêtre (unique IDs, beaucoup plus rapide que map_groups)
    for w in np.unique(windows):
        mask = windows == w
        sub_idx = np.where(mask)[0]

        if len(sub_idx) < 2:
            continue

        # Intervalle local (car irrégulier)
        interval = np.mean(np.diff(mz[sub_idx]))

        # Trouver les pics dans cette fenêtre
        peaks = find_all_peaks(intensity[sub_idx], interval, distance=dist_min)

        # Convertir indices locaux → indices globaux
        all_idx.extend(sub_idx[peaks])

    # Construire DataFrame final avec seulement les pics
    return df[[mz_col, intensity_col]][all_idx,:]


def extract_and_detect_peaks(
    spectra: pl.DataFrame,
    mz_col_name: str,
    intensity_cols: list,
    window: float,
    dist_min: float,
) -> Dict[str, pl.DataFrame]:
    """
    Extracts and detects peaks for all intensity columns in the spectra DataFrame.

    Args:
        spectra: Polars DataFrame containing m/z and intensity columns.
        mz_col_name: Name of the m/z column.
        intensity_cols: List of spectra
        window: Window size for peak extraction.
        dist_min: Minimum distance between peaks.
        top_k: Number of top peaks to keep per window.

    Returns:
        Dictionary mapping each intensity column name to its detected peaks DataFrame.
    """

    if not intensity_cols:
        raise ValueError("No intensity columns found in spectra (except m/z).")

    peaks_dict = {}

    for column in intensity_cols:
    
        spectrum_peaks = extract_peaks_all_windows(
            spectra, mz_col_name, column, window, dist_min
        )
        peaks_dict[column] = spectrum_peaks
    del column
    return peaks_dict


def match_oligos(
    dico_peak: Dict[str, pl.DataFrame],
    library: pl.DataFrame,
    offset: float
) -> List[pl.DataFrame]:
    """
    Match each mass in a spectrum to oligomers in a library within ±offset.
    Returns a dictionary of DataFrames (one per spectrum) with matches, including:
    - All original library columns (except temporary ones)
    - Added columns: 'query_mass', 'Adduct', 'mass_diff'

    Args:
        dico_peak: Dict of spectra (key: spectrum name, value: DataFrame with 'm/z' column).
        library: DataFrame with columns 'Oligo_H+', 'Oligo_Na+', 'Oligo_K+'.
        offset: Mass tolerance for matching.

    Returns:
        Dict[str, pl.DataFrame]: {spectrum_name: matches_df}
    """
    results = {}

    for spectrum_name, spectrum_df in dico_peak.items():
        # Extract mass list (assuming column 0 is 'm/z')
        mass_list = spectrum_df[spectrum_df.columns[0]].to_numpy()
        intensity_list = spectrum_df[spectrum_df.columns[1]].to_numpy()
        matched_oligos = []

        for mass, intensity in zip(mass_list, intensity_list):
            # --- Step 1: Filter candidates ---
            candidates = library.filter(
                (pl.col("Oligo_H+").is_between(mass - offset, mass + offset)) |
                (pl.col("Oligo_Na+").is_between(mass - offset, mass + offset)) |
                (pl.col("Oligo_K+").is_between(mass - offset, mass + offset))
            )

            # --- Step 2: Compute differences ---
            candidates_with_diff = candidates.with_columns(
                pl.lit(mass).alias("query_mass"),
                pl.lit(intensity).alias("intensity"),
                (pl.col("Oligo_H+") - mass).abs().alias("diff_H"),
                (pl.col("Oligo_Na+") - mass).abs().alias("diff_Na"),
                (pl.col("Oligo_K+") - mass).abs().alias("diff_K")
            )

            # --- Step 3: Find best match ---
            best_match = (
                candidates_with_diff
                .with_columns(pl.min_horizontal("diff_H", "diff_Na", "diff_K").alias("mass_diff"))
                .filter(pl.col("mass_diff") == pl.col("mass_diff").min())
                .with_columns(
                    pl.when(pl.col("diff_H") == pl.col("mass_diff")).then(pl.lit("H+"))
                     .when(pl.col("diff_Na") == pl.col("mass_diff")).then(pl.lit("Na+"))
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
    # del spectrum_name, spectrum_df, matched_oligos
    return results


def compare_to_library(dico_peak, offset, unique:bool = True):
   
    library_oligo_unic = read_csv(Path(__file__).parents[1]/"data"/"oligo_combinations_clean.csv", sep=",")
    library_oligo_unic = library_oligo_unic.sort(pl.col("Oligo_mass"), descending=False)

    return match_oligos(dico_peak, library_oligo_unic, offset)


if __name__ == "__main__":
    
    dir = "data/standards/"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(files, dir)
    print(spectra.head(5))

    library_oligo_unic = read_csv(Path(__file__).parents[1]/"data"/"oligo_combinations_clean.csv", sep=",")
    library_oligo_unic = library_oligo_unic.sort(pl.col("Oligo_mass"), descending=False)

    print(library_oligo_unic.collect().head(5))

