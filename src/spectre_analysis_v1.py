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

from pathlib import Path
from joblib import Memory
from scipy.signal import find_peaks as fp


# TODO: Change spectrum name f'Spectrum_{i} in parse_spectra by the file name
memory = Memory("cache_dir", verbose=0)

@memory.cache
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
    )


def parse_spectra(files, dir: str = None):
    dfs = []
    for file in files:
        if isinstance(file, str):
            file = os.path.join(dir, file)
            spectre_name = file.replace('.csv', '').replace(' ', '_')
        else: # if uploader file from streamlit
            spectre_name = file.name.replace('.csv', '').replace(' ', '_')
        dfs.append(read_spectrum(file, spectre_name))
    
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


def find_all_peaks(intensity: np.ndarray, interval, prominence=None, height=None, distance=15):
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





#def plot_spectrum()


if __name__ == "__main__":
    
    dir = "data/standards/"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(files, dir)
    print(spectra.head(5))