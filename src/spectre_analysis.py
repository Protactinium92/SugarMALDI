"""
    Analysis of the MALDI spectrum
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

import polars as pl
import os

from pathlib import Path

# TODO: Change spectrum name f'Spectrum_{i} in parse_spectra by the file name


def read_spectrum(file: str, spectre_name: str, sep: str = ','):
     return (
        pl.scan_csv(file, separator=sep)
        .select([
            pl.col("m/z"),          # Colonne des masses (clé de jointure)
            pl.col("Intensity").alias(spectre_name)        # Colonne d'intensité renommée
        ])
    )



def parse_spectra(dir: str, files:list):
    dfs = []
    for i, file in enumerate(files):
        spectrum_path = os.path.join(dir, file)
        dfs.append(read_spectrum(spectrum_path, f'Spectrum_{i}'))
    
    # Fusionne tous les DataFrames
    df_combined = dfs[0]
    for df in dfs[1:]:
        df_combined = df_combined.join(df, on="m/z", how="full", coalesce=True)

    # Exécute le plan et charge les données
    df_combined = df_combined.collect()

    # Remplace les valeurs manquantes par 0 (après collect)
    return df_combined.fill_null(0)


#def plot_spectrum()


if __name__ == "__main__":
    
    dir = "data/standards/"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(dir, files)
    print(spectra.head(5))
