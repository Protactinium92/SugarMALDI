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

from pathlib import Path

# TODO: Change spectrum name f'Spectrum_{i} in parse_spectra by the file name


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


#def plot_spectrum()


if __name__ == "__main__":
    
    dir = "data/standards/"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(files, dir)
    print(spectra.head(5))