"""
    Analysis of the MALDI spectrum
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

import os
import pandas as pd
import polars as pl
import plotly.express as px

from pathlib import Path

# TODO: Change spectrum name f'Spectrum_{i} in parse_spectra by the file name
# TODO: Add prgress bar


def read_spectrum(file: str, sep: str = ','):
    """
    Lazy reading of the spectrum file....

    Args:
        file (str): _description_
        sep (str, optional): _description_. Defaults to ','.

    Returns:
        _type_: _description_
    """
    return (
        pl.scan_csv(file, separator=sep)
        .select([
            pl.col("m/z"),       # Colonne des masses (clé de jointure)
            pl.col("Intensity")  # Colonne d'intensité renommée
        ])
    )


def parse_spectra(dir: str, files:list):
    spectra = {}
    for i, file in enumerate(files):
        spectrum_path = os.path.join(dir, file)
        spectra[f'Spectrum_{i}'] = read_spectrum(spectrum_path)
    return spectra


def plot_spectrum(spectrum):
    fig = px.line(spectrum, x=list(spectrum['m/z']), y=list(spectrum['Intensity']))
    fig.show



if __name__ == "__main__":
    
    dir = "data/standards/"
    files = [f for f in os.listdir(dir) if f.endswith(".csv")]
    spectra = parse_spectra(dir, files)
    plot_spectrum(spectra['Spectrum_0'].collect())
