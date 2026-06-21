"""
Data Loading Utilities for SugarMALDI
=====================================

Overview
--------
This module provides utility functions for loading reference data used by
the SugarMALDI application.

It includes:
- Elemental mass table loading.
- Cached CSV file reading using Polars lazy execution.
- Streamlit-compatible caching to avoid repeated disk access.

Functions
---------
- element_mass:
    Load elemental monoisotopic masses from the reference mass table.

- read_csv:
    Read a delimited text file as a Polars LazyFrame.

Notes
-----
- All loading functions are cached using `st.cache_data` to improve
  application performance.
- CSV files are loaded lazily via `polars.scan_csv`, allowing query
  optimization and deferred execution.
"""

__author__ = "Pierre-Alexandre HO"

import streamlit as st
import polars as pl

from pathlib import Path


@st.cache_data
def element_mass() -> dict[str, float]:
    """
    Load elemental masses from the reference mass table.

    The reference file is expected to contain one element per line
    formatted as:

        Element<TAB>Mass

    Returns
    -------
    dict[str, float]
        Dictionary mapping element symbols to their corresponding
        monoisotopic masses.

    Notes
    -----
    The result is cached using Streamlit to avoid repeated file reads.
    """
    path_file = Path(__file__).parents[1]/"data"/"mass_elements.txt"
    element_dict = {}
    with open(path_file, "r") as source:
        for line in source:
            element, mass = line.strip().split("\t")
            element_dict[element] = float(mass)
    return element_dict


@st.cache_data
def read_csv(file: Path, sep: str = ",") -> pl.LazyFrame:
    """
    Read a delimited text file as a Polars LazyFrame.

    Parameters
    ----------
    file : Path
        Path to the file to read.
    sep : str, default=","
        Column separator used in the file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representing the file contents.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    The file is loaded lazily using `polars.scan_csv`, which delays
    execution until a collection operation is performed.
    The result is cached using Streamlit.
    """
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    return pl.scan_csv(file, separator=sep)

    
if __name__=="__main__":
    element = element_mass()
    print(len(element))
    for e, m in element.items():
        print(f'{e}: {m}')