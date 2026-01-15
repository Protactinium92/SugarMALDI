"""
Calibration functions 
"""


__author__  = "Pierre-Alexandre HO"
__date__    = "2026-01-14"
__version__ = "1.0"


import spectre_analysis as sp_ana
import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import Dict, Union, Optional, Tuple, List

import datashader as ds
from datashader import transfer_functions as tf
import colorcet as cc


def build_mass_difference_map(
    peaks: pl.LazyFrame,
    oligo_ref: pl.LazyFrame,
    adducts: List[str] = ['labeled_mass_H+', 'labeled_mass_Na+', 'labeled_mass_K+'],
    threshold_ppm: float = 500.0
) -> pl.LazyFrame:
    """
    Build a Mass Difference Map (MDiM) by computing delta_m (ppm) between experimental peaks
    and theoretical oligomers for each adduct.

    Args:
        peaks: LazyFrame containing experimental peaks with 'mz_int' column (integer m/z * 1e6).
        oligo_ref: LazyFrame containing theoretical oligomers with adduct mass columns.
        adducts: List of adduct columns to process (default: H+, Na+, K+).
        threshold_ppm: Mass tolerance in ppm (default: 50.0).


    Returns:
        pl.LazyFrame: Concatenated MDiM with columns:
            - All original columns from peaks
            - theoretical_mass (from oligo_ref)
            - delta_m (mass difference in ppm)
            - adduct (adduct type)

    Note:
        - Uses cross join (filtered in subsequent steps)
        - Avoids .collect() until final concatenation
        - Follows your preference for '_int' suffix and explicit typing
    """
    mdim_list = []
    for adduct in adducts:
        adduct_int = f"{adduct}_int"
        mdim = (
            peaks
            .join(
                oligo_ref.select(pl.col(adduct_int).alias("theoretical_mass")),
                how="cross")
            .with_columns(
                delta_m = (
                    (pl.col("mz_int") - pl.col("theoretical_mass")) / pl.col("theoretical_mass") * 1e6)
                    .alias("delta_m"))
            .filter(pl.col("delta_m").abs() <= threshold_ppm)
        )
        mdim_list.append(mdim)

    return pl.concat(mdim_list) if mdim_list else peaks.lazy().limit(0)


def calculate_error_table(
    mdim_df: pl.LazyFrame,
    mass_col: str = "theoretical_mass",
    delta_col: str = "delta_m",
    window_size: int = 100  # Nombre de masses voisines pour le lissage
) -> pl.LazyFrame:
    """
    Calcule la correction sans grille, en utilisant uniquement les masses mesurées.
    Args:
        mdim_df: Résultat de build_mass_difference_map (avec colonnes mass_col, delta_col)
        window_size: Taille de la fenêtre pour le lissage mobile (en nombre de masses)
    Returns:
        DataFrame avec colonnes: [mass_col, ppm_correction]
    """
    # 1. Trie par masse théorique et calcule une correction lissée
    return (
        mdim_df
        .sort(mass_col) # Mandatory for rooling
        .with_columns(
            # Rolling Average
            pl.col(delta_col)
            .rolling_mean(window_size=window_size, min_samples=1)
            .alias("ppm_correction"))
        .select([mass_col, "ppm_correction"])
    )


def interpolate_corrections(
    spectrum: pl.LazyFrame,  # TOUT le spectre (m/z + Intensity)
    error_table: pl.LazyFrame,  # Pics assignés + ppm_correction
    experimental_mass_col: str = "mz_int",
    new_col_name: str = "new_m/z",
    theorical_mass_col: str = "theoretical_mass"
) -> pl.LazyFrame:
    """
    Interpole les corrections ppm pour TOUTES les masses du spectre.
    Args:
        spectrum: Toutes les masses à corriger (même le bruit).
        error_table: Pics assignés avec leur ppm_correction.
    Returns:
        LazyFrame avec une colonne `ppm_correction` interpolée pour toutes les masses.
    """
    # 1. Collecter les données (nécessaire pour numpy)
    spectrum = sp_ana.normalize_mz_column(
        lf = spectrum,
        mz_col="m/z",
        new_col="mz_int")
    masses = spectrum.select(experimental_mass_col).collect().to_numpy().ravel()
    known_masses = error_table.select(pl.col(theorical_mass_col)).collect().to_numpy().ravel()
    known_corrections = error_table.select(pl.col("ppm_correction")).collect().to_numpy().ravel()

    # ---- Sécurisation : tri obligatoire ----
    order = np.argsort(known_masses)
    known_masses = known_masses[order]
    known_corrections = known_corrections[order]

    # 2. Interpolation linéaire
    corrections_interp = np.interp(
        masses,  # Où interpoler (toutes les masses)
        known_masses,  # Masses connues (assignées)
        known_corrections  # Corrections connues
    )

    # 3. Ajouter les corrections interpolées au DataFrame
    return (spectrum
            .with_columns(ppm_correction=pl.lit(corrections_interp),
                          corrected_mz=pl.col(experimental_mass_col) / (1 + pl.lit(corrections_interp) / 1e6))
            .with_columns((pl.col("corrected_mz") / 1e6).alias(new_col_name))
            .select(["m/z", new_col_name, 'Intensity'])
    )


def save_spectrum(spectrum, path, spectrum_name ):
    file_name = Path(path) / spectrum_name
    spectrum.collect().write_csv(file_name, separator=',')


def process_calibration(
    spectrum_name: str,
    original_spectrum:pl.LazyFrame,
    peaks:pl.LazyFrame,
    oligo_ref: pl.LazyFrame,
    path_to_save: str,
    new_col_name:str = "new_mz"
) -> pl.LazyFrame:
    """
    Process MALDI spectrum calibration pipeline with peak extraction, normalization,
    mass difference mapping, and error correction.

    Args:
        spectrum_name: Identifier for the spectrum being processed
        original_spectrum: LazyFrame containing the original spectrum to calibrate
        peaks: LazyFrame containing the orignal peaks dectection 
        oligo_ref: LazyFrame containing reference oligomers with mass data
        path_to_save: Directory path for saving processed output

    Returns:
        LazyFrame containing the corrected spectrum with new mass column ('all_mono')

    Workflow:
        1. Spectrum extraction → 2. Peak detection → 3. m/z normalization
        4. Mass difference mapping → 5. Error calculation → 6. Interpolation
        7. Memory cleanup → 8. Persistence
    """

    # Mass difference analysis
    print("[INFO] Building mass difference map (500ppm threshold)")
    mdim: pl.LazyFrame = build_mass_difference_map(
        peaks=peaks,
        oligo_ref=oligo_ref,
        threshold_ppm=500
    )

    # Error quantification and correction
    print("[INFO] Calculating correction factors")
    error_table: pl.LazyFrame = calculate_error_table(mdim)

    print("[INFO] Applying calibration corrections")
    corrected_spectrum: pl.LazyFrame = interpolate_corrections(
        spectrum=original_spectrum,
        error_table=error_table,
        new_col_name=new_col_name
    )

    # Persistence with explicit resource management
    print("[INFO] Saving results")
    save_spectrum(corrected_spectrum, path_to_save, spectrum_name)

    # Explicit cleanup 
    del peaks, mdim, error_table

    return corrected_spectrum.lazy()  # Ensure return type consistency

if __name__=="main":
    pass

