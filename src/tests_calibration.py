import spectre_analysis as sp_ana
import calibration as cal
import polars as pl
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import Dict, Union, Optional, Tuple, List

import datashader as ds
from datashader import transfer_functions as tf
import colorcet as cc



if __name__=='__main__':
    
    # I Preparation

    print("[INFO] Initialization")
    workers = sp_ana.find_workers()

    path_std = Path.cwd() / "data" / "standards"
    path_cal = Path.cwd() / "data" / "Standard_Gums_Calibrated_files"

    list_std = os.listdir(path_std)
    list_cal = os.listdir(path_cal)
    list_cal = [i.replace("_Calibrated", "") for i in list_cal]

    # take the commun spectra of the both folders
    spectra = sorted(list(set(list_std) & set(list_cal)))

    if len(spectra) == 0:
        print('No spectra found')
        sys.exit()

    # 1) All monomers
    print("[INFO] load reference library with all monomers")
    monomers_list = ["dHex", "Hex", "HexA", "HexN", "HexNac", "Pent", "Neu", "NeuAc", "NeuGc"]
    library = Path.cwd() / "data" / "oligo_combinations_clean.csv"
    oligo_ref_all = sp_ana.adjust_library(library, monomers_list)
    # Transform the mass in ppm and int to optimize the algorithm
    for adduct in ['labeled_mass_H+', 'labeled_mass_Na+', 'labeled_mass_K+']:
        oligo_ref_all = sp_ana.normalize_mz_column(
        lf = oligo_ref_all,
        mz_col=adduct,
        new_col=f"{adduct}_int"
    )

    # 2) Selection of mmonomers
    print("[INFO] load reference library with aselection of monomers")
    monomers_list = ["Hex", "Pent"]
    library = Path.cwd() / "data" / "oligo_combinations.csv"
    oligo_ref_selection = sp_ana.adjust_library(library, monomers_list)
    # Transform the mass in ppm and int to optimize the algorithm
    for adduct in ['labeled_mass_H+', 'labeled_mass_Na+', 'labeled_mass_K+']:
        oligo_ref_selection = sp_ana.normalize_mz_column(
        lf = oligo_ref_selection,
        mz_col=adduct,
        new_col=f"{adduct}_int"
    )
    
    # Path for saving 
    path_spectrum_all = Path.cwd() / "data" / "calibration" / "spectrum" / "all"
    path_spectrum_selection = Path.cwd() / "data" / "calibration" / "spectrum" / "selection"
    path_peaks_all = Path.cwd() / "data" / "calibration" / "peaks" / "all"
    path_peaks_selection = Path.cwd() / "data" / "calibration" / "peaks" / "selection"

    # II Calibration

    # 1 All monomers

    # print("[INFO] Calibration all monomers")
    # for spectrum_name in spectra:

    #     # Spectrum extraction with progress logging
    #     print(f"[INFO] Starting analysis for {spectrum_name}")
    #     spectrum_file = Path.cwd() / "data" / "standards" / spectrum_name
    #     spectrum_std: pl.LazyFrame = (
    #         sp_ana.read_spectrum(spectrum_file)
    #         .pipe(lambda df: df.lazy())  # Ensure LazyFrame if not already
    #         )
        
    #     # Peak detection pipeline
    #     print("[INFO] Extracting peaks with adaptive parameters")
    #     peaks = (
    #         spectrum_std
    #         .pipe(lambda df: sp_ana.extract_and_detect_peaks(
    #             spectrum_data=df,
    #             window=500,
    #             dist_min=0,
    #             prominence=3e5,
    #             n_workers=workers
    #         ))
    #         .pipe(lambda df: sp_ana.normalize_mz_column(
    #             lf=df,
    #             mz_col="m/z",
    #             new_col="mz_int"
    #         ))
    #     )

    #     corrected_spectrum_all = cal.process_calibration(
    #         spectrum_name,
    #         spectrum_std,
    #         peaks,
    #         oligo_ref_all,
    #         path_spectrum_all,
    #         "all_mono"
    #     )

    #     # Free memory
    #     del peaks, corrected_spectrum_all,
    #     gc.collect()


    # 2 Selection of spectrum

    print("[INFO] Calibration selection monomers")
    for spectrum_name in spectra:
        # Spectrum extraction with progress logging
        print(f"[INFO] Starting analysis for {spectrum_name}")
        spectrum_file = Path.cwd() / "data" / "standards" / spectrum_name
        spectrum_std: pl.LazyFrame = (
            sp_ana.read_spectrum(spectrum_file)
            .pipe(lambda df: df.lazy())  # Ensure LazyFrame if not already
            )
        
        # Peak detection pipeline
        print("[INFO] Extracting peaks with adaptive parameters")
        peaks = (
            spectrum_std
            .pipe(lambda df: sp_ana.extract_and_detect_peaks(
                spectrum_data=df,
                window=500,
                dist_min=0,
                prominence=3e5,
                n_workers=workers
            ))
            .pipe(lambda df: sp_ana.normalize_mz_column(
                lf=df,
                mz_col="m/z",
                new_col="mz_int"
            ))
        )
        corrected_spectrum_selection = cal.process_calibration(
            spectrum_name,
            spectrum_std,
            peaks,
            oligo_ref_selection,
            path_spectrum_selection,
            "selected_mono"
        )
