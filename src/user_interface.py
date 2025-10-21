"""
    User interface for the streatment of the spectra
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-08"
__version__ = "1.0"


import streamlit as st
import spectre_analysis as sp_ana
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
import os
import io, base64
import numpy as np
import atexit
from pathlib import Path
import subprocess
import time


def upload_files() -> pl.DataFrame:
    """
    Handles the upload of MS spectra CSV files via Streamlit's file uploader,
    parses them using `sp_ana.parse_spectra`, and displays a preview of the merged data.

    Returns:
        Dict[str, pl.DataFrame]:
            A dictionary where keys are spectrum names (derived from filenames) and values are Polars DataFrames
            containing the spectrum data (m/z and intensity columns).

    Raises:
        ValueError: Propagates any error from `sp_ana.parse_spectra` (e.g., invalid file format).
    """
    uploaded_files = st.file_uploader(
        "Choose MS spectra",
        type=['csv'],
        accept_multiple_files=True
    )

    spectra = None  # Initialize as None (no files uploaded by default)

    if uploaded_files:
        try:
            spectra = sp_ana.parse_spectra(uploaded_files)
            st.success("✅Files have been successfully imported")
            
        except ValueError as e:
            st.error(f"Error: {str(e)}")
            spectra = None  # Ensure None is returned on error

    return spectra
























































if __name__=="__main__":
    st.title('SugarMALDI')

    spectra = upload_files()

    # Check if their is an ulpoad wherase its crash
    if spectra is not None and len(spectra) > 0:

        st.session_state.spectra_list = list(spectra.keys())
        
        with st.expander("⚙️ Advanced options for peak detection", expanded=False):
            st.subheader("Detection parameters")
            col1, col2 = st.columns([1,1])
            with col1:
                window = st.number_input("Sub-window for looking peak", value=500, step=1)
            with col2:
                dist_min = st.number_input("Distance m/z min between peak", value=15, step=1)
            st.subheader("Match peaks parameters") 
            col3, col4 = st.columns([1,1])
            with col3:          
                offset = st.number_input("Potential offset of the spectre in ppm", value=100_000, step=1)
            with col4:
                rm_threshold = st.number_input('Mass accuracy threshold (ΔM, "ppm")', value=5, step=1,
                                               help=
                                                    r"""
            **Mass Accuracy (ΔM)**:
            Mass Measurement Accuracy / Error, calculated as (1):

            $$\Delta M \ (\mathrm{ppm}) = \left( \frac{|M_{\mathrm{exp}} - M_{\mathrm{theo}}|}{M_{\mathrm{theo}}} \right) \times 10^6$$

            - **$M_{\mathrm{exp}}$**: Experimentally measured mass.
            - **$M_{\mathrm{theo}}$**: Expected theoretical mass.
            - **Typical threshold**: 5 ppm for MALDI-TOF.

            *Example*: If ΔM = 10 ppm and $M_{\mathrm{theo}}$ = 1000 Da,
            the tolerated deviation is **±0.01 Da**.

            *Note*: Lower ΔM = stricter peak matching.

            **Reference:**

            (1) Brenton, A.G.; Godfrey, A.R. *J. Am. Soc. Mass Spectrom.* **2010**, *21*(11), 1821-1835.
            [DOI:10.1016/j.jasms.2010.06.006](https://www.sciencedirect.com/science/article/pii/S1044030510004022)
            """)
        
        # Print Monomer tab
        monomers = sp_ana.read_csv(Path(__file__).parents[1]/"data"/"monomers.csv", sep=";")
        monomers_list = monomers.select(pl.col("Notation")).collect().to_series().to_list()
        st.dataframe(monomers.collect())

        st.markdown("---")
        st.subheader("Process and Download")

        # Initiatilisation of session state to store the treatment
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("🔍 Spectra analysis"):
                with st.spinner('🔄 Treatment ongoing...'):
                    zip_buffer, peak_dict = sp_ana.execute_process(
                        spectra,
                        monomers_list,
                        offset,
                        rm_threshold,
                        window,
                        dist_min
                    )
                    st.session_state.peak_dict = {
                        spectrum_name: peaks_lf.lazy() 
                        for spectrum_name, peaks_lf in peak_dict.items()
                    }
                    st.session_state.analysis_results = zip_buffer

        with col2:
            if st.session_state.analysis_results:
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=st.session_state.analysis_results.getvalue(),
                    file_name="results_maldi.zip",
                    mime="application/zip"
                )

        if 'peak_dict' in st.session_state:
            st.success('✅ Analysis completed !') 
            # Choose the spectra to show
            selected_col_name = st.selectbox("Select spectrum", st.session_state.spectra_list)
            peak_lf = st.session_state.peak_dict[selected_col_name]
            nb_peak = peak_lf.select(pl.len()).collect().item()  # .item() extract the scalar value
            st.write(f'Number of peak detected: {nb_peak}')
            st.dataframe(peak_lf.sort(pl.col("intensity"), descending=False).collect())

            #####
            #   CODE POUR L'AFFICHAGE DES SPECTRES
            #####

    atexit.register(sp_ana.cleanup)
