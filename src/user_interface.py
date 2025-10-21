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
import subprocess
import time

from pathlib import Path
from typing import Dict, Union, Optional, Tuple, List


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


@st.cache_data
def make_background_b64(x: np.ndarray, y: np.ndarray, figsize=(12,4), dpi=150):
    """
    Create the spectrum background non interactive to optimize the display.
    Args:
        x (np.ndarray): m/z values.
        y (np.ndarray): Intensity values.
        figsize (tuple, optional): Figure size. Defaults to (12,4).
        dpi (int, optional): DPI. Defaults to 150.
    Returns:
        str: Base64 encoded SVG image.
        float: x min, x max, y min, y max.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    #ax = fig.add_axes([0, 0, 1, 1])  # Occupe toute la figure
    ax.plot(x, y, linewidth=0.35)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_axis_off()  # Désactive les axes et les bordures
    ax.margins(0)      # Supprime les marges internes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0,  transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}", float(x.min()), float(x.max()), float(y.min()), float(y.max())



def build_plotly_figure(bg_src, df_peaks, mass_col: str, intensity_col: str, x_min, x_max, y_min, y_max, scale_factor=1e-6 ):
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=bg_src,
            xref="x", yref="y",
            x=x_min, y=y_max,
            sizex=x_max-x_min, sizey=y_max-y_min,
            xanchor="left", yanchor="top",
            sizing="stretch", opacity=1.0, layer="below"
        )
    )
    oligo_name = df_peaks["Oligo-Adduct"].to_numpy().reshape(-1, 1)
    fig.add_trace(go.Scattergl(
        x=df_peaks[mass_col].to_numpy()*scale_factor,  # ppm --> Da
        y=df_peaks[intensity_col].to_numpy(),
        mode="markers",
        marker=dict(size=4, color='red'),

        name="Peaks",
        customdata=oligo_name,
        hovertemplate="m/z: %{x}<br>Intensity: %{y}<br>Oligo: %{customdata[0]}<extra></extra>",
    ))
    fig.update_xaxes(title_text="m/z", range=[x_min, x_max])
    fig.update_yaxes(title_text="Intensity", range=[y_min, y_max])
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def define_x_y(
    spectra: Dict[str, pl.LazyFrame],
    selected_spectrum: str,
    mz_name: str = "m/z",
    intensity_name: str = "Intensity"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and filters m/z and intensity arrays from a MALDI spectrum DataFrame.

    Args:
        spectra: Dictrionary of Polars LazyFrame containing MALDI spectra data (must include m/z and Intensity columns)
        selected_spectrum: Name of the spectrum to select from the spectra Dictionary
        mz_name: Name of the m/z column (default: "m/z")
        intensity_name: Name of the intensity column (default: "Intensity")

    Returns:
        Tuple of (x, y) numpy arrays where:
        - x: Filtered m/z values
        - y: Corresponding intensity values
    """
    # Collect the selected spectrum
    spectrum = spectra[selected_spectrum].collect()
    x_all = spectrum[mz_name].to_numpy()

    # Adjust window vizualisation
    mz_min, mz_max = st.slider("Use the slider to zoom with a better rendering", float(x_all.min()), float(x_all.max()),
                        (float(x_all.min()), float(x_all.max())))

    # Create mask for the selected m/z range
    mask = (spectrum[mz_name] >= mz_min) & (spectrum[mz_name] <= mz_max)
    # Apply filter and convert to numpy arrays
    filtered_spectrum = spectrum.filter(mask)
    x = filtered_spectrum[mz_name].to_numpy()
    y = filtered_spectrum[intensity_name].to_numpy()

    del spectrum, mask, filtered_spectrum
    return x, y



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
            selected_spectrum = st.selectbox("Select spectrum", st.session_state.spectra_list)
            peak_lf = st.session_state.peak_dict[selected_spectrum]
            nb_peak = peak_lf.select(pl.len()).collect().item()  # .item() extract the scalar value
            st.write(f'Number of peak detected: {nb_peak}')
            st.dataframe(peak_lf.sort(pl.col("intensity"), descending=False).collect())

            #####
            #   CODE POUR L'AFFICHAGE DES SPECTRES
            #####
            with st.expander("⚙️ Advanced options to display the spectrum", expanded=False):
                dpi = st.selectbox("Image DPI (quality)", [100, 150, 200, 300], index=1)
                      
            # make background image
            x, y = define_x_y(spectra, selected_spectrum,)
            bg_src, x_min, x_max, y_min, y_max = make_background_b64(x, y, dpi=dpi)

            # PLot the mass spectrum
            fig = build_plotly_figure(bg_src, peak_lf.collect(),
                                    mass_col="query_mass",
                                    intensity_col="intensity",
                                    x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max,
                                    scale_factor=1e-6)
            st.plotly_chart(fig, use_container_width=True)

    atexit.register(sp_ana.cleanup)
