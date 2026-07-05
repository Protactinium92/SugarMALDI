# Copyright (C) 2026 Pierre-Alexandre HO
#
# This file is part of SugarMALDI.
#
# Licensed under the CeCILL 2.1 license.
# See LICENSE file for details.

"""
Streamlit utilities for MALDI-MS spectrum visualization and interactive analysis.
=================================================================================   


This module provides helper functions to build interactive spectral visualization
tools within a Streamlit application. It includes utilities for synchronized UI
controls (sliders and numeric inputs), spectrum window selection, background
image generation, and interactive Plotly rendering of MALDI-MS spectra.

Main features
-------------
- Synchronized axis selection (slider + number inputs) with Streamlit session state
- Extraction and filtering of m/z and intensity arrays from Polars LazyFrames
- Generation of optimized background spectrum images (base64 encoded)
- Interactive Plotly spectrum visualization with:
    - Background spectrum overlay
    - Differentiation of identified vs unidentified peaks
    - Hover information for annotations
- Tools for defining and controlling visualization windows on x/y axes

Typical workflow
-----------------
1. Load a spectrum stored as a Polars LazyFrame
2. Use `define_x_y` to select and filter the m/z and intensity range
3. Generate a background image using `make_background_b64`
4. Build an interactive Plotly figure with `build_plotly_figure`
5. Use `axis_selector` utilities to control zoom interactively in Streamlit

Dependencies
------------
- streamlit
- numpy
- polars
- matplotlib
- plotly
- base64 / io
- pathlib

Notes
-----
This module is designed for interactive analytical workflows (e.g. MALDI-MS
data exploration), and assumes a Streamlit session context for state management.
"""

import streamlit as st
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io, base64

from pathlib import Path
from typing import Dict, Union, Tuple


def initialize_axis_state(
    axe: str,
    minimum: float,
    maximum: float,
) -> None:
    """
    Initialize session state for an axis.

    Parameters
    ----------
    axe : str
        Axis name ("x" or "y").
    minimum : float
        Minimum allowed value.
    maximum : float
        Maximum allowed value.
    """

    defaults = {
        f"{axe}_range": (minimum, maximum),
        f"{axe}_min": minimum,
        f"{axe}_max": maximum,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_inputs_from_slider(axe: str) -> None:
    """Update number inputs from slider."""

    st.session_state[f"{axe}_min"], st.session_state[f"{axe}_max"] = (
        st.session_state[f"{axe}_range"]
    )


def update_slider_from_inputs(axe: str) -> None:
    """Update slider from number inputs."""

    minimum = st.session_state[f"{axe}_min"]
    maximum = st.session_state[f"{axe}_max"]

    if minimum > maximum:
        minimum, maximum = maximum, minimum

    st.session_state[f"{axe}_min"] = minimum
    st.session_state[f"{axe}_max"] = maximum
    st.session_state[f"{axe}_range"] = (minimum, maximum)


def axis_selector(
    label: str,
    axe: str,
    minimum: float,
    maximum: float,
    step: float | None = None,
    fmt: str = "%.2f",
) -> Tuple[float, float]:
    """
    Display synchronized slider and number inputs.

    Parameters
    ----------
    label : str
        Axis label.
    axe : str
        Axis name ("x" or "y").
    minimum : float
        Minimum value.
    maximum : float
        Maximum value.
    step : float, optional
        Step of number inputs.
    fmt : str
        Display format.

    Returns
    -------
    tuple[float, float]
        Selected minimum and maximum values.
    """

    initialize_axis_state(axe, minimum, maximum)

    col1, col2 = st.columns([4, 2])

    with col1:
        st.slider(
            label,
            minimum,
            maximum,
            key=f"{axe}_range",
            on_change=update_inputs_from_slider,
            args=(axe,),
        )

    with col2:

        c1, c2 = st.columns(2)

        with c1:
            st.number_input(
                "Min",
                key=f"{axe}_min",
                min_value=minimum,
                max_value=maximum,
                step=step,
                format=fmt,
                on_change=update_slider_from_inputs,
                args=(axe,),
            )

        with c2:
            st.number_input(
                "Max",
                key=f"{axe}_max",
                min_value=minimum,
                max_value=maximum,
                step=step,
                format=fmt,
                on_change=update_slider_from_inputs,
                args=(axe,),
            )

    return st.session_state[f"{axe}_range"]


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


def build_plotly_figure(
    bg_src: Union[str, Path],
    lf_peaks: pl.LazyFrame,
    mass_col: str = "m/z",
    intensity_col: str = "Intensity",
    oligo_col_name: str = "Oligo-Adduct",
    x_min: float = 500,
    x_max: float = 4000,
    y_min: float = 0,
    y_max: float = 1e6,
) -> go.Figure:
    """
    Builds an interactive Plotly figure for MALDI-MS spectra visualization with background image,
    differentiated identified/unidentified peaks.

    Args:
        bg_src: Path to background image file (PNG/JPG) or image data
        lf_peaks: Polars LazyFrame containing peak data with columns:
                 - mass_col (default "m/z")
                 - intensity_col (default "Intensity")
                 - Oligo-Adduct (default peak identification)
        mass_col: Column name for mass/charge values
        intensity_col: Column name for intensity values
        oligo_col_name: Column name for oglicoshaccaride name,
        x_min: Minimum m/z value for x-axis
        x_max: Maximum m/z value for x-axis
        y_min: Minimum intensity for y-axis
        y_max: Maximum intensity for y-axis

    Returns:
        go.Figure: Configured Plotly figure with:
                  - Background image stretched to axis range
                  - Red markers for identified peaks (with oligo info on hover)
                  - Yellow markers for unidentified peaks
                  - Transparent background
                  - Full-width display
    """ 
    # Filter to split the identified peak vs the non identified one
    df_identified = lf_peaks.filter(pl.col(oligo_col_name).is_not_null()).collect()
    df_unidentified = lf_peaks.filter(pl.col(oligo_col_name).is_null()).collect()
    
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
    
    # Put red marker to identified peaks
    oligo_name = (
            df_identified[oligo_col_name]
            .fill_null("")  # Gère les None
            .str.split("/")  # Split par "/"
            .map_elements(
                lambda x: "<br>".join(x) if len(x) > 0 else "",  # Vérifie si la liste n'est pas vide
                return_dtype=pl.String
            )
            .to_list()
        )
    x_formate= [str(x).replace(",", "") for x in df_identified[mass_col].to_list()]
    customdata = np.column_stack((x_formate, oligo_name))  # Custumdata need a

    fig.add_trace(go.Scattergl(
        x=df_identified[mass_col].to_numpy(),
        y=df_identified[intensity_col].to_numpy(),
        mode="markers",
        marker=dict(size=4, color='red'),
        name="Identified Peaks",
        customdata=customdata,
        hovertemplate=(
            "<b>m/z</b>: %{customdata[0]}<br>"
            "<b>Intensity</b>: %{y}<br>"
            "<b>Identification:</b><br>%{customdata[1]}<extra></extra>"
        )))

    # Put yellow marker to unidentified peaks
    x_formate= [str(x).replace(",", "") for x in df_unidentified[mass_col].to_list()]
    fig.add_trace(go.Scattergl(
        x=df_unidentified[mass_col].to_numpy(),
        y=df_unidentified[intensity_col].to_numpy(),
        mode="markers",
        marker=dict(size=4, color='yellow'),
        name="Unidentified Peaks",
        customdata=x_formate,
        hovertemplate="m/z: %{customdata}<br>Intensity: %{y}<br><extra></extra>",
    ))

    fig.update_xaxes(title_text="m/z", range=[x_min, x_max])
    fig.update_yaxes(title_text="Intensity", range=[y_min, y_max])
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40),
                      dragmode=False,
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True)

    # Clean memory
    del df_identified, df_unidentified
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
    y_all = spectrum[intensity_name].to_numpy()

    # Adjust window visualization
    #st.write("Use the slider to zoom with a better rendering")
    with st.expander("Adjust window visualization"):
        x_min, x_max = axis_selector(
            "m/z range",
            "x",
            float(x_all.min()),
            float(x_all.max()),
            step=0.01,
            fmt="%.4f",
        )

        y_min, y_max = axis_selector(
            "Intensity range",
            "y",
            0.0,
            float(y_all.max()),
            step=100.0,
            fmt="%.0f",
        )

    # Create mask for the selected axis range
    mask = (
        (spectrum[mz_name] >= x_min)
        & (spectrum[mz_name] <= x_max)
        & (spectrum[intensity_name] >= y_min)
        & (spectrum[intensity_name] <= y_max)
    )

    # Apply filter and convert to numpy arrays
    filtered_spectrum = spectrum.filter(mask)
    x = filtered_spectrum[mz_name].to_numpy()
    y = filtered_spectrum[intensity_name].to_numpy()

    del spectrum, mask, filtered_spectrum
    return x, y


if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )