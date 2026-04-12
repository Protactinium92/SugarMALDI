"""
Frontend utility functions for the SugarMALDI Streamlit application.

This module contains all Streamlit-specific functions for:
- Data visualization of MALDI-TOF spectra (m/z vs Intensity)
- User interaction handling
- Cached operations for performance optimization

Key Features:
- Polars DataFrame integration for efficient large dataset handling (~1M rows)
- Base64 image generation for transparent spectrum backgrounds
- Streamlit caching decorators (@st.cache_data) for responsive UI

Designed to work with:
- Backend: spectre_analysis.py (data processing)
- Data format: CSV files with 'm/z' and 'Intensity' columns
- Visualization: Matplotlib/Plotly with transparent backgrounds

Dependencies:
    polars>=1.33.0      # Data processing backbone
    streamlit>=1.49.1   # UI framework
    plotly>=6.3.0       # Interactive visualizations
    matplotlib>=3.10.6  # Background graph operation
    numpy>=2.3.2        # Numerical operations

Note:
    All functions are optimized for:
    - Memory efficiency with Polars LazyFrames
    - Streamlit's reactive programming model
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2026-01-14"
__version__ = "1.0"

import streamlit as st
import spectre_analysis as sp_ana
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io, base64
import numpy as np
import logging

from pathlib import Path
from typing import Dict, Union, Tuple, List


def display_logo() -> None:
    """
    Displays the SugarMALDI logo in the Streamlit in function the user theme.
    """
    theme =  st.context.theme.type
    
    if theme == "dark":
        logo_path = Path(__file__).parents[1] / "pictures" / "sugarmaldi_logo_dark_theme.png"
    else:
        logo_path = Path(__file__).parents[1] / "pictures" / "sugarmaldi_logo_light_theme.png"

    if logo_path.exists():
        with st.columns(3)[1]:  # Center the logo on 3 columns
            st.image(str(logo_path), width='stretch')
    else:
        st.write("Logo not found.")


def choose_delimiter() -> str:
    """
    Interactive delimiter selection for file parsing.

    Returns:
        str: The selected delimiter character (converted to actual character for 'Tab'/'Space').
             Possible values: ',', '\t', ' ', ';', ':', '|', '-'.
    """
    delimiter_options = (",", "Tab", "Space", ";", ":", "|", "-")
    mapping = {"Tab": "\t", "Space": " "}  # Conversion for display vs. actual values
    
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.markdown("""
            <div style="line-height:1.2">
                Select delimiter file:<br>
                ⚠️ <span style='color:red; font-style:italic;'>TO DO before uploading files</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        option = st.selectbox(
            label="Select file delimiter", 
            options=delimiter_options, 
            label_visibility="collapsed", 
            key="csv_delimeter")

    # Return converted value if needed, otherwise return as-is
    return mapping.get(option, option)


def upload_files(delimiter: str =",") -> pl.DataFrame:
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
   
    st.markdown("#### Choose MS spectra")
    uploaded_files = st.file_uploader(
        "",
        type=['csv', "tsv", 'txt'],
        accept_multiple_files=True
    )

    css = '''
    <style>
    div[data-testid="stFileUploader"] {
        margin-top: -40px;
    }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)   

    spectra = None  # Initialize as None (no files uploaded by default)

    if uploaded_files:
        try:
            spectra = sp_ana.parse_spectra(uploaded_files, delimiter=delimiter)
            st.success("✅Files have been successfully imported")
            
        except ValueError as e:
            st.error(f"Error: {str(e)}")
            spectra = None  # Ensure None is returned on error

    return spectra


def monomer_selection() ->Tuple[List[str], pl.DataFrame]:
    """
    Interactive monomer selection interface for MALDI spectra analysis.

    Returns:
        List[str]: List of selected monomer notations (empty if none selected).
                  Returns all monomers if selection is not implemented yet.
    """
    st.subheader("Monomer(s) Selection")

    with st.expander("💬 **Information & Guidelines**", expanded=False):
        st.write("Here will be the guidlines for the selection of monomers.")

    # Load and prepare monomer data
    monomers = sp_ana.read_csv(Path(__file__).parents[1] / "data" / "monomers.csv", sep=";")
    monomers_df = (monomers
        .with_columns(pl.lit(False).alias("Selection"))              # Boolean for checkbox
        .select(["Selection"] +  monomers.collect_schema().names())  # Selection column first
        .collect()
    )

    # Interactive DataFrame with selection
    edited_df = st.data_editor(
        monomers_df,
        disabled=monomers_df.columns[1:],  # Disable all except Selection
        hide_index=True,
        width="stretch",
        column_config={
            "Selection": st.column_config.CheckboxColumn(
                "Selection",
                help="Check to include this monomer in analysis"
            )
        }
    )

    # Process selection
    selected_df = edited_df.filter(pl.col("Selection"))
    monomers_list = (selected_df["Notation"].to_list())

    if not monomers_list:
        st.warning("⚠️Please select at least one monomer")
    else:
        badges = " ".join([f":blue-badge[{m}]" for m in monomers_list])
        st.markdown(badges)
    del monomers, edited_df
    return monomers_list, selected_df


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
    Builds an interactive Plotly figure for MALDI-TOF spectra visualization with background image,
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
        name="Undentified Peaks",
        customdata=x_formate,
        hovertemplate="m/z: %{customdata}<br>Intensity: %{y}<br><extra></extra>",
    ))

    fig.update_xaxes(title_text="m/z", range=[x_min, x_max])
    fig.update_yaxes(title_text="Intensity", range=[y_min, y_max])
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40))

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


class StreamlitHandler(logging.Handler):
    """
    A custom logging handler that outputs log messages to a Streamlit placeholder.

    Attributes:
        placeholder (st.delta_generator.DeltaGenerator): The Streamlit container where logs will be displayed.
        logs (List[str]): A list storing all emitted log messages for reference.
    """
    
    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        """
        Initialize the handler with a Streamlit placeholder.

        Args:
            placeholder: A Streamlit container to display log messages.
        """
        super().__init__()
        self.placeholder = placeholder
        self.logs = []

    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Format and emit a log record to the Streamlit placeholder.

        Args:
            record: The log record to process and display.
        """
        msg = self.format(record)
        self.placeholder.text(msg)


def init_streamlit_logger(
    placeholder: st.delta_generator.DeltaGenerator,
    level: int = logging.INFO,
    datefmt: str = "%Y-%m-%d %H:%M:%S"
) -> Tuple[logging.Logger, "StreamlitHandler"]:
    """
    Initialize a logger that outputs logs to a Streamlit placeholder.

    Args:
        placeholder: Streamlit container (e.g., `st.empty()` or `st.container()`) to display logs.
        level: Logging level (default: `logging.INFO`).
        datefmt: Date format for log timestamps (default: `"%Y-%m-%d %H:%M:%S"`).

    Returns:
        Tuple[logging.Logger, StreamlitHandler]:
            - root_logger: Configured root logger with attached handler.
            - handler: The `StreamlitHandler` instance (for later removal if needed).
    """
    # Create Streamlit handler
    handler = StreamlitHandler(placeholder)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt=datefmt
    )
    handler.setFormatter(formatter)

    # Logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove existing handlers
    root_logger.addHandler(handler)

    return root_logger, handler


class StreamlitProgress:
    """
    A Streamlit-compatible progress bar with text display.

    Features:
        - Shows current/total count and percentage
        - Clean layout with text in left column and progress bar in right
        - Methods to update and clear the display
    """
     
    def __init__(self, total: int) -> None:
        """
        Initialize progress tracking with a given total.

        Args:
            total: Total number of items/steps to complete.
        """
        self.total = total
        col_text, col_bar = st.columns([1, 9])
        self.text = col_text.empty()
        self.progress_bar = col_bar.progress(0)
        self._render(0)
        

    def _render(self, current: int) -> None:
        """
        Update the displayed progress text.

        Args:
            current: Current progress count.
        """
        percent = int((current / self.total) * 100) if self.total else 0
        self.text.write(f"📊 {current} / {self.total} ({percent}%)")

    def update(self, current: int) -> None:
        """
        Update the progress bar

        Args:
            current: Current progress count.
        """
        self.progress_bar.progress(current / self.total)
        self._render(current)

    def clear(self):
        """Remove the progress display from the Streamlit interface."""
        self.progress_bar.empty()
        self.text.empty()