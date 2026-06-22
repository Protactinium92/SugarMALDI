"""
Frontend utility functions for the SugarMALDI Streamlit application.
====================================================================

Overview
--------
This module contains all Streamlit-specific functions for:
- Data visualization of MALDI-MS spectra (m/z vs Intensity)
- User interaction handling

Key Features:
- Polars DataFrame integration for efficient large dataset handling (~1M rows)
- Base64 image generation for transparent spectrum backgrounds
- Streamlit caching decorators (@st.cache_data) for responsive UI for spectra rendering

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


import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io, base64
import numpy as np
import logging
import re
import uuid

from src import spectrum_analysis as sp_ana
from src import load_file as load
from src import config as cf
from src import utils as ut


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
                ⚠️ <span style='color:red; font-style:italic;'>TO Change before uploading files</span>
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
   
    st.markdown("##### Choose MS spectra")
    uploaded_files = st.file_uploader(
        label="spectrum_uploader",
        label_visibility="collapsed",
        type=['csv', "tsv", 'txt'],
        accept_multiple_files=True,
    )

    css = '''
    <style>
    div[data-testid="stFileUploader"] {
        margin-top: -10px;
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
    st.markdown("#### Monomer(s) Selection")

    with st.expander("💬 **Information & Guidelines**", expanded=False):
        st.write("Here will be the guidlines for the selection of monomers.")

    # Load and prepare monomer data
    monomers = load.read_csv(Path(__file__).parents[1] / "data" / "monomers.csv", sep=";")
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


def _init_adducts_state() -> None:
    """Initialize session state for adducts with default values from cf.ADDUCT_MASS"""
    if "adducts" not in st.session_state:
        st.session_state.adducts = []
        for adduct_name, mass in cf.ADDUCT_MASS.items():
            st.session_state.adducts.append({
                "id": str(uuid.uuid4()), # Unique ID,  
                "name": adduct_name,
                "mass": mass,
                "formula": adduct_name,  # Default formula same as name
                "is_lost": False,  # Default value for is_lost
            })


def _add_adduct() -> None:
    """Add a new empty adduct."""
    st.session_state.adducts.append(
        {
            "id": str(uuid.uuid4()),
            "name": "",
            "mass": 0.0,
            "formula": "",
            "is_lost": False,
        }
    )


def _delete_adduct(adduct_id: str) -> None:
    """Delete an adduct by its unique ID."""
    if len(st.session_state.adducts) > 1:
        st.session_state.adducts = [
            a for a in st.session_state.adducts if a["id"] != adduct_id
        ]


def print_error_formula(wrong_elements: List[str]):
    elements_str = ", ".join(wrong_elements)
    verb = "are" if len(wrong_elements) > 1 else "is"
    st.error(f"❌ {elements_str} {verb} not in the NIST monoisotopic library")


def markdown_checkbox_title_rendering(text: str) -> None:
    return f"""
        <div style="
            white-space: nowrap;
            font-size: 0.9rem;
            line-height: 2.3;
            margin-bottom: 0.5rem;
        ">
            {text}
        </div>
        """
            

def adduct_selection() -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Render adduct selection UI and return mass + composition dicts.
    """

    on_adduct = st.toggle("Use custom adduct (By default H+ Na+ & K+)")

    if on_adduct:
        st.markdown("ℹ️ *Lost: Check the box to indicate if the adduct is lost during ionization*")
        
        _init_adducts_state()
       
        for i, adduct in enumerate(st.session_state.adducts):
            st.markdown(f"Adduct {i+1}")

            col_name, col_is_lost, col_mass, col_formula, col_delete = st.columns([2, 0.5, 2, 4, 0.5],
                                                    vertical_alignment="bottom")

            adduct["name"] = col_name.text_input(
                "Name",
                value=adduct["name"],
                key=f"name_{adduct['id']}"
            )

            col_is_lost.markdown(markdown_checkbox_title_rendering("Lost"),
                unsafe_allow_html=True
            )
            is_lost = col_is_lost.checkbox(
                "Lost",
                key=f"is_lost_{adduct['id']}",
                label_visibility="collapsed",
                value=False
            )

            adduct["mass"] = col_mass.number_input(
                "m/z",
                format="%.6f",
                key=f"mass_{adduct['id']}",
                max_value=0.0 if is_lost else None,  # No min  but max 0 if is_lost = True forces negative
                min_value=0.0 if not is_lost else None,  # No max but min 0 if is_lost = False forces positive
                value=adduct["mass"] if is_lost == adduct["is_lost"] else adduct["mass"]*(-1),  # Adjust mass if is_lost changed
                step=0.000001
            )
            adduct["is_lost"] = is_lost

            with col_formula:
                sub_col_box, sub_col_formula = st.columns([0.05, 1],
                                                          vertical_alignment="bottom") 
          
            sub_col_box.markdown(markdown_checkbox_title_rendering("Use a brut formula different from name"),
                unsafe_allow_html=True)
            use_formula = sub_col_box.checkbox(
                "Use Formula",
                key=f"use_formula_{adduct['id']}",
                label_visibility="collapsed",
                value=False
            )
            if use_formula:
                adduct["formula"] = sub_col_formula.text_input(
                    "formula",
                    value=adduct["formula"],
                    key=f"formula_{adduct['id']}",
                    placeholder="e.g. NH4",
                    label_visibility="collapsed",
                )
            else:
                adduct["formula"] = adduct["name"]
                sub_col_formula.text_input(
                    "formula",
                    value=adduct["formula"],
                    key=f"formula_{adduct['id']}_disabled",
                    placeholder="e.g. NH4",
                    disabled=True,
                    label_visibility="collapsed"
                )

            col_delete.button(
                label = "❌" if len(st.session_state.adducts) > 1 else "—",
                key=f"delete_{adduct['id']}",
                width=40,
                on_click=_delete_adduct,
                args=(adduct["id"],),
                disabled=(len(adduct) == 1)
            )

        st.button(
            "➕ Add Adduct",
            on_click=_add_adduct,
        )
        field = ["name", "mass", "formula"]
        valid_adducts, error_indices, has_any_input = ut.validate_rows(st.session_state.adducts,
                                                                    required_fields=field)
        adduct_mass, adduct_compo, adduct_is_lost = ut.build_adduct_dicts(valid_adducts)
        
        
        # Warning message

        if error_indices:
            error_adduct = [f"Adduct {i+1}" for i in error_indices]
            st.error(f"❌ Some adducts are incomplete {' / '.join(map(str, error_adduct))}")

        elif not has_any_input:
            st.error("⚠️ Please fill at least one adduct")

        for adduct in valid_adducts:
            composition = adduct_compo.get(adduct["name"], {})

            if adduct["formula"] and not composition:
                st.error(f"❌ Adduct formula '{adduct['formula']}' is invalid")
                adduct_compo = None
                break

            wrong_elements = ut.check_formula(composition)
            if wrong_elements:
                print_error_formula(wrong_elements)
                adduct_compo = None
                break
    else:
        adduct_mass = cf.ADDUCT_MASS
        adduct_compo = cf.ADDUCTS
        adduct_is_lost = {name: False for name in cf.ADDUCT_MASS.keys()}

    st.session_state.adduct_mass = adduct_mass
    st.session_state.adduct_compo = adduct_compo
    st.session_state.adduct_loss = adduct_is_lost


def label_selection():
    col1, col2 = st.columns([0.25, 0.75])
    use_label = col1.toggle("Use label (By default True)", value=True)
    on_label = col2.toggle(
        "Use custom label (By default 3-AQ)",
        value=False,
        disabled=not use_label
    )

    label_name = None
    label_formula = None
    compo = None

    # --- CASE 1: No label ---
    if not use_label:
        pass  # everything already None

    # --- CASE 2: Custom label ---
    elif on_label:
        col_name, col_formula = st.columns([1, 3])

        input_name = col_name.text_input(
            label="Name",
            key="label_name_input",
        )

        input_formula = col_formula.text_input(
            label="Brut or Semi-developped Formula",
            key="label_formula_input",
            placeholder="e.g. CH4O or CH3-OH"
        )

        rows = ({"name": input_name, "formula": input_formula},)

        valid_rows, error_indices, has_any_input = ut.validate_rows(
            rows=rows,
            required_fields=["name", "formula"]
        )

        if error_indices:
            st.error("❌ Both name and formula must be filled")

        elif not has_any_input:
            st.error("⚠️ Please fill the fields")

        else:
            row = valid_rows[0]

            compo_candidate = ut.parse_formula(row["formula"])
            wrong_elements = ut.check_formula(compo_candidate)

            if wrong_elements:
                print_error_formula(wrong_elements)
            else:
                label_name = row["name"]
                label_formula = row["formula"]
                compo = compo_candidate

    # --- CASE 3: Default label ---
    else:
        label_name = "3-AQ"
        label_formula = "C9H8N2"
        compo = cf.LABEL_3AQ

    return use_label, label_name, label_formula, compo


def label_selection():
    col1, col2 = st.columns([0.25, 0.75])
    use_label = col1.toggle("Use label (By default True)", value=True)
    on_label = col2.toggle(
        "Use custom label (By default 3-AQ)",
        value=False,
        disabled=not use_label
    )

    label_name = None
    label_formula = None
    compo = None

    if on_label and use_label:
        col_name, col_formula = st.columns([1, 3])

        label_name = col_name.text_input(
            label="Name",
            key="label_name_input",
        )

        label_formula = col_formula.text_input(
            label="Brut or Semi-developped Fomrula",
            key="label_formula_input",
            placeholder="e.g. CH4O or CH3-OH"
        )

        rows = ({"name": label_name, "formula": label_formula},)

        valid_rows, error_indices, has_any_input = ut.validate_rows(
            rows=rows,
            required_fields=["name", "formula"]
        )

        if error_indices:
            st.error("❌ Both name and formula must be filled")

        elif not has_any_input:
            st.error("⚠️ Please fill the fields")

        else:
            row = valid_rows[0]

            compo = ut.parse_formula(row["formula"])
            wrong_elements = ut.check_formula(compo)

            if wrong_elements:
                print_error_formula(wrong_elements)
                compo = None
            else:
                label_name = row["name"]
                label_formula = row["formula"]

    elif use_label == False:
        label_name = None
        label_formula = None
        compo = None

    else:
        label_name = "3-AQ"
        label_formula = "C9H8N2"
        compo = cf.LABEL_3AQ

    return use_label, label_name, label_formula, compo


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


def build_badges(config):
    return [
        *[f":blue-badge[{m}]" for m in config.monomers_list],
        f":orange-badge[{config.params.iso_score}]",
        *[f":green-badge[{'(-)' if config.adduct_loss.get(name, False) else ''} {name}]" for name in config.adduct_mass.keys()],
        f":red-badge[{config.label_name}]",
    ]


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


if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )