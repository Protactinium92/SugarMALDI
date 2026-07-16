# Copyright (C) 2026 Pierre-Alexandre HO
#
# This file is part of SugarMALDI.
#
# Licensed under the CeCILL 2.1 license.
# See LICENSE file for details.

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

import logging
import uuid

from src import spectrum_analysis as sp_ana
from src import load_file as load
from src import config as cf
from src import utils as ut
from src import helper

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

    st.markdown("""ℹ️ **Select the monomer(s) to include in the analysis.**  
                Select only the most likely monosaccharides for your sample.
                Including unnecessary monomers greatly increases computation time and the number of false-positive identifications.  
                If uncertain, run several targeted searches changing the monomer set instead of one exhaustive analysis.
                """)

    with st.expander("💬 **Guidelines monomers selection**", expanded=False):
        st.markdown(helper.user_guide())

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


def _init_neutral_losses_state() -> None:
    """Initialize session state for neutral losses with default values."""
    if "neutral_losses" not in st.session_state:
        st.session_state.neutral_losses = []
        # Add a default neutral loss (e.g., H2O)
        st.session_state.neutral_losses.append({
            "id": str(uuid.uuid4()),
            "name": "H2O",
            "mass": -18.01056,  # Negative mass for neutral loss
            "formula": "H2O",
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


def _add_neutral_loss() -> None:
    """Add a new empty neutral loss."""
    st.session_state.neutral_losses.append(
        {
            "id": str(uuid.uuid4()),
            "name": "",
            "mass": 0.0,
            "formula": "",
        }
    )


def _delete_adduct(adduct_id: str) -> None:
    """Delete an adduct by its unique ID."""
    if len(st.session_state.adducts) > 1:
        st.session_state.adducts = [
            a for a in st.session_state.adducts if a["id"] != adduct_id
        ]

def _delete_neutral_loss(loss_id: str) -> None: 
    """Delete a neutral loss by its unique ID."""
    if len(st.session_state.neutral_losses) > 1:
        st.session_state.neutral_losses = [
            l for l in st.session_state.neutral_losses if l["id"] != loss_id
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

def _toggle_lost(adduct_id: str) -> None:
    """Invert mass sign when the 'Lost' checkbox changes."""
    mass_key = f"mass_{adduct_id}"
    st.session_state[mass_key] *= -1


def adduct_selection(element_monomere: List[str], label_formula: Dict[str, int] | None
                     ) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Adduct configuration and validation.

    This function renders an interactive interface allowing users to:
    - Define, edit, and delete adducts
    - Configure adduct mass values (m/z)
    - Define chemical formulas or fallback to adduct names
    - Specify whether an adduct is "lost" during ionization
    - Optionally override formulas with custom definitions

    The function manages both:
    - Streamlit session state (UI layer)
    - Internal adduct structure (data layer)

    It also performs validation and builds clean output dictionaries
    used for downstream spectrum analysis.

    Parameters
    ----------
    element_monomere : List[str]
        List of valid elements for monomers, used for formula validation.   
    label_formula : Dict[str, int] | None
        Optional dictionary of label elemental composition for validation.

    Returns
    -------
    adduct_mass : dict[str, float]
        Mapping of adduct names to their corresponding m/z values.

    adduct_compo : dict[str, dict[str, int]]
        Mapping of adduct names to their elemental composition
        (parsed from chemical formulas).

    adduct_is_lost : dict[str, bool]
        Mapping indicating whether each adduct is considered lost
        during ionization (True) or retained (False).

    Notes
    -----
    - Session state is used to persist UI interactions across reruns.
    - Each adduct is identified internally by a UUID stored in session state.
    - The function enforces consistency between user input and validated outputs.
    - Validation is performed on formulas and elemental composition.
    - Invalid adducts or incomplete entries are filtered before output.
    """

    on_adduct = st.toggle("Use custom adduct (By default H+ Na+ & K+) and chemical modifications", value=False)

    if on_adduct:
        st.markdown("ℹ️ *Lost: Check the box to indicate if the adduct is lost during ionization*")
        st.markdown("ℹ️ *Formula: Enter the net elemental composition of the adduct (e.g. H2O, CO, Na, H). Complex expressions such as -H2O+H are not supported; \
                    instead, enter the simplified net composition (HO) with Lost checkbox checked.*")
        
        _init_adducts_state()
       
        for i, adduct in enumerate(st.session_state.adducts):
            # st.markdown(f"Adduct {i+1}")

            # Init session state for each adduct if not already present
            # Use unique keys based on adduct ID to avoid collisions
            # Use session state to persist user input across reruns

            adduct_id = adduct["id"]

            name_key = f"name_{adduct_id}"
            mass_key = f"mass_{adduct_id}"
            formula_key = f"formula_{adduct_id}"
            is_lost_key = f"is_lost_{adduct_id}"
            use_formula_key = f"use_formula_{adduct_id}"

            defaults = {
                name_key: adduct["name"],
                mass_key: adduct["mass"],
                formula_key: adduct["formula"],
                is_lost_key: adduct["is_lost"],
                use_formula_key: adduct["formula"] != adduct["name"],
            }

            for key, default_value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            col_name, col_is_lost, col_mass, col_formula, col_delete = st.columns([2, 0.5, 2, 4, 0.5],
                                                    vertical_alignment="bottom")
            
            col_name.text_input(
                "Name",
                key=name_key,
            )

            col_is_lost.markdown(
                markdown_checkbox_title_rendering("Loss"),
                unsafe_allow_html=True,
            )

            col_is_lost.checkbox(
                "Loss",
                key=is_lost_key,
                label_visibility="collapsed",
                on_change=_toggle_lost,
                args=(adduct_id,),
            )
            is_lost = st.session_state[is_lost_key]
            col_mass.number_input(
                "m/z",
                key=mass_key,
                format="%.6f",
                step=0.000001,
                max_value=0.0 if is_lost else None,
                min_value=0.0 if not is_lost else None, 
            )

            with col_formula:
                sub_col_box, sub_col_formula = st.columns(
                    [0.05, 1],
                    vertical_alignment="bottom",
                )

                sub_col_box.markdown(
                    markdown_checkbox_title_rendering(
                        "Use a brut formula different from name"
                    ),
                    unsafe_allow_html=True,
                )

                sub_col_box.checkbox(
                    "Use Formula",
                    key=use_formula_key,
                    label_visibility="collapsed",
                )

                # -----------------------------
                # FORMULA VALUE LOGIC
                # -----------------------------

                use_formula = st.session_state[use_formula_key]
                # If disabled → formula follows name automatically
                if not use_formula:
                    st.session_state[formula_key] = st.session_state[name_key]

                sub_col_formula.text_input(
                    "formula",
                    key=formula_key,
                    disabled=not use_formula,
                    label_visibility="collapsed",
                )

            col_delete.button(
                label="❌" if len(st.session_state.adducts) > 1 else "—",
                key=f"delete_{adduct_id}",
                width=40,
                on_click=_delete_adduct,
                args=(adduct_id,),
                disabled=(len(st.session_state.adducts) == 1),
            )

            # ------------------------------------------------------------------
            # Synchronize adduct dictionary
            # ------------------------------------------------------------------

            adduct["name"] = st.session_state[name_key]
            adduct["mass"] = st.session_state[mass_key]
            adduct["formula"] = st.session_state[formula_key]
            adduct["is_lost"] = st.session_state[is_lost_key]
           
        st.button(
            "➕ Add Adduct",
            on_click=_add_adduct,
        )
        field = ["name", "mass", "formula"]
        valid_adducts, error_indices, has_any_input = ut.validate_rows(st.session_state.adducts,
                                                                    required_fields=field)
        adduct_mass, adduct_compo, adduct_is_lost = ut.build_entity_dicts(
            valid_adducts,
            entity_type="adduct")

        # ------------------------------------------------------------------
        # Warnings and errors for invalid adducts messages
        # ------------------------------------------------------------------

        if error_indices:
            error_adduct = [f"Adduct {i+1}" for i in error_indices]
            st.error(f"❌ Some adducts are incomplete {' / '.join(map(str, error_adduct))}")

        elif not has_any_input:
            st.error("⚠️ Please fill at least one adduct")

        for adduct in valid_adducts:
            error = ut.validate_entity(
                entity=adduct,
                monomer_elements=element_monomere,
                label_composition=label_formula,
                entity_type="adduct"
            )

            if error:
                st.error(error)
                adduct_compo = None
                break

    else:
        adduct_mass = cf.ADDUCT_MASS
        adduct_compo = cf.ADDUCTS
        adduct_is_lost = {name: False for name in cf.ADDUCT_MASS.keys()}

    st.session_state.adduct_mass = adduct_mass
    st.session_state.adduct_compo = adduct_compo
    st.session_state.adduct_loss = adduct_is_lost


def neutral_loss_selection(
    element_monomere: List[str],
    label_formula: Dict[str, int] | None
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Neutral loss configuration and validation.

    This function renders an interactive interface allowing users to:
    - Define, edit, and delete neutral losses (e.g., H2O, CO2, NH3)
    - Configure their mass values (negative m/z)
    - Define chemical formulas (required)

    Parameters
    ----------
    element_monomere : List[str]
        List of valid elements for monomers, used for formula validation.
    label_formula : Dict[str, int] | None
        Optional dictionary of label elemental composition for validation.

    Returns
    -------
    neutral_loss_mass : dict[str, float]
        Mapping of neutral loss names to their corresponding negative m/z values.
    neutral_loss_compo : dict[str, dict[str, int]]
        Mapping of neutral loss names to their elemental composition.
    """

    on_neutral_loss = st.toggle(
        "Use custom neutral losses (e.g., H2O, CO2, NH3)",
        value=False
    )

    if on_neutral_loss:
        st.markdown(
            "ℹ️ *Neutral losses are molecules lost during ionization (e.g., H₂O, CO₂). "
            "Their m/z values must be **negative** (e.g., -18.01056 for H₂O)."
        )

        _init_neutral_losses_state()

        for i, loss in enumerate(st.session_state.neutral_losses):
            # st.markdown(f"Neutral Loss {i+1}")

            loss_id = loss["id"]
            name_key = f"neutral_loss_name_{loss_id}"
            mass_key = f"neutral_loss_mass_{loss_id}"
            formula_key = f"neutral_loss_formula_{loss_id}"
            use_formula_key = f"use_neutral_loss_formula_{loss_id}"

            # Initialize session state if not present
            defaults = {
                name_key: loss["name"],
                mass_key: loss["mass"],
                formula_key: loss["formula"],
                use_formula_key: loss["formula"] != loss["name"],
            }

            for key, default_value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            col_name, col_mass, col_formula, col_delete = st.columns(
                [2, 2, 4, 0.5],
                vertical_alignment="bottom"
            )

            col_name.text_input(
                "Name",
                key=name_key,
            )

            col_mass.number_input(
                "m/z (must be negative)",
                key=mass_key,
                format="%.6f",
                step=0.000001,
                max_value=0.0,  # Force negative values
            )

            with col_formula:
                sub_col_box, sub_col_formula = st.columns(
                    [0.05, 1],
                    vertical_alignment="bottom",
                )

                sub_col_box.markdown(
                    markdown_checkbox_title_rendering(
                        "Use a brut formula different from name"
                    ),
                    unsafe_allow_html=True,
                )

                sub_col_box.checkbox(
                    "Use Formula",
                    key=use_formula_key,
                    label_visibility="collapsed",
                )

                use_formula = st.session_state[use_formula_key]
                if not use_formula:
                    st.session_state[formula_key] = st.session_state[name_key]

                sub_col_formula.text_input(
                    "Formula",
                    key=formula_key,
                    disabled=not use_formula,
                    label_visibility="collapsed",
                )

            col_delete.button(
                label="❌" if len(st.session_state.neutral_losses) > 1 else "—",
                key=f"delete_neutral_loss_{loss_id}",
                width=40,
                on_click=_delete_neutral_loss,
                args=(loss_id,),
                disabled=(len(st.session_state.neutral_losses) == 1),
            )

            # Synchronize loss dictionary
            loss["name"] = st.session_state[name_key]
            loss["mass"] = st.session_state[mass_key]
            loss["formula"] = st.session_state[formula_key]

        st.button(
            "➕ Add Neutral Loss",
            on_click=_add_neutral_loss,
        )

        # Validation
        field = ["name", "mass", "formula"]
        valid_losses, error_indices, has_any_input = ut.validate_rows(
            st.session_state.neutral_losses,
            required_fields=field
        )

        neutral_loss_mass, neutral_loss_compo, _ = ut.build_entity_dicts(
            valid_losses,
            entity_type="neutral_loss"
        )

        # Warnings
        if error_indices:
            error_losses = [f"Neutral Loss {i+1}" for i in error_indices]
            st.error(f"❌ Some neutral losses are incomplete {' / '.join(error_losses)}")
        elif not has_any_input:
            st.error("⚠️ Please fill at least one neutral loss")

        for neutral_loss in valid_losses:
            error = ut.validate_entity(
                entity=neutral_loss,
                monomer_elements=element_monomere,
                label_composition=label_formula,
                entity_type="neutral_loss"
            )

            if error:
                st.error(error)
                neutral_loss_compo = None
                break

    else:
        # No neutral losses defined, return empty dictionaries
        neutral_loss_mass = {}
        neutral_loss_compo = {}

    return neutral_loss_mass, neutral_loss_compo


def neutral_loss_adduct_association_table(
    neutral_loss_mass: Dict[str, float],
    adducts: Dict[str, float],
) -> Dict[str, List[str]]:
    """
    Render a table to associate neutral losses (rows) with adducts (columns).
    Returns a dictionary: {neutral_loss_name: [list of allowed adducts]}.

    Parameters
    ----------
    neutral_loss_mass : Dict[str, float]
        Mapping of neutral loss names to their negative m/z values.
    adducts : Dict[str, float]
        Dictionary of available adducts (from st.session_state.adduct_mass).

    Returns
    -------
    Dict[str, List[str]]
        Mapping of neutral loss names to their allowed adducts.
    """
    if not neutral_loss_mass  or not adducts:
        st.warning("⚠️ No neutral losses or adducts available for association.")
        return {}

    st.markdown("##### Neutral Loss × Adduct Association Table")
    st.info(
        "ℹ️ Check the boxes to allow combinations between neutral losses (rows) and adducts (columns)."
    )

    # Initialize the association dictionary with empty lists for each neutral loss
    association_dict = {
        loss_name: [] for loss_name in neutral_loss_mass.keys()
    }

    # Create a DataFrame for the table (optional, for display)
    header_columns = ["Neutral Loss"] + list(adducts.keys())
    header = st.columns(len(header_columns))
    for i, col_name in enumerate(header_columns):
        header[i].write(f"**{col_name}**")

    # Lignes du tableau (une par perte de neutre)
    for loss_name in neutral_loss_mass:
        row_columns = st.columns(len(header_columns))

        # Première colonne : nom de la perte de neutre
        row_columns[0].write(loss_name)

        # Colonnes suivantes : checkbox pour chaque adduct
        for j, adduct_name in enumerate(adducts.keys()):
            # Clé unique pour chaque checkbox
            checkbox_key = f"assoc_{loss_name}_{adduct_name}"

            default_value = False  # Default unchecked; can be modified based on previous selections

            # Afficher la checkbox
            is_checked = row_columns[j + 1].checkbox(
                "",
                key=checkbox_key,
                value=default_value,
                label_visibility="collapsed",
            )

            # Mettre à jour association_dict
            if is_checked:
                association_dict[loss_name].append(adduct_name)

    return association_dict


def label_selection():
    """
    Streamlit UI component to select and validate a labeling strategy.

    The user can choose between:
    - No label
    - A custom label (name + chemical formula)
    - A default predefined label (3-AQ)

    Depending on the selection, the function validates the input,
    parses the chemical formula, and checks element consistency.

    Returns
    -------
    use_label : bool
        Whether labeling is enabled or not.
    label_name : str or None
        Name of the selected label. None if no label is used or input is invalid.
    label_formula : str or None
        Chemical formula of the selected label. None if not applicable or invalid.
    compo : dict or None
        Parsed elemental composition of the label.
        - For custom labels: result of `ut.parse_formula`
        - For default label: predefined composition (`cf.LABEL_3AQ`)
        - None if labeling is disabled or invalid input

    Notes
    -----
    Validation steps for custom labels:
    1. Both name and formula must be provided.
    2. The formula is parsed into elemental composition.
    3. Elements are checked against the allowed library via `ut.check_formula`.
    4. Errors are displayed in the Streamlit UI if validation fails.

    UI Behavior
    -----------
    - If `use_label` is False: no label is applied.
    - If `on_label` is True: user-defined label is used.
    - Otherwise: default label "3-AQ" is used.
    """
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


def build_badges(config):
    return [
        *[f":blue-badge[{m}]" for m in config.monomers_list],
        f":orange-badge[{config.params.iso_score}]",
        *[f":green-badge[{name}]" for name in config.adduct_mass.keys()],
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