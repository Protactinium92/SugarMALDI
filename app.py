"""
Streamlit-based user interface for MALDI-MS spectra analysis and visualization.

This application provides an interactive frontend for:
- Uploading and processing MALDI spectra (CSV files with 'm/z' and 'Intensity' columns)
- Visualizing high-resolution spectra (~1M data points)
- Comparing experimental data against oligosaccharide libraries
- Generating analysis reports in standardized CSV format

Key Features:
- Polars integration for memory-efficient data handling (LazyFrames for ~1M+ row datasets)
- Streamlit caching (@st.cache_data) for responsive UI with large datasets
- Interactive visualization using Plotly/Matplotlib with custom styling:
  * Scaled intensity values for comparative analysis
  * Differentiated peak markers (identified vs unidentified)

Technical Implementation:
- Backend separation: Heavy computations delegated to spectre_analysis.py
- Frontend optimization: Cached operations for:
  * Spectrum rendering (base64-encoded images)
  * Peak detection and annotation
  * Library matching results
- Data format compliance:
  * Input: CSV with mandatory 'm/z' and 'Intensity' columns
  * Output: Standardized CSV with 'id', 'masse', and sample metadata

Example Workflow:
    1. Upload multiple spectra → automatic merging on m/z values
    2. Set detection parameters (window size, intensity threshold)
    3. Compare against oligosaccharide library (configurable mass offset)
    4. Download results as ZIP archive with:
       - Processed spectra (1/0 header format)
       - Matching results with mass accuracy metrics
       - Visual overlays of identified peaks

Dependencies:
    see pyproject.toml

"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-08"
__version__ = "1.2"


import streamlit as st
import polars as pl
import plotly.graph_objects as go
import atexit  # Clean closure of tempory files

from src import front_functions as ff
from src import spectrum_analysis as sp_ana
from src import helper
from src import utils as ut
from src import session_manager as ssm


if __name__=="__main__":
    st.set_page_config(page_title="SugarMALDI", page_icon="pictures/favicon.png", layout="wide")
    ff.display_logo()

    ssm.SessionManager.init()

    delimiter = ff.choose_delimiter()
    spectra = ff.upload_files(delimiter)

    # Check if their is an ulpoad wherase its crash
    if spectra is not None and len(spectra) > 0:

        st.session_state.spectra_list = list(spectra.keys())

        snapshot = ssm.SessionManager.get_snapshot()
        has_snapshot = snapshot is not None and st.session_state.analysis_ran

        st.markdown("---")
        st.subheader("Chemical Configuration")

        monomers_list, monomers_df = ff.monomer_selection()

        if has_snapshot:
                ssm.SessionManager.warn_if_any_changed(
                    "Monomers selection",
                    [(monomers_list, snapshot.monomers_list)],
                    condition=(snapshot.monomers_list is not None and monomers_list)
                )

        st.markdown("##### Adduct & Label Selection")

        adduct_container = st.container(border=True)
        with adduct_container:
            ff.adduct_selection()
            adduct_mass = st.session_state.get("adduct_mass", {})
            adduct_compo = st.session_state.get("adduct_compo", {})
     
            if has_snapshot:
                ssm.SessionManager.warn_if_any_changed(
                    "Adduct parameters",
                    [
                        (adduct_mass, snapshot.adduct_mass),
                        (adduct_compo, snapshot.adduct_compo),
                    ]
                )
            
            for adduct in adduct_compo:
                st.write(f"Adduct: {adduct}, Mass: {adduct_mass[adduct]:.6f}, Composition: {adduct_compo[adduct]}")

        label_container = st.container(border=True)
        with label_container:
            label_use, label_name, label_formula, label_dict = ff.label_selection()
             
            if has_snapshot:
                ssm.SessionManager.warn_if_any_changed(
                    "Label parameters",
                    [
                        (label_use, snapshot.label_use),
                        (label_name, snapshot.label_name),
                        (label_formula, snapshot.label_formula),
                    ]
                )

        badges_adduct = " ".join(
            [f":green-badge[{name}: {mass:.6f}]" for name, mass in adduct_mass.items()]
        )
        badges_label = (
            ":red-badge[No Label]"
            if not label_use
            else (
                f":red-badge[{label_name}: {label_formula}]"
                if label_name and label_formula
                else ""
            )
        )
        st.markdown(badges_adduct + badges_label)

        st.markdown("---")
        st.subheader("Process and Download")

        # Get default or saved parameter
        params = ssm.SessionManager.get_params()
        
        with st.expander("⚙️ Advanced options for peak detection", expanded=False):
            # Window parameter only for debuging in expert mode
            window = params.window
            # Use only for segmentation and performance
            # Uncomment bellow for use it in the UI
            # window = st.number_input("Sub-window for peak detection (Da)", value=params.window, step=1,
            #                          min_value=100, max_value=2000,
            #                          help=helper.window())
            st.subheader("Detection parameters 🔎")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                prominence = st.number_input("Prominence", value=params.prominence, step=0.1e5,
                                min_value=0.0,
                                help=helper.prominence())
            with col2:
                height_min = st.number_input("Minimum height of a peak", value=params.height_min, step=0.1e5,
                                           min_value=0.0,
                                           help=helper.peak_height())
            with col3:
                dist_min = st.number_input("Minimum m/z distance between peaks", value=params.dist_min, step=1,
                                           min_value=0,
                                           help=helper.minimum_distance())

            with st.expander(f"💡 Peak Detection Demo"):
                helper.display_peak_detection()

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Match peaks parameter 🎯")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    rm_threshold = st.number_input('Mass accuracy threshold (ΔM in ppm)', value=params.rm_threshold, step=1,
                                                help=helper.rm_threshold())
                with subcol2:
                    number_candidate = st.number_input(
                        "Number of best mass matches selection",
                        value=params.number_candidate,
                        step=1,
                        min_value=1,
                        help=helper.number_candidate()
                    )
            with col2:
                st.subheader("Isotope scoring 🧬")

                subcol1, subcol2 = st.columns(2)

                with subcol1:
                    number_isotope = st.number_input(
                        "Number of isotopes",
                        value=params.number_iso,
                        step=1,
                        help=helper.number_isotopes()
                    )

                with subcol2:
                    method_iso_score = st.selectbox(
                        "Scoring method",
                        options=list(helper.SCORING_HELP.keys()),
                        index=5,
                        help=helper.method_score()
                    )

            with st.expander(f"💡 Details method"):
                 col1, col2 = st.columns([1,1], vertical_alignment="top")
                 with col1:
                    st.markdown(helper.SCORING_HELP[method_iso_score][1])
                 with col2:
                    st.markdown(helper.SCORING_HELP[method_iso_score][0])

            # Create a temporary params object to compare with the saved session params
            current_params = ssm.AnalysisParams(
                window=window,
                prominence=prominence,
                dist_min=dist_min,
                height_min=height_min,
                rm_threshold=rm_threshold,
                number_candidate=number_candidate,
                number_iso=number_isotope,
                iso_score=method_iso_score
            )

            # Warn if params have changed (only if analysis has been run before)
            if has_snapshot:
                ssm.SessionManager.warn_if_any_changed(
                    "Detection parameters",
                    [(current_params, snapshot.params)]
                )

   
        # Create a temporary config object to save
        current_config = ssm.AnalysisConfig(
            params=current_params,
            monomers_list=monomers_list,
            adduct_mass=adduct_mass,
            adduct_compo=adduct_compo,
            label_use=label_use,
            label_name=label_name,
            label_formula=label_formula
        )

        # Check of all field are filled
        is_valid, errors = ut.validate_inputs(
            monomers_list,
            adduct_compo,
            label_use,
            label_dict
        )

        for err in errors:
            st.error(err)

        if not is_valid:
             st.button(
                "🔍 Spectra analysis",
                width="stretch",
                disabled=True
             )        
        else:    
            if st.button(
                "🔍 Spectra analysis",
                on_click=ssm.SessionManager.run_analysis,  # To run first avoiding warning latence
                args=(current_config,),
                width="stretch"
            ):
                with st.spinner('🔄 Treatment ongoing...'):
                    # --- Logger Streamlit ---
                    progress = ff.StreamlitProgress(len(spectra))
                    log_placeholder = st.empty()
                    root_logger, handler = ff.init_streamlit_logger(log_placeholder)

                    zip_buffer, peak_dict = sp_ana.execute_process(
                        spectra,
                        monomers_list,
                        monomers_df,
                        adduct_mass,
                        adduct_compo,
                        label_use,
                        label_dict,
                        rm_threshold,
                        window,
                        prominence,
                        height_min,
                        dist_min,
                        number_candidate,
                        number_isotope,
                        method_iso_score,
                        progress
                    )

                    # Store results in sessions state 
                    # Not in session manager which is dedicated to parameters only
                    st.session_state.peak_dict = {
                        spectrum_name: peaks_lf.lazy() 
                        for spectrum_name, peaks_lf in peak_dict.items()
                    }
                    st.session_state.analysis_results = zip_buffer

                    # Clear logs
                    root_logger.removeHandler(handler)
                    handler.close()
                    log_placeholder.empty()
                    progress.clear()

        if st.session_state.analysis_ran:
            st.success('✅ Analysis completed !') 
            
            with st.expander("⚙️ Advanced options to display the spectrum", expanded=False):
                dpi = st.selectbox("Image DPI (quality)", [100, 150, 200, 300], index=1, key="dpi")
            
            # Choose the spectra to show
            selected_spectrum = st.selectbox("Select spectrum", st.session_state.spectra_list, key="spectrum_selection")
            peak_lf = st.session_state.peak_dict[selected_spectrum]
            nb_peak = peak_lf.select(pl.len()).collect().item()  # .item() extract the scalar value
            nb_id_peak = peak_lf.filter(pl.col("Identifications").is_not_null()).select(pl.len()).collect().item()  # .item() extract the scalar value
            
            st.write(f'Number of peak identified vs detected: {nb_id_peak} / {nb_peak}')

            # Show specytrum plot
                                
            # make background image
            x, y = ff.define_x_y(spectra, selected_spectrum,)
            bg_src, x_min, x_max, y_min, y_max = ff.make_background_b64(x, y, dpi=dpi)

            # PLot the mass spectrum
            fig = ff.build_plotly_figure(bg_src, peak_lf,
                                    mass_col="m/z",
                                    intensity_col="Intensity",
                                    oligo_col_name="Identifications",
                                    x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max,)
            st.plotly_chart(fig, width='stretch')
            
            # Show peak table
            st.write("### **Identified peaks**")
            col1, col2 = st.columns([1,1])
            with col1:
                    st.markdown("""
                        <div>
                            <span style='font-style: italic;'>
                                Delta_M in ppm, m/z in dalton, query_mass in ppm<br>
                                Note: The formula and the mass contain the adduit and the label 3-AQ
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
            with col2:
                badges = ff.build_badges(snapshot)
                st.markdown("Information selections: " + " ".join(badges))

            st.dataframe(peak_lf
                         .filter(pl.col("Identifications").is_not_null())
                         .sort(pl.col("Intensity"), descending=False)
                         .drop(pl.col("query_mass"))
                         .collect())

            if st.session_state.analysis_results:
                st.download_button(
                    label="⬇️ Download all results ZIP",
                    data=st.session_state.analysis_results.getvalue(),
                    file_name="results_maldi.zip",
                    mime="application/zip",
                    width="stretch",
                )

    atexit.register(sp_ana.cleanup)
