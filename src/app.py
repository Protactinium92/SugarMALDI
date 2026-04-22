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
__version__ = "1.0"


import streamlit as st
import polars as pl
import plotly.graph_objects as go
import atexit  # Clean closure of tempory files

import front_functions as ff
import spectrum_analysis as sp_ana
import helper


if __name__=="__main__":
    st.set_page_config(page_title="SugarMALDI", page_icon="pictures/favicon.png", layout="wide")
    ff.display_logo()

    delimiter = ff.choose_delimiter()
    spectra = ff.upload_files(delimiter)

    # Check if their is an ulpoad wherase its crash
    if spectra is not None and len(spectra) > 0:

        st.session_state.spectra_list = list(spectra.keys())

        st.markdown("---")
        st.subheader("Chemical Configuration")

        if "monomers_list" not in st.session_state:
            st.session_state.monomers_list = None
            st.session_state.monomers_df = None
        monomers_list, monomers_df = ff.monomer_selection()

        if (st.session_state.monomers_list is not None and     # 1 Check the 1st launch
            monomers_list and                                  # 2 chek if not empty (warning in the function)
            monomers_list != st.session_state.monomers_list):  # 3 check if the list changes
            st.warning("⚠️ Warning: List of monomers changes. Please re-run the analysis to apply the new settings.")

        st.markdown("##### Adduct & Label Selection")

        adduct_container = st.container(border=True)
        with adduct_container:
            ff.adduct_selection()
            adduct_mass = st.session_state.get("adduct_mass", {})
            adduct_compo = st.session_state.get("adduct_compo", {})
        
        label_container = st.container(border=True)
        with label_container:
            label_name, label_formula, label_dict = ff.label_selection()
        
        # TODO: Put warnings
        badges_adduct = " ".join(
            [f":green-badge[{name}: {mass:.6f}]" for name, mass in adduct_mass.items()]
        )
        badges_label = (
            f":red-badge[{label_name}: {label_formula}]"
            if label_name and label_formula
            else ""
        )
        st.markdown(badges_adduct + badges_label)

        st.markdown("---")
        st.subheader("Process and Download")

        # Initialize session state for parameters if not already set
        # Default values, used for the first run
        # Allowed to generate a warning if the user change them
        if 'window' not in st.session_state:
            st.session_state.window = 500
        if 'prominence' not in st.session_state:
            st.session_state.prominence = 3e5  # Float 
        if 'dist_min' not in st.session_state:
            st.session_state.dist_min = 0
        if 'height_min' not in st.session_state:
            st.session_state.height_min = 5e5
        if 'rm_threshold' not in st.session_state:
            st.session_state.rm_threshold = 5
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'number_candidate' not in st.session_state:
            st.session_state.number_candidate = 4
        if "number_iso" not in st.session_state:
            st.session_state.number_iso = 5
        if "iso_score" not in st.session_state:
            st.session_state.iso_score = None
        if "scoring_expander_open" not in st.session_state:
            st.session_state.scoring_expander_open = False
        
        with st.expander("⚙️ Advanced options for peak detection", expanded=False):
            # Window parameter only for debuging in expert mode
            window = st.session_state.window
            # Use only for segmentation and performance
            # Uncomment bellow for use it in the UI
            # window = st.number_input("Sub-window for peak detection (Da)", value=st.session_state.window, step=1,
            #                          min_value=100, max_value=2000,
            #                          help=helper.window())
            st.subheader("Detection parameters 🔎")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                prominence = st.number_input("Prominence", value=st.session_state.prominence, step=0.1e5,
                                min_value=0.0,
                                help=helper.prominence())
            with col2:
                height_min = st.number_input("Minimum height of a peak", value=st.session_state.height_min, step=0.1e5,
                                           min_value=0.0,
                                           help=helper.peak_height())
            with col3:
                dist_min = st.number_input("Minimum m/z distance between peaks", value=st.session_state.dist_min, step=1,
                                           min_value=0,
                                           help=helper.minimum_distance())

            with st.expander(f"💡 Peak Detection Demo"):
                 helper.display_peak_detection()

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Match peaks parameter 🎯")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    rm_threshold = st.number_input('Mass accuracy threshold (ΔM in ppm)', value=st.session_state.rm_threshold, step=1,
                                                help=helper.rm_threshold())
                with subcol2:
                    number_candidate = st.number_input(
                        "Number of best mass matches selection",
                        value=st.session_state.number_candidate,
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
                        value=st.session_state.number_iso,
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
                    
            # Detect changes and warn the user
            if (((window != st.session_state.window) or
                (prominence != st.session_state.prominence) or
                (height_min != st.session_state.height_min) or 
                (dist_min != st.session_state.dist_min) or
                (rm_threshold != st.session_state.rm_threshold) or  
                (number_candidate != st.session_state.number_candidate) or
                (number_isotope != st.session_state.number_iso) or
                (method_iso_score != st.session_state.iso_score))and
                (st.session_state.analysis_results != None)):
                st.warning("⚠️ Warning: Parameter(s) changed. Please re-run the analysis to apply the new settings.")

        if monomers_list and adduct_mass and label_dict:
            
            if st.button("🔍 Spectra analysis", use_container_width=True):
                with st.spinner('🔄 Treatment ongoing...'):

                    # --- Logger Streamlit ---
                    progress = ff.StreamlitProgress(len(spectra))
                    log_placeholder = st.empty()
                    root_logger, handler = ff.init_streamlit_logger(log_placeholder)

                    zip_buffer, peak_dict = sp_ana.execute_process(
                        spectra,
                        monomers_list,
                        monomers_df,
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
                    st.session_state.peak_dict = {
                        spectrum_name: peaks_lf.lazy() 
                        for spectrum_name, peaks_lf in peak_dict.items()
                    }
                    # Update session state with new values
                    # For warning messages
                    st.session_state.analysis_results = zip_buffer
                    st.session_state.window = window
                    st.session_state.prominence = prominence
                    st.session_state.height_min = height_min
                    st.session_state.dist_min = dist_min
                    st.session_state.rm_threshold = rm_threshold
                    st.session_state.monomers_list = monomers_list
                    st.session_state.monomers_df = monomers_df
                    st.session_state.number_candidate = number_candidate
                    st.session_state.number_iso = number_isotope
                    st.session_state.iso_score = method_iso_score

                    # Clear logs
                    root_logger.removeHandler(handler)
                    handler.close()
                    log_placeholder.empty()
                    progress.clear()

        if 'peak_dict' in st.session_state:
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
                badges = " ".join([f":blue-badge[{m}]" for m in monomers_list])
                badges = badges + f" :orange-badge[{method_iso_score}]"
                st.markdown(f"Information selections: {badges}")
            st.dataframe(peak_lf
                         .filter(pl.col("Identifications").is_not_null())
                         .sort(pl.col("Intensity"), descending=False)
                         .collect())


            if st.session_state.analysis_results:
                st.download_button(
                    label="⬇️ Download all results ZIP",
                    data=st.session_state.analysis_results.getvalue(),
                    file_name="results_maldi.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    atexit.register(sp_ana.cleanup)
