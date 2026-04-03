"""
Streamlit-based user interface for MALDI-TOF spectra analysis and visualization.

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
import spectre_analysis as sp_ana


if __name__=="__main__":
    st.set_page_config(page_title="SugarMALDI", page_icon="pictures/favicon.png", layout="wide")
    ff.display_logo()

    delimiter = ff.choose_delimiter()
    spectra = ff.upload_files(delimiter)

    # Check if their is an ulpoad wherase its crash
    if spectra is not None and len(spectra) > 0:

        st.session_state.spectra_list = list(spectra.keys())
       
        if "monomers_list" not in st.session_state:
            st.session_state.monomers_list = None
            st.session_state.monomers_df = None
        monomers_list, monomers_df = ff.monomer_selection()

        if (st.session_state.monomers_list is not None and     # 1 Check the 1st launch
            monomers_list and                                  # 2 chek if not empty (warning in the function)
            monomers_list != st.session_state.monomers_list):  # 3 check if the list changes
            st.warning("⚠️ Warning: List of monomers changes. Please re-run the analysis to apply the new settings.")

        
        # Initialize session state for parameters if not already set
        # Default values, used for the first run
        # Allowed to generate a warning if the user change them
        if 'window' not in st.session_state:
            st.session_state.window = 500
        if 'dist_min' not in st.session_state:
            st.session_state.dist_min = 0
        if 'rm_threshold' not in st.session_state:
            st.session_state.rm_threshold = 5
        if 'prominence' not in st.session_state:
            st.session_state.prominence = 3e5  # Float 
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if "number_iso" not in st.session_state:
            st.session_state.number_iso = 5
        if "iso_score" not in st.session_state:
            st.session_state.iso_score = None
        
        # Parameter window
        with st.expander("⚙️ Advanced options for peak detection", expanded=False):
            st.subheader("Detection parameters")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                window = st.number_input("Sub-window for peak detection (Da)", value=st.session_state.window, step=1)
            with col2:
                dist_min = st.number_input("Minimum m/z distance between peaks", value=st.session_state.dist_min, step=1)
            with col3:
                prominence = st.number_input("Prominence", value=st.session_state.prominence, step=0.1e5,
                                              help="Minimum height difference between peak and its neighbores")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("🎯 Match peaks parameter")
                rm_threshold = st.number_input('Mass accuracy threshold (ΔM in ppm)', value=st.session_state.rm_threshold, step=1,
                                               help=ff.help_rm_threshold())
            with col2:
                st.subheader("🧬 Isotope scoring")

                subcol1, subcol2 = st.columns(2)

                with subcol1:
                    number_isotope = st.number_input(
                        "Number of isotopes",
                        value=st.session_state.number_iso,
                        step=1,
                        help="Number of isotopic peaks used in scoring"
                    )

                with subcol2:
                    method_iso_score = st.selectbox(
                        "Scoring method",
                        options=list(ff.SCORING_HELP.keys()),
                        index=5,
                        help=ff.help_method_score()
                    )

            with st.expander(f"💡 Details method {method_iso_score}"):
                st.markdown(ff.SCORING_HELP[method_iso_score])
                    
            # Detect changes and warn the user
            if (((window != st.session_state.window) or
                (dist_min != st.session_state.dist_min) or
                (rm_threshold != st.session_state.rm_threshold) or 
                (number_isotope != st.session_state.number_iso) or
                (method_iso_score != st.session_state.iso_score))and
                (st.session_state.analysis_results != None)):
                st.warning("⚠️ Warning: Parameter(s) changed. Please re-run the analysis to apply the new settings.")


        st.markdown("---")
        st.subheader("Process and Download")
        if monomers_list:
            
            if st.button("🔍 Spectra analysis"):
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
                        dist_min,
                        prominence,
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
                    st.session_state.dist_min = dist_min
                    st.session_state.rm_threshold = rm_threshold
                    st.session_state.monomers_list = monomers_list
                    st.session_state.monomers_df = monomers_df
                    st.session_state.prominence = prominence
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
            col1, col2 = st.columns([1,2])
            with col1:
                st.write("*Delta_M in ppm, m/z in datlton, query_mass in ppm*")
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
