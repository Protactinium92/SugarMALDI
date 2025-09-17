"""
    User interface for the streatment of the spectra
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-08"
__version__ = "1.0"


import streamlit as st
import spectre_analysis_v1 as sp_ana
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io, base64
import numpy as np
from pathlib import Path



def upload_files() -> pl.DataFrame:
    """
    Handles the upload of MS spectra CSV files via Streamlit's file uploader,
    parses them using `sp_ana.parse_spectra`, and displays a preview of the merged data.

    Returns:
        pl.DataFrame:
            - The merged spectra DataFrame if files are uploaded and parsed successfully.
            - None if no files are uploaded or if an error occurs.

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
            st.subheader("Files have been successfully imported")
            st.write("### See spectrum data")
            st.dataframe(spectra.head(100))  # Show preview of the first 100 rows
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
    #ax.margins(0)      # Supprime les marges internes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0,  transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}", float(x.min()), float(x.max()), float(y.min()), float(y.max())


# --- peak picking simple (local maxima + top_k) ---
@st.cache_data
def filter_top_peak(df: pl.DataFrame, mass_col: str, intensity_col: str, top_k: int):
    df_filtered = df.sort(intensity_col, descending=True).head(top_k).sort(mass_col)
    return df_filtered


def build_plotly_figure(bg_src, mass_col: str, intensity_col: str, x_min, x_max, y_min, y_max, df_peaks):
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
    fig.add_trace(go.Scattergl(
        x=df_peaks[mass_col].to_numpy(),
        y=df_peaks[intensity_col].to_numpy(),
        mode="markers",
        marker=dict(size=4, color='red'),

        # mode="lines",  # Affiche uniquement les lignes (pas de markers)
        # line=dict(width=1),

        name="Peaks",
        hovertemplate="m/z: %{x}<br>Intensity: %{y}<extra></extra>"
    ))
    fig.update_xaxes(title_text="m/z", range=[x_min, x_max])
    fig.update_yaxes(title_text="Intensity", range=[y_min, y_max])
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40))
    return fig


if __name__ == "__main__":
    st.title('SugarMALDI')
    spectra = upload_files()
    
    # Check if their is an ulpoad wherase its crash
    if spectra is not None and len(spectra) > 0:

        # Récupérer la première colonne (m/z)
        mz_col_name = spectra.columns[0]

        # Lister les autres colonnes disponibles
        other_cols = spectra.columns[1:]

        # Choisir une colonne parmi celles-ci (UI Streamlit par exemple)
        selected_col_name = st.selectbox("Select spectrum", other_cols)

        with st.expander("⚙️ Advanced options to display the spectrum", expanded=False):
            col1, col2 = st.columns([1,1])
            with col1:
                window = st.number_input("Sub-window for looking peak", value=500, step=1)
                dist_min = st.number_input("Distance m/z min between peak", value=15, step=1)
            with col2:
                top_k = st.slider("Show number of peaks (top K)", 1, 500, 500, step=1)
                dpi = st.selectbox("Image DPI (quality)", [100, 150, 200, 300], index=1)
             
        peaks_dict = sp_ana.extract_and_detect_peaks(
            spectra,
            mz_col_name,
            other_cols,
            window,
            dist_min
        )

        # Show only the Top k peaks in the selected zone

        # Créer un DataFrame avec m/z + spectre choisi
        spectrum = spectra.select([mz_col_name, selected_col_name]) 

        x_all = spectrum[mz_col_name].to_numpy()
        y_all = spectrum[selected_col_name].to_numpy()
        
        # UI controls
        mz_min, mz_max = st.slider("Use the slider to zoom with a better rendering", float(x_all.min()), float(x_all.max()),
                                (float(x_all.min()), float(x_all.max())))
        mask = (spectrum[mz_col_name] >= mz_min) & (spectrum[mz_col_name] <= mz_max)
        spectrum = spectrum.filter(mask)
        x = spectrum[mz_col_name].to_numpy()
        y = spectrum[selected_col_name].to_numpy()
        
        # make background image
        bg_src, x_min, x_max, y_min, y_max = make_background_b64(x, y, dpi=dpi)

        # Selection of top k peaks
        df_peaks = filter_top_peak(
            peaks_dict[selected_col_name],
            mz_col_name,
            selected_col_name,
            top_k=top_k)

        # PLot the mass spectrum
        fig = build_plotly_figure(bg_src, mz_col_name,
            selected_col_name, x_min, x_max, y_min, y_max, df_peaks)
        st.plotly_chart(fig, use_container_width=True)

        monomers = sp_ana.read_csv(Path(__file__).parents[1]/"data"/"monomers.csv", sep=";")
        st.dataframe(monomers.collect())

        with st.expander("⚙️ Advanced options for plotting", expanded=False):
            offset = st.number_input("Potential offset of the spectre in ppm", value=5, step=1)
            top_k2 = st.slider("Number of peaks to display (top K)", 1, 500, 500, step=1)
            
        # Comparison to clean library
        check_oligo_dict = sp_ana.compare_to_library(
            peaks_dict,
            offset * 10e-6
        )
        
        oligo = check_oligo_dict[selected_col_name].collect()
        st.write(f"Number of oligosacharride found: {oligo.select(pl.len()).item()} / "
                 f"{peaks_dict[selected_col_name].select(pl.len()).item()} peaks")

        show_oligo = filter_top_peak(
            oligo,
            "query_mass",
            "intensity",
            top_k=top_k2)
        
        st.dataframe(show_oligo)



