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
from scipy.signal import find_peaks as fp


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



@st.cache_data
def find_all_peaks(intensity: np.ndarray, height=None, distance=None):
    # Renvoie les indices des peaks
    peaks, _ = fp(intensity, height=height, distance=distance)
    return peaks


# --- peak picking simple (local maxima + top_k) ---
@st.cache_data
def detect_peaks_polars(df: pl.DataFrame, mass_col: str, intensity_col: str, top_k: int, quantile_thresh: float):
    qval = float(df[intensity_col].quantile(quantile_thresh))
    df_filtered = df.filter(pl.col(intensity_col) >= qval)
    df_filtered = df_filtered.sort(intensity_col, descending=True).head(top_k).sort(mass_col)
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
        marker=dict(size=1),

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
        selected_col_name = st.selectbox("Sélectionner le spectre à visualiser", other_cols)

        # Créer un DataFrame avec m/z + spectre choisi
        spectrum_0 = spectra.select([mz_col_name, selected_col_name]) 

        x_all = spectrum_0[mz_col_name].to_numpy()
        y_all = spectrum_0[selected_col_name].to_numpy()
        
        
        # UI controls
        col1, col2 = st.columns([3,1])
        with col2:
            top_k = st.slider("Nombre de pics (top K)", 100, 5000, 1000, step=100)
            q = st.slider("Seuil quantile", 0.90, 0.999, 0.99, step=0.001)
            dpi = st.selectbox("DPI image (qualité)", [100, 150, 200, 300], index=1)

        # Optionnel : zoom window (re-génère image pour la fenêtre) 
        use_window = st.checkbox("Activer fenêtre m/z pour re-générer fond (meilleur rendu au zoom)", value=False)
        if use_window:
            mz_min, mz_max = st.slider("Fenêtre m/z", float(x_all.min()), float(x_all.max()),
                                    (float(x_all.min()), float(x_all.max())))
            mask = (spectrum_0[mz_col_name] >= mz_min) & (spectrum_0[mz_col_name] <= mz_max)
            sub = spectrum_0.filter(mask)
            x = sub[mz_col_name].to_numpy()
            y = sub[selected_col_name].to_numpy()
        else:
            sub = spectrum_0
            x = x_all; y = y_all

        # make background image
        bg_src, x_min, x_max, y_min, y_max = make_background_b64(x, y, dpi=dpi)

        peak_indices = find_all_peaks(sub[selected_col_name])
        spectrum_peaks = sub[peak_indices]
        df_peaks = detect_peaks_polars(
            spectrum_peaks,
            mz_col_name,
            selected_col_name,
            top_k=top_k,
            quantile_thresh=q)
        fig = build_plotly_figure(bg_src, mz_col_name,
            selected_col_name, x_min, x_max, y_min, y_max, df_peaks)
        st.plotly_chart(fig, use_container_width=True)

        # Too heavy
        # fig = build_plotly_figure(bg_src, mz_col_name,
        #     selected_col_name, x_min, x_max, y_min, y_max, spectrum_0)
        # st.plotly_chart(fig, use_container_width=True)

    
    




