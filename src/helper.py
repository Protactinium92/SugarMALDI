
__author__  = "Pierre-Alexandre HO"


import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

SCORING_HELP = {
        "cosine": """
    ### Cosine similarity
    - Compare the angle between intensity vectors
    - Range: 0 → 1
    - Goal: 1 = perfect match, 0 = no similarity  
    ✔ Robust to scaling  
    ✔ Fast and stable  
    """,

        "pearson": """
    ### Pearson correlation
    - Measures linear correlation between patterns
    - Range: -1 → 1 (1 = perfect correlation) 
    - Goal: 1 = identical patterns in shape, 0 = no correlation, -1 = inverse correlation   
    ✔ Sensitive to pattern shape  
    ✖ Unstable with few peaks  
    """,

        "chi2": """
    ### Chi-square distance
    - Measures deviation between distributions
    - Range: 0 → ∞ (1 / (1 + χ²) used as score → 0 → 1)
    - Goal: 1 = identical patterns, 0 = large deviations  
    ✔ Penalizes large errors strongly  
    ✔ Good mismatch detector  
    ✖ Sensitive to noise  
    """,

        "kl": """
    ### Kullback-Leibler divergence
    - Measures information loss between distributions 
    - Range: 0 → ∞ (1 / (1 + KL) used as score → 0 → 1)
    - Goal: 1 = identical distributions, 0 = large information loss  
    ✔ Very discriminative  
    ✖ Penalizes missing peaks strongly  
    """,

        "wasserstein": """
    ### Wasserstein distance (Earth Mover’s Distance)
    - Measures distance between distributions  
    - Range: 0 → ∞ (1 / (1 + W) used as score → 0 → 1)
    - Goal: 1 = identical distributions, 0 = large distance  
    ✔ Robust to small m/z shifts  
    ✔ Good for noisy spectra  
    ✖ Less sensitive to intensity differences  
    """,

        "hybrid": """
    ### Hybrid score
    - Combines multiple metrics (0.5 × cosine + 0.4 × coverage - 0.3 × penalty)
    - Range: approximately 0 → 1
    - Goal: 1 = best overall isotopic match, 0 = poor match 
    ✔ Most robust  
    ✔ Best for automatic selection  
    """
}


def window():
    return r"""
            **Definition**: Size of the moving window (in Daltons) used to detect local maxima (peaks) in the spectrum.  
            **Recommendation**: 500 Da  
            **Impact**: A smaller window increases sensitivity to small peaks but may detect noise.  
            """

def minium_distance():
    return r"""
            **Definition**: Minimum distance (in m/z units) required between two adjacent peaks to be considered distinct.  
            **Recommendation**: 1 m/z
            *Note: The function can not accept a value lower than 1 m/z*
             """

def rm_threshold():
    return r"""
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
            """

def prominence():
    return r"""
            **Definition**: Minimum vertical distance (in intensity units) between a peak and the lowest contour line encircling it without crossing any higher terrain.  
            **Purpose**: Filters out "shoulder peaks" or noise that are not significantly taller than surrounding data points.  

            **How it works**:  

            - A peak is only retained if its height exceeds the highest of its two adjacent valleys by at least this value.  
            - Example: If `prominence=5e5`, a peak with intensity 1,000,000 must have valleys ≤ 500,000 on both sides to be detected.  

            **Typical values**:  

            - **Low prominence (1e4–1e5)**: Detects minor peaks (risk of noise).  
            - **Moderate prominence (1e5–5e5)**: Balances sensitivity and specificity (recommended for MALDI-TOF).  
            - **High prominence (>5e5)**: Only major peaks (may miss low-abundance signals).  

            **Tip**: Start with 10% of your highest peak intensity, then adjust based on visual inspection of the spectrum.
            """

def display_full_demo_spectrum():
    # Generate m/z axis
    np.random.seed(42)
    x = np.linspace(50, 1050, 8000)

    n_peaks = 50

    # Regular spacing
    base_positions = np.linspace(100, 1000, n_peaks)

    # Small jitter
    peak_positions = base_positions + np.random.normal(0, 1.0, n_peaks)

    # Narrow MS-like widths
    peak_widths = np.random.uniform(1.5, 5, n_peaks)

    # Alternating intensity pattern (key change 🔥)
    peak_heights = np.zeros(n_peaks)
    for i in range(n_peaks):
        if i % 2 == 0:
            peak_heights[i] = np.random.uniform(6e6, 1e7)  # big peak
        else:
            peak_heights[i] = np.random.uniform(2e5, 1e6)  # small peak

    y = np.zeros_like(x)

    for pos, width, height in zip(peak_positions, peak_widths, peak_heights):
        y += height * np.exp(-(x - pos) ** 2 / (2 * width ** 2))

    # Stronger noise + baseline
    noise = np.random.normal(0, 2e5, size=x.shape)
    baseline = 1e5 * np.sin(x / 50) + 5e4 * np.random.normal(size=x.shape)
    y += noise + baseline


    col1, col2 = st.columns(2)
    with col1:
        prominence =  st.slider(
            "Prominence (intensity units)",
            min_value=0.0,
            max_value=2e6,
            value=1.0e5,
            step=0.1e5,
        )
    with col2:
        distance = st.slider(
            "Minimum m/z distance between peaks",
            min_value=0,
            max_value=500,
            value=50,
            step=1,
            help="Minimum horizontal distance (in m/z units) between adjacent peaks."
        )
  
    # Peak detection
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)

    # Plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Spectrum'
    ))

    fig.add_trace(go.Scatter(
        x=x[peaks],
        y=y[peaks],
        mode='markers',
        marker=dict(size=8),
        name='Detected Peaks'
    ))

    fig.update_layout(
        title="Synthetic Spectrum with Peak Detection",
        xaxis_title="m/z or Time",
        yaxis_title="Intensity",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=0)
    )
    return fig, peaks


def add_double_arrow_with_lines(
    fig: go.Figure,
    x_pos: float,
    y_base: float,
    height: float,
    color: str = "orange",
    line_width: float = 2,
    line_length: float = 15
) -> None:
    """
    Add a vertical double arrow with horizontal dashed lines at the top and bottom.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to which the arrow and lines will be added.
    x_pos : float
        X position of the arrow.
    y_base : float
        Y position of the base (bottom) of the arrow.
    height : float
        Height of the arrow.
    color : str, optional
        Color of the arrow and lines, by default "orange".
    line_width : float, optional
        Width of the horizontal dashed lines, by default 2.
    line_length : float, optional
        Half-length of the horizontal lines, by default 15.
    """

    # Upward arrow (base -> top)
    fig.add_annotation(
        x=x_pos,
        y=y_base + height,  # tip at the top
        ax=x_pos,
        ay=y_base,          # start at the base
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowcolor=color,
    )

    # Downward arrow (top -> base) to create double arrow
    fig.add_annotation(
        x=x_pos,
        y=y_base,           # tip at the base
        ax=x_pos,
        ay=y_base + height, # start at the top
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowcolor=color,
    )

    # Bottom horizontal dashed line
    fig.add_shape(
        type="line",
        x0=x_pos - line_length,
        x1=x_pos + line_length,
        y0=y_base,
        y1=y_base,
        line=dict(color=color, dash="dash", width=line_width)
    )

    # Top horizontal dashed line
    fig.add_shape(
        type="line",
        x0=x_pos - line_length,
        x1=x_pos + line_length,
        y0=y_base + height,
        y1=y_base + height,
        line=dict(color=color, dash="dash", width=line_width)
    )

def display_easy_schema() -> str:
    """
    Display a simple educational 3-peak schema showing the effect of
    prominence and distance in peak detection.

    Returns
    -------
    str
        Markdown explanation for the schema.
    """
    # Create signal
    # 3 peaks (controlled)
    x = np.linspace(0, 100, 2000)
    peaks_pos = [30, 50, 70]
    heights = [6e5, 2e6, 10e5]
    widths = [3, 4, 3]

    y = np.zeros_like(x)
    for p, h, w in zip(peaks_pos, heights, widths):
        y += h * np.exp(-(x - p) ** 2 / (2 * w ** 2))
    # === Vallée entre pic 1 et pic 2 ===
    valley_center = (peaks_pos[0] + peaks_pos[1]) / 2
    valley_width = peaks_pos[1] - peaks_pos[0]

    # Offset maximum de la vallée
    valley_offset_max = 2e5

    # Créer une gaussienne inversée pour la vallée
    valley_mask = (x > peaks_pos[0]) & (x < peaks_pos[1])
    valley_x = x[valley_mask]
    valley_offset = valley_offset_max * np.exp(-((valley_x - valley_center) ** 2) / (2 * (valley_width/4) ** 2))

    # Ajouter l'offset à la vallée
    y[valley_mask] += valley_offset


    y += np.random.normal(0, 2e4, size=x.shape)  # slight noise
    dx = np.mean(np.diff(x))

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        prominence = st.slider("Prominence", 0.0, 2e6, 3e5, step=5e4)
    with col2:
        distance = st.slider("Distance (m/z)", 1, 100, 10)
        distance_points = max(1, int((distance / dx)))

    detected, _ = find_peaks(y, prominence=prominence, distance=distance_points)

    fig = go.Figure()

    # Signal
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Signal"))

    # Detected peaks
    fig.add_trace(go.Scatter(
        x=x[detected],
        y=y[detected],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Detected peaks"
    ))

    # === PROMINENCE ANNOTATION (main peak) ===

    # Vertical arrow for prominence
    # === Position de la vallée pour la flèche de Prominence ===
    valley_idx = np.argmin(y[valley_mask])  # point le plus bas de la vallée
    valley_x_pos = valley_x[valley_idx]
    valley_y_pos = y[valley_mask][valley_idx]
    add_double_arrow_with_lines(
        fig=fig,
        x_pos=valley_x_pos,
        y_base=valley_y_pos,
        height=prominence,
        color="orange"
    )
    # === Detect valley between pic 2 and pic 3 ===
    valley_2_3_mask = (x > peaks_pos[1]) & (x < peaks_pos[2])
    valley_2_3_x = x[valley_2_3_mask]
    valley_2_3_y = y[valley_2_3_mask]

    # Find the lowest point of the valley
    valley_2_3_idx = np.argmin(valley_2_3_y)
    valley_2_3_x_pos = valley_2_3_x[valley_2_3_idx]
    valley_2_3_y_pos = valley_2_3_y[valley_2_3_idx]

    add_double_arrow_with_lines(
        fig=fig,
        x_pos=valley_2_3_x_pos,
        y_base=valley_2_3_y_pos,
        height=prominence,
        color="orange"
    )

    # === Dynamic Distance Arrow ===
    peak_x = peaks_pos[1]
    peak_y = heights[1]
    left_arrow_x = peak_x
    right_arrow_x = (peak_x + distance)

    fig.add_annotation(
        x=right_arrow_x, y=peak_y*0.5,
        ax=left_arrow_x, ay=peak_y*0.5,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowside='start+end',
        arrowcolor='green'
    )
    fig.add_shape(
        type="line",
        x0=right_arrow_x,
        x1=right_arrow_x,
        y0=0,
        y1=peak_y,
        line=dict(color="green", dash="dash")
    )
    fig.add_shape(
        type="rect",
        x0=peak_x,
        x1=peak_x-distance,
        y0=0,
        y1=max(y),
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=peak_x,
        x1=peak_x+distance,
        y0=0,
        y1=max(y),
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

    fig.update_layout(title="Demonstration of Peak Prominences and Distances",
                      template='plotly_white', margin=dict(l=20,r=20,t=40,b=20),
                      xaxis_title='m/z', yaxis_title='Intensity',
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # === Markdown explanation ===
    explanation = (
        "### How to read this schema\n"
        "- 🔴 Detected peaks = pass prominence & distance\n"
        "- 🟠 Vertical arrow = Prominence of main peak\n"
        "- 🟢 Horizontal arrow = Minimum distance constraint around main peak\n"
    )

    return explanation

def display_peak_detection() -> None:
    """
    Display an interactive peak detection demo using Streamlit.

    Generates a synthetic spectrum with peaks of varying intensities
    (range up to 10M) and allows the user to explore the influence
    of `prominence` and `distance` parameters from scipy.signal.find_peaks.
    """

    col1, col2 = st.columns([1,2])

    with col2:
        tab1, tab2 = st.tabs(["🔍 Illustration", "📊 Full demo spectrum"])
        with tab1:
            instruction = display_easy_schema()
        with tab2:
            # Display info
            fig, peaks = display_full_demo_spectrum()
            st.markdown(f"**Number of peaks detected:** {len(peaks)}")
            st.plotly_chart(fig, use_container_width=True)

    with col1:
        st.markdown("""
        ### Interpretation
        - **Prominence** controls how much a peak stands out from surrounding noise.
        - **Distance** enforces a minimum spacing between detected peaks.
        
        Try increasing prominence to remove small peaks, or increasing distance to merge close peaks.
        """)
        st.markdown(instruction)


def method_score():
    return r"""
            Compare experimental vs theoretical isotopic patterns.  
            
            - cosine: fast, robust
            - pearson: pattern correlation
            - chi2: penalizes differences
            - kl: highly discriminative
            - wasserstein: robust to shifts
            - hybrid: best overall (recommended, default)
        """
