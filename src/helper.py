# Copyright (C) 2026 Pierre-Alexandre HO
#
# This file is part of SugarMALDI.
#
# Licensed under the CeCILL 2.1 license.
# See LICENSE file for details.

"""
Streamlit Helper Module for Interactive Scientific UI
=====================================================

Overview
--------
This module provides helper functions for a Streamlit-based application.
It is designed to support the main interface by offering interactive
educational content, visual explanations, and parameter guidance.

The primary goal is to improve user understanding of core signal processing
and mass spectrometry concepts through interactive visualization tools.

Main Purpose
------------
This module acts as a "help system" for the application, providing:

1. Interactive Help Components
   - Explanations of key algorithmic parameters
   - Contextual guidance embedded in the UI
   - Streamlit-based tooltips and sections

2. Visual and Educational Demonstrations
   - Synthetic signal generation for demonstration purposes
   - Interactive peak detection visualization
   - Graphical interpretation of key concepts:
       * Peak prominence
       * Peak height threshold
       * Minimum distance between peaks

3. Scientific Concept Explanations
   - Mass spectrometry signal processing concepts
   - Isotopic scoring functions and similarity metrics
   - Mathematical formulations with intuitive interpretations

4. Interactive Parameter Exploration
   - Real-time adjustment of signal processing parameters
   - Immediate visual feedback using Plotly
   - Educational comparison of parameter effects

Scoring Methods Included
------------------------
The module documents and explains several isotopic similarity metrics:

- Cosine similarity
- Pearson correlation
- Chi-square distance
- Kullback-Leibler divergence
- Wasserstein distance
- Hybrid scoring approach

Dependencies
------------
- numpy
- streamlit
- scipy.signal (find_peaks)
- plotly.graph_objects

Usage
-----
This file is intended to be imported into a Streamlit application:

    import src.helper as helper

and used to display interactive help sections:

    helper.display_peak_detection()

Notes
-----
- This module is strictly for UI/UX support and educational purposes.
- It does not perform production-level signal processing.
- All signals are synthetic and generated for visualization only.
- The modification of parameters in this module does not affect the main analysis pipeline.
"""

__author__  = "Pierre-Alexandre HO"


import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

SCORING_HELP = {
            "cosine": [
        r"""
$$
S_{cosine} = \frac{\sum_{i=1}^{n} I_i^{\mathrm{exp}} \cdot I_i^{\mathrm{theo}}}
{\sqrt{\sum_{i=1}^{n} (I_i^{\mathrm{exp}})^2} \cdot \sqrt{\sum_{i=1}^{n} (I_i^{\mathrm{theo}})^2}}
$$

with:  

- $I_i^{\mathrm{exp}}$    : normalized intensity of the $i$-th experimental isotope  
- $I_i^{\mathrm{theo}}$   : normalized intensity of the $i$-th theoretical isotope  
- $n$                     : number of isotopes considered
""",
        """
### Cosine similarity

- Compares the angle between intensity vectors.
- Range: 0 → 1  
- Goal: 1 = perfect match, 0 = no similarity  
✔ Robust to scaling  
✔ Fast and stable  
"""
    ],

    "pearson": [
        r"""
$$
S_{pearson} = \frac{\sum_{i=1}^{n} (I_i^{\mathrm{exp}} - \bar{I}^{\mathrm{exp}})(I_i^{\mathrm{theo}} - \bar{I}^{\mathrm{theo}})}
{\sqrt{\sum_{i=1}^{n} (I_i^{\mathrm{exp}} - \bar{I}^{\mathrm{exp}})^2} \cdot \sqrt{\sum_{i=1}^{n} (I_i^{\mathrm{theo}} - \bar{I}^{\mathrm{theo}})^2}}
$$

with:    
$\bar{I}^{\mathrm{exp}} = \frac{1}{n} \sum_{i=1}^{n} I_i^{\mathrm{exp}} ,\quad
\bar{I}^{\mathrm{theo}} = \frac{1}{n} \sum_{i=1}^{n} I_i^{\mathrm{theo}}$

- $I_i^{\mathrm{exp}}$    : normalized intensity of the $i$-th experimental isotope  
- $I_i^{\mathrm{theo}}$   : normalized intensity of the $i$-th theoretical isotope  
- $n$                     : number of isotopes considered

""",
        """
### Pearson correlation

- Measures linear correlation between patterns.
- Range: -1 → 1  
- Goal: 1 = identical shapes, 0 = no correlation, -1 = inverse correlation  
✔ Sensitive to pattern shape  
✖ Unstable with few peaks  
"""
    ],

    "chi2": [
        r"""
$$
\chi^2 = \sum_{i=1}^{n} \frac{(I_i^{\mathrm{exp}} - I_i^{\mathrm{theo}})^2}{I_i^{\mathrm{theo}} + \epsilon} \quad \text{then}
\quad S_{\chi^2} = \frac{1}{1 + \chi^2}
$$
with:

- $I_i^{\mathrm{exp}}$    : normalized intensity of the $i$-th experimental isotope  
- $I_i^{\mathrm{theo}}$   : normalized intensity of the $i$-th theoretical isotope  
- $n$                     : number of isotopes considered
- \epsilon$               : small constant to avoid division by zero.
""",
        r"""
### Chi-square distance

- Measures deviation between experimental and theoretical intensities.
- Range: 0 → ∞ (score transformed as $1/(1+\chi^2)$ → 0 → 1)
- Goal: 1 = identical patterns, 0 = large deviations  
✔ Penalizes large errors strongly  
✔ Good mismatch detector  
✖ Sensitive to noise  
"""
    ],

    "KL": [
        r"""
$$
D_{\mathrm{KL}}(Q \parallel P) = \sum_{i=1}^{n} Q_i \log \frac{Q_i}{P_i + \epsilon} \quad \text{then}
\quad S_\mathrm{KL} = \frac{1}{1 + D_\mathrm{KL}}
$$

with:

$Q_i = \frac{I_i^{\mathrm{theo}}}{\sum_{j=1}^{n} I_j^{\mathrm{theo}}} , \quad
P_i = \frac{I_i^{\mathrm{exp}}}{\sum_{j=1}^{n} I_j^{\mathrm{exp}}}$

- $I_i^{\mathrm{exp}}$    : normalized intensity of the $i$-th experimental isotope  
- $I_i^{\mathrm{theo}}$   : normalized intensity of the $i$-th theoretical isotope  
- $n$                     : number of isotopes considered
- $\epsilon$              : small constant to avoid log(0)
""",
        """
### Kullback-Leibler (KL) divergence

- Measures information loss between distributions.  
- Range: 0 → ∞ (score transformed as $1/(1+KL)$ → 0 → 1)
- Goal: 1 = identical distributions, 0 = large information loss  
✔ Very discriminative  
✖ Penalizes missing peaks strongly  
"""
    ],

    "wasserstein": [
        r"""
$$
W_1(P,Q) = \min_{\gamma \in \Gamma} \sum_{i,j} \gamma_{ij} | m_i^{\mathrm{exp}} - m_j^{\mathrm{theo}} | \quad \text{then}
\quad S_\mathrm{W} = \frac{1}{1 + W_1}
$$

where:

$m_i^{\mathrm{exp}}$ : m/z of the $i$-th experimental isotope  

$m_j^{\mathrm{theo}}$ : m/z of the $j$-th theoretical isotope  

$\Gamma$ : set of valid transport plans  

$\gamma_{ij}$ : fraction of “mass” transported from $m_i^{\mathrm{exp}}$ to $m_j^{\mathrm{theo}}$
""",
        """
### Wasserstein distance (Earth Mover's Distance)

- Measures the minimal “cost” to transform the experimental distribution into the theoretical one.
- Range: 0 → ∞ (score transformed as $1/(1+W)$ → 0 → 1)
- Goal: 1 = identical distributions, 0 = large distance  
✔ Robust to small m/z shifts  
✔ Good for noisy spectra  
✖ Less sensitive to intensity differences  
"""
    ],
    "hybrid": [
        r"""
$$
S_\mathrm{hybrid} = 0.5 \cdot \text{Cosine} + 0.4 \cdot \text{Coverage} - 0.3 \cdot \text{Penalty}
$$

where:

**Cosine** : cosine similarity between experimental and theoretical isotope intensities  

**Coverage** : fraction of matched isotopic peaks
$$
\text{Coverage} = \frac{\sum_{i=1}^{N} \mathbf{1}_{\{I_i^{\mathrm{exp}} > 0\}}}{N}
$$
$N$ : number of theoretical isotopes considered  
$\mathbf{1}_{\{I_i^{\mathrm{exp}} > 0\}}$ : indicator function, 1 if the $i$-th theoretical isotope has a corresponding experimental peak, 0 otherwise  

**Penalty** : penalizes missing critical isotopes, e.g. monoisotopic peak
$$
\text{Penalty} = \sum_{i=1}^{N} \mathbf{1}_{\{I_i^{\mathrm{exp}} < \epsilon\}}
$$
$\epsilon$ : small intensity threshold to consider a peak as “detected”
""",
        """
### Hybrid score

- Combines multiple metrics (cosine similarity, coverage, penalty)
- Range: approximately 0 → 1
- Goal: 1 = best overall isotopic match, 0 = poor match  
✔ Most robust  
✔ Best for automatic candidate ranking  
"""
    ]
}


def window() -> str:
    return r"""
    **Definition**: Size of the m/z window (in Da) used to segment the spectrum into independent regions for processing.  
    **Recommendation**: Adjust to ensure sufficient data points per window (typically ≥100 points).  
    **Impact**: Smaller windows improve local peak detection and parallel efficiency but may suffer from edge effects
    and reduced statistical robustness; larger windows provide more stable peak detection but reduce parallelization efficiency
    and may smooth local spectral features.
    """


def minimum_distance() -> str:
    return r"""
    **Definition**: Minimum separation (in m/z units) required between two peaks to be considered distinct.  
    **Impact**: Smaller values increase peak resolution and allow detection of closely spaced ions but may introduce noise,
    whereas larger values reduce false positives and computational load but may merge or discard nearby peaks.
    """


def rm_threshold() -> str:
    return r"""
    **Definition**: Mass Measurement Accuracy / Error, calculated as (1):

    $$\Delta M \ (\mathrm{ppm}) = \left( \frac{|M_{\mathrm{exp}} - M_{\mathrm{theo}}|}{M_{\mathrm{theo}}} \right) \times 10^6$$

    - **$M_{\mathrm{exp}}$**: Experimentally measured mass.
    - **$M_{\mathrm{theo}}$**: Expected theoretical mass.
    - **Typical threshold**: 5 ppm for MALDI-MS.

    *Example*: If ΔM = 10 ppm and $M_{\mathrm{theo}}$ = 1000 Da,
    the tolerated deviation is **±0.01 Da**.

    **Impact**: Lower thresholds enforce stricter mass filtering, increasing identification confidence but potentially excluding
    true candidates due to mass measurement errors or calibration drift; higher thresholds improve sensitivity and candidate recall
    but increase ambiguity and downstream computational cost..

    **Reference:**

    (1) Brenton, A.G.; Godfrey, A.R. *J. Am. Soc. Mass Spectrom.* **2010**, *21*(11), 1821-1835.
    [DOI:10.1016/j.jasms.2010.06.006](https://www.sciencedirect.com/science/article/pii/S1044030510004022)
    """

def prominence() -> str:
    return r"""
    **Definition**: Minimum intensity difference between a peak apex and its surrounding baseline (local minima).  
    **Impact**: Lower values increase sensitivity to low-abundance and shoulder peaks but may introduce noise,
    whereas higher values restrict detection to dominant signals, improving robustness but potentially missing minor species.
    """


def peak_height() -> str:
    return r"""
    **Definition**: Minimum intensity required for a peak to be detected from intensity axis (0).  
    **Impact**: Lower values increase sensitivity to low-abundance signals but may detect noise and artefacts,
    whereas higher values restrict detection to dominant peaks, improving robustness but potentially missing minor species.  
    **Note**: If the prominence threshold is set higher than the height threshold, the height criterion becomes effectively redundant,
    as only peaks exceeding the prominence constraint will be retained.  
    """


def number_candidate() -> str:
    return r"""
    **Definition**: Number of top candidate oligosaccharides retained after mass filtering (ΔM) and prior to isotopic pattern scoring.  
    **Impact**: Increasing this value improves the probability of retaining the correct candidate in ambiguous regions,
    but increases computational cost; decreasing it accelerates processing but risks excluding relevant candidates.
    """


def number_isotopes() -> str:
    return r"""
    **Definition**: Number of isotopic peaks (e.g., M, M+1, M+2, ...) included in the isotopic envelope used for scoring.  
    **Impact**: Considering more isotopic peaks improves discrimination by better representing the theoretical isotopic distribution,
    but increases sensitivity to noise and computational cost; fewer peaks reduce complexity but may limit resolving power between candidates.
    """


def method_score() -> str:
    return r"""
    **Definition**: Metric used to compare experimental and theoretical isotopic envelopes.  
    
    **Options**:  
    - cosine: Fast and robust similarity metric.  
    - pearson: Measures linear correlation between intensity profiles.  
    - chi2: Emphasizes deviations in peak intensities.  
    - kl: Divergence-based metric, highly sensitive to distribution differences.  
    - wasserstein: Distance-based metric, robust to small m/z shifts.  
    - hybrid: Combination of multiple metrics for improved robustness (recommended, default).  
    
    **Impact**: The selected metric determines sensitivity to intensity variations, noise, and peak misalignment,
    directly influencing identification confidence and computational performance.
    """


def display_full_demo_spectrum():
    # Generate m/z axis
    np.random.seed(42)
    x = np.linspace(0, 10, 8000)
    dx = np.mean(np.diff(x))
    
    n_peaks = 50
    # Regular spacing
    base_positions = np.linspace(0.5, 10, n_peaks)

    # Small jitter
    peak_positions = base_positions + np.random.normal(0, 0.01, n_peaks)

    # Narrow MS-like widths
    peak_widths = np.random.uniform(0.015, 0.05, n_peaks)

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
    baseline = 1e5 * np.sin(x / 0.5) + 5e4 * np.random.normal(size=x.shape)
    y += noise + baseline


    col1, col2, col3 = st.columns(3)
    with col1:
        prominence =  st.slider(
            "Prominence",
            min_value=0.0,
            max_value=2e6,
            value=0.0e5,
            step=0.1e5,
        )
    with col2:
        height_min = st.slider(
            "Minimum height",
            min_value=0.0,
            max_value=10e6,
            value=0.0e5,
            step=0.1e5,
        )
    with col3:
        distance = st.slider(
            "Minimum m/z distance",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.001,
        )
        #Conversion distance (float) to indice (int) with a minium value of 1 unit 
        distance_points = max(1, int((distance / dx)))
  
    # Peak detection
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance_points, height=height_min)

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
        margin=dict(l=40, r=20, t=40, b=0),
        showlegend=False
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
    peaks_pos = [15, 30, 50, 70]
    heights = [3e5, 6e5, 2e6, 10e5]
    widths = [2, 3, 4, 3]

    y = np.zeros_like(x)
    for p, h, w in zip(peaks_pos, heights, widths):
        y += h * np.exp(-(x - p) ** 2 / (2 * w ** 2))
    # === Vallée entre pic 1 et pic 2 ===
    valley_center = (peaks_pos[1] + peaks_pos[2]) / 2
    valley_width = peaks_pos[2] - peaks_pos[1]

    # Offset maximum de la vallée
    valley_offset_max = 2e5

    # Créer une gaussienne inversée pour la vallée
    valley_mask = (x > peaks_pos[1]) & (x < peaks_pos[2])
    valley_x = x[valley_mask]
    valley_offset = valley_offset_max * np.exp(-((valley_x - valley_center) ** 2) / (2 * (valley_width/4) ** 2))

    # Ajouter l'offset à la vallée
    y[valley_mask] += valley_offset


    y += np.random.normal(0, 2e4, size=x.shape)  # slight noise
    dx = np.mean(np.diff(x))

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        prominence = st.slider("Prominence", 0.0, 2e6, 3e5, step=5e4)
    with col2:
        height_min = st.slider("Minimum height", 0.0, 2e6, 3e5, step=5e4)
    with col3:
        distance = st.slider("Minimum m/z distance", 0, 100, 10)
        distance_points = max(1, int((distance / dx)))

    detected, _ = find_peaks(y, prominence=prominence, distance=distance_points, height=height_min)

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
    # === Detect valley between pic 3 and pic 4 ===
    valley_3_4_mask = (x > peaks_pos[2]) & (x < peaks_pos[3])
    valley_3_4_x = x[valley_3_4_mask]
    valley_3_4_y = y[valley_3_4_mask]

    # Find the lowest point of the valley
    valley_2_3_idx = np.argmin(valley_3_4_y)
    valley_3_4_x_pos = valley_3_4_x[valley_2_3_idx]
    valley_3_4_y_pos = valley_3_4_y[valley_2_3_idx]

    add_double_arrow_with_lines(
        fig=fig,
        x_pos=valley_3_4_x_pos,
        y_base=valley_3_4_y_pos,
        height=prominence,
        color="orange"
    )

    # === Dynamic height horizontal line ===
    fig.add_shape(
        type="line",
        x0=min(x),
        x1=max(x),
        y0=height_min,
        y1=height_min,
        line=dict(color="purple", dash="dash")
    )
    fig.add_shape(
        type="rect",
        x0=min(x),
        x1=max(x),
        y0=0,
        y1=height_min,
        fillcolor="gray",
        opacity=0.2,
        line_width=0
    )

    # === Dynamic Distance Arrow ===
    peak_x = peaks_pos[2]
    peak_y = heights[2]
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
    st.plotly_chart(fig, width="stretch")

    # === Markdown explanation ===
    explanation = (
        "### How to read this schema\n"
        "- 🔴 Detected peaks = pass prominence & distance\n"
        "- 🟠 Vertical arrow = Prominence of main peak\n"
        "- 🟣 Horizontal line = Minimum peak height\n"
        "- 🟢 Horizontal arrow = Minimum distance constraint around main peak\n"
        "- ⬜ Rejected zone (distance constraint)\n"
    )

    return explanation


@st.fragment
def display_peak_detection() -> None:
    """
    Display an interactive peak detection demo using Streamlit.

    Generates a synthetic spectrum with peaks of varying intensities
    (range up to 10M) and allows the user to explore the influence
    of `prominence` and `distance` parameters from scipy.signal.find_peaks.
    """

    col1, col2 = st.columns([1,1])

    with col2:
        tab1, tab2 = st.tabs(["🔍 Illustration", "📊 Full demo spectrum"])
        with tab1:
            instruction = display_easy_schema()
        with tab2:
            # Display info
            fig, peaks = display_full_demo_spectrum()
            st.markdown(f"**Number of peaks detected:** {len(peaks)}")
            st.plotly_chart(fig, width="stretch")

    with col1:
        st.markdown("""
        ### Interpretation
        - **Prominence**  (intensity units) controls how much a peak stands out from surrounding noise.
                    A peak is only retained if its height exceeds the highest of its two adjacent valleys by at least this value.
        - **Height**  (intensity units) minimum intensity of a peak. The height is measured from 0 to the maximum of the peak. If prominence > height, height becomes redundant.  
        - **Distance** enforces a minimum spacing between detected peaks.
        
        Try increasing prominence to remove small peaks, or increasing distance to merge close peaks.
        """)
        st.markdown(instruction)


if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )

