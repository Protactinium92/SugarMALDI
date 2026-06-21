"""
SugarMALDI source package.

This package contains the core components of the SugarMALDI application,
including Streamlit frontend utilities, session management, file I/O,
and MALDI-MS spectral analysis pipelines.

The codebase is structured into three main layers:

- Frontend layer:
    Streamlit-based UI components for visualization and user interaction.

- Session layer:
    Management of application state, configuration, and analysis parameters.

- Core analysis layer:
    Scientific processing of MALDI-MS spectra, including peak detection,
    noise filtering, and oligosaccharide library matching.

Design principles:
- Separation of UI and scientific logic
- Memory-efficient processing using Polars LazyFrames
- Reproducible analysis pipelines for high-throughput MS data

This package is intended to be used primarily through the Streamlit
application entry point (app.py).
"""