# Copyright (C) 2026 Pierre-Alexandre HO
#
# This file is part of SugarMALDI.
#
# Licensed under the CeCILL 2.1 license.
# See LICENSE file for details.

"""
Session State Management for SugarMALDI Analysis Application
=============================================================

Overview
--------
This module manages Streamlit session state for the SugarMALDI application.
It provides structured handling of analysis parameters, configuration
snapshots, and execution state tracking.

The goal is to ensure reproducibility and consistency between user-defined
parameters and executed analyses by maintaining a persistent session-level
state.

Main Responsibilities
---------------------
1. Analysis Parameter Management
   - Stores and retrieves user-defined parameters for spectrum analysis
   - Encapsulated in the `AnalysisParams` dataclass

2. Configuration Snapshot System
   - Captures a full analysis configuration at execution time
   - Enables reproducibility of results
   - Encapsulated in the `AnalysisConfig` dataclass

3. Streamlit Session State Handling
   - Initializes and maintains session variables
   - Tracks whether an analysis has been executed
   - Stores current parameters and saved snapshots

4. Change Detection Utilities
   - Compares current vs reference configurations
   - Detects parameter modifications using dataclass comparison
   - Supports warnings when re-analysis is required

5. User Feedback Helpers
   - Provides warning mechanisms when parameters change
   - Ensures users are informed when results may no longer be valid

Core Data Structures
--------------------
- AnalysisParams:
    Defines all tunable parameters for spectrum analysis
    (peak detection, isotopic scoring, filtering thresholds, etc.)

- AnalysisConfig:
    Immutable snapshot combining:
    - analysis parameters
    - chemical configuration (monomers, adducts)
    - labeling options

Design Philosophy
-----------------
- Streamlit-native session state management
- Reproducibility through snapshot-based execution
- Minimal but robust state tracking
- Defensive comparison of dataclass configurations

Usage
-----
Typical usage within a Streamlit app:

    SessionManager.init()
    params = SessionManager.get_params()
    SessionManager.run_analysis(config)

Notes
-----
- This module does not perform analysis itself.
- It only manages state and configuration consistency.
- Designed to support interactive scientific workflows.
"""


__author__  = "Pierre-Alexandre HO"

import streamlit as st
from dataclasses import dataclass, asdict
from typing import Optional


# ---------- PARAMS ----------

@dataclass
class AnalysisParams:
    """Parameters for spectrum analysis.

    Attributes:
        window (int): Window size for peak detection.
        prominence (float): Minimum prominence for peaks.
        dist_min (int): Minimum distance between peaks.
        height_min (float): Minimum height for peaks.
        rm_threshold (float): Threshold for removing peaks.
        number_candidate (int): Number of candidate peaks.
        number_iso (int): Number of isotopes.
        iso_score (Optional[str]): Isotope scoring method.
    """
    window: int = 500
    prominence: float = 1e5
    dist_min: int = 0
    height_min: float = 2e5
    rm_threshold: float = 5
    number_candidate: int = 4
    number_iso: int = 5
    iso_score: Optional[str] = None


# ---------- GLOBAL CONFIG SNAPSHOT ----------

@dataclass
class AnalysisConfig:
    """Configuration snapshot for analysis.

    Attributes:
        params (AnalysisParams): Analysis parameters.
        monomers_list (list[str]): List of monomers.
        adduct_mass (dict): Adduct masses.
        adduct_compo (dict): Adduct compositions.
        neutral_loss_use (bool): Whether to use neutral loss.
        associated_neutral_loss_mass (dict): Neutral loss masses associated with adducts.
        associated_neutral_loss_compo (dict): Neutral loss compositions associated with adducts.
        label_use (bool): Whether to use labels.
        label_name (str): Label name.
        label_formula (Optional[str]): Label formula.
        label_compo (Optional[dict]): Label composition.
    """
    params: AnalysisParams
    monomers_list: list[str]
    adduct_mass: dict
    adduct_compo: dict
    neutral_loss_use: bool
    associated_neutral_loss_mass: dict
    associated_neutral_loss_compo: dict
    label_use: bool
    label_name: str
    label_formula: Optional[str]
    label_compo: Optional[dict]


# ---------- SESSION MANAGER (MINIMAL) ----------

class SessionManager:
    """Manages Streamlit session state for analysis parameters and snapshots."""

    @staticmethod
    def init():
        """Initialize session state variables if not present."""
        if "params" not in st.session_state:
            st.session_state.params = AnalysisParams()

        if "snapshot" not in st.session_state:
            st.session_state.snapshot = None

        if "analysis_ran" not in st.session_state:
            st.session_state.analysis_ran = False

    @staticmethod
    def get_params() -> AnalysisParams:
        """Retrieve current analysis parameters from session state.

        Returns:
            AnalysisParams: Current analysis parameters.
        """
        return st.session_state.params

    @staticmethod
    def save_snapshot(config: AnalysisConfig):
        """Save the analysis configuration snapshot to session state.

        Args:
            config (AnalysisConfig): Configuration to save.
        """
        st.session_state.snapshot = config
        st.session_state.analysis_ran = True

    @staticmethod
    def get_snapshot() -> Optional[AnalysisConfig]:
        """Retrieve the saved analysis configuration snapshot.

        Returns:
            Optional[AnalysisConfig]: Saved configuration or None.
        """
        return st.session_state.snapshot
    
    @staticmethod
    def run_analysis(config: AnalysisConfig):
        """Run analysis by saving the configuration snapshot.

        Args:
            config (AnalysisConfig): Configuration for analysis.
        """
        SessionManager.save_snapshot(config)
        st.session_state.analysis_ran = True
    
    # ---------- DIFF ----------

    @staticmethod
    def has_changed(current, reference) -> bool:
        """Check if current value has changed compared to reference.

        Args:
            current: Current value.
            reference: Reference value.

        Returns:
            bool: True if changed, False otherwise.
        """
        if reference is None:
            return False

        try:
            return asdict(current) != asdict(reference)
        except Exception:
            return current != reference

    # ---------- WARNING HELPER ----------

    @staticmethod
    def warn_if_any_changed(label: str, pairs: list[tuple], condition: bool = True):
        """Warn if any value in pairs has changed, under given condition.

        Args:
            label (str): Warning label.
            pairs (list[tuple]): List of (current, reference) pairs.
            condition (bool): Condition to check warning.
        """
        if not condition:
            return

        for current, reference in pairs:
            if SessionManager.has_changed(current, reference):
                st.warning(f"⚠️ {label} changed. Please re-run the analysis.")
                return

