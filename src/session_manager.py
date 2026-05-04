# =============================
# session_manager.py
# =============================

import streamlit as st
from dataclasses import dataclass, asdict
from typing import Optional


# ---------- PARAMS ----------

@dataclass
class AnalysisParams:
    window: int = 500
    prominence: float = 3e5
    dist_min: int = 0
    height_min: float = 5e5
    rm_threshold: float = 5
    number_candidate: int = 4
    number_iso: int = 5
    iso_score: Optional[str] = None


# ---------- GLOBAL CONFIG SNAPSHOT ----------

@dataclass
class AnalysisConfig:
    params: AnalysisParams
    monomers_list: list[str]
    adduct_mass: dict
    adduct_compo: dict
    label_use: bool
    label_name: str
    label_formula: Optional[str]


# ---------- SESSION MANAGER (MINIMAL) ----------

class SessionManager:

    @staticmethod
    def init():
        if "params" not in st.session_state:
            st.session_state.params = AnalysisParams()

        if "snapshot" not in st.session_state:
            st.session_state.snapshot = None

        if "analysis_ran" not in st.session_state:
            st.session_state.analysis_ran = False

    @staticmethod
    def get_params() -> AnalysisParams:
        return st.session_state.params

    @staticmethod
    def save_snapshot(config: AnalysisConfig):
        st.session_state.snapshot = config
        st.session_state.analysis_ran = True

    @staticmethod
    def get_snapshot() -> Optional[AnalysisConfig]:
        return st.session_state.snapshot
    
    @staticmethod
    def run_analysis(config: AnalysisConfig):
        SessionManager.save_snapshot(config)
        st.session_state.analysis_ran = True
    
    # ---------- DIFF ----------

    @staticmethod
    def has_changed(current, reference) -> bool:
        if reference is None:
            return False

        try:
            return asdict(current) != asdict(reference)
        except Exception:
            return current != reference

    # ---------- WARNING HELPER ----------

    @staticmethod
    def warn_if_any_changed(label: str, pairs: list[tuple], condition: bool = True):
        if not condition:
            return

        for current, reference in pairs:
            if SessionManager.has_changed(current, reference):
                st.warning(f"⚠️ {label} changed. Please re-run the analysis.")
                return

