import polars as pl
import pytest

from src.spectrum_analysis import calculate_chemical_formula



def test_3hex_proton_formula_without_label(
    monomer_df,
):
    """
    Test:
    3 Hex + H+

    Expected:
    3 x C6H12O6 - 2 H2O + H+
    = C18H33O16+
    """

    oligo_df = {
        "Hex": 3
    }

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct="H+",
        adduct_compo={"H+": {"H": 1}},
        use_label_3aq=False,
    )

    assert result == "C18H33O16"

def test_3hex_label_3aq_proton_formula(
    monomer_df,
    label_3aq_compo,
):
    """
    Test formula calculation for:
    3 Hex + 3AQ label + H+
    
    Expected:
    3 Hex (C6H12O6)
    + 3AQ label (C9H8N2)
    - 3 H2O condensation
    + H+
    
    = C27H39O15N2
    """

    # 3 Hex units
    oligo_df = {
        "Hex": 3
    }


    # Proton adduct
    adduct_compo = {
        "H+": {"H": 1}
    }

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct="H+",
        adduct_compo=adduct_compo,
        use_label_3aq=True,
        label_compo=label_3aq_compo,
    )

    assert result == "C27H39N2O15"


def test_3hex_label_3aq_water_loss_proton_adduct(
    monomer_df,
    label_3aq_compo,
):
    """
    Test:
    3 Hex + 3AQ label + H+(-H2O)

    Expected:
    C27H37N2O14
    """

    oligo_df = {
        "Hex": 3
    }

    adduct_compo = {
            "H+(-H2O)": {"H": -1,"O": -1}
        }

    result = calculate_chemical_formula(
        oligo_df=oligo_df,
        monomer_df=monomer_df,
        adduct="H+(-H2O)",
        adduct_compo=adduct_compo,
        use_label_3aq=True,
        label_compo=label_3aq_compo,
    )

    assert result == "C27H37N2O14"