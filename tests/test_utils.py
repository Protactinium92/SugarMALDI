import pytest
from typing import Dict, List, Tuple, Optional
from src.utils import (
    parse_formula,
    validate_rows,
    build_entity_dicts,
    check_formula,
    validate_inputs,
    validate_lost_adduct,
    validate_entity,
    validate_association,
    parse_association_mass,
    parse_association_formula,
)
from unittest.mock import patch
from src import config as cf

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_monomer_elements():
    """Sample monomer elements for testing."""
    return ["C", "H", "O", "N"]


# =============================================================================
# Tests for parse_formula
# =============================================================================

def test_parse_formula_basic():
    """Test parsing a basic chemical formula."""
    assert parse_formula("H2O") == {"H": 2, "O": 1}

def test_parse_formula_single_element():
    """Test parsing a single-element formula."""
    assert parse_formula("C") == {"C": 1}

def test_parse_formula_with_subscripts():
    """Test parsing a formula with multi-digit subscripts."""
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}

def test_parse_formula_lost():
    """Test parsing a formula marked as lost."""
    assert parse_formula("H2O", lost=True) == {"H": -2, "O": -1}

def test_parse_formula_empty():
    """Test parsing an empty formula."""
    assert parse_formula("") == {}

def test_parse_formula_case_insensitive():
    """Test that element symbols are case-sensitive (e.g., 'Na' vs 'NA')."""
    assert parse_formula("NaCl") == {"Na": 1, "Cl": 1}
    assert parse_formula("NACL") == {"N": 1, "A": 1, "C": 1, "L": 1}

def test_parse_formula_zero_counts_removed():
    """Test that zero counts are removed from the composition."""
    # Example: "H2O" with lost=True and H count becomes 0 (not possible in practice, but edge case)
    assert parse_formula("H0O", lost=True) == {"O": -1}

def test_parse_formula_with_plus_sign():
    """Test parsing a formula with a '+' sign (e.g., 'H+')."""
    # The '+' is not handled by the current regex, so it is ignored.
    assert parse_formula("H+") == {"H": 1}  # Only 'H' is captured; '+' is ignored.

def test_parse_formula_with_minus_sign():
    """Test parsing a formula with a '-' sign (e.g., 'H-')."""
    # The '-' is not handled by the current regex, so it is ignored.
    assert parse_formula("H-") == {"H": 1}  # Only 'H' is captured; '-' is ignored.

def test_parse_formula_with_parentheses():
    """Test parsing a formula with parentheses (e.g., 'Ca(OH)2')."""
    # Parentheses are not handled by the current regex.
    # The regex captures 'Ca', 'O', 'H' but ignores parentheses and the '2' after ')'.
    # Expected result: {'Ca': 1, 'O': 1, 'H': 1} (the '2' after ')' is ignored).
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 1, "H": 1}

def test_parse_formula_with_invalid_characters():
    """Test parsing a formula with invalid characters (e.g., 'H@O')."""
    # The '@' is not captured by the regex, so only 'H' and 'O' are captured.
    assert parse_formula("H@O") == {"H": 1, "O": 1}

def test_parse_formula_with_spaces():
    """Test parsing a formula with spaces (e.g., 'H 2 O')."""
    # Spaces are not handled by the regex, so only 'H', '2', 'O' are captured.
    # The '2' is interpreted as a count for 'H'.
    assert parse_formula("H 2 O") == {"H": 2, "O": 1}

def test_parse_formula_with_mixed_case_and_numbers():
    """Test parsing a formula with mixed case and numbers (e.g., 'h2o')."""
    # The regex is case-sensitive for element symbols (must start with uppercase).
    # 'h2o' is not captured correctly because 'h' is lowercase.
    assert parse_formula("h2o") == {}  # No match because 'h' is not uppercase.

def test_parse_formula_with_subscript_zero():
    """Test parsing a formula with a subscript of 0 (e.g., 'H0O')."""
    # The '0' is captured as a count, so H:0 is ignored (0 * sign = 0).
    assert parse_formula("H0O") == {"O": 1}

def test_parse_formula_with_complex_invalid_formula():
    """Test parsing a complex invalid formula (e.g., 'H2+O4--')."""
    # Only 'H2' and 'O4' are captured; '+' and '-' are ignored.
    assert parse_formula("H2+O4--") == {"H": 2, "O": 4}

def test_parse_formula_with_non_standard_elements():
    """Test parsing a formula with non-standard elements (e.g., 'XxYy2')."""
    # Non-standard elements are captured if they follow the [A-Z][a-z]* format.
    assert parse_formula("XxYy2") == {"Xx": 1, "Yy": 2}

# =============================================================================
# Tests for validate_rows
# =============================================================================

def test_validate_rows_all_valid(sample_adducts):
    """Test validation with all rows valid."""
    required_fields = ["name", "mass", "formula"]
    valid_rows, error_indices, has_any_input = validate_rows(sample_adducts, required_fields)
    assert valid_rows == sample_adducts
    assert error_indices == []
    assert has_any_input is True

def test_validate_rows_partial_input():
    """Test validation with partial input in some rows."""
    rows = [
        {"name": "H+", "mass": 1.007276, "formula": "H", "is_lost": False},
        {"name": "Na+", "mass": None, "formula": "Na", "is_lost": False},  # Invalid: mass is None
        {"name": "", "mass": 0, "formula": "", "is_lost": False},  # Empty row
    ]
    required_fields = ["name", "mass", "formula"]
    valid_rows, error_indices, has_any_input = validate_rows(rows, required_fields)
    assert valid_rows == [rows[0]]
    assert error_indices == [1]
    assert has_any_input is True

def test_validate_rows_empty_input():
    """Test validation with all rows empty."""
    rows = [
        {"name": "", "mass": 0, "formula": "", "is_lost": False},
        {"name": None, "mass": None, "formula": None, "is_lost": False},
    ]
    required_fields = ["name", "mass", "formula"]
    valid_rows, error_indices, has_any_input = validate_rows(rows, required_fields)
    assert valid_rows == []
    assert error_indices == []
    assert has_any_input is False

# =============================================================================
# Tests for build_entity_dicts
# =============================================================================

def test_build_entity_dicts_adducts(sample_adducts):
    """Test building dictionaries for adducts."""
    entity_mass, entity_compo, entity_is_lost = build_entity_dicts(sample_adducts, "adduct")
    assert entity_mass == {"H+": 1.007276, "Na+": 22.989221, "K+": 38.963158}
    assert entity_compo == {"H+": {"H": 1}, "Na+": {"Na": 1}, "K+": {"K": 1}}
    assert entity_is_lost == {"H+": False, "Na+": False, "K+": False}

def test_build_entity_dicts_neutral_losses(sample_neutral_losses):
    """Test building dictionaries for neutral losses."""
    entity_mass, entity_compo, entity_is_lost = build_entity_dicts(sample_neutral_losses, "neutral_loss")
    assert entity_mass == {"(-)H2O": -18.010564}
    assert entity_compo == {"(-)H2O": {"H": -2, "O": -1}}
    assert entity_is_lost == {"(-)H2O": True}

def test_build_entity_dicts_lost_adduct():
    """Test building dictionaries for a lost adduct."""
    entities = [{"name": "NH4+", "mass": 18.034374, "formula": "NH4", "is_lost": True}]
    entity_mass, entity_compo, entity_is_lost = build_entity_dicts(entities, "adduct")
    assert entity_mass == {"(-)NH4+": -18.034374}
    assert entity_compo == {"(-)NH4+": {"N": -1, "H": -4}}
    assert entity_is_lost == {"(-)NH4+": True}


# =============================================================================
# Tests for check_formula
# =============================================================================

def test_check_formula_valid_elements():
    """Test checking a formula with valid elements."""
    composition = {"C": 6, "H": 12, "O": 6}
    assert check_formula(composition) == []

def test_check_formula_invalid_elements(monkeypatch):
    """Test checking a formula with invalid elements."""
    composition = {"C": 6, "H": 12, "Xx": 1}  # Xx is invalid
    assert check_formula(composition) == ["Xx"]

# =============================================================================
# Tests for validate_inputs
# =============================================================================

def test_validate_inputs_valid():
    """Test validation with all inputs valid."""
    monomers_list = ["Hex", "Pent"]
    adduct_compo = {"H+": {"H": 1}}
    neutral_loss_use = False
    neutral_loss_compo = {}
    association = {}
    use_label = False
    label_compo = None
    is_valid, errors = validate_inputs(
        monomers_list, adduct_compo, neutral_loss_use, neutral_loss_compo,
        association, use_label, label_compo
    )
    assert is_valid is True
    assert errors == []

def test_validate_inputs_no_monomers():
    """Test validation with no monomers."""
    is_valid, errors = validate_inputs(
        [], {"H+": {"H": 1}}, False, {}, {}, False, None
    )
    assert is_valid is False
    assert "No monomers provided" in errors

def test_validate_inputs_no_adduct_compo():
    """Test validation with no adduct composition."""
    is_valid, errors = validate_inputs(
        ["Glc"], {}, False, {}, {}, False, None
    )
    assert is_valid is False
    assert "Adduct composition is missing, empty or invalid" in errors

def test_validate_inputs_label_enabled_but_invalid():
    """Test validation with label enabled but invalid."""
    is_valid, errors = validate_inputs(
        ["Glc"], {"H+": {"H": 1}}, False, {}, {}, True, None
    )
    assert is_valid is False
    assert "Label is enabled but invalid" in errors

def test_validate_inputs_neutral_loss_enabled_but_invalid():
    """Test validation with neutral loss enabled but invalid."""
    is_valid, errors = validate_inputs(
        ["Glc"], {"H+": {"H": 1}}, True, {}, {}, False, None
    )
    assert is_valid is False
    assert "Chemical modification is enabled but invalid" in errors

def test_validate_inputs_neutral_loss_not_associated():
    """Test validation with neutral loss not associated with any adduct."""
    association = {"H2O": []}  # Not associated with any adduct
    is_valid, errors = validate_inputs(
        ["Glc"], {"H+": {"H": 1}}, True, {"H2O": {"H": 2, "O": 1}},
        association, False, None
    )
    assert is_valid is False
    assert "One or more chemical modifications provided are not associated with any adduct." in errors

# =============================================================================
# Tests for validate_lost_adduct
# =============================================================================

def test_validate_lost_adduct_not_lost():
    """Test validation for an adduct that is not lost."""
    composition = {"H": 1, "O": 1}
    is_lost = False
    monomer_elements = ["H", "O", "C"]
    label_composition = None
    assert validate_lost_adduct(composition, is_lost, monomer_elements, label_composition) == []

def test_validate_lost_adduct_lost_valid():
    """Test validation for a lost adduct with valid elements."""
    composition = {"H": 1, "O": 1}
    is_lost = True
    monomer_elements = ["H", "O", "C"]
    label_composition = None
    assert validate_lost_adduct(composition, is_lost, monomer_elements, label_composition) == []

def test_validate_lost_adduct_lost_invalid():
    """Test validation for a lost adduct with invalid elements."""
    composition = {"H": 1, "O": 1, "Xx": 1}  # Xx is invalid
    is_lost = True
    monomer_elements = ["H", "O", "C"]
    label_composition = None
    assert validate_lost_adduct(composition, is_lost, monomer_elements, label_composition) == ["Xx"]

def test_validate_lost_adduct_lost_valid_with_label():
    """Test validation for a lost adduct with elements in the label."""
    composition = {"H": 1, "C": 1}
    is_lost = True
    monomer_elements = ["H", "O"]
    label_composition = {"C": 1}
    assert validate_lost_adduct(composition, is_lost, monomer_elements, label_composition) == []

# =============================================================================
# Tests for validate_entity
# =============================================================================

def test_validate_entity_valid_adduct(sample_monomer_elements, sample_label_composition):
    """Test validation for a valid adduct."""
    entity = {"name": "H+", "mass": 1.007276, "formula": "H", "is_lost": False}
    assert validate_entity(entity, sample_monomer_elements, sample_label_composition, "adduct") is None

def test_validate_entity_invalid_formula():
    """Test validation for an entity with an invalid formula."""
    entity = {"name": "Xx+", "mass": 1.0, "formula": "Xx", "is_lost": False}
    # Patch check_formula to return ["Xx"]
    assert validate_entity(entity, ["H", "O"], None, "adduct") == "❌ Xx is not in the NIST monoisotopic library."

def test_validate_entity_lost_invalid_elements(sample_monomer_elements):
    """Test validation for a lost entity with invalid elements."""
    entity = {"name": "Xx-", "mass": 1.0, "formula": "Xx", "is_lost": True}
    # Patch check_formula to return [] (assume Xx is in the library)
    # But validate_lost_adduct should catch it
    assert validate_entity(entity, sample_monomer_elements, None, "adduct") == (
        "❌ Lost Adduct 'Xx' contains element(s) that are not present in the monomers or the selected label: Xx."
    )

def test_validate_entity_neutral_loss():
    """Test validation for a neutral loss (is_lost is forced to True)."""
    entity = {"name": "H2O", "mass": 18.010564, "formula": "H2O", "is_lost": False}  # is_lost is ignored for neutral_loss
    assert validate_entity(entity, ["H", "O"], None, "neutral_loss") is None

# =============================================================================
# Tests for validate_association
# =============================================================================

def test_validate_association_valid(sample_association):
    """Test validation for a valid association."""
    assert validate_association(sample_association) == []

def test_validate_association_invalid():
    """Test validation for an invalid association (neutral loss not associated)."""
    association = {"H2O": [], "NH3": ["K+"]}  # H2O is not associated
    errors = validate_association(association)
    assert len(errors) == 1
    assert "Neutral loss 'H2O' is not associated with any adduct." in errors[0]

# =============================================================================
# Tests for parse_association_mass
# =============================================================================

def test_parse_association_mass(sample_association, sample_adduct_mass, sample_neutral_loss_mass):
    """Test parsing association for masses."""
    result = parse_association_mass(sample_association, sample_adduct_mass, sample_neutral_loss_mass)
    expected = {
        "H+H2O": sample_adduct_mass["H+"] + sample_neutral_loss_mass["H2O"],
        "Na+H2O": sample_adduct_mass["Na+"] + sample_neutral_loss_mass["H2O"],
        "K+NH3": sample_adduct_mass["K+"] + sample_neutral_loss_mass["NH3"],
    }
    assert result == expected

def test_parse_association_mass_missing_keys():
    """Test parsing association with missing keys in mass dictionaries."""
    association = {"H2O": ["H+", "UnknownAdduct"]}
    adduct_mass = {"H+": 1.007276}
    neutral_loss_mass = {"H2O": 18.010564}
    result = parse_association_mass(association, adduct_mass, neutral_loss_mass)
    assert result == {"H+H2O": 1.007276 + 18.010564, "UnknownAdductH2O": 0 + 18.010564}

# =============================================================================
# Tests for parse_association_formula
# =============================================================================

def test_parse_association_formula(sample_association, sample_adduct_compo, sample_neutral_loss_compo):
    """Test parsing association for compositions."""
    result = parse_association_formula(sample_association, sample_adduct_compo, sample_neutral_loss_compo)
    expected = {
        "H+H2O": {"H": sample_adduct_compo["H+"]["H"] + sample_neutral_loss_compo["H2O"]["H"],
                  "O": sample_neutral_loss_compo["H2O"]["O"]},
        "Na+H2O": {"Na": sample_adduct_compo["Na+"]["Na"],
                   "H": sample_neutral_loss_compo["H2O"]["H"],
                   "O": sample_neutral_loss_compo["H2O"]["O"]},
        "K+NH3": {"K": sample_adduct_compo["K+"]["K"],
                  "N": sample_neutral_loss_compo["NH3"]["N"],
                  "H": sample_neutral_loss_compo["NH3"]["H"]},
    }
    assert result == expected

def test_parse_association_formula_missing_keys():
    """Test parsing association with missing keys in composition dictionaries."""
    association = {"H2O": ["H+", "UnknownAdduct"]}
    adduct_compo = {"H+": {"H": 1}}
    neutral_loss_compo = {"H2O": {"H": 2, "O": 1}}
    result = parse_association_formula(association, adduct_compo, neutral_loss_compo)
    assert result == {
        "H+H2O": {"H": 1 + 2, "O": 1},
        "UnknownAdductH2O": {"H": 2, "O": 1},  # UnknownAdduct has no composition
    }