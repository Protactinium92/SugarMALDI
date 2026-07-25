import pytest
import polars as pl


@pytest.fixture
def mock_element_mass():
    return {
        "H": 1.007825,
        "C": 12.0,
        "O": 15.994915,
        "N": 14.003074,
        "Na": 22.989769,
        "K": 38.963707,
    }

@pytest.fixture
def sample_adducts():
    """Valid sample adducts for testing."""
    return [
        {"name": "H+", "mass": 1.007276, "formula": "H", "is_lost": False},
        {"name": "Na+", "mass": 22.989221, "formula": "Na", "is_lost": False},
        {"name": "K+", "mass": 38.963158, "formula": "K", "is_lost": False},
    ]

@pytest.fixture
def sample_neutral_losses():
    """Valid sample neutral losses for testing."""
    return [
        {"name": "H2O", "mass": 18.010564, "formula": "H2O", "is_lost": True},
    ]

@pytest.fixture
def sample_association():
    """Sample association dictionary for testing."""
    return {
        "H2O": ["H+", "Na+"],
        "NH3": ["K+"],
    }

@pytest.fixture
def sample_adduct_mass():
    """Sample adduct masses for testing."""
    return {"H+": 1.007276, "Na+": 22.989221, "K+": 38.963158}

@pytest.fixture
def sample_neutral_loss_mass():
    """Sample neutral loss masses for testing."""
    return {"H2O": 18.010564, "NH3": 17.026549}

@pytest.fixture
def sample_adduct_compo():
    """Sample adduct compositions for testing."""
    return {
        "H+": {"H": 1},
        "Na+": {"Na": 1},
        "K+": {"K": 1},
    }

@pytest.fixture
def sample_neutral_loss_compo():
    """Sample neutral loss compositions for testing."""
    return {
        "H2O": {"H": 2, "O": 1},
        "NH3": {"N": 1, "H": 3},
    }

@pytest.fixture
def monomer_df():
    """Minimal monomer database containing Hex."""
    return pl.DataFrame(
        {
            "Selected": [True],
            "Notation": ["Hex"],
            "Description": ["Hexose"],
            "C": [6],
            "H": [12],
            "O": [6],
            "N": [0],
        }
    )

@pytest.fixture
def label_3aq_compo():
    """Elemental composition of 3-AQ label."""
    return {
        "C": 9,
        "H": 8,
        "N": 2,
    }

@pytest.fixture
def sample_monomer_elements():
    """Elemental composition of monomeres."""
    return ['C', 'H', 'O']


