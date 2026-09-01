import pytest
import polars as pl
import io
import numpy as np



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
def label_3aq_compo():
    """Elemental composition of 3-AQ label."""
    return {
        "C": 9,
        "H": 8,
        "N": 2,
    }


@pytest.fixture
def sample_neutral_loss_compo():
    """Sample neutral loss compositions for testing."""
    return {
        "H2O": {"H": 2, "O": 1},
        "NH3": {"N": 1, "H": 3},
    }

@pytest.fixture
def monomer_df_hex_selected():
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
def monomer_df():
    """Sample monomer DataFrame."""
    return pl.DataFrame({
        "Selected": [False, True, False],
        "Notation": ["dHex", "Hex", "HexA"],
        "Description": ["Dhexose", "Hexose", "HexA"],
        "C": [6, 6, 6],
        "H": [10, 12, 10],
        "O": [5, 6, 6],
        "N": [0, 0, 1]
    })


@pytest.fixture
def sample_monomer_elements():
    """Elemental composition of monomeres without N for testing."""
    return ['C', 'H', 'O']

@pytest.fixture
def sample_spectrum_csv(tmp_path):
    """Create a temporary CSV file with sample spectrum data."""
    data = """m/z,Intensity
100.0,1000
101.0,2000
102.0,1500
103.0,500
"""
    file_path = tmp_path / "test_spectrum.csv"
    file_path.write_text(data)
    return str(file_path)

@pytest.fixture
def sample_spectrum_data():
    """Sample spectrum data as numpy arrays."""
    return {
        "m/z": np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        "Intensity": np.array([1000.0, 2000.0, 1500.0, 500.0, 3000.0, 2500.0])
    }

@pytest.fixture
def sample_lazyframe_spectrum(sample_spectrum_data):
    """Sample Polars LazyFrame for testing."""
    return pl.LazyFrame(sample_spectrum_data)

@pytest.fixture
def sample_dataframe_spectrum(sample_spectrum_data):
    """Sample Polars DataFrame for testing."""
    return pl.DataFrame(sample_spectrum_data)


@pytest.fixture
def sample_full_library_monomer():
    """Sample library LazyFrame."""
    return pl.LazyFrame({
        "column_id": [1, 2, 3],
        "dHex": [1, 0, 2],
        "Hex": [0, 1, 1],
        "HexA": [0, 0, 0],
        "Neutral_unlabeled_mass": [100.0, 150.0, 200.0]
    })


class MockUploadedFile(io.BytesIO):
    def __init__(self, content: str, name: str):
        super().__init__(content.encode("utf-8"))
        self.name = name


@pytest.fixture
def sample_uploaded_file():
    data = """m/z,Intensity
100.0,1000
101.0,2000
"""

    return MockUploadedFile(
        content=data,
        name="test_spectrum.csv"
    )
