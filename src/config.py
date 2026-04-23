import tempfile
from pathlib import Path


TEMP_DIR = Path(tempfile.mkdtemp())  # Temporate files
SCALE = 1_000_000  # 1e6 to convert float mass in int
ADDUCT_MASS = {'H+': 1.007276, 'Na+': 22.989221, 'K+': 38.963158}
ADDUCTS = {
    "H+": {"H": 1},
    "Na+": {"Na": 1},
    "K+": {"K": 1},
}
WATER = {'H': 2, 'O': 1}
LABEL_3AQ = {'C': 9, 'H': 8, 'N': 2}
NAME_COLUMN_MASS_OLIGO = "Neutral_unlabeled_mass"  # Mass without adduct and label 