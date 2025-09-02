"""
    Generate a set of theoretical masses for oligosaccharides composed of hexoses and pentoses.

    These masses are intended for mass spectrometric analysis, where the oligosaccharides 
    are derivatized and adducts may be added during the measurement.

    General formula for the oligosaccharides:
        x Pentose + y Hexose → Oligosaccharide + (x + y - 1) H2O

    The corresponding mass-to-charge ratio (m/z) is given by:
        m/z = x·m(Pentose) + y·m(Hexose) - (x + y - 1)·m(H2O)
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

# import
import csv
from typing import Dict, List, Tuple


# Constant

MONOISOTOPIC_MASS = {'C': 12.000000, 'H': 1.007825, 'O': 15.994915, 'N': 14.003074}
ADDUC_MASS = {'H+': 1.007276, 'Na+': 22.989769, 'K+': 38.963707}
HEXOSE = {'C': 6, 'H': 12, 'O': 6}
PENTOSE = {'C': 5, 'H': 10, 'O': 5}
WATER = {'H': 2, 'O': 1}


def calc_mass_mol(mol: Dict[str, int]) -> float:
    """
    Calculate the molecular mass of a molecule (mol) from its atomic composition.

    Args:
        mol: Dictionary representing the atomic composition of the molecule.
             Keys: atomic symbols (e.g., 'C', 'H', 'O').
             Values: number of atoms of each type.

    Returns:
        Molecular mass of the molecule, in atomic mass units (u).
    """
    return sum(MONOISOTOPIC_MASS[elem] * mol[elem] for elem in mol)


def calc_mass_oligo(number_pentose: int = 0, number_hexose: int = 0) -> float:
    """
    Calculate the molecular mass of an oligosaccharide, accounting for pentoses,
    hexoses, and water loss during glycosidic bond formation.

    Args:
        number_pentose: Number of pentose residues in the oligosaccharide.
        number_hexose: Number of hexose residues in the oligosaccharide.

    Returns:
        Total molecular mass of the oligosaccharide, in atomic mass units (u).
    
    Raises:
        ValueError: If neither pentose nor hexose residues are provided.
    """
    try:
        if number_pentose < 0 or number_hexose < 0:
            raise ValueError("Residue counts must be non-negative.")
        if number_pentose == 0 and number_hexose == 0:
            raise ValueError("At least one pentose or hexose residue must be provided.")
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")
    
    mass_pentose = number_pentose * calc_mass_mol(PENTOSE)
    mass_hexose = number_hexose * calc_mass_mol(HEXOSE)
    return mass_pentose + mass_hexose - (number_hexose + number_pentose - 1) * calc_mass_mol(WATER)


def generate_oligo_combinations(max_mass: float) -> List[Tuple[int, int, float]]:
    """
    Generate all possible combinations of pentose and hexose residues whose molecular mass
    is below a given maximum mass (max_mass). Each combination is represented as a tuple
    (number_pentose, number_hexose, molecular_mass).

    Args:
        max_mass: Maximum allowed molecular mass (in atomic mass units, u).

    Returns:
        A list of tuples, where each tuple contains:
        - number of pentose residues (int)
        - number of hexose residues (int)
        - molecular mass of the combination (float)
    """
    combinations = []  # List to store valid combinations
    number_pentose = 0
    follow = True  # Flag statement to avoid infinit loop

    while follow:
        oligo_mass = 0
        number_hexose = 0 if number_pentose > 0 else 1  # Skip (0, 0) combination
        while oligo_mass < max_mass:
            oligo_mass = calc_mass_oligo(number_pentose, number_hexose)
            
            #  If only pentose is used and the mass is already bigger than the limit
            if number_hexose == 0 and oligo_mass > max_mass:
                follow = False
                break
            
            # Add valid combination to the list
            combinations.append((number_pentose, number_hexose, round(oligo_mass, 6)))
            print(f"Pentoses: {number_pentose}, Hexoses: {number_hexose} -> Mass: {oligo_mass:.3f} u")
            number_hexose += 1
        number_pentose += 1

    return combinations


def write_oligo_combinations_to_csv(
    combinations: List[Tuple[int, int, float]],
    filename: str = "oligo_combinations.csv"
) -> None:
    """
    Writes oligosaccharide combinations to a CSV file.

    Args:
        combinations: List of tuples (number_pentose, number_hexose, mass_oligo) to write.
        filename: Name of the output CSV file. Defaults to "oligo_combinations.csv".

    Example:
        >>> combinations = [(0, 1, 162.053), (0, 2, 322.119), (1, 0, 132.042)]
        >>> write_oligo_combinations_to_csv(combinations)
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["number_pentose", "number_hexose", "mass_oligo"])
        # Write data rows
        writer.writerows(combinations)


if __name__ == "__main__":
    spectrum_threshold = 2500  # Maximum limit of the MS
    combinations = generate_oligo_combinations(spectrum_threshold)
    write_oligo_combinations_to_csv(combinations)





