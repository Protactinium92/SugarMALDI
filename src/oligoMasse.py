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
import pandas as pd
import polars as pl
from typing import Dict, List, Tuple
from pathlib import Path

# Constant

MONOISOTOPIC_MASS = {'C': 12.000000, 'H': 1.007825, 'O': 15.994915, 'N': 14.003074}
ADDUC_MASS = {'H+': 1.007276, 'Na+': 22.989769, 'K+': 38.963707}
# HEXOSE = {'C': 6, 'H': 12, 'O': 6}
# PENTOSE = {'C': 5, 'H': 10, 'O': 5}
WATER = {'H': 2, 'O': 1}
LABEL_3AQ = {'C': 9, 'H': 8, 'N': 2}


def csv_to_df(file: str, sep: str):
    return pl.read_csv(file, separator = sep)


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


def calculate_molecular_mass(
    df: pl.DataFrame,
    ignore_columns: List[str] = None
) -> pl.DataFrame:
    """
    Calculate the molecular mass for each row in a Polars DataFrame using a global
    MONOISOTOPIC_MASS dictionary. The function sums the contributions of each atom
    (columns) to compute the total molecular mass.

    Args:
        df: Input DataFrame. The first two columns are assumed to be metadata
            (e.g., 'Notation', 'Description'), and the rest are atomic counts.
        ignore_columns: List of column names to exclude from the calculation.
            Defaults to ['Notation', 'Description'] if not provided.

    Returns:
        A new DataFrame with an additional 'mass' column containing the calculated
        molecular mass for each row.

    Raises:
        ValueError: If any atomic column in the DataFrame is missing from MONOISOTOPIC_MASS.

    Note:
        We used the vectorization property of polars
    """
    # Set default ignore_columns if not provided
    if ignore_columns is None:
        ignore_columns = ["Notation", "Description"]

    # Select atoms in the df
    # and check if every atoms are include in MONOISOTOPIC_MASS
    atom_cols = [col for col in df.columns if col not in ignore_columns]
    
    missing_atoms = set(atom_cols) - set(MONOISOTOPIC_MASS.keys())
    if missing_atoms:
        raise ValueError(
            f"Missing monoisotopic mass for atoms: {missing_atoms}. "
            "Please update the global MONOISOTOPIC_MASS dictionary."
        )

    # Build the expression to sum (count * mass) for each atom
    mass_expr = pl.lit(0.0)  # initialisation of the column
    for col in atom_cols:
        mass_expr += pl.col(col) * MONOISOTOPIC_MASS[col]

    # Add the 'mass' column to the DataFrame
    return df.with_columns(mass_expr.alias("mass"))


# def calc_mass_oligo(number_pentose: int = 0, number_hexose: int = 0) -> float:
#     """
#     Calculate the molecular mass of an oligosaccharide, accounting for pentoses,
#     hexoses, and water loss during glycosidic bond formation.

#     Args:
#         number_pentose: Number of pentose residues in the oligosaccharide.
#         number_hexose: Number of hexose residues in the oligosaccharide.

#     Returns:
#         Total molecular mass of the oligosaccharide, in atomic mass units (u).
    
#     Raises:
#         ValueError: If neither pentose nor hexose residues are provided.
#     """
#     try:
#         if number_pentose < 0 or number_hexose < 0:
#             raise ValueError("Residue counts must be non-negative.")
#         if number_pentose == 0 and number_hexose == 0:
#             raise ValueError("At least one pentose or hexose residue must be provided.")
#     except ValueError as e:
#         raise ValueError(f"Invalid input: {e}")
    
#     mass_pentose = number_pentose * calc_mass_mol(PENTOSE)
#     mass_hexose = number_hexose * calc_mass_mol(HEXOSE)
#     return mass_pentose + mass_hexose - (number_hexose + number_pentose - 1) * calc_mass_mol(WATER)


def generate_oligo_combinations(df: pl.DataFrame, max_mass: float) -> List[Tuple[int, int, float]]:
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
    monomers_df = csv_to_df(Path("data/monomers.csv"), sep =";")
    monomers_df = calculate_molecular_mass(monomers_df)
    spectrum_threshold = 2500  # Maximum limit of the MS
    #combinations = generate_oligo_combinations(spectrum_threshold)
    #write_oligo_combinations_to_csv(combinations)
    print(monomers_df)
    monomers_list = list(monomers_df["Notation"])
    number_monomers = len(monomers_list)
    # df = pd.DataFrame(columns=monomers_list)
    
    # mass_list = []
    # mass = 0
    # i = 0
    # while mass < spectrum_threshold:
    #     new_row = [i+1] + [0] * (len(monomers_list))
    #     df.loc[i] = new_row
        
    
    
    # print(df)

    # Crée un DataFrame vide avec les colonnes (y compris "masse")
    df = pd.DataFrame(columns=monomers_list + ["mass_oligo"])


    # def calc_mass_oligo(list_nb_mono, monomers_df):
    #     mass = 0
    #     for j, monomer in enumerate(monomers_list):
    #         # Récupère la masse du monomère depuis monomers_df
    #         monomer_mass =  monomers_df.filter(pl.col("Notation") == monomer)["mass"].item()
    #         # Ajoute la contribution de ce monomère à la masse totale
    #         mass += list_nb_mono[j] * monomer_mass

    #     # Soustrait (nombre de monomères - 1) * mH2O
    #     mass -= (sum(list_nb_mono) - 1) * calc_mass_mol(WATER)
    #     return mass

    # for mono in range(number_monomers):
        
    #     mass = 0
    #     i = 0
    #     while mass < spectrum_threshold:
    #         # Crée la nouvelle ligne : [i+1, 0, 0, ...]
    #         new_row_values = [0] * (number_monomers)
    #         new_row_values[mono] = i+1
    #         # Calcule la masse de la ligne
    #         mass = calc_mass_oligo(new_row_values, monomers_df)

    #         # Ajoute la masse calculée à la ligne
    #         new_row_values.append(mass)

    #         # Ajoute la ligne au DataFrame
    #         df.loc[len(df)] = new_row_values

    #         # Passe à l'itération suivante
    #         i += 1

    # print(df)


    def calc_mass_oligo(counts):
        mass = sum(count * monomer_masses[monomer] for count, monomer in zip(counts, monomers_list))
        total_monomers = sum(counts)
        if total_monomers > 0:
            mass -= (total_monomers - 1) * calc_mass_mol(WATER)
        return mass

    # Liste pour stocker les résultats
    results = []

    # Utilise une pile pour générer les combinaisons
    stack = []
    # Chaque élément de la pile est une tuple (combinaison, index du monomère à incrémenter)
    stack.append(([0] * len(monomers_list), 0))
    monomer_masses = {
        row["Notation"]: row["mass"]
        for row in monomers_df.iter_rows(named=True)
        }
    while stack:
        current_combination, index = stack.pop()

        if index == len(monomers_list):
            # Calcule la masse de la combinaison actuelle
            mass = calc_mass_oligo(current_combination)
            # Ajoute la combinaison et la masse aux résultats
            results.append(current_combination + [mass])
        else:
            # Essaye toutes les valeurs possibles pour ce monomère (0, 1, 2, ...)
            # On s'arrête si la masse dépasse le seuil pour éviter les combinaisons inutiles
            count = 0
            while True:
                new_combination = current_combination.copy()
                new_combination[index] = count
                # Estime la masse minimale possible pour cette combinaison partielle
                partial_mass = sum(new_combination[j] * monomer_masses[monomers_list[j]] for j in range(index + 1))
                if partial_mass >= spectrum_threshold:
                    break  # Pas besoin d'aller plus loin
                stack.append((new_combination, index + 1))
                count += 1

    # Crée un DataFrame Pandas avec les résultats
    columns = monomers_list + ["masse"]
    df = pd.DataFrame(results, columns=columns)

    # Filtre les lignes où la masse ne dépasse pas le seuil
    df = df[df["masse"] <= spectrum_threshold]

    print(df)





