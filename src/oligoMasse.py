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
ADDUCT_MASS = {'H+': 1.007276, 'Na+': 22.989769, 'K+': 38.963707}
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


def calc_oligo_mass(monomer_combination: List[int], monomer_masses: Dict[str, float]) -> float:
    """
    Calculate the total mass of an oligomer from a given combination of monomers,
    accounting for water loss during polymerization (one water molecule lost per bond).

    Args:
        monomer_combination: List of monomer counts (e.g., [2, 1, 0] for 2 of monomer 1, 1 of monomer 2, etc.).
        monomer_masses: Dictionary mapping each monomer notation to its mass.

    Returns:
        float: Total mass of the oligomer, after subtracting water loss.
    """
    mass = sum(
        nb_monomer_i * monomer_masses[monomer_i]
        for nb_monomer_i, monomer_i in zip(monomer_combination, monomers_list)
    )
    total_monomers = sum(monomer_combination)
    if total_monomers > 0:
        mass -= (total_monomers - 1) * calc_mass_mol(WATER)
    return mass


def generate_oligo_combinations(
    monomers_df: pl.DataFrame,
    monomers_list: List[str],
    spectrum_threshold: List[int],
) -> pd.DataFrame:
    """
    Generate all possible oligomer combinations (with monomer counts) whose total mass does not exceed a threshold.
    Uses a stack-based approach to iteratively explore combinations and prune invalid branches early.

    Args:
        monomers_df: DataFrame containing monomer data, with columns "Notation" and "mass".
        monomers_list: List of monomer notations (e.g., ["A", "B", "C"]).
        spectrum_threshold: Minimum and Maximum allowed mass for an oligomer combination.

    Returns:
        pd.DataFrame: A DataFrame where each row is a combination of monomer counts + its total mass.
                      Columns: monomer notations + "masse" (total mass).
                      Only includes combinations with mass <= spectrum_threshold.

    Note:
        The generation of all combinaison is perfomred with the backtracking algorythm
    """
    # Check that spectrum_threshold has exactly two elements
    if len(spectrum_threshold) != 2:
        raise ValueError("spectrum_threshold must be a list of two floats: [min_mass, max_mass]")
    _, max_mass = spectrum_threshold

    # List to store valid combinations and their masses
    results: List[List] = []
    # Stack for iterative backtracking: each element is (current_combination, next_monomer_index)
    stack: List[tuple] = []
    # Initialize stack with all-zero combination and starting index
    stack.append(([0] * len(monomers_list), 0))

    # Map monomer notation to its mass
    monomer_masses: Dict[str, float] = {
        row["Notation"]: row["mass"]
        for row in monomers_df.iter_rows(named=True)
    }

    while stack:
        current_combination, index = stack.pop()

        if index == len(monomers_list):
            # Full combination: calculate mass and store
            mass = calc_oligo_mass(current_combination, monomer_masses)
            results.append(current_combination + [round(mass, 6)])
        else:
            # Try all possible counts for the current monomer (0, 1, 2, ...)
            count = 0
            while True:
                new_combination = current_combination.copy()
                new_combination[index] = count

                # Calculate partial mass (automatically accounts for water loss)
                partial_mass = calc_oligo_mass(new_combination, monomer_masses)

                # Prune if partial mass exceeds threshold
                if partial_mass >= max_mass:
                    break

                # Explore next monomer
                stack.append((new_combination, index + 1))
                count += 1

    # Create Polars DataFrame and filter by mass threshold
    columns = monomers_list + ["Oligo_mass"]
    df = pl.DataFrame(results, schema=columns,  orient="row")
    df = df.filter((pl.col("Oligo_mass") <= max_mass))

    return df


def add_adduct_columns(
    df: pl.DataFrame,
    adduct_masses: Dict[str, float],
    mass_column_name: str = "Oligo_mass"
) -> pl.DataFrame:
    """
    Add columns for each adduct's total mass to a Polars DataFrame of oligomer combinations.

    Args:
        df: Polars DataFrame containing oligomer combinations and their base mass.
            Must include a column with the base mass (default: "masse").
        adduct_masses: Dictionary mapping adduct names to their masses (e.g., {"[M+H]+": 1.007, "[M+Na]+": 22.99}).
        mass_column: Name of the column containing the base mass (default: "Oligo_mass").

    Returns:
        pl.DataFrame: Input DataFrame with additional columns for each adduct's total mass.
                      New columns are named "Oligo_[adduct]".
    """
    label_mass = calc_mass_mol(LABEL_3AQ)
    for adduct, adduct_mass in adduct_masses.items():
        df = df.with_columns(
            (pl.col(mass_column_name) + label_mass + adduct_mass).alias(f"Oligo_{adduct}")
        )
    return df


def write_oligo_combinations_to_csv(
    df: pl.DataFrame,
    output_path: str,
    separator: str = ",",
    include_header: bool = True,
) -> None:
    """
    Write a Polars DataFrame of oligomer combinations to a CSV file.

    Args:
        df: Polars DataFrame containing oligomer combinations and their masses.
            Expected columns: monomer notations + "masse".
        output_path: Path to the output CSV file (e.g., "output/combinations.csv").
        separator: CSV field separator (default: ",").
        include_header: Whether to write the column names as the first row (default: True).

    Returns:
        None

    Raises:
        ValueError: If the output directory does not exist and cannot be created.
    """
    # Ensure the output directory exists
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write the DataFrame to CSV
    df.write_csv(
        output_path,
        separator=separator,
        include_header=include_header
    )


if __name__ == "__main__":
    spectrum_threshold = [500, 2500]  # Min / Maximum limit of the MS
    
    monomers_df = csv_to_df(Path("data/monomers.csv"), sep =";")
    monomers_df = calculate_molecular_mass(monomers_df)
    print(monomers_df)
    monomers_list = list(monomers_df["Notation"])
    combinations_df = generate_oligo_combinations(monomers_df, monomers_list, spectrum_threshold)
    combinations_df = add_adduct_columns(combinations_df, ADDUCT_MASS)
    
    write_oligo_combinations_to_csv(combinations_df, Path("data/oligo_combinations.csv"))
    