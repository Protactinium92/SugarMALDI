"""
    Generate a set of theoretical masses for oligosaccharides composed of hexoses and pentoses.

    These masses are intended for mass spectrometric analysis, where the oligosaccharides 
    are derivatized and adducts may be added during the measurement.

    General formula for the oligosaccharides:
        x Pentose + y Hexose → Oligosaccharide + (x + y - 1) H2O

    The corresponding mass-to-charge ratio (m/z) is given by:
        m/z = x·m(Pentose) + y·m(Hexose) - (x + y - 1)·m(H2O)

    The label used is the 3-AQ, which is involved in a condensation reaction with the reducing end
    of the oligosaccharide.

    Adduct mass are added at the end of the process.

    
"""

__author__  = "Pierre-Alexandre HO"
__date__    = "2025-09-02"
__version__ = "1.0"

# import
import polars as pl
from typing import Dict, List, Set, Union, Tuple
from pathlib import Path

# Constant

MONOISOTOPIC_MASS = {'C': 12.000000, 'H': 1.007825, 'O': 15.994915, 'N': 14.003074}
ADDUCT_MASS = {'H+': 1.007276, 'Na+': 22.989221, 'K+': 38.963158}
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
    return round(mass, 6)


def ensure_lazy(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    """
    Convert input to a Polars LazyFrame if it is not already lazy.

    Args:
        df: Polars DataFrame or LazyFrame.

    Returns:
        LazyFrame version of the input.
    """
    return df if isinstance(df, pl.LazyFrame) else df.lazy()


def generate_oligo_combinations(
    monomers_df: pl.DataFrame,
    monomers_list: List[str],
    spectrum_threshold: List[int],
) -> pl.LazyFrame:
    """
    Generate all possible oligomer combinations (with monomer counts) whose total mass does not exceed a threshold.
    Uses a stack-based approach to iteratively explore combinations and prune invalid branches early.

    Args:
        monomers_df: DataFrame containing monomer data, with columns "Notation" and "mass".
        monomers_list: List of monomer notations (e.g., ["A", "B", "C"]).
        spectrum_threshold: Minimum and Maximum allowed mass for an oligomer combination.

    Returns:
        pl.LazyFrame: A DataFrame where each row is a combination of monomer counts + its total mass.
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
    # Initialize stack with all-zero combination, starting index and partial masss
    stack: List[tuple] = [([0] * len(monomers_list), 0, 0.0)]
     

    # Map monomer notation to its mass
    monomer_masses: Dict[str, float] = {
        row["Notation"]: row["mass"]
        for row in monomers_df.iter_rows(named=True)
    }

    while stack:
        current_combination, index, partial_mass = stack.pop()

        if index == len(monomers_list):
            # Full combination: calculate mass and store
            results.append(current_combination + [partial_mass])
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
                stack.append((new_combination, index + 1, round(partial_mass, 6)))
                count += 1

    # Create Polars DataFrame and filter by mass threshold
    columns = monomers_list + ["Oligo_mass"]
    df = pl.DataFrame(results, schema=columns,  orient="row").lazy()
    df = df.filter((pl.col("Oligo_mass") <= max_mass))

    return df


def add_adduct_columns(
    df: Union[pl.DataFrame, pl.LazyFrame],
    adduct_masses: Dict[str, float],
    mass_column_name: str = "Oligo_mass"
) -> pl.LazyFrame:
    """
    Add columns for each adduct's total mass to a Polars DataFrame of oligomer combinations.
    Addition of the label 3AQ which is a condensation reaction with the oligomer

    Args:
        df: Polars DataFrame containing oligomer combinations and their base mass.
            Must include a column with the base mass (default: "masse").
        adduct_masses: Dictionary mapping adduct names to their masses (e.g., {"[M+H]+": 1.007, "[M+Na]+": 22.99}).
        mass_column: Name of the column containing the base mass (default: "Oligo_mass").

    Returns:
        pl.LazyFrame: Input LazyFrame with additional columns for each adduct's total mass.
                      New columns are named "Oligo_[adduct]".
    """
    df_lazy = ensure_lazy(df)

    label_mass = calc_mass_mol(LABEL_3AQ)
    water = calc_mass_mol(WATER)
    # Build all adduct columns in one call using list comprehension
    adduct_columns = [
        (pl.col(mass_column_name) + label_mass - water + adduct_mass).round(6).alias(f"Oligo_{adduct}")
        for adduct, adduct_mass in adduct_masses.items()
    ]

    # Add all columns lazily
    return df_lazy.with_columns(adduct_columns)


def add_column_id(df: pl.LazyFrame, monomers: List[str]) -> pl.LazyFrame:
    """
    Adds a 'column_id' column to the LazyFrame by concatenating the specified columns with "-".
    The 'column_id' column is moved to the first position.

    Args:
        df: Input Polars LazyFrame.
        monomers: List of column names to concatenate.

    Returns:
        Polars LazyFrame with 'column_id' as the first column.
    """
    
    df_lazy = ensure_lazy(df)
    column_names = df.collect_schema().names()
    df_lazy = (df_lazy
               .with_columns(pl.concat_str(monomers, separator="-").alias("column_id"))
               .select(["column_id"] + column_names)
    )
    return df_lazy


def get_duplicate(
    df: Union[pl.DataFrame, pl.LazyFrame],
    id_col: str = "column_id",
    columns_to_check: List[str] | None = None) -> List[str]:
    """
    Returns a set of `id_col` values corresponding to rows that contain 
    at least one value that appears more than once in any of the specified columns.
    
    Fully lazy and vectorized horizontal version:

    Args:
        df (pl.DataFrame): The Polars DataFrame to analyze.
        id_col (str, optional): The name of the unique identifier column. Defaults to "column_id".
        columns_to_check (List[str] | None, optional): List of columns to check for duplicated values.
            If None, use the key of ADDUCT_MASS dict.

    Returns:
        List[str]: A list of `id_col` values for rows containing duplicated values.
    """
    if columns_to_check is None:
        columns_to_check = [f'Oligo_{key}' for key in list(ADDUCT_MASS.keys())]

    # Convert to lazy frame
    df_lazy = ensure_lazy(df)

    # Step 1: For each column, find which values appear more than once (lazy, vectorized)
    duplicated_values_exprs = [
        pl.col(c).is_in(  # rerurn bool for each row, whether the value in column c is in the given set of values
            df_lazy.select(pl.col(c), allow_object=True)
                   .group_by(c)  # separate DataFrame into groups of identical values.
                   .agg(pl.len().alias("cnt"))  # count how many rows are in each group.
                   .filter(pl.col("cnt") > 1)
                   .collect()[c]
                   .to_list()
        )
        for c in columns_to_check
    ]

    # Step 2: Combine all boolean masks horizontally with pl.any_horizontal
    mask_global = pl.any_horizontal(duplicated_values_exprs)

    # Step 3: Filter original DataFrame lazily and collect id_col values
    duplicate_ids = (
        df_lazy.filter(mask_global)
               .select(id_col)
               .unique()
               .collect()[id_col]
               .to_list()
    )

    return duplicate_ids


def reduce_row(
    combinations_df: pl.DataFrame | pl.LazyFrame,
    duplicate_ids: list[str],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Clean the dataframe by:
      - keeping all rows whose `column_id` is not duplicated,
      - keeping only the first occurrence of rows whose `column_id` belongs 
        to the duplicated set (based on identical masses).

    Args:
        combinations_df (pl.DataFrame | pl.LazyFrame): 
            Input DataFrame containing oligomer combinations.
        duplicate_ids (list[str]): 
            List of `column_id` values that correspond to duplicated masses.

    Returns:
        pl.DataFrame | pl.LazyFrame:
            Cleaned DataFrame with:
              - a new column 'Duplicate' tagging each row as "Unique" or "Shared",
              - one row per duplicated mass group (first occurrence only).
    """

    # Convert to lazy frame
    combinations_df = ensure_lazy(combinations_df)

    # Keep all unique rows
    unique = (
        combinations_df
        .filter(~pl.col("column_id").is_in(duplicate_ids))
        .with_columns(pl.lit("Unique").alias("Multiple_Oligo"))
    )

    # Keep only first occurrence of duplicate rows
    duplicate = (
        combinations_df
        .filter(pl.col("column_id").is_in(duplicate_ids))
        .unique(
            subset=[f"Oligo_{key}" for key in ADDUCT_MASS.keys()],
            keep="first",
            maintain_order=True,
        )
        .with_columns(pl.lit("Shared").alias("Multiple_Oligo"))
    )

    # Concatenate both
    return pl.concat([unique, duplicate])


def write_pl_df_to_csv(
    df: Union[pl.DataFrame, pl.LazyFrame],
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

    df = df.collect() if isinstance(df, pl.LazyFrame) else df

    # Write the DataFrame to CSV
    df.write_csv(
        output_path,
        separator=separator,
        include_header=include_header
    )


if __name__ == "__main__":
    spectrum_threshold = [500, 4000]  # Min / Maximum limit of the MS
    
    # Monomers treatement
    monomers_df = csv_to_df(Path("data/monomers.csv"), sep =";")
    monomers_df = calculate_molecular_mass(monomers_df)
    print(monomers_df)
    monomers_list = list(monomers_df["Notation"])

    # Generation of all combination
    print(f'Generation of combination please wait')
    combinations_df = (
        generate_oligo_combinations(monomers_df, monomers_list, spectrum_threshold)
        .pipe(add_adduct_columns, ADDUCT_MASS)
        .pipe(add_column_id, monomers_list)
    )

    size_combination = combinations_df.select(pl.len()).collect().item()
    write_pl_df_to_csv(combinations_df, Path("data/oligo_combinations.csv"))

    # Duplicate treatment
    print(f'Calculation of mass duplicates please wait')
    duplicate_ids = get_duplicate(combinations_df)
    print(f'Size of cobination: {size_combination:,} include {len(duplicate_ids):,} lines with '
          'at least one mass identic to another one in the table')

    # Clean dataframe
    print(f'Datacuration please wait')
    final_oligomers = reduce_row(combinations_df, duplicate_ids).collect()
    print(f"Final size of the database: {len(final_oligomers):,} * {len(ADDUCT_MASS)} mass")
    write_pl_df_to_csv(final_oligomers, Path("data/oligo_combinations_clean.csv"))

