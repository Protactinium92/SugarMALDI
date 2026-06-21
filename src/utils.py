"""
Utility functions for SugarMALDI data processing and validation.

This module provides reusable helper functions for:
- Chemical formula parsing into elemental composition
- Adduct processing and dictionary construction
- Input validation for MALDI-MS analysis workflows
- Row-level validation of user-provided structured data

Functions in this module operate on generic Python data structures
(dict, list, str) and do not require Streamlit state or rendering.
"""

__author__  = "Pierre-Alexandre HO"


import re
from typing import Dict, Union, Tuple, List
from src import config as cf
from src import load_file as load


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into its elemental composition.

    This function converts a string representation of a chemical formula
    into a dictionary where keys are element symbols and values are their
    respective counts.

    Parameters
    ----------
    formula : str
        Chemical formula string (e.g., "NH4", "C6H12O6").

    Returns
    -------
    Dict[str, int]
        Dictionary mapping element symbols to integer counts.
        Example: "NH4" -> {"N": 1, "H": 4}

    Notes
    -----
    This parser assumes a simple chemical notation without nested
    groups or parentheses.
    """
    pattern = r"([A-Z][a-z]*)(\d*)"
    matches = re.findall(pattern, formula)
    
    composition = {}
    for element, count in matches:
        composition[element] = composition.get(element, 0) + int(count or 1)
    
    return composition


def validate_rows(
    rows: List[dict],
    required_fields: List[str],
) -> Tuple[List[dict], List[int], bool]:
    """
    Validate structured input rows based on required fields.

    This function checks a list of dictionaries (rows) and separates:
    - Fully valid rows (all required fields filled)
    - Invalid rows (partial input)
    - Completely empty rows

    Parameters
    ----------
    rows : list of dict
        Input data
    required_fields : list of str
        Fields that must be filled for a row to be valid

    Returns
    -------
    valid_rows : List[dict]
        Rows where all required fields are properly filled.
    error_indices : List[int]
        Indices of rows that contain partial/incomplete data.
    has_any_input : bool
        True if at least one row contains any non-empty input.
    """

    valid_rows = []
    error_indices = []
    has_any_input = False

    for i, row in enumerate(rows):

        values = [row.get(f) for f in required_fields]

        # Empty line
        if all(v in cf.EMPTY_VALUES for v in values):
            continue
        
        # At least one value somewhere
        has_any_input = True

        # True if all required fields are filled
        if all(v not in cf.EMPTY_VALUES for v in values):
            valid_rows.append(row)
        else:
            error_indices.append(i)

    return valid_rows, error_indices, has_any_input


def build_adduct_dicts(adducts: List[dict]
                       )-> Tuple[Dict[str, float], Dict[str, Dict[str, int]], Dict[str, bool]]:
    """
    Build mass and composition dictionaries from a list of adduct definitions.

    This function processes a list of adduct dictionaries and constructs:
    - A mapping of adduct names to their masses
    - A mapping of adduct names to their elemental compositions
    - A mapping of adduct names to their lost status

    Parameters
    ----------
    adducts : List[dict]
        List of adduct dictionaries. Each dictionary must contain:
        - "name" (str): Adduct identifier
        - "mass" (float): Adduct mass
        - "formula" (str): Chemical formula of the adduct
        - "is_lost" (bool): Whether the adduct is lost during ionization

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, Dict[str, int]], Dict[str, bool]]
        - adduct_mass: mapping of adduct names to masses
        - adduct_compo: mapping of adduct names to elemental compositions
        - adduct_is_lost: mapping of adduct names to their lost status
    """
    
    adduct_mass: Dict[str, float] = {}
    adduct_compo: Dict[str, Dict[str, int]] = {}
    adduct_is_lost: Dict[str, bool] = {}

    for adduct in adducts:
        name = adduct["name"]
        mass = adduct["mass"]
        formula = adduct["formula"]
        is_lost = adduct["is_lost"]
        
        if name and formula:
            adduct_mass[name] = mass
            adduct_compo[name] = parse_formula(formula)
            adduct_is_lost[name] = is_lost

    return adduct_mass, adduct_compo, adduct_is_lost


def check_formula(dict_formula_compo: dict) -> List[str]:
    """
    Check whether all elements in a chemical formula are present
    in the reference elemental mass library.

    This function compares the keys of a parsed chemical formula
    against the available monoisotopic mass database and returns
    a list of unknown or unsupported elements.

    Parameters
    ----------
    dict_formula_compo : dict
        Dictionary representing a chemical formula composition,
        where keys are element symbols (e.g., "C", "H", "O")
        and values are their counts.

    Returns
    -------
    List[str]
        List of element symbols that are not found in the
        elemental mass library. Returns an empty list if all
        elements are valid.

    """
    element_mass_library = load.element_mass()
    return [
        element for element in dict_formula_compo.keys()
        if element not in element_mass_library
    ]


def validate_inputs(
    monomers_list: List[str],
    adduct_compo: Dict[str, int],
    use_label: bool,
    label_compo: Dict[str, int],
) -> Tuple[bool, List[str]]:
    """
    Validate input parameters for MALDI-MS spectral analysis.

    This function checks the consistency and completeness of user-provided
    inputs required for downstream analysis.

    Parameters
    ----------
    monomers_list : List[str]
        List of monomer identifiers.
    adduct_compo : Dict[str, int]
        Adduct composition dictionary.
    use_label : bool
        Whether isotopic or chemical labeling is enabled.
    label_compo : Dict[str, int]
        Composition dictionary of the label (if used).

    Returns
    -------
    is_valid : bool
        Whether all conditions are satisfied
    errors : list of str
        List of error messages
    """
    errors = []

    if not monomers_list:
        errors.append("No monomers provided")

    if not adduct_compo:
        errors.append("Adduct composition is missing or empty")

    if use_label and label_compo is None:
        errors.append("Label is enabled but invalid")

    return len(errors) == 0, errors


if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )