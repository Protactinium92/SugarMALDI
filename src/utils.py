# Copyright (C) 2026 Pierre-Alexandre HO
#
# This file is part of SugarMALDI.
#
# Licensed under the CeCILL 2.1 license.
# See LICENSE file for details.

"""
Utility functions for SugarMALDI data processing and validation.
================================================================

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
from typing import Dict, Union, Tuple, List, Optional
from src import config as cf
from src import load_file as load


def parse_formula(formula: str, lost: bool = False) -> Dict[str, int]:
    """
    Parse a chemical formula into its elemental composition.

    Parameters
    ----------
    formula : str
        Chemical formula string representing a net elemental composition.
    lost : bool, optional
        Whether the adduct is lost during ionization.
        If ``True``, all element counts are negated.
        If ``False``, the formula is kept unchanged.

    Returns
    -------
    Dict[str, int]
        Elemental composition with signed counts.
    """
    # global sign variable to determine if the adduct is lost
    sign = -1 if lost else 1

    pattern = r"([A-Z][a-z]*)(\d*)"
    matches = re.findall(pattern, formula)

    composition: Dict[str, int] = {}
    for element, count in matches:
        value = int(count or 1) * sign
        composition[element] = composition.get(element, 0) + value

    # remove zero entries (optional clean-up)
    composition = {k: v for k, v in composition.items() if v != 0}

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

    Lost adducts are automatically prefixed with "(-)" in the output
    dictionaries.

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
        
        if not (name and formula):
            continue

        if is_lost:
            name = f"(-){name}"

        adduct_mass[name] = mass
        adduct_compo[name] = parse_formula(formula, is_lost)
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
        errors.append("Adduct composition is missing, empty or invalid")

    if use_label and label_compo is None:
        errors.append("Label is enabled but invalid")

    return len(errors) == 0, errors


def validate_lost_adduct(
    composition: Dict[str, int],
    is_lost: bool,
    monomer_elements: List[str],
    label_composition: Optional[Dict[str, int]],
) -> List[str]:
    """
    Validate that a lost adduct only contains removable elements.

    When an adduct is marked as lost during ionization, every element
    in its composition must be present either in the monomer elemental
    composition or in the label composition.

    Parameters
    ----------
    composition : dict[str, int]
        Elemental composition of the adduct.
    is_lost : bool
        Whether the adduct is lost during ionization.
    monomer_elements : list[str]
        List of element symbols present in the monomers
        (e.g. ["C", "H", "N", "O"]).
    label_composition : dict[str, int] or None
        Elemental composition of the selected label. If no label is used,
        this should be None.

    Returns
    -------
    list[str]
        Elements that cannot be removed. An empty list indicates that
        the lost adduct is valid.
    """
    if not is_lost:
        return []

    allowed_elements = set(monomer_elements)

    if label_composition is not None:
        allowed_elements.update(label_composition.keys())

    return sorted(
        element
        for element in composition
        if element not in allowed_elements
    )



def validate_adduct(
    adduct: Dict[str, object],
    monomer_elements: List[str],
    label_composition: Optional[Dict[str, int]],
) -> Optional[str]:
    """
    Validate a single adduct.

    The validation checks:
    - the chemical formula can be parsed;
    - all elements belong to the supported element list;
    - if the adduct is marked as lost, all its elements can originate
      from either the monomers or the selected label.

    Parameters
    ----------
    adduct : dict
        Adduct definition containing at least the keys ``"formula"``
        and ``"is_lost"``.
    monomer_elements : list[str]
        List of element symbols present in the monomers.
    label_composition : dict[str, int] or None
        Elemental composition of the selected label. If no label is used,
        this should be ``None``.

    Returns
    -------
    error : str or None
        Error message if validation failed, otherwise ``None``.
    """
    composition = parse_formula(
        adduct["formula"],
        adduct["is_lost"],
    )

    if adduct["formula"] and not composition:
        return (
            f"❌ Adduct formula '{adduct['formula']}' is invalid."
        )

    wrong_elements = check_formula(composition)
    if wrong_elements:
        elements_str = ", ".join(wrong_elements)
        verb = "are" if len(wrong_elements) > 1 else "is"

        return (
            f"❌ {elements_str} {verb} not in the NIST monoisotopic library."
        )

    invalid_elements = validate_lost_adduct(
        composition=composition,
        is_lost=adduct["is_lost"],
        monomer_elements=monomer_elements,
        label_composition=label_composition,
    )

    if invalid_elements:
        return (
            f"❌ Lost adduct '{adduct['formula']}' contains element(s) "
            "that are not present in the monomers or the selected label: "
            f"{', '.join(invalid_elements)}."
        )

    return  None

if __name__ == "__main__":
    print(
        "This module is not designed to be run standalone.\n"
        "It should be imported in `app.py` to provide utility functions.\n"
    )