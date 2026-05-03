from typing import Dict, Union, Tuple, List

def validate_inputs(
    monomers_list: List[str],
    adduct_compo: Dict[str, int],
    use_label: bool,
    label_compo: Dict[str, int],
) -> Tuple[bool, List[str]]:
    """
    Validate all inputs for spectra analysis.

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