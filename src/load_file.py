import streamlit as st
import polars as pl

from pathlib import Path



@st.cache_data
def element_mass():
    path_file = Path(__file__).parents[1]/"data"/"mass_elements.txt"
    element_dict = {}
    with open(path_file, "r") as source:
        for line in source:
            element, mass = line.strip().split("\t")
            element_dict[element] = float(mass)
    return element_dict


@st.cache_data
def read_csv(file:str, sep: str = ','):
    """
    Read a lambda csv file

    Args:
        file (str): _description_
        sep (str, optional): _description_. Defaults to ','.

    Returns:
        _type_: _description_
    """
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    return pl.scan_csv(file, separator=sep)

    
if __name__=="__main__":
    element = element_mass()
    print(len(element))
    for e, m in element.items():
        print(f'{e}: {m}')