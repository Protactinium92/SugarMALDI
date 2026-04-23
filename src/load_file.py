import streamlit as st

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
    

if __name__=="__main__":
    element = element_mass()
    print(len(element))
    for e, m in element.items():
        print(f'{e}: {m}')