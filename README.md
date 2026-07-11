# SugarMALDI v0.1.5

A Spectra Analysis Tool (Draft)

---

## User-friendly interface  

Access the app via Streamlit:  
*(add your deployed link here if available)*

## Try it on your machine  

### Installation  

#### 1. Get the code  

```bash
git clone https://github.com/Protactinium92/SugarMALDI.git
cd SugarMALDI
```

#### Option 1 — Installation with uv (recommended 🚀)

##### Install uv (if not already installed)

```bash
pip install uv
```

Or (Linux / Mac):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```



> uv is a fast Python package and project manager that can replace pip, virtualenv, and more.
> More information on [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)  


##### Install the project

```bash
uv sync
```

This will:
- create a virtual environment automatically  
- install all dependencies from `pyproject.toml`  

#### Option 2 — Standard installation (pip + venv)

Create a virtual environment
```bash
python -m venv .venv
```
Activate it

1. Linux / Mac
```bash
source .venv/bin/activate
```

2. Windows
```bash
.venv\Scripts\activate
```

Upgrade pip
```bash
pip install --upgrade pip
```

Install the project
```bash
pip install .
```

### Run the app  

Directly with UV:
```bash
uv run streamlit run app.py
```

Or in virtual environnment:

```bash
streamlit run app.py
```

## Important information:

> [!IMPORTANT]  
>Before starting the first analysis please create the oligosacaride library by running this command:
```bash
# UV:
uv run src/oligoMass.py

# In virtual envioronnement
src/oligoMass.py
```

## Project structure  

- `pyproject.toml` → dependency and project configuration  
- `app.py` → main Streamlit application  


## Example

Example spectra can be downloaded from the dedicated Google Drive folder [here](https://drive.google.com/drive/folders/1_okD6vXmOA4NmZ66R5Os8q0V76rWMidL) 

It contains 4 examples:  
- Arabinogalactan standard: Non-calibrated / Calibrated
- Galactomannan standard: Non-calibrated / Calibrated


## Notes  

- This project uses `pyproject.toml` (PEP 517/518)  
- Recommended Python version: **>= 3.9**  


The application relies on cached functions distributed across multiple modules. In some execution contexts, Streamlit may display the following warning:

```text id="9l95cl"
WARNING streamlit.runtime.caching.cache_data_api:
No runtime found, using MemoryCacheStorageManager
```

In this case, Streamlit automatically uses an in-memory cache backend. This behavior is expected and does not affect the application.


# License

SugarMALDI is distributed under the CeCILL 2.1 License.

The full license text is available in the LICENSE file and at
https://cecill.info.


## Contributing  

Feel free to open issues to improve the project 🚀  