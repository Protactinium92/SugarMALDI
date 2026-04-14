# SugarMALDI  

A Spectra Analysis Tool (Draft)

---

## User-friendly interface  

Access the app via Streamlit:  
*(add your deployed link here if available)*

---

## Try it on your machine  

---

### Installation  

#### 1. Get the code  

```bash
git clone https://github.com/Protactinium92/SugarMALDI.git
cd SugarMALDI
```

---

#### Option 1 — Installation with uv (recommended 🚀)

##### Install uv (if not already installed)

```bash
pip install uv
```

Or (Linux / Mac):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

More information on [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/getting-started/installation/#pypi)

> uv is a fast Python package and project manager that can replace pip, virtualenv, and more. :contentReference[oaicite:0]{index=0}  


---

##### Install the project

```bash
uv sync
```

This will:
- create a virtual environment automatically  
- install all dependencies from `pyproject.toml`  

---

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

---

### Run the app  

With UV:
```bash
uv run streamlit run src/app.py
```

Or in virtual environnment:

```bash
streamlit run src/app.py
```

---

## Project structure  

- `pyproject.toml` → dependency and project configuration  
- `app.py` → main Streamlit application  

---

## Notes  

- This project uses `pyproject.toml` (PEP 517/518), so no `requirements.txt` is needed.  
- Recommended Python version: **>= 3.9**  


---

## Contributing  

Feel free to open issues or submit pull requests to improve the project 🚀  