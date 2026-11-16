# Library Agent

## Set up on Windows

From this directory, create and activate a virtual environment, then install
the project dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate the simulated SQLite database and run the test suite:

```powershell
python seed.py
python -m pytest
streamlit run app.py
```

`library-agent/` is the project directory; `library_agent/` is the single
importable Python package (underscores are required in package names). The
project has one test suite in `tests/`; seeded runtime data is stored in
`data/`.

## Agent and environment

The agent loop is **perceive** (loans, returns, queries), then **reason** (rank,
predict, detect), then **act** (recommend, alert, answer). Its environment is
the catalog, students, lending rules, and loan inventory stored in SQLite.
Student and librarian views are a demo interface and do not provide account
authentication.
