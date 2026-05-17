# WALTzER simulator

Spectral and detector simulation for the WALTzER instrument. The main entry point is `src/waltzer_simulator.py`.

## Required components

These are the third-party libraries the code expects (see also `tests/config/requirements.txt` for pinned-style versions used in CI/tests):

| Component    | Role |
|-------------|------|
| **Python**  | 3.10+ recommended (development uses current 3.x). |
| **numpy**   | Arrays and numerics throughout. |
| **scipy**   | Interpolation, special functions, splines (e.g. instrument/background). |
| **matplotlib** | Flux/debug plots when enabled in config. |
| **astropy** | Coordinates, time, units, FITS I/O, table IO. |
| **astroquery** | Gaia / archive queries (target and background stars when those code paths run). |
| **openpyxl** | Reading Excel parameter and mapping files (`loaders/load_excel.py` and helpers). |
| **pandas**  | Used in Gaia helper scripts and some utilities (`gaiahelper/`). |

Install in a virtual environment:

```bash
pip install -r tests/config/requirements.txt
```

Some features (Gaia lookups, SIMBAD name resolution via `astroquery`) need **network access** at runtime.

## Repository layout

- **`configs/`** — Global and per-channel instrument configuration.
- **`data/`** — Efficiency curves, zodiacal light, stellar model grids, etc. See `data/readme.txt` for the file checklist.
- **`input/`** — Example parameter files (e.g. `input/parameters/`).
- **`src/`** — Application code; photon spectral pipeline: `src/flux/photon_flux_pipeline.py`.

## Running

Run commands from the **repository root** so relative paths such as `input/` and `configs/` match what the simulator expects.

When you run `python src/waltzer_simulator.py`, Python places the **directory that contains that script** (the `src/` folder) at the front of `sys.path`. Imports from top-level packages under `src/` (for example `loaders` and `configs`) therefore work **without** setting `PYTHONPATH`. Set `PYTHONPATH=src` if you import those modules without going through this script (for example one-off `python -c` from another working directory).

**CLI:** at most one optional argument — the path to the parameter file. If you omit it, the simulator loads **`input/parameters.txt`**.

Example using the default file:

```bash
python src/waltzer_simulator.py
```

Example passing a specific parameter file:

```bash
python src/waltzer_simulator.py input/parameters/parameter_030.txt
```

## Tests

```bash
python -m pytest tests/
```
