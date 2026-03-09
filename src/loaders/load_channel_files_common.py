from pathlib import Path
import logging
import numpy as np
from loaders.run_waltzer_context import get_repo_root
from utils.helpers import resolve_path_under

def load_effective_area_file(effective_area_filename: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load effective area calibration table.

    Resolves file path as:
        repo_root / "data" / effective_area_filename

    Returns:
        wavelength_angstrom: np.ndarray
        effective_area_cm2: np.ndarray
        pixel_scale: float  (mandatory; taken from '# Pixel scale: <value>' header line)

    Reads numeric table as whitespace-delimited. Ignores middle columns by taking:
        first numeric column  -> wavelength
        last  numeric column  -> effective area

    Hard fails with a clear error message (and logs) if:
        file missing
        pixel scale missing
        no numeric data rows
        wavelength/effective area length mismatch
    """
    repo_root = get_repo_root()
    path = resolve_path_under(repo_root, "data", effective_area_filename)

    logging.info("Loading effective area file: %s", path)

    if not path.exists():
        msg = f"Effective area file not found: {path}"
        logging.error(msg)
        raise ValueError(msg)

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    pixel_scale = _parse_pixel_scale(text, path)
    skiprows = find_first_numeric_row_index(text, path)

    try:
        data = np.loadtxt(path, comments="#", skiprows=skiprows)
    except Exception as exc:
        msg = f"Failed to parse numeric data from effective area file: {path}"
        logging.error(msg)
        raise ValueError(msg) from exc

    if data.ndim == 1:
        msg = (
            f"Invalid effective area table structure in file: {path}. "
            "Expected at least 2 rows and 2 columns."
        )
        logging.error(msg)
        raise ValueError(msg)


    if data.size == 0 or data.shape[0] == 0:
        msg = f"No numeric data rows found in effective area file: {path}"
        logging.error(msg)
        raise ValueError(msg)

    wavelength = data[:, 0].astype(float, copy=False)

    eff_area = data[:, -1].astype(float, copy=False)

    if wavelength.shape[0] != eff_area.shape[0]:
        msg = (
            "Wavelength and effective area length mismatch "
            f"({wavelength.shape[0]} vs {eff_area.shape[0]}) in file: {path}"
        )
        logging.error(msg)
        raise ValueError(msg)

    logging.info("Effective area loaded (%s): Rows=%d, pixel_scale=%s", effective_area_filename, wavelength.shape[0], pixel_scale)
    logging.info("Effective area summary: file=%s rows(WL)=%d rows(EA)=%d pixel_scale=%s", effective_area_filename, wavelength.shape[0], eff_area.shape[0], pixel_scale)

    return wavelength, eff_area, pixel_scale

def load_background_file(background_filename: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load background spectrum table.

    Resolves file path as:
        repo_root / "data" / background_filename

    Expected format:
        whitespace-delimited numeric table
        column 0 -> wavelength
        column 1 -> background flux

    Hard fails if:
        file missing
        table structure invalid
        no numeric rows
        wavelength/flux length mismatch
    """
    repo_root = get_repo_root()

    if not background_filename or background_filename.strip() == "":
        logging.info("No background file configured.")
        return None, None

    path = resolve_path_under(repo_root, "data", background_filename)

    logging.info("Loading background file: %s", path)

    if not path.exists():
        msg = f"Background file not found: {path}"
        logging.error(msg)
        raise ValueError(msg)

    try:
        data = np.loadtxt(path)
    except Exception as exc:
        msg = f"Failed to parse numeric data from background file: {path}"
        logging.error(msg)
        raise ValueError(msg) from exc

    if data.ndim != 2 or data.shape[1] < 2:
        msg = f"Invalid background table structure in file: {path}"
        logging.error(msg)
        raise ValueError(msg)

    if data.shape[0] == 0:
        msg = f"No numeric data rows found in background file: {path}"
        logging.error(msg)
        raise ValueError(msg)

    wavelength = data[:, 0].astype(float, copy=False)
    flux = data[:, 1].astype(float, copy=False)

    if wavelength.shape[0] != flux.shape[0]:
        msg = f"Wavelength and flux length mismatch in background file: {path}"
        logging.error(msg)
        raise ValueError(msg)

    logging.info("Background loaded (%s): rows=%d", background_filename, wavelength.shape[0])

    return wavelength, flux

def _parse_pixel_scale(lines: list[str], path: Path) -> float:
    for line in lines:
        s = line.strip()
        if s.startswith("# Pixel scale:"):
            try:
                value_str = s.split(":", 1)[1].strip()
                pixel_scale = float(value_str)
                logging.info("Parsed pixel scale from effective area header: %s (file: %s)", pixel_scale, path)
                return pixel_scale
            except Exception as exc:
                msg = f"Invalid pixel scale value in effective area file header: {path}"
                logging.error(msg)
                raise ValueError(msg) from exc

    msg = f"Missing required header line '# Pixel scale: <value>' in effective area file: {path}"
    logging.error(msg)
    raise ValueError(msg)

def find_first_numeric_row_index(lines: list[str], path: Path) -> int:
    """
    Returns the line index to use as skiprows for np.loadtxt so that the first
    unskipped line is numeric data.

    We skip:
      empty lines
      comment lines (#...)
      the column header line ("Wavelength ...")
    """
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue

        # Try to parse first token as float -> numeric data starts here
        first = s.split()[0]
        try:
            float(first)
            return idx
        except Exception:
            # Non-numeric line (e.g. column header). Keep scanning.
            continue

    msg = f"Could not find any numeric data rows in effective area file: {path}"
    logging.error(msg)
    raise ValueError(msg)

def parse_spread_header_wavelengths(lines: list[str], path: Path, channel_name: str) -> np.ndarray:
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.lower().startswith("pixels"):
            parts = s.split()
            if len(parts) < 2:
                logging.error("Channel %s: spread header has no wavelength columns in %s", channel_name, path)
                raise ValueError(f"Spread header has no wavelength columns in file: {path}")
            try:
                wavelength_header = np.array([float(x) for x in parts[1:]], dtype=float)
                logging.info("Channel %s: spread header wavelengths count=%d values=%s", channel_name, wavelength_header.shape[0], wavelength_header)
                return wavelength_header
            except Exception as exc:
                logging.error("Channel %s: failed to parse spread header wavelength columns in %s", channel_name, path)
                raise ValueError(f"Failed to parse spread header wavelength columns in file: {path}") from exc

    logging.error("Channel %s: no 'pixels ...' header line found in %s", channel_name, path)
    raise ValueError(f"No 'pixels ...' header line found in spread profile file: {path}")

def load_zod_dist_file(filename: str) -> np.ndarray | None:
    repo_root = get_repo_root()
    if not filename or filename.strip() == "":
        logging.info("No zodiacal distribution file configured.")
        return None

    path = resolve_path_under(repo_root, "data", filename)
    if not path.exists():
        msg = f"Zodiacal distribution file not found: {path}"
        logging.error(msg)
        raise ValueError(msg)

    try:
        data = np.loadtxt(path).T
    except Exception as exc:
        msg = f"Failed to parse zodiacal distribution file: {path}"
        logging.error(msg)
        raise ValueError(msg) from exc
    if data.ndim != 2:
        msg = f"Invalid zodiacal distribution table: {path}"
        logging.error(msg)
        raise ValueError(msg)

    logging.info("Zodiacal distribution file loaded: filename=%s path=%s shape=%s", filename, path, data.shape)
    return data

def load_zod_spectrum_file(filename: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    repo_root = get_repo_root()
    if not filename or filename.strip() == "":
        logging.info("No zodiacal spectrum file configured.")
        return None, None

    path = resolve_path_under(repo_root, "data", filename)
    if not path.exists():
        msg = f"Zodiacal spectrum file not found: {path}"
        logging.error(msg)
        raise ValueError(msg)

    try:
        data = np.loadtxt(path)
    except Exception as exc:
        msg = f"Failed to parse zodiacal spectrum file: {path}"
        logging.error(msg)
        raise ValueError(msg) from exc
    if data.ndim != 2 or data.shape[1] < 2:
        msg = f"Invalid zodiacal spectrum table: {path}"
        logging.error(msg)
        raise ValueError(msg)

    wavelength = data[:, 0].astype(float, copy=False)
    spectrum = data[:, 1].astype(float, copy=False)
    logging.info("Zodiacal spectrum file loaded: filename=%s path=%s rows=%d", filename, path, wavelength.shape[0])
    return wavelength, spectrum
