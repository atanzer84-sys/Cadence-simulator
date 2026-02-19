import logging
from pathlib import Path
import numpy as np
from loaders.run_setup import get_repo_root


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
    path = (repo_root / "data" / effective_area_filename).resolve()

    logging.info("Loading effective area file: %s", path)

    if not path.exists():
        msg = f"Effective area file not found: {path}"
        logging.error(msg)
        raise ValueError(msg)

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    pixel_scale = _parse_pixel_scale(text, path)
    skiprows = _find_first_numeric_row_index(text, path)

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

    logging.info(
        "Effective area loaded (%s): Rows=%d, pixel_scale=%s",
        effective_area_filename,
        wavelength.shape[0],
        pixel_scale,
    )

    return wavelength, eff_area, pixel_scale

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

def _find_first_numeric_row_index(lines: list[str], path: Path) -> int:
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

def load_spread_profile_file(spread_filename: str, channel_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    repo_root = get_repo_root()
    
    if not spread_filename or spread_filename.strip() == "": 
        logging.info("Channel %s: no spread profile configured.", channel_name)
        return None, None, None

    path = (repo_root / "data" / spread_filename).resolve()
    logging.info("Channel %s: loading spread profile file: %s", channel_name, path)

    if not path.exists():
        logging.error("Channel %s: spread profile file not found: %s", channel_name, path)
        raise ValueError(f"Spread profile file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    wavelength_header = _parse_spread_header_wavelengths(lines, path, channel_name)
    skiprows = _find_first_numeric_row_index(lines, path)

    try:
        data = np.loadtxt(path, skiprows=skiprows)
    except Exception as exc:
        logging.error("Channel %s: failed to parse numeric spread table from %s", channel_name, path)
        raise ValueError(f"Failed to parse numeric spread table from file: {path}") from exc

    if data.ndim != 2 or data.shape[1] < 2:
        logging.error("Channel %s: invalid spread table structure in %s (need dy + >=1 weight col)", channel_name, path)
        raise ValueError(f"Invalid spread table structure in file: {path} (need dy + >=1 weight col)")

    positions = data[:, 0].astype(float, copy=False)
    weights_matrix = data[:, 1:].astype(float, copy=False)

    if weights_matrix.shape[1] != wavelength_header.shape[0]:
        logging.error("Channel %s: spread header wavelength count != weight columns in %s", channel_name, path)
        raise ValueError(f"Spread header wavelength count does not match weight columns in file: {path}")

    logging.info("Channel %s: spread loaded rows=%d weight_cols=%d", channel_name, positions.shape[0], weights_matrix.shape[1])
    logging.info("Channel %s: spread first vertical dispersion dy values=%s", channel_name, positions[:10])
    logging.info("Channel %s: spread first row weights=%s", channel_name, weights_matrix[0, :])
    logging.info("Channel %s: spread center row dy=%g weights=%s", channel_name, positions[len(positions)//2], weights_matrix[len(positions)//2, :])

    return positions, weights_matrix, wavelength_header


def _parse_spread_header_wavelengths(lines: list[str], path: Path, channel_name: str) -> np.ndarray:
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