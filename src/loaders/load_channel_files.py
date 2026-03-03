from loaders.run_waltzer_context import get_repo_root
from pathlib import Path
import logging
import numpy as np


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

    logging.info(
        "Effective area summary: file=%s rows(WL)=%d rows(EA)=%d pixel_scale=%s",
        effective_area_filename,
        wavelength.shape[0],
        eff_area.shape[0],
        pixel_scale,
    )

    return wavelength, eff_area, pixel_scale


def load_spread_profile_file_spectroscopy(spread_filename: str, channel_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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
    if weights_matrix.shape[0] != positions.shape[0]:
        logging.error("Spread File Error: channel=%s spread_y_positions and spread_y_weights row mismatch", channel_name)
        raise ValueError("Spread profile row mismatch")

    logging.info("Channel %s: spread loaded rows=%d weight_cols=%d", channel_name, positions.shape[0], weights_matrix.shape[1])
    logging.info("Channel %s: spread first vertical dispersion dy values=%s", channel_name, positions[:10])
    logging.info("Channel %s: spread first row weights=%s", channel_name, weights_matrix[0, :])
    logging.info("Channel %s: spread center row dy=%g weights=%s", channel_name, positions[len(positions)//2], weights_matrix[len(positions)//2, :])

    return positions, weights_matrix, wavelength_header

def load_background_file(background_filename: str) -> tuple[np.ndarray, np.ndarray]:
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

    path = (repo_root / "data" / background_filename).resolve()

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

    logging.info(
        "Background loaded (%s): rows=%d",
        background_filename,
        wavelength.shape[0],
    )

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


def load_zod_dist_file(filename: str) -> np.ndarray:
    repo_root = get_repo_root()
    if not filename or filename.strip() == "":
        logging.info("No zodiacal distribution file configured.")
        return None

    path = (repo_root / "data" / filename).resolve()
    logging.info("Loading zodiacal distribution file: %s", path)
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

    logging.info("Zodiacal distribution loaded (%s): shape=%s", filename, data.shape)
    return data


def load_zod_spectrum_file(filename: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    repo_root = get_repo_root()
    if not filename or filename.strip() == "":
        logging.info("No zodiacal spectrum file configured.")
        return None, None

    path = (repo_root / "data" / filename).resolve()
    logging.info("Loading zodiacal spectrum file: %s", path)
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
    logging.info("Zodiacal spectrum loaded (%s): rows=%d", filename, wavelength.shape[0])
    return wavelength, spectrum



def load_psf_profile_file(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load radial PSF profile file.

    Expected format:
        column 0 -> radial distance in pixels
        column 1 -> normalized PSF intensity

    Returns:
        psf_radial_distance: np.ndarray
        psf_radial_flux: np.ndarray
    """
    repo_root = get_repo_root()

    if not filename or filename.strip() == "":
        raise ValueError("PSF profile file not configured.")

    path = (repo_root / "data" / filename).resolve()
    logging.info("Loading PSF profile file: %s", path)

    if not path.exists():
        raise ValueError(f"PSF profile file not found: {path}")

    try:
        data = np.loadtxt(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse PSF profile file: {path}") from exc

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid PSF profile table structure: {path}")

    rad = data[:, 0].astype(float, copy=False)
    flux = data[:, 1].astype(float, copy=False)

    mask = rad >= 0.0
    psf_radial_distance = rad[mask]
    psf_radial_flux = flux[mask]

    if psf_radial_distance.shape[0] == 0:
        raise ValueError(f"No positive radial values found in PSF file: {path}")

    logging.info("PSF profile loaded (%s): rows=%d", filename, psf_radial_distance.shape[0])

    return psf_radial_distance, psf_radial_flux

def load_spread_profile_file_photometry(
    spread_filename: str,
    channel_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Photometry spread file loader.

    Expected header:
        pixels 0_Y 0_X 115_Y 115_X 230_Y 230_X 324_Y 324_X

    Returns:
        spread_positions: (Ny,) dy offsets (from 'pixels')
        spread_y_weights: (Ny, Nanchors) from '*_Y' columns
        spread_x_weights: (Ny, Nanchors) from '*_X' columns
        spread_anchors:   (Nanchors,) anchor values parsed from headers (0,115,230,324)
    """
    repo_root = get_repo_root()

    if not spread_filename or spread_filename.strip() == "":
        logging.info("Channel %s: no spread profile configured.", channel_name)
        return None, None, None, None

    path = (repo_root / "data" / spread_filename).resolve()
    logging.info("Channel %s: loading spread profile file: %s", channel_name, path)

    if not path.exists():
        logging.error("Channel %s: spread profile file not found: %s", channel_name, path)
        raise ValueError(f"Spread profile file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    skiprows = _find_first_numeric_row_index(lines, path)

    header_line = None
    for line in lines[:skiprows]:
        s = line.strip()
        if s.startswith("pixels"):
            header_line = s
            break

    if header_line is None:
        raise ValueError(f"Spread header line starting with 'pixels' not found in file: {path}")

    headers = header_line.split()

    y_col_indices: list[int] = []
    x_col_indices: list[int] = []
    y_anchors: list[float] = []
    x_anchors: list[float] = []

    for idx, h in enumerate(headers):
        if idx == 0:
            continue

        if h.endswith("_Y") or h.endswith("_X"):
            try:
                a = float(h.split("_")[0])
            except Exception as exc:
                raise ValueError(f"Failed to parse anchor from spread header column: {h} in {path}") from exc

            if h.endswith("_Y"):
                y_col_indices.append(idx)
                y_anchors.append(a)
            else:
                x_col_indices.append(idx)
                x_anchors.append(a)

    if len(y_col_indices) < 1:
        raise ValueError(f"No *_Y columns found in spread header in file: {path}")
    if len(x_col_indices) < 1:
        raise ValueError(f"No *_X columns found in spread header in file: {path}")

    spread_anchors_y = np.array(y_anchors, dtype=float)
    spread_anchors_x = np.array(x_anchors, dtype=float)

    if spread_anchors_y.shape != spread_anchors_x.shape or np.any(spread_anchors_y != spread_anchors_x):
        raise ValueError(
            f"Spread file anchor mismatch between *_Y and *_X columns in {path}. "
            f"Y={spread_anchors_y}, X={spread_anchors_x}"
        )

    spread_anchors = spread_anchors_y

    try:
        data = np.loadtxt(path, skiprows=skiprows)
    except Exception as exc:
        raise ValueError(f"Failed to parse numeric spread table from file: {path}") from exc

    if data.ndim != 2:
        raise ValueError(f"Invalid spread table structure in file: {path}")

    if data.shape[1] < max(max(y_col_indices), max(x_col_indices)) + 1:
        raise ValueError(f"Spread table missing required columns in file: {path}")

    spread_positions = data[:, 0].astype(float, copy=False)
    spread_y_weights = data[:, y_col_indices].astype(float, copy=False)
    spread_x_weights = data[:, x_col_indices].astype(float, copy=False)

    if spread_y_weights.shape[0] != spread_positions.shape[0] or spread_x_weights.shape[0] != spread_positions.shape[0]:
        raise ValueError("Spread profile row mismatch")

    logging.info("Channel %s: spread loaded rows=%d anchors=%d anchors=%s", channel_name, spread_positions.shape[0], spread_anchors.shape[0], spread_anchors)
    logging.info("Channel %s: spread column sums Y=%s", channel_name, spread_y_weights.sum(axis=0))
    logging.info("Channel %s: spread column sums X=%s", channel_name, spread_x_weights.sum(axis=0))

    return spread_positions, spread_y_weights, spread_x_weights, spread_anchors