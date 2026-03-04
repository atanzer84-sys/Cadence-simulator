from loaders.run_waltzer_context import get_repo_root, RunContext
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

    logging.info("Effective area loaded (%s): Rows=%d, pixel_scale=%s", effective_area_filename, wavelength.shape[0], pixel_scale)
    logging.info("Effective area summary: file=%s rows(WL)=%d rows(EA)=%d pixel_scale=%s", effective_area_filename, wavelength.shape[0], eff_area.shape[0], pixel_scale)

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

def load_psf_image_file(filename: str, channel_name: str, ctx: RunContext, min_cols: int = 20, stability_rows: int = 3, ) -> tuple[np.ndarray, int, int]:

    repo_root = get_repo_root()

    if not filename or filename.strip() == "":
        raise ValueError("PSF image file not configured.")

    path = (repo_root / "data" / filename).resolve()
    logging.info("Channel %s: loading PSF image file: %s", channel_name, path)

    if not path.exists():
        raise ValueError(f"PSF image file not found: {path}")

    text_lines = path.read_text(encoding="utf-16", errors="replace").splitlines()

    def _try_parse_row(raw: str) -> list[float] | None:
        parts = raw.strip().split()
        if len(parts) < min_cols:
            return None
        try:
            return [float(x) for x in parts]
        except Exception:
            return None

    # 1) find first stable numeric grid start
    grid_start_line = None
    grid_cols = None

    for i in range(len(text_lines)):
        row0 = _try_parse_row(text_lines[i])
        if row0 is None:
            continue

        n = len(row0)

        stable = True
        for k in range(1, stability_rows + 1):
            if i + k >= len(text_lines):
                stable = False
                break
            rowk = _try_parse_row(text_lines[i + k])
            if rowk is None or len(rowk) != n:
                stable = False
                break

        if stable:
            grid_start_line = i
            grid_cols = n
            break

    if grid_start_line is None or grid_cols is None:
        raise ValueError(f"Channel {channel_name}: could not locate PSF numeric grid in file: {path}")

    # 2) read numeric grid
    rows: list[list[float]] = []
    for j in range(grid_start_line, len(text_lines)):
        r = _try_parse_row(text_lines[j])
        if r is None or len(r) != grid_cols:
            break
        rows.append(r)

    psf = np.asarray(rows, dtype=float)

    # 3) read center from header if present, else fallback to peak
    psf_center_x = None
    psf_center_y = None

    header_lines = text_lines[:grid_start_line]
    for line in header_lines:
        if "center" not in line.casefold():
            continue

        ints: list[int] = []
        for tok in line.replace(",", " ").split():
            try:
                ints.append(int(tok))
            except Exception:
                pass

        if len(ints) >= 2:
            # assume file center is 1-based -> convert to 0-based for numpy
            psf_center_x = ints[0] - 1
            psf_center_y = ints[1] - 1
            break

    if psf_center_x is None or psf_center_y is None:
        peak_y, peak_x = np.unravel_index(int(np.nanargmax(psf)), psf.shape)
        psf_center_y = int(peak_y)
        psf_center_x = int(peak_x)

    # 4) normalize so sum == 1
    total_sum = float(np.nansum(psf))
    if total_sum <= 0.0:
        raise ValueError(f"Channel {channel_name}: PSF sum invalid: {total_sum}")

    psf /= total_sum

    logging.info("Channel %s: PSF loaded shape=(%d,%d) raw_sum=%g norm_sum=%g center=(y=%d,x=%d) grid_start_line=%d", channel_name, psf.shape[0], psf.shape[1], total_sum, float(np.nansum(psf)), int(psf_center_y), int(psf_center_x), int(grid_start_line))

    return psf, int(psf_center_y), int(psf_center_x)