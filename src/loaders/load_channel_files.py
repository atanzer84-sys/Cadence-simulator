from loaders.run_waltzer_context import get_repo_root, RunContext
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

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

def load_spread_profile_file_photometry(spread_filename: str, channel_name: str, ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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

# def load_psf_image_file(filename: str, channel_name: str, min_cols: int = 20, stability_rows: int = 3) -> tuple[np.ndarray, int, int]:
#     repo_root = get_repo_root()

#     if not filename or filename.strip() == "":
#         raise ValueError("PSF image file not configured.")

#     path = (repo_root / "data" / filename).resolve()
#     logging.info("Channel %s: starting PSF image load from %s", channel_name, path)

#     if not path.exists():
#         msg = f"PSF image file not found: {path}"
#         logging.error(msg)
#         raise ValueError(msg)

#     text = path.read_text(encoding="utf-16", errors="replace").splitlines()
#     logging.info("Channel %s: PSF file read complete with %d total lines", channel_name, len(text))

#     def _try_parse_row(raw: str) -> list[float] | None:
#         parts = raw.strip().split()
#         if len(parts) < min_cols:
#             return None
#         try:
#             return [float(x) for x in parts]
#         except Exception:
#             return None

#     logging.info("Channel %s: searching first stable numeric PSF block (min_cols=%d, stability_rows=%d)", channel_name, min_cols, stability_rows)

#     start_idx = None
#     ncols = None

#     for i in range(len(text)):
#         row0 = _try_parse_row(text[i])
#         if row0 is None:
#             continue

#         n = len(row0)

#         ok = True
#         for k in range(1, stability_rows + 1):
#             if i + k >= len(text):
#                 ok = False
#                 break
#             rowk = _try_parse_row(text[i + k])
#             if rowk is None or len(rowk) != n:
#                 ok = False
#                 break

#         if ok:
#             start_idx = i
#             ncols = n
#             break

#     if start_idx is None or ncols is None:
#         msg = f"Channel {channel_name}: could not locate PSF numeric grid in file: {path}"
#         logging.error(msg)
#         logging.error("Channel %s: no stable numeric grid found using constraints min_cols=%d and stability_rows=%d", channel_name, min_cols, stability_rows)
#         raise ValueError(msg)

#     logging.info("Channel %s: detected PSF numeric grid start at line %d with %d columns", channel_name, start_idx, ncols)

#     rows: list[list[float]] = []
#     for j in range(start_idx, len(text)):
#         r = _try_parse_row(text[j])
#         if r is None or len(r) != ncols:
#             logging.info("Channel %s: PSF grid terminated at line %d due to non-numeric row or column-count change", channel_name, j)
#             break
#         rows.append(r)

#     psf = np.asarray(rows, dtype=float)

#     # 2) center from header (if present), else fallback to peak
#     psf_center_y = None
#     psf_center_x = None

#     logging.info("Channel %s: scanning %d header lines for PSF center metadata", channel_name, start_idx)
#     header_lines = text[:start_idx]
#     for line in header_lines:
#         s = line.casefold()
#         if "center" not in s:
#             continue

#         nums: list[int] = []
#         for tok in line.replace(",", " ").split():
#             try:
#                 nums.append(int(tok))
#             except Exception:
#                 pass

#         if len(nums) >= 2:
#             # assume 1-based in file -> convert to 0-based
#             psf_center_x = nums[0] - 1
#             psf_center_y = nums[1] - 1
#             logging.info("Channel %s: found PSF center candidate in header and converted to 0-based coordinates", channel_name)
#             break

#     if psf_center_y is None or psf_center_x is None:
#         iy, ix = np.unravel_index(int(np.nanargmax(psf)), psf.shape)
#         psf_center_y = int(iy)
#         psf_center_x = int(ix)
#         logging.info("Channel %s: PSF center not found in header, using peak at (y=%d, x=%d)", channel_name, psf_center_y, psf_center_x)
#     else:
#         logging.info("Channel %s: PSF center read from header as (y=%d, x=%d) [0-based]", channel_name, psf_center_y, psf_center_x)

#     # 3) normalize so sum = 1
#     total_sum = float(np.nansum(psf))
#     if total_sum <= 0.0:
#         raise ValueError(f"Channel {channel_name}: PSF total intensity sum is invalid: {total_sum}")

#     psf /= total_sum

#     # 4) recenter PSF so paste_stamp_center uses the correct optical center
#     target_x = psf.shape[1] // 2
#     target_y = psf.shape[0] // 2
#     dx = int(target_x - psf_center_x)
#     dy = int(target_y - psf_center_y)

#     psf = np.roll(psf, shift=dy, axis=0)
#     psf = np.roll(psf, shift=dx, axis=1)

#     psf_center_x = int(target_x)
#     psf_center_y = int(target_y)

#     logging.info(
#         "Channel %s: PSF loaded rows=%d cols=%d sum_norm=%g center=(y=%d,x=%d) shift=(dy=%d,dx=%d)",
#         channel_name,
#         psf.shape[0],
#         psf.shape[1],
#         float(np.nansum(psf)),
#         psf_center_y,
#         psf_center_x,
#         dy,
#         dx,
#     )

#     return psf, psf_center_y, psf_center_x

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
    psf_txt_path = ctx.output_dir / f"{channel_name}_psf_matrix.txt"
    psf_png_path = ctx.output_dir / f"{channel_name}_psf_image.png"
    np.savetxt(psf_txt_path, psf, fmt="%.18e")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(psf, origin="lower", aspect="equal", cmap="gray")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.set_title(f"{channel_name} PSF")
    fig.savefig(psf_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Channel %s: PSF debug outputs written txt=%s png=%s", channel_name, psf_txt_path, psf_png_path)

    return psf, int(psf_center_y), int(psf_center_x)