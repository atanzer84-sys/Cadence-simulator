from loaders.run_setup import get_repo_root
from pathlib import Path
import logging
import numpy as np
from configs.channel import SpectroscopyChannel, PhotometryChannel
from configs.user_config import UserConfig

def load_channels_config(user_cfg: UserConfig):
    repo_root = get_repo_root()
    # load channel config, not cached, pass it through
    nuv_channel = load_channel_config(repo_root / "configs" / "waltzer_nuv.cfg", user_cfg.exposure_NUV_s)
    vis_channel = load_channel_config(repo_root / "configs" / "waltzer_vis.cfg", user_cfg.exposure_VIS_s)
    ir_channel  = load_channel_config(repo_root / "configs" / "waltzer_ir.cfg", user_cfg.exposure_IR_s)

    return nuv_channel, vis_channel, ir_channel



def load_channel_config(path: Path, exposure_s:float):
    logging.info("Reading channel config from %s", path)

    raw = _parse_simple_kv(path)

    channel_name=str(raw["channel_name"]).strip()
    x_pixels=_as_int(raw["x_pixels"], key="x_pixels")
    y_pixels=_as_int(raw["y_pixels"], key="y_pixels")
    resolution_factor=_as_float(raw["resolution_factor"], key="resolution_factor")
    dark_noise=_as_float(raw["dark_noise"], key="dark_noise")
    dark_current_sigma=_as_float(raw["dark_current_sigma"], key="dark_current_sigma")
    read_noise=_as_float(raw["read_noise"], key="read_noise")
    effective_area_file=raw["effective_area_file"]
    bias_offset=_as_float(raw.get("bias_offset", 0.0), key="bias_offset")
    ccd_gain=_as_float(raw.get("ccd_gain", 1.0), key="ccd_gain")
    mode=_as_int(raw["mode"], key="mode")
    spread_profile_file=str(raw.get("spread_profile_file", "")).strip()
    spread_half_height_pix=_as_optional_int(raw.get("spread_half_height_pix", None)) or 0
    source_file=str(path)

    if channel_name == "IR":
        return PhotometryChannel(
            channel_name=channel_name,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
            resolution_factor=resolution_factor,
            dark_noise=dark_noise,
            dark_current_sigma=dark_current_sigma,
            read_noise=read_noise,
            bias_offset=bias_offset,
            ccd_gain=ccd_gain,
            exposure_s=exposure_s,
            source_file=source_file
        )

    wavelength, effective_area, pixel_scale = load_effective_area_file(effective_area_file)
    if len(wavelength) != x_pixels:
        logging.error("%s: effective_area_file=%s len(wavelength)=%d != x_pixels=%d source_file=%s", channel_name, effective_area_file, len (wavelength), x_pixels, source_file, )
        raise ValueError(
            f"{channel_name}: effective_area_file={effective_area_file} "
            f"len(wavelength)={len(wavelength)} != x_pixels={x_pixels} "
            f"source_file={source_file}"
        )
        
    spread_pos, spread_w, spread_wl_header = load_spread_profile_file(spread_profile_file, channel_name)

    return SpectroscopyChannel(
        channel_name=channel_name,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        resolution_factor=resolution_factor,
        dark_noise=dark_noise,
        dark_current_sigma=dark_current_sigma,
        read_noise=read_noise,
        bias_offset=bias_offset,
        effective_area_file=effective_area_file,
        ccd_gain=ccd_gain,
        exposure_s=exposure_s,
        mode=mode,
        spread_profile_file=spread_profile_file,
        spread_half_height_pix=spread_half_height_pix,
        wavelength=wavelength,
        effective_area=effective_area,
        pixel_scale=pixel_scale,
        spread_y_positions=spread_pos,
        spread_y_weights=spread_w,
        spread_y_wavelengths=spread_wl_header,
        source_file=source_file
    )  


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
    if weights_matrix.shape[0] != positions.shape[0]:
        logging.error("Spread File Error: channel=%s spread_y_positions and spread_y_weights row mismatch", channel_name)
        raise ValueError("Spread profile row mismatch")

    logging.info("Channel %s: spread loaded rows=%d weight_cols=%d", channel_name, positions.shape[0], weights_matrix.shape[1])
    logging.info("Channel %s: spread first vertical dispersion dy values=%s", channel_name, positions[:10])
    logging.info("Channel %s: spread first row weights=%s", channel_name, weights_matrix[0, :])
    logging.info("Channel %s: spread center row dy=%g weights=%s", channel_name, positions[len(positions)//2], weights_matrix[len(positions)//2, :])

    return positions, weights_matrix, wavelength_header


def _as_optional_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.casefold() == "none":
        return None
    try:
        return int(s)
    except Exception as exc:
        logging.error("Invalid int value: %r", value)
        raise ValueError(f"Invalid int value: {value!r}") from exc

def _as_int(value, *, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        logging.error("Invalid int for key '%s': %r", key, value)
        raise ValueError(f"Invalid int for key '{key}': {value!r}") from exc


def _as_float(value, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc


def _parse_simple_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.error("Channel config file not found at %s", path)
        raise FileNotFoundError(f"Config not found: {path}")

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if "=" not in s:
            continue
        k, v = (p.strip() for p in s.split("=", 1))
        data[k] = v
    return data



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