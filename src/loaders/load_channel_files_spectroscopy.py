import logging
import numpy as np
from loaders.run_waltzer_context import get_repo_root
from utils.helpers import resolve_path_under
from loaders.load_channel_files_common import (
    parse_spread_header_wavelengths,
    find_first_numeric_row_index,
    read_text_lines_with_fallback,
)

def load_spread_profile_file_spectroscopy(spread_filename: str, channel_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    repo_root = get_repo_root()
    
    if not spread_filename or spread_filename.strip() == "": 
        logging.info("Channel %s: no spread profile configured.", channel_name)
        return None, None, None

    path = resolve_path_under(repo_root, "data", spread_filename)
    if not path.exists():
        logging.error("Channel %s: spread profile file not found: %s", channel_name, path)
        raise ValueError(f"Spread profile file not found: {path}")

    lines = read_text_lines_with_fallback(
        path,
        encodings=("utf-8", "utf-8-sig", "utf-16"),
        context=f"Channel {channel_name} spread profile",
    )
    wavelength_header = parse_spread_header_wavelengths(lines, path, channel_name)
    skiprows = find_first_numeric_row_index(lines, path)

    try:
        data = np.loadtxt(lines[skiprows:])
    except Exception as exc:
        logging.error("Channel %s: failed to parse numeric spread table from %s", channel_name, path)
        raise ValueError(f"Failed to parse numeric spread table from file: {path}") from exc

    if data.ndim != 2 or data.shape[1] < 2:
        logging.error("Channel %s: invalid spread table structure in %s (need dy + >=1 weight col)", channel_name, path)
        raise ValueError(f"Invalid spread table structure in file: {path} (need dy + >=1 weight col)")

    positions = data[:, 0].astype(np.float32, copy=False)
    weights_matrix = data[:, 1:].astype(np.float32, copy=False)

    if weights_matrix.shape[1] != wavelength_header.shape[0]:
        logging.error("Channel %s: spread header wavelength count != weight columns in %s", channel_name, path)
        raise ValueError(f"Spread header wavelength count does not match weight columns in file: {path}")
    if weights_matrix.shape[0] != positions.shape[0]:
        logging.error("Spread File Error: channel=%s spread_y_positions and spread_y_weights row mismatch", channel_name)
        raise ValueError("Spread profile row mismatch")

    logging.info("Spread profile loaded: channel=%s rows=%d wavelength_bins=%d", channel_name, positions.shape[0], wavelength_header.shape[0])
    return positions, weights_matrix, wavelength_header

def load_polarization_file(filename: str, channel_name: str):

    if not filename or filename.strip() == "":
        logging.info("Channel %s: no polarization delta file configured.", channel_name)
        return None, None

    repo_root = get_repo_root()
    path = resolve_path_under(repo_root, "data", filename)

    if not path.exists():
        logging.error("Channel %s: polarization file not found: %s", channel_name, path)
        raise ValueError(f"Polarization file not found: {path}")

    try:
        data = np.loadtxt(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse polarization file: {path}") from exc

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid polarization file format: {path}")

    wavelength = data[:, 0].astype(np.float32)
    delta = data[:, 1].astype(np.float32)

    if not np.all(np.isfinite(delta)):
        raise ValueError(f"{channel_name}: polarization delta contains non-finite values")

    if np.any(delta < 0.0) or np.any(delta > 1.0):
        raise ValueError(f"{channel_name}: polarization delta must be in range [0, 1]")
        
    logging.info("Polarization file loaded: channel=%s rows=%d", channel_name, len(wavelength))

    return wavelength, delta

def validate_polarization_config(channel_name: str, observation_mode: str, pol_wl, pol_delta, beam_separation_pix: int, y_pixels: int):

    if observation_mode == "spectropolarimetry":
        if pol_wl is None or pol_delta is None:
            raise ValueError(f"{channel_name}: spectropolarimetry requires polarization_file")
    
        if beam_separation_pix < 1:
            raise ValueError(f"{channel_name}: beam_separation_pix must be >= 1")
    
        if beam_separation_pix >= y_pixels:
            raise ValueError(f"{channel_name}: beam_separation_pix={beam_separation_pix} too large for detector height={y_pixels}")
    
    elif observation_mode != "spectroscopy":
        raise ValueError(f"{channel_name}: invalid observation_mode={observation_mode}")