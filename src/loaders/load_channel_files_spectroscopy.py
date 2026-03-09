from loaders.run_waltzer_context import get_repo_root
import logging
import numpy as np
from utils.helpers import resolve_path_under
from loaders.load_channel_files_common import parse_spread_header_wavelengths, find_first_numeric_row_index

def load_spread_profile_file_spectroscopy(spread_filename: str, channel_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    repo_root = get_repo_root()
    
    if not spread_filename or spread_filename.strip() == "": 
        logging.info("Channel %s: no spread profile configured.", channel_name)
        return None, None, None

    path = resolve_path_under(repo_root, "data", spread_filename)
    if not path.exists():
        logging.error("Channel %s: spread profile file not found: %s", channel_name, path)
        raise ValueError(f"Spread profile file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    wavelength_header = parse_spread_header_wavelengths(lines, path, channel_name)
    skiprows = find_first_numeric_row_index(lines, path)

    try:
        data = np.loadtxt(path, skiprows=skiprows)
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