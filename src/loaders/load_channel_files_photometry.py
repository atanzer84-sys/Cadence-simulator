from loaders.run_waltzer_context import get_repo_root, RunContext
import logging
import numpy as np

def load_psf_image_file(filename: str, channel_name: str, ctx: RunContext, min_cols: int = 20, stability_rows: int = 3, ) -> tuple[np.ndarray, int, int]:

    repo_root = get_repo_root()

    if not filename or filename.strip() == "":
        raise ValueError("PSF image file not configured.")

    path = resolve_path_under(repo_root, "data", filename)
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