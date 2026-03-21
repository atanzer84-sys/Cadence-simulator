import numpy as np
from pathlib import Path
from utils.constants import (debug_wavelength_range_nuv, debug_wavelength_range_vis, debug_wavelength_range_ir, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR)
from configs.global_config import get_global_config

CHANNEL_NUV = "NUV"
CHANNEL_VIS = "VIS"
CHANNEL_NIR = "NIR"


def dump_3d_array(array, output_dir, star_name: str, tag: str, perChannel: bool = False, full: bool = False, zoom: bool = False, fmt="%.18e"):
    # print(f"[DEBUG] dump_spectrum_snapshots: star='{star_name}', tag='{tag}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wl = array[:, 0]

    def _dump(filename, wmin, wmax):
        out = array[(wl >= wmin) & (wl <= wmax)]
        np.savetxt(output_dir / filename, out, fmt=fmt)

    cfg = get_global_config()
    channels = {name for name, enabled in ((CHANNEL_NUV, cfg.run_nuv), (CHANNEL_VIS, cfg.run_vis), (CHANNEL_NIR, cfg.run_nir)) if enabled}

    if perChannel:
        if CHANNEL_NUV in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_NUV}.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1])
        if CHANNEL_VIS in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_VIS}.txt", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1])
        if CHANNEL_NIR in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_NIR}.txt", debug_wavelength_range_ir[0], debug_wavelength_range_ir[1])
    if full:
        _dump(f"{star_name}_{tag}_full.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        if CHANNEL_NUV in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_NUV}_zoom.txt", *DEBUG_WL_A_NUV)
        if CHANNEL_VIS in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_VIS}_zoom.txt", *DEBUG_WL_A_VIS)
        if CHANNEL_NIR in channels:
            _dump(f"{star_name}_{tag}_{CHANNEL_NIR}_zoom.txt", *DEBUG_WL_A_NIR)



def dump_1d_array(wave, array, output_dir, star_name: str, tag: str, perChannel: bool = False, full: bool = False, zoom: bool = False, fmt="%.18e"):
    cfg = get_global_config()
    channels = {name for name, enabled in ((CHANNEL_NUV, cfg.run_nuv), (CHANNEL_VIS, cfg.run_vis), (CHANNEL_NIR, cfg.run_nir)) if enabled}

    if perChannel:
        if CHANNEL_NUV in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_NUV}.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1], fmt)
        if CHANNEL_VIS in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_VIS}.txt", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1], fmt)
        if CHANNEL_NIR in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_NIR}.txt", debug_wavelength_range_ir[0], debug_wavelength_range_ir[1], fmt)

    if full:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_full.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1], fmt)

    if zoom:
        if CHANNEL_NUV in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_NUV}_zoom.txt", *DEBUG_WL_A_NUV, fmt)
        if CHANNEL_VIS in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_VIS}_zoom.txt", *DEBUG_WL_A_VIS, fmt)
        if CHANNEL_NIR in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{CHANNEL_NIR}_zoom.txt", *DEBUG_WL_A_NIR, fmt)


def dump_masked_1d(wave, array, output_dir, filename, wmin, wmax, fmt="%.18e"):

    mask = (wave >= wmin) & (wave <= wmax)
    out = np.column_stack((wave[mask], array[mask]))
    if out.size == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / filename, out, fmt=fmt)


def dump_1d_for_channel(wave, array, output_dir, star_name: str, tag: str, channel_name: str, full: bool = False, zoom: bool = False, fmt="%.18e"):
    if wave.shape != array.shape:
        raise ValueError("wave / values shape mismatch")

    if channel_name == CHANNEL_NUV:
        full_range = debug_wavelength_range_nuv
        zoom_range = DEBUG_WL_A_NUV
    elif channel_name == CHANNEL_VIS:
        full_range = debug_wavelength_range_vis
        zoom_range = DEBUG_WL_A_VIS
    elif channel_name == CHANNEL_NIR:
        full_range = debug_wavelength_range_ir
        zoom_range = DEBUG_WL_A_NIR
    else:
        raise ValueError(f"channel_name must be {CHANNEL_NUV!r}, {CHANNEL_VIS!r}, or {CHANNEL_NIR!r}")

    if full:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{channel_name}.txt", full_range[0], full_range[1], fmt)

    if zoom:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{channel_name}_zoom.txt", zoom_range[0], zoom_range[1], fmt)


def dump_npz_snapshot(output_dir, filename, **arrays):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / filename
    converted = {}
    for key, value in arrays.items():
        if np.isscalar(value):
            converted[key] = np.asarray(value, dtype=np.float64)
        else:
            converted[key] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(out_file, **converted)
    return out_file


def dump_cropped_image_npz(output_dir, filename, image, y_center, half_height=500):
    y_center_i = int(round(y_center))
    y_min = max(0, y_center_i - int(half_height))
    y_max = min(image.shape[0], y_center_i + int(half_height) + 1)
    image_cropped = image[y_min:y_max, :].astype(np.float32, copy=False)
    return dump_npz_snapshot(output_dir, filename, image=image_cropped, y0=np.int32(y_center_i), y_min=np.int32(y_min), y_max=np.int32(y_max))


def dump_effective_area_txt(output_dir, channel_name, effective_area_wavelength, effective_area, pixel_scale):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((effective_area_wavelength, effective_area))
    out_file = output_dir / f"{channel_name}_effective_area.txt"
    np.savetxt(out_file, table, fmt="%.18e", header="pixel_scale=" + f"{float(pixel_scale):.18e}" + "\n" + "effective_area_wavelength effective_area")
    return out_file