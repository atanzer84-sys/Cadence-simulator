import numpy as np
from pathlib import Path
from utils.constants import (debug_wavelength_range_nuv, debug_wavelength_range_vis, debug_wavelength_range_ir, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR)
from configs.global_config import get_global_config



def dump_3d_array(array, output_dir, star_name: str, tag: str, perChannel: bool = False, full: bool = False, zoom: bool = False, fmt="%.18e"):
    # print(f"[DEBUG] dump_spectrum_snapshots: star='{star_name}', tag='{tag}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wl = array[:, 0]

    def _dump(filename, wmin, wmax):
        out = array[(wl >= wmin) & (wl <= wmax)]
        np.savetxt(output_dir / filename, out, fmt=fmt)

    cfg = get_global_config()
    channels = {name for name, enabled in (("NUV", cfg.run_nuv), ("VIS", cfg.run_vis), ("NIR", cfg.run_nir)) if enabled}

    if perChannel:
        if "NUV" in channels:
            _dump(f"{star_name}_{tag}_NUV.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1])
        if "VIS" in channels:
            _dump(f"{star_name}_{tag}_VIS.txt", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1])
        if "NIR" in channels:
            _dump(f"{star_name}_{tag}_NIR.txt", debug_wavelength_range_ir[0], debug_wavelength_range_ir[1])
    if full:
        _dump(f"{star_name}_{tag}_full.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        if "NUV" in channels:
            _dump(f"{star_name}_{tag}_NUV_zoom.txt", *DEBUG_WL_A_NUV)
        if "VIS" in channels:
            _dump(f"{star_name}_{tag}_VIS_zoom.txt", *DEBUG_WL_A_VIS)
        if "NIR" in channels:
            _dump(f"{star_name}_{tag}_IR_zoom.txt",  *DEBUG_WL_A_NIR)



def dump_1d_array(wave, array, output_dir, star_name: str, tag: str, perChannel: bool = False, full: bool = False, zoom: bool = False, fmt="%.18e"):
    cfg = get_global_config()
    channels = {name for name, enabled in (("NUV", cfg.run_nuv), ("VIS", cfg.run_vis), ("NIR", cfg.run_nir)) if enabled}

    if perChannel:
        if "NUV" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_NUV.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1], fmt)
        if "VIS" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_VIS.txt", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1], fmt)
        if "NIR" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_IR.txt",  debug_wavelength_range_ir[0],  debug_wavelength_range_ir[1],  fmt)

    if full:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_full.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1], fmt)

    if zoom:
        if "NUV" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_NUV_zoom.txt", *DEBUG_WL_A_NUV, fmt)
        if "VIS" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_VIS_zoom.txt", *DEBUG_WL_A_VIS, fmt)
        if "NIR" in channels:
            dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_IR_zoom.txt",  *DEBUG_WL_A_NIR,  fmt)


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

    if channel_name == "NUV":
        full_range = debug_wavelength_range_nuv
        zoom_range = DEBUG_WL_A_NUV
    elif channel_name == "VIS":
        full_range = debug_wavelength_range_vis
        zoom_range = DEBUG_WL_A_VIS
    elif channel_name == "NIR":
        full_range = debug_wavelength_range_ir
        zoom_range = DEBUG_WL_A_NIR
    else:
        raise ValueError("channel_name must be 'NUV', 'VIS', or 'IR'")

    if full:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{channel_name}.txt", full_range[0], full_range[1], fmt)

    if zoom:
        dump_masked_1d(wave, array, output_dir, f"{star_name}_{tag}_{channel_name}_zoom.txt", zoom_range[0], zoom_range[1], fmt)