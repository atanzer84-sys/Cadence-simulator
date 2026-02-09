import numpy as np
from pathlib import Path
from utils.constants import DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR, debug_wavelength_range_nuv, debug_wavelength_range_ir

def dump_3d_array(array, output_dir, star_name: str, tag: str, full: bool = True, zoom: bool = True, fmt="%.18e"):
    # print(f"[DEBUG] dump_spectrum_snapshots: star='{star_name}', tag='{tag}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wl = array[:, 0]

    def _dump(filename, wmin, wmax):
        out = array[(wl >= wmin) & (wl <= wmax)]
        np.savetxt(output_dir / filename, out, fmt=fmt)

    if full:
        _dump(f"{star_name}_{tag}_complete.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        _dump(f"{star_name}_{tag}_NUV.txt", *DEBUG_WL_A_NUV)
        _dump(f"{star_name}_{tag}_VIS.txt", *DEBUG_WL_A_VIS)
        _dump(f"{star_name}_{tag}_IR.txt",  *DEBUG_WL_A_IR)

def dump_1d_array(wave, values, output_dir, star_name: str, tag: str, full: bool = True, zoom: bool = True, fmt="%.18e"):
    # print(f"[DEBUG] dump_spectrum_snapshots_1d: star='{star_name}', tag='{tag}'")


    if wave.shape != values.shape:
        raise ValueError("wave / values shape mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(filename, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = np.column_stack((wave[mask], values[mask]))
        np.savetxt(output_dir / filename, out, fmt=fmt)

    if full:
        _dump(f"{star_name}_{tag}_complete.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        _dump(f"{star_name}_{tag}_NUV.txt", *DEBUG_WL_A_NUV)
        _dump(f"{star_name}_{tag}_VIS.txt", *DEBUG_WL_A_VIS)
        _dump(f"{star_name}_{tag}_IR.txt",  *DEBUG_WL_A_IR)


def dump_diff_3d_array(spectrum_before, spectrum_after, output_dir, star_name, tag, full: bool = True, zoom: bool = True, fmt="%.18e"):
    # print(f"[DEBUG] dump_diff_windows_3d: star='{star_name}', tag='{tag}'")
    wave = spectrum_after[:, 0]
    flux_before = spectrum_before[:, 1]
    flux_after  = spectrum_after[:, 1]

    if wave.shape != flux_before.shape or wave.shape != flux_after.shape:
        raise ValueError("before/after shape mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diff = flux_after - flux_before

    def _dump(win, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = np.column_stack((wave[mask], diff[mask]))
        np.savetxt(output_dir / f"{star_name}_{tag}_DIFF_{win}.txt", out, fmt=fmt)

    if full:
        _dump("FULL", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        _dump("NUV", *DEBUG_WL_A_NUV)
        _dump("VIS", *DEBUG_WL_A_VIS)
        _dump("IR",  *DEBUG_WL_A_IR)

def dump_diff_1d_array(wave, before, after, output_dir, star_name, tag, full: bool = True, zoom: bool = True, fmt="%.18e"):
    # print(f"[DEBUG] dump_diff_windows_1d: star='{star_name}', tag='{tag}'")

    if wave.shape != before.shape or wave.shape != after.shape:
        raise ValueError("wave / before / after shape mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diff = after - before

    def _dump(win, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = np.column_stack((wave[mask], diff[mask]))
        np.savetxt(output_dir / f"{star_name}_{tag}_DIFF_{win}.txt", out, fmt=fmt)

    if full:
        _dump("FULL", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        _dump("NUV", *DEBUG_WL_A_NUV)
        _dump("VIS", *DEBUG_WL_A_VIS)
        _dump("IR",  *DEBUG_WL_A_IR)
