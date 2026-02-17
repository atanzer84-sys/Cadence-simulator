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

def dump_1d_array(wave, array, output_dir, star_name: str, tag: str, full: bool = True, zoom: bool = True, fmt="%.18e"):
    # print(f"[DEBUG] dump_spectrum_snapshots_1d: star='{star_name}', tag='{tag}'")


    if wave.shape != array.shape:
        raise ValueError("wave / values shape mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(filename, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = np.column_stack((wave[mask], array[mask]))
        np.savetxt(output_dir / filename, out, fmt=fmt)

    if full:
        _dump(f"{star_name}_{tag}_complete.txt", debug_wavelength_range_nuv[0], debug_wavelength_range_ir[1])
    if zoom:
        _dump(f"{star_name}_{tag}_NUV.txt", *DEBUG_WL_A_NUV)
        _dump(f"{star_name}_{tag}_VIS.txt", *DEBUG_WL_A_VIS)
        _dump(f"{star_name}_{tag}_IR.txt",  *DEBUG_WL_A_IR)

