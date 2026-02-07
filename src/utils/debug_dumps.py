import numpy as np
from pathlib import Path
from utils.constants import WL_NUV_max, WL_IR_max, WL_IR_min, WL_NUV_min, WL_VIS_max, WL_VIS_min

def dump_spectrum_snapshots(array, output_dir, star_name: str, tag: str, dump_full: bool = True, fmt="%.18e", ):
    """
    Dump a standard set of wavelength window snapshots for legacy comparison.
    Assumes wavelength in column 0.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wl = array[:, 0]

    def _dump(filename, wmin=None, wmax=None):
        if wmin is not None and wmax is not None:
            out = array[(wl >= wmin) & (wl <= wmax)]
        else:
            out = array
        np.savetxt(output_dir / filename, out, fmt=fmt)

    if dump_full:
        _dump(f"{star_name}_{tag}_complete.txt")

    _dump(f"{star_name}_{tag}_NUV.txt", WL_NUV_min, WL_NUV_max)
    _dump(f"{star_name}_{tag}_VIS.txt", WL_VIS_min, WL_VIS_max)
    _dump(f"{star_name}_{tag}_IR.txt",  WL_IR_min,  WL_IR_max)

def dump_spectrum_snapshots_1d(wave, values, output_dir, star_name: str, tag: str, dump_full: bool = True, fmt="%.18e",):
    """
    Dump a standard set of wavelength window snapshots for 1D spectra.
    """
    if wave.shape != values.shape:
        raise ValueError("wave / values shape mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(filename, wmin=None, wmax=None):
        if wmin is not None and wmax is not None:
            mask = (wave >= wmin) & (wave <= wmax)
            out = np.column_stack((wave[mask], values[mask]))
        else:
            out = np.column_stack((wave, values))
        np.savetxt(output_dir / filename, out, fmt=fmt)

    if dump_full:
        _dump(f"{star_name}_{tag}_complete.txt")

    _dump(f"{star_name}_{tag}_NUV.txt", WL_NUV_min, WL_NUV_max)
    _dump(f"{star_name}_{tag}_VIS.txt", WL_VIS_min, WL_VIS_max)
    _dump(f"{star_name}_{tag}_IR.txt",  WL_IR_min,  WL_IR_max)

def dump_diff_windows_1d(wave, before, after, output_dir, star_name, tag, fmt="%.18e"):
    if wave.shape != before.shape or wave.shape != after.shape:
        raise ValueError("wave / before / after shape mismatch")

    diff = after - before

    # --- FULL diff dump ---
    out_full = np.column_stack((wave, diff))
    dump_array(
        out_full,
        output_dir,
        filename=f"{star_name}_DIFF_{tag}_diff_full.txt",
        fmt=fmt,
    )

    # --- WINDOWED diffs ---
    def _dump(win, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = (
            np.column_stack((wave[mask], diff[mask]))
            if np.any(mask)
            else np.empty((0, 2))
        )
        dump_array(
            out,
            output_dir,
            filename=f"{star_name}_DIFF_{tag}_diff_{win}.txt",
            fmt=fmt,
        )

    _dump("NUV", WL_NUV_min, WL_NUV_max)
    _dump("VIS", WL_VIS_min, WL_VIS_max)
    _dump("IR",  WL_IR_min,  WL_IR_max)

def dump_diff_windows_3d(spectrum_before, spectrum_after, output_dir, star_name, tag, fmt="%.18e", ):
    """
    Diff for a multi-column spectrum.

    spectrum_before : full array BEFORE (wave in col 0, flux in col 1)
    spectrum_after  : full array AFTER  (same wavelength grid)

    Produces:
      STAR_<tag>_flux_diff_full.txt
      STAR_<tag>_flux_diff_NUV.txt
      STAR_<tag>_flux_diff_VIS.txt
      STAR_<tag>_flux_diff_IR.txt
    """

    wave = spectrum_after[:, 0]
    flux_before = spectrum_before[:, 1]
    flux_after  = spectrum_after[:, 1]

    if wave.shape != flux_before.shape or wave.shape != flux_after.shape:
        raise ValueError("before/after shape mismatch")

    diff = flux_after - flux_before

    # --- FULL diff ---
    out_full = np.column_stack((wave, diff))
    dump_array(
        out_full,
        output_dir,
        filename=f"{star_name}_{tag}_flux_diff_full.txt",
        fmt=fmt,
    )

    # --- WINDOWED diffs ---
    def _dump(win, wmin, wmax):
        mask = (wave >= wmin) & (wave <= wmax)
        out = (
            np.column_stack((wave[mask], diff[mask]))
            if np.any(mask)
            else np.empty((0, 2))
        )
        dump_array(
            out,
            output_dir,
            filename=f"{star_name}_{tag}_flux_diff_{win}.txt",
            fmt=fmt,
        )

    _dump("NUV", WL_NUV_min, WL_NUV_max)
    _dump("VIS", WL_VIS_min, WL_VIS_max)
    _dump("IR",  WL_IR_min,  WL_IR_max)



def dump_array(array, output_dir, filename, wl_min=None, wl_max=None, fmt="%.18e"):
    """
    Dump a spectrum array to disk.

    If wl_min and wl_max are provided, dump only wavelengths in [wl_min, wl_max].
    If either is None, dump the full array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if wl_min is not None and wl_max is not None:
        wl = array[:, 0]
        out_array = array[(wl >= wl_min) & (wl <= wl_max)]
    else:
        out_array = array

    np.savetxt(output_dir / filename, out_array, fmt=fmt)

    return out_array

def dump_spectrum_snapshots(
    array,
    output_dir,
    star_name: str,
    tag: str,
    dump_full: bool = True,
) -> None:
    """
    Dump a standard set of wavelength window snapshots for legacy comparison.
    """
    if dump_full:
        dump_array(array, output_dir, filename=f"{star_name}_{tag}_complete.txt")

    dump_array(array, output_dir, filename=f"{star_name}_{tag}_NUV.txt", wl_min=WL_NUV_min, wl_max=WL_NUV_max)
    dump_array(array, output_dir, filename=f"{star_name}_{tag}_VIS.txt", wl_min=WL_VIS_min, wl_max=WL_VIS_max)
    dump_array(array, output_dir, filename=f"{star_name}_{tag}_IR.txt", wl_min=WL_IR_min, wl_max=WL_IR_max)
