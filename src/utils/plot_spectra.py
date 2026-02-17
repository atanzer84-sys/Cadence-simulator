from typing import Any
import logging
import matplotlib.pyplot as plt
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR

def plot_flux_and_photons_windows(wavelengths, values, output_dir, star, filename_tag, title_text,
 y_label,cut=True):
    if cut:
        ranges = {
            "nuv": tuple[Any, ...](debug_wavelength_range_nuv),
            "vis": tuple(debug_wavelength_range_vis),
            "ir":  tuple(debug_wavelength_range_ir),
            # zoom windows (explicit min/max constants)
            # zoom / debug windows
            "nuv_zoom": DEBUG_WL_A_NUV,
            "vis_zoom": DEBUG_WL_A_VIS,
            "ir_zoom":  DEBUG_WL_A_IR,
        }
    else:
        ranges = {
            "full": (float(wavelengths.min()), float(wavelengths.max()))
        }

    print(f"Producing plots for {star.name}")
    logging.info("Producing plots for %s", star.name)

    for key, (wmin, wmax) in ranges.items():
        mask = (wavelengths >= wmin) & (wavelengths <= wmax)

        wl = wavelengths[mask]
        flux = values[mask]

        fig, ax = plt.subplots(figsize=(12, 4))

        band = key.split("_")[0]  # "nuv", "vis", "ir"
        colors = {"nuv": "darkblue", "vis": "darkgreen", "ir": "darkred"}
        color = colors.get(band, "black")
        ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
        
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel(y_label)

        ax.set_title(f"{star.name}: {title_text} | {wmin}–{wmax} Å, M={star.mass} M☉, d={star.distance_pc} pc", fontsize=11)
        fig.savefig(output_dir / f"{star.name}_{filename_tag}_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
