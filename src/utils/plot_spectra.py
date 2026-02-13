from typing import Any
import logging
import matplotlib.pyplot as plt
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR

def plot_flux_and_photons_windows(wavelengths, flux_star, output_dir, star, quantity_tag, y_label):

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
    print(f"Producing plots for {star.name}")
    logging.info("Producing plots for %s", star.name)

    for key, (wmin, wmax) in ranges.items():
        mask = (wavelengths >= wmin) & (wavelengths <= wmax)

        wl = wavelengths[mask]
        flux = flux_star[mask]

        fig, ax = plt.subplots()

        band = key.split("_")[0]  # "nuv", "vis", "ir"
        colors = {"nuv": "darkblue", "vis": "darkgreen", "ir": "darkred"}
        color = colors.get(band, "black")
        ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
        
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")
        ax.set_ylabel(y_label)

        ax.set_title(f"{star.name}: {quantity_tag} | {wmin}–{wmax} Å, M={star.mass} M☉, d={star.distance_pc} pc", fontsize=11)
        fig.savefig(output_dir / f"{star.name}_{quantity_tag}_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
