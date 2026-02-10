from typing import Any
import logging
import matplotlib.pyplot as plt
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR

def plot_flux_and_photons_windows(wavelengths, flux_undiluted, photons_star, output_dir, star):
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
    print(f"Producing photon plots for {star.name}")
    logging.info("Producing photon plots for %s", star.name)

    for key, (wmin, wmax) in ranges.items():
        mask = (wavelengths >= wmin) & (wavelengths <= wmax)


        wl = wavelengths[mask]
        # flux_u = flux_undiluted[mask]
        photons = photons_star[mask]

        # # plot: undiluted flux vs photons_star
        # fig, ax = plt.subplots()
        # ax.plot(wl, flux_u, label="undiluted flux")
        # ax.plot(wl, photons, label="photons_star")
        # ax.set_xlabel("Wavelength (A)")
        # ax.set_ylabel("Flux / Photons")
        # ax.set_title(f"{star.name} {key.upper()} [{wmin}-{wmax} A]  " f"M={star.mass} Msun  d={star.distance_pc} pc")
        # ax.legend()
        # fig.savefig(output_dir / f"{star.name}_{key}_flux_vs_photons.png", dpi=200, bbox_inches="tight")
        # plt.close(fig)

        # plot: photons_star only
        fig, ax = plt.subplots()
        ax.plot(wl, photons, color="darkred", linewidth=0.4, alpha=0.6)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")
        ax.set_title(f"{star.name}: {wmin}–{wmax} Å, M={star.mass} M☉, d={star.distance_pc} pc", fontsize=11)
        fig.savefig(output_dir / f"{star.name}_{key}_photons.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
