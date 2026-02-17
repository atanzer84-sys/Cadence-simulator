import logging
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import Any
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR

def write_frames_fits(frames, headers, frame_type, channel_tag, output_dir):

    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write FITS: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, output_dir)

    for k, (frame, header) in enumerate(zip(frames, headers)):
        filename = output_dir / f"WALTzER_{channel_tag}_{frame_type}_{k:05d}.fits"
        fits.PrimaryHDU(data=frame, header=header).writeto(filename, overwrite=True)
        logging.debug("Wrote %s", filename)

    logging.info("Finished writing %d FITS file(s)", n_frames)



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
