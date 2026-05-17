import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR
from domain.star import Star
from configs.global_config import get_global_config
from utils.images_common import format_star_metadata, normalize_target_name
import matplotlib.ticker as ticker


def plot_flux_and_photons_windows(wavelengths, values, output_dir, star: Star, filename_tag, title_text, y_label, perChannel: bool = True, full: bool = False, zoom: bool = True):
    print(f"Producing plots for {star.name}")
    logging.info("Producing plots for %s", star.name)

    cfg = get_global_config()
    channels = {name for name, enabled in (("NUV", cfg.run_nuv), ("VIS", cfg.run_vis), ("NIR", cfg.run_nir)) if enabled}

    if perChannel:
        if "NUV" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1])
        if "VIS" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1])
        if "NIR" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NIR",  debug_wavelength_range_ir[0],  debug_wavelength_range_ir[1])

    if full:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "full", float(wavelengths.min()), float(wavelengths.max()))

    if zoom:
        if "NUV" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV_zoom", *DEBUG_WL_A_NUV)
        if "VIS" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS_zoom", *DEBUG_WL_A_VIS)
        if "NIR" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NIR_zoom",  *DEBUG_WL_A_NIR)


def plot_1d_for_channel(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name: str, full: bool = False, zoom: bool = False, noise_floor: float | None = None):
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
        raise ValueError("channel_name must be 'NUV', 'VIS', or 'NIR'")

    if full:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name, full_range[0], full_range[1], noise_floor=noise_floor)

    if zoom:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, f"{channel_name}_zoom", zoom_range[0], zoom_range[1], noise_floor=noise_floor)


def _plot_photon_flux(wavelengths, values, output_dir, star : Star, filename_tag, title_text, y_label, key, wmin, wmax, noise_floor: float | None = None):

    mask = (wavelengths >= wmin) & (wavelengths <= wmax)
    logging.info("Plotting %s for star %s in window '%s' (%.1f–%.1f Å); %d wavelength bins", filename_tag, star.name, key, wmin, wmax, int(mask.sum()))

    wl = wavelengths[mask]
    flux = values[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    band = key.split("_")[0].lower()
    colors = {"nuv": "darkblue", "vis": "darkgreen", "nir": "darkred"}
    color = colors.get(band, "black")

    ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
    if noise_floor is not None:
        ax.axhline(noise_floor, color="red", linestyle="--", linewidth=1.0, label=f"Bias + dark = {noise_floor:.1f} e⁻ px⁻¹")
        ax.legend(fontsize=9)
    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    meta = format_star_metadata(star)
    ax.set_title(rf"{star.name}: {title_text} | {wmin:.2f}–{wmax:.2f} $\mathrm{{\AA}}$, {meta}", fontsize=11)
    safe_name = normalize_target_name(star.name)
    fig.savefig(Path(output_dir) / f"{safe_name}_{filename_tag}_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)




def plot_model_input(model_data, wl_min, wl_max, output_dir, star: Star, filename_tag="FluxCalc_1_model_input", title_text="Model Input Spectrum"):
    wavelength = model_data[:, 0]
    flux = model_data[:, 1]
    continuum = model_data[:, 2]

    mask = (wavelength >= wl_min) & (wavelength <= wl_max)

    wl = wavelength[mask]
    flux_w = flux[mask]
    cont_w = continuum[mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))


    # 1) full resolution, very faint
    ax.plot(wl, flux_w, color="black", linewidth=0.1, alpha=0.1)

    # 2) downsampled, visible structure
    step = 175
    ax.plot(wl[::step], flux_w[::step], color="black", linewidth=0.2, label="Flux")

    ax.plot(wl, cont_w, color="red", linewidth=0.6, label="Continuum")

    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Model spectrum [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

    meta = format_star_metadata(star)
    ax.set_title(rf"{star.name}: {title_text} | {wl_min:.2f}–{wl_max:.2f} $\mathrm{{\AA}}$, {meta}", fontsize=11)

    ax.legend()

    safe_name = normalize_target_name(star.name)
    out_path = Path(output_dir) / f"{safe_name}_{filename_tag}_{int(wl_min)}_{int(wl_max)}.png"

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)