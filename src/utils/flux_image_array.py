import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR, MgII1w, MgII2w
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
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NIR", debug_wavelength_range_ir[0], debug_wavelength_range_ir[1])

    if full:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "full", float(wavelengths.min()), float(wavelengths.max()))

    if zoom:
        if "NUV" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV_zoom", *DEBUG_WL_A_NUV)
        if "VIS" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS_zoom", *DEBUG_WL_A_VIS)
        if "NIR" in channels:
            _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NIR_zoom", *DEBUG_WL_A_NIR)


def plot_1d_for_channel(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name: str, full: bool = False, zoom: bool = False, noise_floor: float | None = None, noise_sigma: float | None = None):
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
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name, full_range[0], full_range[1], noise_floor=noise_floor, noise_sigma=noise_sigma)


    if zoom:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, f"{channel_name}_zoom", zoom_range[0], zoom_range[1], noise_floor=noise_floor)


def _wavelength_ticks(ax, plot_key: str) -> None:
    """Readable wavelength labels: fixed spacing per band where helpful; mpl chooses for wide spans."""

    if "_" in plot_key:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
        return

    stem = plot_key.split("_")[0]

    if stem == "full":
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=11, prune="both"))
        return

    if stem == "NUV":
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        return

    if stem == "VIS":
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        return

    if stem == "NIR":
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=11, prune="both"))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))


def _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, key, wmin, wmax, noise_floor: float | None = None, noise_sigma: float | None = None):

    mask = (wavelengths >= wmin) & (wavelengths <= wmax)
    logging.info("Plotting %s for star %s in window '%s' (%.1f–%.1f Å); %d wavelength bins", filename_tag, star.name, key, wmin, wmax, int(mask.sum()))

    wl = wavelengths[mask]
    flux = values[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    band = key.split("_")[0].lower()
    colors = {"nuv": "darkblue", "vis": "darkgreen", "nir": "darkred"}
    color = colors.get(band, "black")

    ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6, label=rf"{band.upper()} ({wmin:.0f}–{wmax:.0f} $\mathrm{{\AA}}$)")
    
    if noise_floor is not None:
        ax.axhline(noise_floor, color="red", linestyle="--", linewidth=1.0, label=rf"Baseline (bias + dark) = {noise_floor:.1f} e$^{{-}}$ px$^{{-1}}$")
        if noise_sigma is not None:
            ax.axhspan(noise_floor - noise_sigma, noise_floor + noise_sigma, color="red", alpha=0.15, label=rf"$\pm\sigma$ (read + dark noise) = {noise_sigma:.1f} e$^{{-}}$ px$^{{-1}}$")



    _wavelength_ticks(ax, key)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="x", which="major", length=6)

    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ax.set_ylabel(y_label)

    meta = format_star_metadata(star)
    ax.set_title(rf"{star.name}: {title_text} | {wmin:.2f}–{wmax:.2f} $\mathrm{{\AA}}$, {meta}", fontsize=11)

    legend_loc = {"nuv": "upper left", "vis": "upper right", "nir": "upper right"}
    ax.legend(loc=legend_loc.get(band, "upper right"), fontsize=9, framealpha=0.8)

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

def plot_lce_comparison_four_panel(wavelengths, flux_before, flux_after, output_dir, star, k_min, k_max, h_min, h_max, filename_tag, title_text, sigmaMg21, sigmaMg22):
    mask_k = (wavelengths >= k_min) & (wavelengths <= k_max)
    mask_h = (wavelengths >= h_min) & (wavelengths <= h_max)

    wl_k = wavelengths[mask_k]
    wl_h = wavelengths[mask_h]

    flux_before_k = flux_before[mask_k]
    flux_before_h = flux_before[mask_h]
    flux_after_k = flux_after[mask_k]
    flux_after_h = flux_after[mask_h]

    flux_before_k = np.clip(flux_before_k, 1e-30, None)
    flux_before_h = np.clip(flux_before_h, 1e-30, None)
    flux_after_k  = np.clip(flux_after_k, 1e-30, None)
    flux_after_h  = np.clip(flux_after_h, 1e-30, None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", sharey=True)
        
    def annotate_reference_line(ax, wavelength, label):
        ax.annotate(label, xy=(wavelength, 0.98), xycoords=("data", "axes fraction"), xytext=(6, -2), textcoords="offset points", va="top", ha="left", color="red", fontsize=10)

    for ax in axes.flat:
        ax.set_yscale("log")

    axes[0, 0].plot(wl_k, flux_before_k, color="black", linewidth=1)
    axes[0, 0].axvline(MgII1w, color="red", linestyle="--", linewidth=0.8)
    # for s in [1, 2]:
    #     axes[0, 0].axvline(MgII1w + s*sigmaMg21, color="blue", linestyle=":", alpha=0.5)
    #     axes[0, 0].axvline(MgII1w - s*sigmaMg21, color="blue", linestyle=":", alpha=0.5)
    axes[0, 0].set_ylabel(r"Spectral luminosity [erg s$^{-1}$ $\mathrm{\AA}^{-1}$]")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    annotate_reference_line(axes[0, 0], MgII1w, rf"Mg II k = {MgII1w:.4f} $\mathrm{{\AA}}$")

    axes[0, 1].plot(wl_h, flux_before_h, color="black", linewidth=1)
    axes[0, 1].axvline(MgII2w, color="red", linestyle="--", linewidth=0.8)
    # for s in [1, 2]:
    #     axes[0, 1].axvline(MgII2w + s*sigmaMg22, color="blue", linestyle=":", alpha=0.5)
    #     axes[0, 1].axvline(MgII2w - s*sigmaMg22, color="blue", linestyle=":", alpha=0.5)
    
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    annotate_reference_line(axes[0, 1], MgII2w, rf"Mg II h = {MgII2w:.4f} $\mathrm{{\AA}}$")

    axes[1, 0].plot(wl_k, flux_after_k, color="black", linewidth=1)
    axes[1, 0].axvline(MgII1w, color="red", linestyle="--", linewidth=0.8)
    # for s in [1, 2]:
    #     axes[1, 0].axvline(MgII1w + s*sigmaMg21, color="blue", linestyle=":", alpha=0.5)
    #     axes[1, 0].axvline(MgII1w - s*sigmaMg21, color="blue", linestyle=":", alpha=0.5)
    axes[1, 0].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    axes[1, 0].set_ylabel(r"Spectral luminosity [erg s$^{-1}$ $\mathrm{\AA}^{-1}$]")

    axes[1, 1].plot(wl_h, flux_after_h, color="black", linewidth=1)
    axes[1, 1].axvline(MgII2w, color="red", linestyle="--", linewidth=0.8)
    # for s in [1, 2]:
    #     axes[1, 1].axvline(MgII2w + s*sigmaMg22, color="blue", linestyle=":", alpha=0.5)
    #     axes[1, 1].axvline(MgII2w - s*sigmaMg22, color="blue", linestyle=":", alpha=0.5)
    axes[1, 1].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")

    for ax in axes.flat:
        ax.set_yscale("log")


    meta = format_star_metadata(star)
    fig.suptitle(rf"{star.name}: {title_text} | k: {k_min:.3f}–{k_max:.3f} $\mathrm{{\AA}}$, h: {h_min:.3f}–{h_max:.3f} $\mathrm{{\AA}}$, {meta}", fontsize=11)

    safe_name = normalize_target_name(star.name)
    out_path = Path(output_dir) / f"{safe_name}_{filename_tag}.png"

    for ax in axes[:, 0]:
        ax.set_xlim(k_min, k_max)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    for ax in axes[:, 1]:
        ax.set_xlim(h_min, h_max)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_two_line_comparison_four_panel(wavelengths, flux_before, flux_after, output_dir, star: Star, line1_min, line1_max, line1_wavelength, line1_label, line2_min, line2_max, line2_wavelength, line2_label, filename_tag, title_text):
    mask_1 = (wavelengths >= line1_min) & (wavelengths <= line1_max)
    mask_2 = (wavelengths >= line2_min) & (wavelengths <= line2_max)

    wl_1 = wavelengths[mask_1]
    wl_2 = wavelengths[mask_2]

    flux_before_1 = flux_before[mask_1]
    flux_before_2 = flux_before[mask_2]
    flux_after_1 = flux_after[mask_1]
    flux_after_2 = flux_after[mask_2]

    flux_before_1 = np.clip(flux_before_1, 1e-30, None)
    flux_after_1  = np.clip(flux_after_1, 1e-30, None)
    flux_before_2 = np.clip(flux_before_2, 1e-30, None)
    flux_after_2  = np.clip(flux_after_2, 1e-30, None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", sharey=True)

    def annotate_reference_line(ax, wavelength, label):
        ax.annotate(label, xy=(wavelength, 0.98), xycoords=("data", "axes fraction"), xytext=(6, -2), textcoords="offset points", va="top", ha="left", color="red", fontsize=10)

    for ax in axes.flat:
        ax.set_yscale("log")

    axes[0, 0].plot(wl_1, flux_before_1, color="black", linewidth=1)
    axes[0, 0].axvline(line1_wavelength, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_ylabel(r"Spectral luminosity [erg s$^{-1}$ $\mathrm{\AA}^{-1}$]")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    annotate_reference_line(axes[0, 0], line1_wavelength, rf"{line1_label} = {line1_wavelength:.4f} $\mathrm{{\AA}}$")

    axes[0, 1].plot(wl_2, flux_before_2, color="black", linewidth=1)
    axes[0, 1].axvline(line2_wavelength, color="red", linestyle="--", linewidth=0.8)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    annotate_reference_line(axes[0, 1], line2_wavelength, rf"{line2_label} = {line2_wavelength:.4f} $\mathrm{{\AA}}$")

    axes[1, 0].plot(wl_1, flux_after_1, color="black", linewidth=1)
    axes[1, 0].axvline(line1_wavelength, color="red", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    axes[1, 0].set_ylabel(r"Spectral luminosity [erg s$^{-1}$ $\mathrm{\AA}^{-1}$]")

    axes[1, 1].plot(wl_2, flux_after_2, color="black", linewidth=1)
    axes[1, 1].axvline(line2_wavelength, color="red", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel(r"Wavelength ($\mathrm{\AA}$)")

    for ax in axes.flat:
        ax.set_yscale("log")

    meta = format_star_metadata(star)
    fig.suptitle(
        rf"{star.name}: {title_text} | "
        rf"{line1_label}: {line1_min:.3f}–{line1_max:.3f} $\mathrm{{\AA}}$, "
        rf"{line2_label}: {line2_min:.3f}–{line2_max:.3f} $\mathrm{{\AA}}$, "
        rf"{meta}",
        fontsize=11,
    )

    safe_name = normalize_target_name(star.name)
    out_path = Path(output_dir) / f"{safe_name}_{filename_tag}.png"

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ism_transmission_four_panel(wavelengths, flux_before, flux_after, output_dir, star, line1_min, line1_max, line1_wavelength, line1_label, line2_min, line2_max, line2_wavelength, line2_label, filename_tag, title_text):
    transmission = flux_after / flux_before

    mask_1 = (wavelengths >= line1_min) & (wavelengths <= line1_max)
    mask_2 = (wavelengths >= line2_min) & (wavelengths <= line2_max)

    wl_1 = wavelengths[mask_1]
    wl_2 = wavelengths[mask_2]

    trans_1 = transmission[mask_1]
    trans_2 = transmission[mask_2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # left panel
    axes[0].plot(wl_1, trans_1, color="black")
    axes[0].axvline(line1_wavelength, color="red", linestyle="--")
    axes[0].set_xlabel("Wavelength (Å)")
    axes[0].set_ylabel("Transmission")
    axes[0].set_ylim(0, 1.05)

    # right panel
    axes[1].plot(wl_2, trans_2, color="black")
    axes[1].axvline(line2_wavelength, color="red", linestyle="--")
    axes[1].set_xlabel("Wavelength (Å)")
    axes[1].set_ylim(0, 1.05)

    # clean ticks
    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig.suptitle(f"{star.name}: {title_text}")

    safe_name = normalize_target_name(star.name)
    out_path = Path(output_dir) / f"{safe_name}_{filename_tag}.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)