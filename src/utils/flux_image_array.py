import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR
from domain.star import Star
from configs.global_config import get_global_config
from utils.images_common import format_star_metadata, normalize_target_name
from domain.star_catalog import StarCatalog
from configs.channel_config import Channel
from loaders.run_waltzer_context import RunContext


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


def plot_1d_for_channel(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name: str, full: bool = False, zoom: bool = False):
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
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name, full_range[0], full_range[1])

    if zoom:
        _plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, f"{channel_name}_zoom", zoom_range[0], zoom_range[1])


def _plot_photon_flux(wavelengths, values, output_dir, star : Star, filename_tag, title_text, y_label, key, wmin, wmax):

    mask = (wavelengths >= wmin) & (wavelengths <= wmax)
    logging.info("Plotting %s for star %s in window '%s' (%.1f–%.1f Å); %d wavelength bins", filename_tag, star.name, key, wmin, wmax, int(mask.sum()))

    wl = wavelengths[mask]
    flux = values[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    band = key.split("_")[0].lower()
    colors = {"nuv": "darkblue", "vis": "darkgreen", "nir": "darkred"}
    color = colors.get(band, "black")

    ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel(y_label)
    meta = format_star_metadata(star)
    ax.set_title(f"{star.name}: {title_text} | {wmin:.2f}–{wmax:.2f} Å, {meta}", fontsize=11)
    safe_name = normalize_target_name(star.name)
    fig.savefig(Path(output_dir) / f"{safe_name}_{filename_tag}_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)




def plot_background_star_counts(background_stars_catalog: StarCatalog, channel: Channel | None, ctx: RunContext):

    if channel is None:
        return

    wavelength = channel.effective_area_wavelength
    stars_sorted = sorted(background_stars_catalog.stars_by_id.items(), key=lambda item: item[1].gaia_magnitude)
    total = len(stars_sorted)
    safe_target = normalize_target_name(ctx.target_name)

    for start in range(0, total, 5):

        subset = stars_sorted[start:start + 5]

        plt.figure()

        for star_id, bg_star in subset:
            counts_s_px = background_stars_catalog.counts_by_id_and_band[(star_id, channel.channel_name)]
            counts_exp = counts_s_px * channel.exposure_s

            label = f"G={bg_star.gaia_magnitude:.2f}"
            plt.plot(wavelength, counts_exp, label=label, linewidth=0.4, alpha=0.6)

        plt.axhline(channel.read_noise, linestyle="--", color="black", label=f"Read noise={channel.read_noise:g} e⁻")
        plt.axhline(channel.dark_noise * channel.exposure_s, linestyle=":", color="black", label=f"Dark={channel.dark_noise:g} e⁻/s ({channel.dark_noise * channel.exposure_s:g} e⁻)")
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Counts exposure^-1 pixel^-1")
        plt.legend()

        title = f"{ctx.target_name}: Background stars vs noise ({channel.channel_name}, {channel.exposure_s}s)"
        plt.title(title)

        filename = ctx.output_dir / f"{safe_target}_background_stars_{channel.channel_name}_{start}.png"

        plt.savefig(filename)
        plt.close()
    
    logging.info("Background star count plots finished: channel=%s", channel.channel_name)