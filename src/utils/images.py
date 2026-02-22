import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR
from loaders.run_waltzer_context import RunContext
from domain.star import Star

STATS_KEYS = {
    "BIAS": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET"],
    "DARK": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "DARKVAL", "DARKSIG", "EXPTIME", "B_OFFSET", "RNOISE"],
    "SCIENCE": ["EXPTIME", "DARKVAL", "DARKSIG", "B_OFFSET", "RNOISE"],
}


def _header_val(header, key):
    if hasattr(header, "get"):
        return header.get(key)
    for item in header:
        if len(item) >= 2 and item[0] == key:
            return item[1]
    return None


def _stats_keys_for_header(header):
    ft = _header_val(header, "FILETYPE")
    filetype = str(ft).upper() if ft else "BIAS"
    return STATS_KEYS.get(filetype, STATS_KEYS["BIAS"])


def write_frames_png(frames, headers, frame_type, channel_tag, ctx: RunContext, star: Star, show_stats=False):

    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write PNG: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s PNG frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, ctx.output_dir)

    for k, (frame, header) in enumerate(zip(frames, headers)):

        filename = ctx.output_dir / f"WALTzER_{channel_tag}_{frame_type}_{k:05d}.png"

        ny, nx = frame.shape
        width_in = 10.0
        img_h_in = max(2.0, width_in * (ny / nx))
        text_h_in = 0.7

        fig = plt.figure(figsize=(width_in, img_h_in + text_h_in))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[img_h_in, text_h_in])

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(frame, origin="lower", aspect="equal", cmap="gray")
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(f"{star.name}: {channel_tag} {frame_type} | M={star.mass} M☉, d={star.distance_pc} pc", fontsize=11)

        ax_txt = fig.add_subplot(gs[1, 0])
        ax_txt.axis("off")

        if show_stats:
            keys = _stats_keys_for_header(header)
            parts = []
            for i, k in enumerate(keys):
                fmt = "" if k in ("RNOISE", "B_OFFSET") else ".2f"
                parts.append(format_header(header, k, fmt))
                if (i + 1) % 5 == 0 and i + 1 < len(keys):
                    parts.append("\n")
                elif i + 1 < len(keys):
                    parts.append("    ")
            stats_text = "".join(parts)

            ax_txt.text(
                0.5, 0.5, stats_text,
                ha="center", va="center",
                fontsize=10,
                transform=ax_txt.transAxes,
            )

        fig.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

        logging.debug("Wrote %s", filename)

    logging.info("Finished writing %d PNG file(s)", n_frames)


def format_header(header, key, fmt_str=".2f"):
    val = _header_val(header, key)
    if val is None:
        return f"{key}=n/a"
    return f"{key}={format(val, fmt_str)}" if fmt_str else f"{key}={val}"


def plot_photon_flux(wavelengths, values, output_dir, star : Star, filename_tag, title_text, y_label, key, wmin, wmax):

    mask = (wavelengths >= wmin) & (wavelengths <= wmax)
    logging.info("Plotting %s for star %s in window '%s' (%.1f–%.1f Å); %d wavelength bins", filename_tag, star.name, key, wmin, wmax, int(mask.sum()))

    wl = wavelengths[mask]
    flux = values[mask]

    fig, ax = plt.subplots(figsize=(12, 4))

    band = key.split("_")[0].lower()
    colors = {"nuv": "darkblue", "vis": "darkgreen", "ir": "darkred"}
    color = colors.get(band, "black")

    ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{star.name}: {title_text} | {wmin}–{wmax} Å, M={star.mass} M☉, d={star.distance_pc} pc", fontsize=11)
    fig.savefig(Path(output_dir) / f"{star.name}_{filename_tag}_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_flux_and_photons_windows(wavelengths, values, output_dir, star: Star, filename_tag, title_text, y_label, perChannel: bool = True, full: bool = False, zoom: bool = True):
    print(f"Producing plots for {star.name}")
    logging.info("Producing plots for %s", star.name)

    if perChannel:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1])
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1])
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "IR",  debug_wavelength_range_ir[0],  debug_wavelength_range_ir[1])

    if full:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "full", float(wavelengths.min()), float(wavelengths.max()))

    if zoom:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV_zoom", *DEBUG_WL_A_NUV)
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS_zoom", *DEBUG_WL_A_VIS)
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "IR_zoom",  *DEBUG_WL_A_IR)


def plot_1d_for_channel(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name: str, full: bool = False, zoom: bool = False):
    if channel_name == "NUV":
        full_range = debug_wavelength_range_nuv
        zoom_range = DEBUG_WL_A_NUV
    elif channel_name == "VIS":
        full_range = debug_wavelength_range_vis
        zoom_range = DEBUG_WL_A_VIS
    elif channel_name == "IR":
        full_range = debug_wavelength_range_ir
        zoom_range = DEBUG_WL_A_IR
    else:
        raise ValueError("channel_name must be 'NUV', 'VIS', or 'IR'")

    if full:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name, full_range[0], full_range[1])

    if zoom:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, f"{channel_name}_zoom", zoom_range[0], zoom_range[1])