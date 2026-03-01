import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_IR
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import numpy as np
from configs.channel_config import SpectroscopyChannel


STATS_KEYS = {
    "BIAS": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET"],
    "DARK": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "DARKVAL", "DARKSIG", "EXPTIME", "B_OFFSET", "RNOISE"],
    "SCIENCE": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "DARKVAL", "DARKSIG", "B_OFFSET", "RNOISE", "EXPTIME"],
}

# Shared layout constants so single-image and multi-frame PNGs match.
_WIDTH_IN = 10.0
_TEXT_H_IN = 0.7
_GAP_IN = 0.8


def _save_single_frame_png(array: np.ndarray, filename: Path, title: str, stats_text: str | None = None) -> None:
    """Draw one 2D image with optional stats line; save to filename. Shared by write_image_png and write_frames_png."""
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin = np.percentile(array, 1)
    vmax = np.percentile(array, 99.9)
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel("pixels", labelpad=8)
    ax.set_ylabel("pixels", labelpad=8)
    ax.set_title(title, fontsize=11)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    if stats_text:
        ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=10, transform=ax_txt.transAxes)

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.debug("Wrote %s", filename)


def write_image_png(array, frame_type: str, ctx: RunContext, channel: SpectroscopyChannel, show_stats: bool = True) -> None:

    logging.info("WRITE_IMAGE_PNG called | frame_type=%s | channel=%s | shape=%s", frame_type, channel.channel_name, array.shape)

    star_name = str(ctx.target_name).replace(" ", "_")
    filename = ctx.output_dir / f"{star_name}_{channel.channel_name}_{frame_type}_image.png"
    title = f"{ctx.target_name}: {channel.channel_name} {frame_type}"

    stats_text = None
    if show_stats:
        mean, median = float(np.mean(array)), float(np.median(array))
        std = float(np.std(array, ddof=0))
        vmin, vmax = float(np.min(array)), float(np.max(array))
        rn = getattr(channel, "read_noise", None)
        bo = getattr(channel, "bias_offset", None)
        dv = getattr(channel, "dark_noise", None)
        ds = getattr(channel, "dark_current_sigma", None)
        ex = getattr(channel, "exposure_s", None)
        def _v(x, fmt=""):
            return (format(x, fmt) if fmt else str(x)) if x is not None else "n/a"
        stats_text = (
            f"MEAN={mean:.3f}  MEDIAN={median:.3f}  STDDEV={std:.2f}  MIN={vmin:.2f}  MAX={vmax:.2f}\n"
            f"RNOISE={_v(rn)}  B_OFFSET={_v(bo)}  DARKVAL={_v(dv, '.3f')}  DARKSIG={_v(ds, '.3f')}  EXPTIME={_v(ex, '.3f')}"
        )

    _save_single_frame_png(array, filename, title, stats_text)




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

    star_name = str(star.name).replace(" ", "_")
    title_base = f"{star.name}: {channel_tag} {frame_type} | M={star.effective_temperature} K, d={star.distance_pc} pc"

    for k, (frame, header) in enumerate(zip(frames, headers)):
        filename = ctx.output_dir / f"WALTzER_{star_name}_{channel_tag}_{frame_type}_{k:05d}.png"

        stats_text = None
        if show_stats:
            keys = _stats_keys_for_header(header)
            parts = []
            for i, key in enumerate(keys):
                if key in ("RNOISE", "B_OFFSET"):
                    fmt = ""
                elif key in ("MEAN", "MEDIAN", "DARKSIG", "DARKVAL"):
                    fmt = ".3f"
                else:
                    fmt = ".2f"
                parts.append(format_header(header, key, fmt))
                if (i + 1) % 5 == 0 and i + 1 < len(keys):
                    parts.append("\n")
                elif i + 1 < len(keys):
                    parts.append("    ")
            stats_text = "".join(parts)

        _save_single_frame_png(frame, filename, title_base, stats_text)

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
    ax.set_title(f"{star.name}: {title_text} | {wmin}–{wmax} Å, M={star.effective_temperature} K, d={star.distance_pc} pc", fontsize=11)
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