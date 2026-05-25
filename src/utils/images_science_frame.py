import logging
from loaders.run_cadence_context import RunContext
from domain.star import Star
from utils.images_common import format_frame_title, build_stats_row, build_png_filename, format_stats_text
from configs.channel_config import Channel, SpectroscopyChannel
from configs.global_config import GlobalConfig
from instrument.spectrum_spread import get_target_star_detector_position
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for fast batch PNG writing
import matplotlib.pyplot as plt


_SPECTRUM_STRIP_MARGIN_FRAC = 0.2
# Shared layout constants so single-image and multi-frame PNGs match.
_WIDTH_IN = 10.0
_TEXT_H_IN = 0.7
_GAP_IN = 0.8
_TITLE_FONTSIZE = 11
_STATS_FONTSIZE = 10
_FIGURE_COMMON_DPI = 150
# None avoids expensive tight-bbox computation; we use tight_layout() for spacing instead.
_BBOX_INCHES = None
# Max pixels used for percentile scaling; larger arrays are subsampled to reduce runtime.
_PERCENTILE_MAX_PIXELS = 250_000


def _imshow_science_array(ax, array: np.ndarray, vmin: float, vmax: float, y_detector_lo: int | None = None) -> None:
    """Plot detector array with pixel-centered coordinates; if y_detector_lo is set (cropped strip), y-axis is full-detector row index."""
    ny, nx = array.shape
    if y_detector_lo is None:
        ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
    else:
        y_hi_det = y_detector_lo + ny
        extent = (-0.5, nx - 0.5, y_detector_lo - 0.5, y_hi_det - 0.5)
        ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(y_detector_lo - 0.5, y_hi_det - 0.5)


def write_science_frame_png(detector_data, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, phot=None, index: int | None = None) -> None:
    filetype = "science"
    channel_name = channel.channel_name
    inverted = cfg.invert_science_frames

    # Prepare data for PNG: optionally crop to spectrum strip for spectroscopy
    data_for_png = detector_data
    y_detector_lo: int | None = None
    if cfg.science_frame_png_crop_spectrum_region and isinstance(channel, SpectroscopyChannel):
        ny = channel.y_pixels
        _, y0 = get_target_star_detector_position(channel)  # same position used when placing spectrum on detector
        half_strip_pix = int(round(channel.slit_half_length_arcsec / channel.pixel_scale))
        strip_height = 2 * half_strip_pix + 1
        margin = int(round(_SPECTRUM_STRIP_MARGIN_FRAC * strip_height))
        half_height = half_strip_pix + margin
        y_lo = max(0, y0 - half_height)
        y_hi = min(ny, y0 + half_height + 1)
        data_for_png = detector_data[y_lo:y_hi, :]
        y_detector_lo = y_lo

    title = format_frame_title(channel_name, filetype, star)

    frame_to_plot = (data_for_png.max() - data_for_png) if inverted else data_for_png

    stats_values, stats_keys = build_stats_row(detector_data, channel, filetype)
    stats_text = format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    filename = build_png_filename(ctx.output_dir, star.name, channel_name, filetype, channel.exposure_s, index, waltzer_prefix=True)

    if channel_name == "VIS":
        if cfg.science_frame_png_crop_spectrum_region:
            save_single_frame_png_VIS_cropped(frame_to_plot, filename, title, stats_text, y_detector_lo)
        else:
            save_single_frame_png_VIS(frame_to_plot, filename, title, stats_text, y_detector_lo)
    elif channel_name == "NIR":
        save_single_frame_png_NIR(frame_to_plot, filename, title, stats_text, phot=phot, draw_aperture_photometry_overlay=channel.draw_aperture_photometry_overlay)
    elif channel_name == "NUV":
        save_single_frame_png_NUV(frame_to_plot, filename, title, stats_text, y_detector_lo)
    else:
        raise ValueError(f"Unsupported channel for science PNG: {channel_name}")



def save_single_frame_png_NIR(array: np.ndarray, filename: Path, title: str, stats_text: str, phot=None, draw_aperture_photometry_overlay: bool = False) -> None:
    STATS_FONTSIZE_NIR = 13
    TITLE_FONTSIZE_NIR = 18
    GAP_IN_NIR = 0.35
    TEXT_H_IN_NIR = 0.9
    NIR_LABEL_FONTSIZE = 12
    _FIGURE_NIR_DPI = 100


    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + GAP_IN_NIR + TEXT_H_IN_NIR))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, GAP_IN_NIR, TEXT_H_IN_NIR], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    # vmin, vmax = _calculate_percentile_scales(array)
    # vmin = float(np.min(array))
    # vmax = float(np.max(array))
    vmin = float(np.percentile(array, 1))
    vmax = float(np.percentile(array, 99.9))
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)

    counts_star = None
    counts_star_noise = None

    if phot is not None and draw_aperture_photometry_overlay:
        counts_star, counts_star_noise, x0, y0, aperture_radius, annulus_outer_radius = phot

        aperture_circle = plt.Circle((x0, y0), aperture_radius, fill=False, linewidth=1.0, color="#00ff00")
        annulus_outer_circle = plt.Circle((x0, y0), annulus_outer_radius, fill=False, linewidth=1.0, color="#ff0000")

        from matplotlib.lines import Line2D

        ax.add_patch(aperture_circle)
        ax.add_patch(annulus_outer_circle)

        legend_handles = [
            Line2D([0], [0], color="#00ff00", lw=2, label="stellar aperture"),
            Line2D([0], [0], color="#ff0000", lw=2, label="background annulus"),
        ]

        ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xlabel("pixels", labelpad=8, fontsize=NIR_LABEL_FONTSIZE)
    ax.set_ylabel("pixels", labelpad=8, fontsize=NIR_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=NIR_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE_NIR, pad = 10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")

    stats_lines = stats_text.splitlines(); 
    if counts_star is not None and counts_star_noise is not None:
        stats_lines.append(f"C_STAR={int(round(counts_star)):,}".replace(",", " ") + " e⁻   " + f"C_STAR_NOISE={int(round(counts_star_noise)):,}".replace(",", " ") + " e⁻")

    n_lines = len(stats_lines)
    y_top = 0.82
    y_bottom = 0.18

    if n_lines == 1:
        y_positions = [0.5]
    else:
        y_positions = np.linspace(y_top, y_bottom, n_lines)

    for line, y in zip(stats_lines, y_positions):
        color = "#ff0000" if (line.startswith("C_STAR=") and not line.startswith("C_STAR_NOISE=") and counts_star < 5000) else "black"
        ax_txt.text(0.5, y, line, ha="center", va="center", fontsize=STATS_FONTSIZE_NIR, color=color, transform=ax_txt.transAxes)
        
    fig.tight_layout()
    fig.savefig(filename, dpi=_FIGURE_NIR_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def save_single_frame_png_NUV(array: np.ndarray, filename: Path, title: str, stats_text: str, y_detector_lo: int | None = None) -> None:
    
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    _imshow_science_array(ax, array, vmin, vmax, y_detector_lo)

    ax.set_xlabel("pixels", labelpad=8)
    ax.set_ylabel("pixels", labelpad=8)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad = 10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
    fig.savefig(filename, dpi=_FIGURE_COMMON_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def save_single_frame_png_VIS(array: np.ndarray, filename: Path, title: str, stats_text: str, y_detector_lo: int | None = None) -> None:
    
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    _imshow_science_array(ax, array, vmin, vmax, y_detector_lo)

    ax.set_xlabel("pixels", labelpad=8)
    ax.set_ylabel("pixels", labelpad=8)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.tight_layout()
    fig.savefig(filename, dpi=_FIGURE_COMMON_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def save_single_frame_png_VIS_cropped(array: np.ndarray, filename: Path, title: str, stats_text: str, y_detector_lo: int | None = None) -> None:
    GAP_IN_VIS = 0.8
    TEXT_H_IN_VIS = 0.28

    ny, nx = array.shape
    img_h_in = max(1.2, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + GAP_IN_VIS + TEXT_H_IN_VIS), facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, GAP_IN_VIS, TEXT_H_IN_VIS], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    _imshow_science_array(ax, array, vmin, vmax, y_detector_lo)

    ax.set_xlabel("pixels", labelpad=6)
    ax.set_ylabel("pixels", labelpad=6)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.08)
    fig.savefig(filename, dpi=_FIGURE_COMMON_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def _calculate_percentile_scales(array: np.ndarray, low_pct: float = 1, high_pct: float = 99.9) -> tuple[float, float]:
    """Robust (vmin, vmax) for image display: percentiles with fallback for non-finite or flat arrays.
    Large arrays are subsampled for percentile computation to reduce runtime."""
    n = array.size
    if n > _PERCENTILE_MAX_PIXELS:
        step = max(1, int(np.ceil(np.sqrt(n / _PERCENTILE_MAX_PIXELS))))
        arr = array[::step, ::step]
    else:
        arr = array
    vmin = float(np.percentile(arr, low_pct))
    vmax = float(np.percentile(arr, high_pct))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax

