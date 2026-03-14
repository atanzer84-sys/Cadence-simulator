"""Diagnostic plots: star counts vs read noise, dark current and bias for spectroscopy and photometry."""
import logging
import numpy as np
import matplotlib.pyplot as plt

from configs.channel_config import Channel, PhotometryChannel
from domain.star import Star
from loaders.run_context import RunContext
from utils.images_common import normalize_target_name, format_frame_title

_COUNTS_VS_NOISE_DPI = 100
_COUNTS_VS_NOISE_FIGSIZE = (12, 5)  # inches; output pixels = figsize * dpi → 1200×500
_COUNTS_VS_NOISE_BBOX = "tight"
_FONTSIZE_LABEL = 16
_FONTSIZE_TICK = 14
_FONTSIZE_TITLE = 16
_FONTSIZE_LEGEND = 14


def plot_star_counts_vs_noise_spectroscopy(wavelength: np.ndarray, counts_s_px: np.ndarray, channel: Channel, ctx: RunContext, *, filename_tag: str = "target_star_counts_vs_noise", already_per_second: bool = False, star: Star | None = None) -> None:
    """Plot peak counts per wavelength vs noise levels for one spectroscopy channel; save PNG to ctx.output_dir."""
    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)

    counts_exp_px = counts_s_px * channel.exposure_s

    if counts_exp_px.size != wavelength.size:
        counts_exp_px = np.full_like(wavelength, float(np.squeeze(counts_exp_px)), dtype=float)

    plt.plot(wavelength, counts_exp_px, color="darkblue", linewidth=0.6, alpha=0.8, label="Peak pixel count")

    title = format_frame_title(ctx.target_name, channel.channel_name, f"Peak Pixel Counts ({channel.exposure_s}s)", star)
    _style_counts_vs_noise_figure(channel, title)
    plt.xlabel("Wavelength [Å]", fontsize=_FONTSIZE_LABEL)
    filename = ctx.output_dir / f"{safe_target}_{filename_tag}_{channel.channel_name}.png"
    plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
    plt.close()
    logging.debug("Wrote %s", filename)


def plot_star_counts_vs_noise_photometry(rate_image_e_s: np.ndarray, channel: PhotometryChannel, ctx: RunContext, *, filename_tag: str = "target_star_counts_vs_noise", star: Star | None = None) -> None:
    """Plot peak pixel count vs noise levels for one photometry channel; save PNG to ctx.output_dir."""
    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)
    peak_pixel_e = np.float32(np.max(rate_image_e_s) * channel.exposure_s)
    x = np.array([0.0], dtype=np.float32)

    plt.scatter(x, [peak_pixel_e], color="darkblue", s=40, label="Peak pixel count", zorder=3)
    
    title = format_frame_title(ctx.target_name, channel.channel_name, f"Peak Pixel Counts ({channel.exposure_s}s)", star)
    _style_counts_vs_noise_figure(channel, title)
    plt.xlabel("Photometric band", fontsize=_FONTSIZE_LABEL)
    plt.xlim(-0.5, 0.5)
    plt.xticks(x, [channel.channel_name])
    filename = ctx.output_dir / f"{safe_target}_{filename_tag}_{channel.channel_name}.png"
    plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
    plt.close()


def _style_counts_vs_noise_figure(channel: Channel, title: str) -> None:
    """Shared axes, labels, noise reference lines and title for counts-vs-noise diagnostic plots. Dark and bias are separate (dark is not dark+bias)."""
    plt.axhline(channel.read_noise, linestyle="--", color="black", label=f"Read noise={channel.read_noise:g} e⁻")
    plt.axhline(channel.bias_offset, linestyle="-", color="gray", label=f"Bias={channel.bias_offset:g} e⁻")
    dark_per_exp = channel.dark_noise * channel.exposure_s
    plt.axhline(dark_per_exp, linestyle=":", color="black", label=f"Dark={channel.dark_noise:g} e⁻/s ({dark_per_exp:g} e⁻)")
    plt.yscale("log")
    plt.ylabel("Counts (exposure⁻¹ pixel⁻¹)", fontsize=_FONTSIZE_LABEL)
    plt.title(title, fontsize=_FONTSIZE_TITLE)
    plt.legend(fontsize=_FONTSIZE_LEGEND)
    plt.gca().tick_params(axis="both", labelsize=_FONTSIZE_TICK)
