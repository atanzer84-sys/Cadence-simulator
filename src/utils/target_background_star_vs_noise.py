import logging
import numpy as np
import matplotlib.pyplot as plt

from configs.channel_config import Channel, PhotometryChannel
from domain.star import Star
from loaders.run_context import RunContext
from utils.images_common import normalize_target_name

_COUNTS_VS_NOISE_DPI = 100
_COUNTS_VS_NOISE_FIGSIZE = (12, 5)  # (width_in, height_in); pixel size = figsize * dpi
_COUNTS_VS_NOISE_BBOX = "tight"


def _title_for_channel(star: Star | None, channel: Channel) -> str:
    """Build plot title: star.name: Magnitude - Peak Pixel Counts (Channel, Exposure)."""
    suffix = f"Peak Pixel Counts ({channel.channel_name}, {channel.exposure_s}s)"
    if star is None:
        return suffix
    if star.gaia_magnitude is not None:
        return f"{star.name}: {star.gaia_magnitude:.2f} - {suffix}"
    return f"{star.name} - {suffix}"


def plot_star_counts_vs_noise_spectroscopy(wavelength: np.ndarray, counts_s_px: np.ndarray, channel: Channel, ctx: RunContext, *, filename_tag: str = "target_star_counts_vs_noise", already_per_second: bool = False, star: Star | None = None) -> None:
    
    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)

    counts_exp_px = counts_s_px * channel.exposure_s

    if counts_exp_px.size != wavelength.size:
        counts_exp_px = np.full_like(wavelength, float(np.squeeze(counts_exp_px)), dtype=float)

    plt.plot(wavelength, counts_exp_px, color="darkblue", linewidth=0.6, alpha=0.8, label="Counts")

    _style_counts_vs_noise_figure(channel, _title_for_channel(star, channel))
    filename = ctx.output_dir / f"{safe_target}_{filename_tag}_{channel.channel_name}.png"
    plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
    plt.close()
    logging.debug("Wrote %s", filename)


def plot_star_counts_vs_noise_photometry(rate_image_e_s: np.ndarray, channel: PhotometryChannel, ctx: RunContext, *, filename_tag: str = "target_star_counts_vs_noise", star: Star | None = None) -> None:
    
    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)
    peak_pixel_e = np.float32(np.max(rate_image_e_s) * channel.exposure_s)
    x = np.array([0.0], dtype=np.float32)

    plt.scatter(x, [peak_pixel_e], color="darkblue", s=40, label="Target peak pixel", zorder=3)
    plt.axhline(channel.read_noise, linestyle="--", color="black", label=f"Read noise={channel.read_noise:g} e⁻")
    plt.axhline(channel.bias_offset, linestyle="-", color="gray", label=f"Bias={channel.bias_offset:g} e⁻")
    dark_per_exp = channel.dark_noise * channel.exposure_s
    plt.axhline(dark_per_exp, linestyle=":", color="black", label=f"Dark={channel.dark_noise:g} e⁻/s ({dark_per_exp:g} e⁻)")
    plt.yscale("log")
    plt.xlim(-0.5, 0.5)
    plt.xticks(x, [channel.channel_name])
    plt.xlabel("Photometric band")
    plt.ylabel("Counts (exposure⁻¹ pixel⁻¹)")
    plt.title(_title_for_channel(star, channel))
    plt.legend()
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
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Counts (exposure⁻¹ pixel⁻¹)")
    plt.title(title)
    plt.legend()
