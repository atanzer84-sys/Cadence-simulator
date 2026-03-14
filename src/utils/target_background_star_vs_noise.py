import logging
import numpy as np
import matplotlib.pyplot as plt

from configs.channel_config import Channel, PhotometryChannel
from domain.star_catalog import StarCatalog
from loaders.run_context import RunContext
from utils.images_common import normalize_target_name

_COUNTS_VS_NOISE_DPI = 100
_COUNTS_VS_NOISE_FIGSIZE = (12, 5)  # (width_in, height_in); pixel size = figsize * dpi
_COUNTS_VS_NOISE_BBOX = "tight"


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



def plot_target_star_counts_vs_noise_photometry(rate_image_e_s: np.ndarray, channel: PhotometryChannel, ctx: RunContext, *, title: str = "Target star peak pixel vs noise", filename_tag: str = "target_star_counts_vs_noise") -> None:

    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)

    peak_pixel_e = float(np.max(rate_image_e_s) * channel.exposure_s)

    x = np.array([0.0])
    labels = [channel.channel_name]

    plt.scatter(x, [peak_pixel_e], color="darkblue", s=40, label="Target peak pixel", zorder=3)

    read_noise_e = float(channel.read_noise)
    bias_e = float(channel.bias_offset)
    dark_e = float(channel.dark_noise * channel.exposure_s)

    plt.axhline(read_noise_e, color="red", linewidth=0.8, linestyle="--", label="Read noise")
    plt.axhline(bias_e, color="green", linewidth=0.8, linestyle="--", label="Bias")
    plt.axhline(dark_e, color="orange", linewidth=0.8, linestyle="--", label="Dark")

    plt.xticks(x, labels)
    plt.xlabel("Photometric band")
    plt.ylabel("Electrons / pixel / exposure")
    plt.title(f"{title} ({channel.channel_name}, {channel.exposure_s}s)")
    plt.grid(alpha=0.25)
    plt.legend()

    filename = ctx.output_dir / f"{safe_target}_{filename_tag}_{channel.channel_name}.png"
    plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
    plt.close()

    
def plot_target_star_counts_vs_noise_spectroscopy(wavelength: np.ndarray, counts_s_px: np.ndarray, channel: Channel, ctx: RunContext, *, title: str = "Counts vs noise", filename_tag: str = "target_star_counts_vs_noise", already_per_second: bool = False) -> None:
    
    safe_target = normalize_target_name(ctx.target_name)
    plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)

    counts_exp_px = counts_s_px * channel.exposure_s

    if counts_exp_px.size != wavelength.size:
        counts_exp_px = np.full_like(wavelength, float(np.squeeze(counts_exp_px)), dtype=float)

    plt.plot(wavelength, counts_exp_px, color="darkblue", linewidth=0.6, alpha=0.8, label="Counts")

    _style_counts_vs_noise_figure(channel, f"{title} ({channel.channel_name}, {channel.exposure_s}s)")
    filename = ctx.output_dir / f"{safe_target}_{filename_tag}_{channel.channel_name}.png"
    plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
    plt.close()
    logging.debug("Wrote %s", filename)

def plot_background_star_counts(background_stars_catalog: StarCatalog, channel: Channel | None, ctx: RunContext):

    if channel is None:
        return

    wavelength = channel.effective_area_wavelength
    stars_sorted = sorted(background_stars_catalog.stars_by_id.items(), key=lambda item: item[1].gaia_magnitude)
    total = len(stars_sorted)
    safe_target = normalize_target_name(ctx.target_name)

    for start in range(0, total, 5):

        subset = stars_sorted[start:start + 5]

        plt.figure(figsize=_COUNTS_VS_NOISE_FIGSIZE)

        if isinstance(channel, PhotometryChannel) and channel.psf_image is not None:
            psf_sum = float(np.sum(channel.psf_image))
            peak_fraction = (float(np.max(channel.psf_image)) / psf_sum) if psf_sum > 0 else 0.0
        else:
            peak_fraction = None

        for star_id, bg_star in subset:
            counts_s_px = background_stars_catalog.counts_by_id_and_band[(star_id, channel.channel_name)]
            total_flux_exposure = float(np.asarray(counts_s_px).sum() * channel.exposure_s)
            if isinstance(channel, PhotometryChannel) and peak_fraction is not None:
                expected_peak_e = total_flux_exposure * peak_fraction
                counts_exp = np.full_like(wavelength, expected_peak_e)
            else:
                counts_exp = np.atleast_1d(np.asarray(counts_s_px) * channel.exposure_s)
                if counts_exp.size != wavelength.size:
                    counts_exp = np.full_like(wavelength, float(np.squeeze(counts_exp)))

            label = f"G={bg_star.gaia_magnitude:.2f} (name: {bg_star.name})"
            plt.plot(wavelength, counts_exp, label=label, linewidth=0.6, alpha=0.8)

        plot_title = f"{ctx.target_name}: Background stars vs noise ({channel.channel_name}, {channel.exposure_s}s)"
        _style_counts_vs_noise_figure(channel, plot_title)

        filename = ctx.output_dir / f"{safe_target}_background_stars_{channel.channel_name}_{start}.png"
        plt.savefig(filename, dpi=_COUNTS_VS_NOISE_DPI, bbox_inches=_COUNTS_VS_NOISE_BBOX)
        plt.close()
    
    logging.info("Background star count plots finished: channel=%s", channel.channel_name)