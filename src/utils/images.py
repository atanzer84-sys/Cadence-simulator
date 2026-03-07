import logging
import matplotlib.pyplot as plt
from pathlib import Path
from utils.constants import debug_wavelength_range_ir, debug_wavelength_range_nuv, debug_wavelength_range_vis, DEBUG_WL_A_NUV, DEBUG_WL_A_VIS, DEBUG_WL_A_NIR
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import numpy as np
from configs.channel_config import Channel
from domain.star_catalog import StarCatalog

STATS_KEYS = {
    "BIAS":     ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET"],
    "DARK":     ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET", "DARKVAL", "DARKSIG", "EXPTIME"],
    "SCIENCE":  ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET", "DARKVAL", "DARKSIG", "EXPTIME"],
}

STATS_KEY_FORMAT = {
    "RNOISE": "",
    "B_OFFSET": "",
    "MEAN": ".8f",
    "MEDIAN": ".8f",
    "DARKVAL": ".3f",
    "DARKSIG": ".3f",
    "STDDEV": ".8f",
    "MIN": ".5f",
    "MAX": ".5f",
    "EXPTIME": ".2f",
}

# Units for second-line stats (first line MEAN/MEDIAN/STDDEV/MIN/MAX are ADU)
STATS_KEY_UNIT = {
    "RNOISE": "e⁻",
    "B_OFFSET": "e⁻",
    "DARKVAL": "e⁻/s",
    "DARKSIG": "e⁻/s",
    "EXPTIME": "s",
}

# Shared layout constants so single-image and multi-frame PNGs match.
_WIDTH_IN = 10.0
_TEXT_H_IN = 0.7
_GAP_IN = 0.8

def write_image_png(array, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool = True, star: Star | None = None, index: int | None = None) -> None:
    """Write 2D array as PNG. Uses percentile scaling (1–99.9) like write_frames_png. Optional index for multi-frame output (e.g. frame_00042.png)."""
    logging.info("WRITE_IMAGE_PNG called | frame_type=%s | channel=%s | shape=%s | index=%s", frame_type, channel.channel_name, array.shape, index)

    title = _format_frame_title(ctx.target_name, channel.channel_name, frame_type, star)
    stats_values = None
    stats_keys = []
    if show_stats:
        filetype = _infer_stats_filetype(frame_type)
        stats_keys = STATS_KEYS.get(filetype, STATS_KEYS["SCIENCE"])
        stats_values = _stats_from_array_channel(array, channel)

    _write_one_frame_png(array, ctx.output_dir, ctx.target_name, channel.channel_name, frame_type, title, stats_values, stats_keys, index=index, waltzer_prefix=False)


def write_frames_png(frames, headers, frame_type, channel_tag, ctx: RunContext, star: Star, show_stats=False):

    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write PNG: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s PNG frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, ctx.output_dir)

    title_base = _format_frame_title(star.name, channel_tag, frame_type, star)

    for k, (frame, header) in enumerate(zip(frames, headers)):
        stats_values = None
        stats_keys = []
        if show_stats:
            stats_keys = _stats_keys_for_header(header)
            stats_values = _stats_from_header(header, stats_keys)

        _write_one_frame_png(frame, ctx.output_dir, star.name, channel_tag, frame_type, title_base, stats_values, stats_keys, index=k)

    logging.info("Finished writing %d PNG file(s)", n_frames)

def plot_flux_and_photons_windows(wavelengths, values, output_dir, star: Star, filename_tag, title_text, y_label, perChannel: bool = True, full: bool = False, zoom: bool = True):
    print(f"Producing plots for {star.name}")
    logging.info("Producing plots for %s", star.name)

    if perChannel:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV", debug_wavelength_range_nuv[0], debug_wavelength_range_nuv[1])
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS", debug_wavelength_range_vis[0], debug_wavelength_range_vis[1])
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NIR",  debug_wavelength_range_ir[0],  debug_wavelength_range_ir[1])

    if full:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "full", float(wavelengths.min()), float(wavelengths.max()))

    if zoom:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "NUV_zoom", *DEBUG_WL_A_NUV)
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "VIS_zoom", *DEBUG_WL_A_VIS)
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, "IR_zoom",  *DEBUG_WL_A_NIR)


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
        raise ValueError("channel_name must be 'NUV', 'VIS', or 'IR'")

    if full:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, channel_name, full_range[0], full_range[1])

    if zoom:
        plot_photon_flux(wavelengths, values, output_dir, star, filename_tag, title_text, y_label, f"{channel_name}_zoom", zoom_range[0], zoom_range[1])




def _normalize_target_name(name: str) -> str:
    """Normalize target/star name for filenames (spaces → underscores). Used by PNG writes and plots."""
    return str(name).replace(" ", "_")


def _build_png_filename(output_dir: Path, target_name: str, channel_tag: str, frame_type: str, index: int | None = None, *, waltzer_prefix: bool = True) -> Path:
    """Build PNG filename. waltzer_prefix=True: FITS→PNG (WALTzER_...). waltzer_prefix=False: debug PNGs ({target}_{channel}_{frame_type}.png, no WALTzER)."""
    safe = _normalize_target_name(target_name)
    prefix = "WALTzER_" if waltzer_prefix else ""
    if index is not None:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}_{index:05d}.png"
    if waltzer_prefix:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}.png"
    return output_dir / f"{safe}_{channel_tag}_{frame_type}_image.png"


def _write_one_frame_png(array: np.ndarray, output_dir: Path, target_name: str, channel_tag: str, frame_type: str, title: str, stats_values: dict | None, stats_keys: list[str], index: int | None = None, *, waltzer_prefix: bool = True, use_asinh_scale: bool = False) -> None:
    """Write one PNG with unified filename, title, and stats. Used by write_image_png and write_frames_png."""
    filename = _build_png_filename(output_dir, target_name, channel_tag, frame_type, index, waltzer_prefix=waltzer_prefix)
    stats_text = _format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    _save_single_frame_png(array, filename, title, stats_text, use_asinh_scale=use_asinh_scale)

def _format_stats_text(values: dict, keys: list[str]) -> str:
    """Format key=value pairs. Use N/A for missing/None values. Add units for second-line items."""
    pairs = [(k, values.get(k)) for k in keys]
    if not pairs:
        return ""
    parts = []
    for i, (key, val) in enumerate(pairs):
        if val is None:
            s = f"{key}=N/A"
        else:
            fmt = STATS_KEY_FORMAT.get(key, ".2f")
            s = f"{key}={format(val, fmt)}" if fmt else f"{key}={val}"
            unit = STATS_KEY_UNIT.get(key, "")
            if unit:
                s += f" {unit}"
        parts.append(s)
        if (i + 1) % 5 == 0 and i + 1 < len(pairs):
            parts.append("\n")
        elif i + 1 < len(pairs):
            parts.append("    ")
    return "".join(parts)


def _stats_from_array_channel(array: np.ndarray, channel: Channel) -> dict:
    """Build stats dict from array and channel (for write_image_png)."""
    return {
        "MEAN": float(np.mean(array)),
        "MEDIAN": float(np.median(array)),
        "STDDEV": float(np.std(array, ddof=0)),
        "MIN": float(np.min(array)),
        "MAX": float(np.max(array)),
        "RNOISE": getattr(channel, "read_noise", None),
        "B_OFFSET": getattr(channel, "bias_offset", None),
        "DARKVAL": getattr(channel, "dark_noise", None),
        "DARKSIG": getattr(channel, "dark_current_sigma", None),
        "EXPTIME": getattr(channel, "exposure_s", None),
    }


def _stats_from_header(header, keys: list[str]) -> dict:
    """Build stats dict from FITS header. Values not in header become None (filtered later)."""
    return {k: _header_val(header, k) for k in keys}


def _save_single_frame_png(array: np.ndarray, filename: Path, title: str, stats_text: str | None = None, use_asinh_scale: bool = False) -> None:
    """Draw one 2D image with optional stats line; save to filename. Shared by write_image_png and write_frames_png."""
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin = np.percentile(array, 1)
    vmax = np.percentile(array, 99.9)
    # Fallback for sparse/flat images: scaling bounds can collapse
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(array))
        vmax = float(np.max(array))
        if vmax <= vmin:
            vmax = vmin + 1.0
    if use_asinh_scale:
        arr_display = np.arcsinh(array)  # arcsinh(0)=0, no masking; linear for faint, log-like for bright
        vmin = np.percentile(arr_display, 1)
        vmax = np.percentile(arr_display, 93)
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin = float(np.min(arr_display))
            vmax = float(np.max(arr_display))
            if vmax <= vmin:
                vmax = vmin + 1.0
        ax.imshow(arr_display, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    else:
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


def _format_star_meta(star: Star | None) -> str:
    """Format Teff and distance for titles. Returns empty string if star is None."""
    if star is None:
        return ""
    teff_str = f"{int(round(star.effective_temperature))}" if star.effective_temperature is not None else "—"
    dist_str = f"{int(round(star.distance_pc))}" if star.distance_pc is not None else "—"
    return f"$T_{{\\mathrm{{eff}}}}$={teff_str} K, $d$={dist_str} pc"


def _format_frame_title(target_name: str, channel_tag: str, frame_type: str, star: Star | None) -> str:
    """Build title with optional Teff and distance when star is provided."""
    base = f"{target_name}: {channel_tag} {frame_type}"
    meta = _format_star_meta(star)
    return f"{base} | {meta}" if meta else base


def _infer_stats_filetype(frame_type: str) -> str:
    """Infer BIAS/DARK/SCIENCE from frame_type for stats key selection."""
    u = frame_type.upper()
    if "BIAS" in u:
        return "BIAS"
    if "DARK" in u:
        return "DARK"
    return "SCIENCE"


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
    colors = {"nuv": "darkblue", "vis": "darkgreen", "nir": "darkred"}
    color = colors.get(band, "black")

    ax.plot(wl, flux, color=color, linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel(y_label)
    meta = _format_star_meta(star)
    ax.set_title(f"{star.name}: {title_text} | {wmin:.2f}–{wmax:.2f} Å, {meta}", fontsize=11)
    safe_name = _normalize_target_name(star.name)
    fig.savefig(Path(output_dir) / f"{safe_name}_{filename_tag}_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def write_background_star_visibility_tests(merged_image: np.ndarray, spectra_bgstars_image: np.ndarray, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool = True, star: Star | None = None, index: int | None = None) -> None:
    """Write one 4-panel diagnostic PNG: merged percentile, background-stars-only, merged plus bg-star footprint, merged asinh."""
    logging.info("WRITE_BACKGROUND_STAR_VISIBILITY_TESTS called | frame_type=%s | channel=%s | merged_shape=%s | spectra_bgstars_shape=%s | index=%s", frame_type, channel.channel_name, merged_image.shape, spectra_bgstars_image.shape, index)
    title = _format_frame_title(ctx.target_name, channel.channel_name, frame_type, star)
    stats_values = None
    stats_keys = []
    if show_stats:
        filetype = _infer_stats_filetype(frame_type)
        stats_keys = STATS_KEYS.get(filetype, STATS_KEYS["SCIENCE"])
        stats_values = _stats_from_array_channel(merged_image, channel)
    _write_background_star_visibility_panel_png(merged_image, spectra_bgstars_image, ctx.output_dir, ctx.target_name, channel.channel_name, f"{frame_type}_PANEL", title, stats_values, stats_keys, index=index)

def _write_background_star_visibility_panel_png(merged_image: np.ndarray, spectra_bgstars_image: np.ndarray, output_dir: Path, target_name: str, channel_tag: str, frame_type: str, title: str, stats_values: dict | None, stats_keys: list[str], index: int | None = None) -> None:
    filename = _build_png_filename(output_dir, target_name, channel_tag, frame_type, index=index, waltzer_prefix=False)
    ny, nx = merged_image.shape
    img_h_in_single = max(2.0, _WIDTH_IN * (ny / nx))
    panel_h_in = max(6.5, img_h_in_single * 1.9)
    stats_text = _format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    fig = plt.figure(figsize=(2.0 * _WIDTH_IN, panel_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, _TEXT_H_IN / panel_h_in], hspace=0.22, wspace=0.12)
    merged_vmin = np.percentile(merged_image, 1)
    merged_vmax = np.percentile(merged_image, 99.9)
    if (not np.isfinite(merged_vmin)) or (not np.isfinite(merged_vmax)) or (merged_vmax <= merged_vmin):
        merged_vmin = float(np.min(merged_image))
        merged_vmax = float(np.max(merged_image))
        if merged_vmax <= merged_vmin:
            merged_vmax = merged_vmin + 1.0
    bg_arr_display = np.arcsinh(spectra_bgstars_image)
    bg_positive = bg_arr_display[spectra_bgstars_image > 0]
    if bg_positive.size > 0:
        bg_vmin = np.percentile(bg_positive, 1.0)
        bg_vmax = np.percentile(bg_positive, 99.0)
    else:
        bg_vmin = 0.0
        bg_vmax = 1.0
    display_floor = np.median(merged_image)
    merged_asinh_input = np.clip(merged_image - display_floor, 0.0, None)
    merged_asinh = np.arcsinh(merged_asinh_input)
    merged_asinh_positive = merged_asinh[merged_asinh_input > 0]
    if merged_asinh_positive.size > 0:
        merged_asinh_vmin = np.percentile(merged_asinh_positive, 1.0)
        merged_asinh_vmax = np.percentile(merged_asinh_positive, 99.5)
    else:
        merged_asinh_vmin = 0.0
        merged_asinh_vmax = 1.0
    mask = spectra_bgstars_image > 0.0
    has_bg = np.any(mask)
    mask_dilated = mask.copy()
    mask_dilated[1:, :] |= mask[:-1, :]
    mask_dilated[:-1, :] |= mask[1:, :]
    mask_dilated[:, 1:] |= mask[:, :-1]
    mask_dilated[:, :-1] |= mask[:, 1:]
    overlay = np.where(mask_dilated, 1.0, np.nan)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(merged_image, origin="lower", aspect="equal", cmap="gray", vmin=merged_vmin, vmax=merged_vmax)
    ax1.set_xlim(-0.5, nx - 0.5)
    ax1.set_ylim(-0.5, ny - 0.5)
    ax1.set_xlabel("pixels", labelpad=8)
    ax1.set_ylabel("pixels", labelpad=8)
    ax1.set_title("Merged percentile", fontsize=11)
    ax2 = fig.add_subplot(gs[0, 1])

    if has_bg:
        ax2.imshow(bg_arr_display, origin="lower", aspect="equal", cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
    else:
        ax2.text(0.5, 0.5, "No background stars in this frame", ha="center", va="center", fontsize=12, transform=ax2.transAxes)

    ax2.set_xlim(-0.5, nx - 0.5)
    ax2.set_ylim(-0.5, ny - 0.5)
    ax2.set_xlabel("pixels", labelpad=8)
    ax2.set_ylabel("pixels", labelpad=8)
    ax2.set_title("Background stars only", fontsize=11)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(merged_image, origin="lower", aspect="equal", cmap="gray", vmin=merged_vmin, vmax=merged_vmax)
    
    
    if np.any(mask_dilated):
        ax3.imshow(overlay, origin="lower", aspect="equal", cmap="Reds", alpha=0.08, interpolation="nearest")
        ax3.contour(mask_dilated.astype(float), levels=[0.5], colors="red", linewidths=0.45, alpha=0.5)
    
    ax3.set_xlim(-0.5, nx - 0.5)
    ax3.set_ylim(-0.5, ny - 0.5)
    ax3.set_xlabel("pixels", labelpad=8)
    ax3.set_ylabel("pixels", labelpad=8)
    ax3.set_title("Merged + bg star footprint", fontsize=11)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(merged_asinh, origin="lower", aspect="equal", cmap="gray", vmin=merged_asinh_vmin, vmax=merged_asinh_vmax)
    if np.any(mask_dilated):
        ax4.imshow(overlay, origin="lower", aspect="equal", cmap="Reds", alpha=0.15, interpolation="nearest")
        ax4.contour(mask_dilated.astype(float), levels=[0.5], colors="red", linewidths=0.5, alpha=0.6)
    ax4.set_xlim(-0.5, nx - 0.5)
    ax4.set_ylim(-0.5, ny - 0.5)
    ax4.set_xlabel("pixels", labelpad=8)
    ax4.set_ylabel("pixels", labelpad=8)
    ax4.set_title("Merged asinh", fontsize=11)


    fig.suptitle(title, fontsize=13)
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    if stats_text:
        ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=10, transform=ax_txt.transAxes)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.debug("Wrote %s", filename)







def plot_background_star_counts(background_stars_catalog: StarCatalog, channel: Channel, ctx: RunContext):

    wavelength = channel.effective_area_wavelength
    stars_sorted = sorted(background_stars_catalog.stars_by_id.items(), key=lambda item: item[1].gaia_magnitude)
    total = len(stars_sorted)
    safe_target = _normalize_target_name(ctx.target_name)

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