import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import numpy as np
from configs.channel_config import Channel, PhotometryChannel
from matplotlib.lines import Line2D
from utils.images_common import normalize_target_name, format_frame_title

STATS_KEYS = {
    "BIAS":     ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET"],
    "DARK":     ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET", "DARKVAL", "DARKSIG", "EXPTIME"],
    "SCIENCE":  ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET", "DARKVAL", "DARKSIG", "EXPTIME"],
    "BG_STARS": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "EXPTIME"],
}

_FRAME_TYPE_TO_STATS_FILETYPE = (("BIAS", "BIAS"), ("DARK", "DARK"))
_DEFAULT_FRAME_STATS_FILETYPE = "SCIENCE"

STATS_KEY_FORMAT = {
    "RNOISE": "",
    "B_OFFSET": "",
    "MEAN": ".2f",
    "MEDIAN": ".2f",
    "DARKVAL": ".2f",
    "DARKSIG": ".2f",
    "STDDEV": ".2f",
    "MIN": ".2f",
    "MAX": ".2f",
    "EXPTIME": ".2f",
}

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




def generate_background_star_visibility_on_science_frame(merged_image: np.ndarray, spectra_bgstars_image: np.ndarray, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool = True, star: Star | None = None, index: int | None = None, background_star_bands: dict[str, dict[str, float]] | None = None, background_star_arcs: dict[str, list[tuple[int, int]]] | None = None, inverted: bool = False) -> None:

    title, per_panel_stats, layout = _build_background_visibility_context(merged_image, spectra_bgstars_image, frame_type, ctx, channel, show_stats, star)
    filename = _build_png_filename(ctx.output_dir, ctx.target_name, channel.channel_name, f"{frame_type}_PANEL", index=index, waltzer_prefix=True)
    ny, nx = merged_image.shape
    fig = plt.figure(figsize=(layout["fig_w"], layout["fig_h"]))
    gs = fig.add_gridspec(nrows=11, ncols=1, height_ratios=layout["height_ratios"], hspace=0.0)
    stats_fontsize = 9

    merged_display = (merged_image.max() - merged_image) if inverted else merged_image
    merged_vmin, merged_vmax = _compute_display_range(merged_display)
    bg_arr_display, bg_vmin, bg_vmax = _compute_bg_display_range(spectra_bgstars_image)
    if inverted:
        bg_arr_display = bg_arr_display.max() - bg_arr_display
        bg_vmin, bg_vmax = _compute_display_range(bg_arr_display)

    # mask_dilated, overlay, has_bg, use_band_overlay = _compute_bg_mask_overlay(spectra_bgstars_image, background_star_bands)
    mask_dilated, overlay, has_bg, use_band_overlay = _compute_bg_mask_overlay(spectra_bgstars_image, channel, background_star_bands, background_star_arcs)
    # Panel 1 : Science

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(merged_display, origin="lower", aspect="equal", cmap="gray", vmin=merged_vmin, vmax=merged_vmax)
    _style_detector_panel_axis(ax1, nx, ny, "Science Frame")

    # Panel 2 : Background Stars

    ax2 = fig.add_subplot(gs[4, 0])
    _background_star_axis(ax2, bg_arr_display, has_bg, bg_vmin, bg_vmax, nx, ny)

    # Panel 3 : Science + Background Stars
    # _science_frame_overlay_background_star_axis(fig, gs, merged_image, merged_vmin, merged_vmax, use_band_overlay, background_star_bands, ny, nx, mask_dilated, overlay)
    _science_frame_overlay_background_star_axis(fig, gs, merged_display, merged_vmin, merged_vmax, use_band_overlay, background_star_bands, background_star_arcs, ny, nx, mask_dilated, overlay)
    _render_panel_stats_rows(fig, gs, per_panel_stats, stats_fontsize)

    fig.suptitle(title, fontsize=12, y=0.90)
    fig.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logging.debug("Wrote %s", filename)






def _build_png_filename(output_dir: Path, target_name: str, channel_tag: str, frame_type: str, index: int | None = None, *, waltzer_prefix: bool = True) -> Path:
    """Build PNG filename. waltzer_prefix=True: FITS→PNG (WALTzER_...). waltzer_prefix=False: debug PNGs ({target}_{channel}_{frame_type}.png, no WALTzER)."""
    safe = normalize_target_name(target_name)
    prefix = "WALTzER_" if waltzer_prefix else ""
    if index is not None:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}_{index:05d}.png"
    if waltzer_prefix:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}.png"
    return output_dir / f"{safe}_{channel_tag}_{frame_type}_image.png"


def write_one_frame_png(array: np.ndarray, output_dir: Path, target_name: str, channel_tag: str, frame_type: str, title: str, stats_values: dict | None, stats_keys: list[str], index: int | None = None, *, waltzer_prefix: bool = True, axis_label_fontsize: int | None = None, tick_label_fontsize: int | None = None) -> None:
    """Write one PNG with unified filename, title, and stats. Used by write_calibration_frame_png and write_science_frames_png."""
    filename = _build_png_filename(output_dir, target_name, channel_tag, frame_type, index, waltzer_prefix=waltzer_prefix)
    stats_text = _format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    _save_single_frame_png(array, filename, title, stats_text, axis_label_fontsize=axis_label_fontsize, tick_label_fontsize=tick_label_fontsize)

def _format_stats_text(values: dict, keys: list[str], *, use_scientific_for_small: bool = False) -> str:
    """Format key=value pairs. Use N/A for missing/None values. Add units for second-line items.
    If use_scientific_for_small, use scientific notation (e.g. 1.23e-08) for |val| in (0, 1e-4) or |val| >= 1e6."""
    pairs = [(k, values.get(k)) for k in keys]
    if not pairs:
        return ""
    parts = []
    for i, (key, val) in enumerate(pairs):
        if val is None:
            s = f"{key}=N/A"
        else:
            fmt = STATS_KEY_FORMAT.get(key, ".2f")
            if use_scientific_for_small and isinstance(val, (int, float)) and val != 0:
                if abs(val) < 1e-4 or abs(val) >= 1e6:
                    fmt = ".2e"
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


def stats_from_array_channel(array: np.ndarray, channel: Channel) -> dict:
    """Build stats dict from array and channel (for write_calibration_frame_png)."""
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


def _stats_from_array_only(array: np.ndarray) -> dict:
    """Build stats dict from array only (no instrument params). For spectra/bg-star-only images."""
    return {
        "MEAN": float(np.mean(array)),
        "MEDIAN": float(np.median(array)),
        "STDDEV": float(np.std(array, ddof=0)),
        "MIN": float(np.min(array)),
        "MAX": float(np.max(array)),
    }




def _save_single_frame_png(array: np.ndarray, filename: Path, title: str, stats_text: str | None = None, axis_label_fontsize: int | None = None, tick_label_fontsize: int | None = None) -> None:
    """Draw one 2D image with optional stats line; save to filename. Shared by write_calibration_frame_png and write_science_frames_png."""
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
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    if axis_label_fontsize is None:
        ax.set_xlabel("pixels", labelpad=8)
        ax.set_ylabel("pixels", labelpad=8)
    else:
        ax.set_xlabel("pixels", labelpad=8, fontsize=axis_label_fontsize)
        ax.set_ylabel("pixels", labelpad=8, fontsize=axis_label_fontsize)
    if tick_label_fontsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)
    ax.set_title(title, fontsize=11)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    if stats_text:
        ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=10, transform=ax_txt.transAxes)

    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.debug("Wrote %s", filename)


def stats_keys_for_filetype(filetype: str, default_filetype: str = _DEFAULT_FRAME_STATS_FILETYPE) -> list[str]:
    return STATS_KEYS.get(filetype, STATS_KEYS[default_filetype])


def _build_background_visibility_context(merged_image: np.ndarray, spectra_bgstars_image: np.ndarray, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool, star: Star | None) -> tuple[str, list[tuple[dict | None, list[str]]], dict[str, float | list[float]]]:
    title = format_frame_title(ctx.target_name, channel.channel_name, frame_type, star)
    per_panel_stats = _prepare_background_star_panel_stats(merged_image, spectra_bgstars_image, channel, show_stats)
    panel_shapes = [merged_image.shape, spectra_bgstars_image.shape, merged_image.shape]
    layout = _compute_panel_layout(panel_shapes)
    return title, per_panel_stats, layout

def _style_detector_panel_axis(ax: Any, nx: int, ny: int, title: str) -> None:
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel("pixels", labelpad=10)
    ax.set_ylabel("pixels", labelpad=10)
    ax.set_title(title, fontsize=11)

def _render_panel_stats_rows(fig: Any, gs: Any, per_panel_stats: list[tuple[dict | None, list[str]]], stats_fontsize: int) -> None:
    for i, (vals, keys) in enumerate(per_panel_stats):
        if not keys:
            continue
        if vals is None:
            vals = {k: None for k in keys}
        use_sci = keys == STATS_KEYS["BG_STARS"]
        stats_text = _format_stats_text(vals, keys, use_scientific_for_small=use_sci)
        stat_row = 2 + 4 * i
        ax_stat = fig.add_subplot(gs[stat_row, 0])
        ax_stat.axis("off")
        ax_stat.set_facecolor("white")
        ax_stat.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=stats_fontsize, transform=ax_stat.transAxes)

def _background_star_axis(ax: Any, bg_arr_display: np.ndarray, has_bg: bool, bg_vmin: float, bg_vmax: float, nx: int, ny: int) -> None:
    if has_bg:
        ax.imshow(bg_arr_display, origin="lower", aspect="equal", cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
    else:
        blank = np.ones((ny, nx))
        ax.imshow(blank, origin="lower", aspect="equal", cmap="gray", vmin=0.0, vmax=1.0)
        ax.text(0.5, 0.5, "No background stars in this frame", ha="center", va="center", fontsize=12, color="black", transform=ax.transAxes)
    _style_detector_panel_axis(ax, nx, ny, "Background Stars")

def _science_frame_overlay_background_star_axis(fig: Any, gs: Any, merged_image: np.ndarray, merged_vmin: float, merged_vmax: float, use_band_overlay: bool, background_star_bands: dict[str, dict[str, float]] | None, background_star_arcs: dict[str, list[tuple[int, int]]] | None, ny: int, nx: int, mask_dilated: np.ndarray, overlay: np.ndarray) -> None:
    ax3 = fig.add_subplot(gs[8, 0])
    ax3.imshow(merged_image, origin="lower", aspect="equal", cmap="gray", vmin=merged_vmin, vmax=merged_vmax)
    if use_band_overlay:
        for band in background_star_bands.values():
            y0 = float(band["y0"])
            sigma = float(band["sigma"])
            # y_m2 = y0 - 2.0 * sigma
            # y_p2 = y0 + 2.0 * sigma
            y_m3 = y0 - 3.0 * sigma
            y_p3 = y0 + 3.0 * sigma
            # if 0.0 <= y_m2 <= ny - 1:
            #     ax3.axhline(y_m2, color="red", linewidth=0.6, alpha=0.95)
            # if 0.0 <= y_p2 <= ny - 1:
            #     ax3.axhline(y_p2, color="red", linewidth=0.6, alpha=0.95)
            if 0.0 <= y_m3 <= ny - 1:
                ax3.axhline(y_m3, color="#00FF66", linewidth=0.3, alpha=0.95)
            if 0.0 <= y_p3 <= ny - 1:
                ax3.axhline(y_p3, color="#00FF66", linewidth=0.3, alpha=0.95)
        # legend_lines = [Line2D([0], [0], color="#00FF66", lw=1.2, label="Background spectra ±2σ (cross-dispersion)"), Line2D([0], [0], color="red", lw=1.2, label="Background spectra ±3σ (cross-dispersion)")]
        legend_lines = [Line2D([0], [0], color="#00FF66", lw=1.2, label="Background spectra ±3σ (cross-dispersion)")]
        ax3.legend(handles=legend_lines, loc="upper right", fontsize=8, frameon=True)
    elif background_star_arcs is not None and len(background_star_arcs) > 0:
        if np.any(mask_dilated):
            ax3.imshow(overlay, origin="lower", aspect="equal", cmap="Reds", alpha=0.2, interpolation="nearest")
            ax3.contour(mask_dilated.astype(float), levels=[0.5], colors="#00FF66", linewidths=0.6, alpha=0.95)
        legend_lines = [Line2D([0], [0], color="#00FF66", lw=1.2, label="Background Star Arc (90% Flux)")]
        ax3.legend(handles=legend_lines, loc="upper right", fontsize=8, frameon=True)
    else:
        if np.any(mask_dilated):
            ax3.imshow(overlay, origin="lower", aspect="equal", cmap="Reds", alpha=0.2, interpolation="nearest")
            ax3.contour(mask_dilated.astype(float), levels=[0.5], colors="#00FF66", linewidths=0.6, alpha=0.95)
    _style_detector_panel_axis(ax3, nx, ny, "Science Frame with Background Stars Footprint")

def _compute_display_range(image: np.ndarray) -> tuple[float, float]:
    vmin = np.percentile(image, 1.0)
    vmax = np.percentile(image, 99.9)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)

def _compute_bg_display_range(spectra_bgstars_image: np.ndarray) -> tuple[np.ndarray, float, float]:
    bg_arr_display = np.arcsinh(spectra_bgstars_image)
    bg_positive = bg_arr_display[spectra_bgstars_image > 0.0]
    if bg_positive.size > 0:
        bg_vmin = float(np.percentile(bg_positive, 1.0))
        bg_vmax = float(np.percentile(bg_positive, 99.0))
    else:
        bg_vmin = 0.0
        bg_vmax = 1.0
    return bg_arr_display, bg_vmin, bg_vmax

def _compute_bg_mask_overlay(spectra_bgstars_image: np.ndarray, channel: Channel, background_star_bands: dict[str, dict[str, float]] | None, background_star_arcs: dict[str, list[tuple[int, int]]] | None) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    mask = spectra_bgstars_image > 0.0
    has_bg = bool(np.any(mask))
    use_band_overlay = background_star_bands is not None and len(background_star_bands) > 0
    if background_star_arcs is not None and len(background_star_arcs) > 0:
        r_eff_px = _compute_psf_r90_px(channel)
        ny, nx = spectra_bgstars_image.shape
        mask_dilated = _build_arc_overlay_mask(ny, nx, background_star_arcs, r_eff_px)
        overlay = np.where(mask_dilated, 1.0, np.nan)
        return mask_dilated, overlay, has_bg, use_band_overlay

    mask_dilated = mask.copy()
    mask_dilated[1:, :] |= mask[:-1, :]
    mask_dilated[:-1, :] |= mask[1:, :]
    mask_dilated[:, 1:] |= mask[:, :-1]
    mask_dilated[:, :-1] |= mask[:, 1:]
    overlay = np.where(mask_dilated, 1.0, np.nan)
    return mask_dilated, overlay, has_bg, use_band_overlay

def _build_arc_overlay_mask(ny: int, nx: int, background_star_arcs: dict[str, list[tuple[int, int]]], r_eff_px: float) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    r_int = int(np.ceil(r_eff_px))
    r_sq = float(r_eff_px) * float(r_eff_px)

    for positions in background_star_arcs.values():
        for x0, y0 in positions:
            x_min = max(0, x0 - r_int)
            x_max = min(nx - 1, x0 + r_int)
            y_min = max(0, y0 - r_int)
            y_max = min(ny - 1, y0 + r_int)

            yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
            local_mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_sq
            mask[y_min:y_max + 1, x_min:x_max + 1] |= local_mask

    return mask

def _compute_panel_layout(panel_shapes: list[tuple[int, int]]) -> dict[str, float | list[float]]:
    max_nx = max(nx for _, nx in panel_shapes)
    panel_w_in = max(8.0, min(10.0, 0.1 * max_nx))
    panel_heights = [panel_w_in * (ny / nx) for ny, nx in panel_shapes]
    gap_panel_to_stats_h = 0.3
    stats_h = 0.3
    gap_between_panels_h = 0.3
    title_h = 3.5
    fig_w = panel_w_in
    fig_h = sum(panel_heights) + 3.0 * stats_h + 3.0 * gap_panel_to_stats_h + 2.0 * gap_between_panels_h + title_h
    height_ratios = [
        panel_heights[0], gap_panel_to_stats_h, stats_h, gap_between_panels_h,
        panel_heights[1], gap_panel_to_stats_h, stats_h, gap_between_panels_h,
        panel_heights[2], gap_panel_to_stats_h, stats_h,
    ]
    return {
        "panel_w_in": panel_w_in,
        "panel_heights_in": panel_heights,
        "stats_h": stats_h,
        "gap_panel_to_stats_h": gap_panel_to_stats_h,
        "gap_between_panels_h": gap_between_panels_h,
        "title_h": title_h,
        "fig_w": fig_w,
        "fig_h": fig_h,
        "height_ratios": height_ratios,
    }

def _prepare_background_star_panel_stats(merged_image: np.ndarray, spectra_bgstars_image: np.ndarray, channel: Channel, show_stats: bool) -> list[tuple[dict | None, list[str]]]:

    per_panel_stats: list[tuple[dict | None, list[str]]] = [(None, []), (None, []), (None, [])]
    if not show_stats:
        return per_panel_stats

    per_panel_stats[0] = _build_channel_panel_stats_row(merged_image, channel)
    per_panel_stats[1] = _build_bg_star_panel_stats_row(spectra_bgstars_image, channel)
    per_panel_stats[2] = _build_channel_panel_stats_row(merged_image, channel)
    return per_panel_stats

def _build_channel_panel_stats_row(array: np.ndarray, channel: Channel) -> tuple[dict, list[str]]:
    science_keys = stats_keys_for_filetype("SCIENCE")
    return stats_from_array_channel(array, channel), science_keys

def _build_bg_star_panel_stats_row(spectra_bgstars_image: np.ndarray, channel: Channel) -> tuple[dict, list[str]]:
    bg_keys = stats_keys_for_filetype("BG_STARS")
    has_bg = np.any(spectra_bgstars_image > 0.0)
    if has_bg:
        values = _stats_from_array_only(spectra_bgstars_image)
        values["EXPTIME"] = getattr(channel, "exposure_s", None)
        return values, bg_keys
    return {k: None for k in bg_keys}, bg_keys


def _compute_psf_r90_px(channel: PhotometryChannel) -> float:
    psf = channel.psf_image
    cx = int(channel.psf_center_x)
    cy = int(channel.psf_center_y)

    yy, xx = np.indices(psf.shape, dtype=float)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    flat_psf = psf.ravel()
    flat_r = rr.ravel()

    order = np.argsort(flat_psf)[::-1]
    psf_sorted = flat_psf[order]
    r_sorted = flat_r[order]

    cumulative = np.cumsum(psf_sorted)

    idx = int(np.searchsorted(cumulative, 0.90, side="left"))

    return float(np.max(r_sorted[:idx + 1]))