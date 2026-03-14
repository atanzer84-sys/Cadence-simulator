from domain.star import Star
from configs.channel_config import Channel
import numpy as np
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for fast batch PNG writing
import matplotlib.pyplot as plt


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

STATS_KEYS = {
    "BIAS":     ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET"],
    "SCIENCE":  ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "RNOISE", "B_OFFSET", "DARKVAL", "DARKSIG", "EXPTIME"],
    "BG_STARS": ["MEAN", "MEDIAN", "STDDEV", "MIN", "MAX", "EXPTIME"],
}

# Shared layout constants so single-image and multi-frame PNGs match.
_WIDTH_IN = 10.0
_TEXT_H_IN = 0.7
_GAP_IN = 0.8
_TITLE_FONTSIZE = 11
_STATS_FONTSIZE = 10
_FIGURE_DPI = 100
# None avoids expensive tight-bbox computation; we use tight_layout() for spacing instead.
_BBOX_INCHES = None
# Max pixels used for percentile scaling; larger arrays are subsampled to reduce runtime.
_PERCENTILE_MAX_PIXELS = 250_000


def format_star_metadata(star: Star | None) -> str:
    """Format Teff, distance and Gaia G magnitude for titles. Returns empty string if star is None."""
    if star is None:
        return ""
    teff_str = f"{int(round(star.effective_temperature))}" if star.effective_temperature is not None else "—"
    dist_str = f"{int(round(star.distance_pc))}" if star.distance_pc is not None else "—"
    g_str = f"{star.gaia_magnitude:.1f}" if star.gaia_magnitude is not None else "—"
    return f"$T_{{\\mathrm{{eff}}}}$={teff_str} K, $d$={dist_str} pc, $G_{{\\mathrm{{Mag}}}}$={g_str}"

def format_frame_title(target_name: str, channel_tag: str, frame_type: str, star: Star | None) -> str:
    """Build title with optional Teff and distance when star is provided."""
    base = f"{target_name} | {channel_tag} {frame_type}"
    meta = format_star_metadata(star)
    return f"{base} | {meta}" if meta else base

def normalize_target_name(name: str) -> str:
    """Normalize target/star name for filenames (spaces → underscores). Used by PNG writes and plots."""
    return str(name).replace(" ", "_")

def build_stats_row(array: np.ndarray, channel: Channel, frame_type: str) -> tuple[dict, list[str]]:
    filetype = "BIAS" if "BIAS" in frame_type.upper() else "SCIENCE"
    stats_keys = STATS_KEYS[filetype]
    stats_values = {
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
    return stats_values, stats_keys

def build_png_filename(output_dir: Path, target_name: str, channel_tag: str, frame_type: str, index: int | None = None, *, waltzer_prefix: bool = True) -> Path:
    safe = normalize_target_name(target_name)
    prefix = "WALTzER_" if waltzer_prefix else ""
    if index is not None:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}_{index:05d}.png"
    if waltzer_prefix:
        return output_dir / f"{prefix}{safe}_{channel_tag}_{frame_type}.png"
    return output_dir / f"{safe}_{channel_tag}_{frame_type}_image.png"

def format_stats_text(values: dict, keys: list[str], *, use_scientific_for_small: bool = False) -> str:
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

def save_single_frame_png_NIR(array: np.ndarray, filename: Path, title: str, stats_text: str,  channel_name:str) -> None:
    STATS_FONTSIZE_NIR = 13
    TITLE_FONTSIZE_NIR = 18
    GAP_IN_NIR = 0.35
    TEXT_H_IN_NIR = 0.45
    NIR_LABEL_FONTSIZE = 16

    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + GAP_IN_NIR + TEXT_H_IN_NIR))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, GAP_IN_NIR, TEXT_H_IN_NIR], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xlabel("pixels", labelpad=8, fontsize=NIR_LABEL_FONTSIZE)
    ax.set_ylabel("pixels", labelpad=8, fontsize=NIR_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=NIR_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE_NIR, pad = 10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=STATS_FONTSIZE_NIR, transform=ax_txt.transAxes)

    fig.tight_layout()
    fig.savefig(filename, dpi=_FIGURE_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def save_single_frame_png_NUV(array: np.ndarray, filename: Path, title: str, stats_text: str,  channel_name:str) -> None:
    FIGURE_DPI = 175
    
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xlabel("pixels", labelpad=8)
    ax.set_ylabel("pixels", labelpad=8)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad = 10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
    fig.savefig(filename, dpi=FIGURE_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)

def save_single_frame_png_VIS(array: np.ndarray, filename: Path, title: str, stats_text: str,  channel_name:str) -> None:
    
    ny, nx = array.shape
    img_h_in = max(2.0, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + _GAP_IN + _TEXT_H_IN))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, _GAP_IN, _TEXT_H_IN], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xlabel("pixels", labelpad=8)
    ax.set_ylabel("pixels", labelpad=8)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.tight_layout()
    fig.savefig(filename, dpi=_FIGURE_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)


def save_single_frame_png_VIS_cropped(array: np.ndarray, filename: Path, title: str, stats_text: str, channel_name: str) -> None:
    GAP_IN_VIS = 0.8
    TEXT_H_IN_VIS = 0.28

    ny, nx = array.shape
    img_h_in = max(1.2, _WIDTH_IN * (ny / nx))

    fig = plt.figure(figsize=(_WIDTH_IN, img_h_in + GAP_IN_VIS + TEXT_H_IN_VIS), facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[img_h_in, GAP_IN_VIS, TEXT_H_IN_VIS], hspace=0)

    ax = fig.add_subplot(gs[0, 0])
    vmin, vmax = _calculate_percentile_scales(array)
    ax.imshow(array, origin="lower", aspect="equal", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.set_xlabel("pixels", labelpad=6)
    ax.set_ylabel("pixels", labelpad=6)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE, pad=10)

    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=_STATS_FONTSIZE, transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.85, bottom=0.08)
    fig.savefig(filename, dpi=_FIGURE_DPI, bbox_inches=_BBOX_INCHES)
    plt.close(fig)
    logging.debug("Wrote %s", filename)