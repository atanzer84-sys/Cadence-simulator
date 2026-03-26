from domain.star import Star
from configs.channel_config import Channel
import numpy as np
from pathlib import Path


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


def format_frame_title(channel_tag: str, frame_type: str, star: Star) -> str:
    """Build title with Teff and distance metadata from star."""
    base = f"{star.name} | {channel_tag} {frame_type}"
    meta = format_star_metadata(star)
    return f"{base} | {meta}"

def format_star_metadata(star: Star) -> str:
    """Format Teff, distance and Gaia G magnitude for titles."""
    teff_str = f"{int(round(star.effective_temperature))}" if star.effective_temperature is not None else "—"
    dist_str = f"{int(round(star.distance_pc))}" if star.distance_pc is not None else "—"
    g_str = f"{star.gaia_magnitude:.1f}" if star.gaia_magnitude is not None else "—"
    return f"$T_{{\\mathrm{{eff}}}}$={teff_str} K, $d$={dist_str} pc, $G_{{\\mathrm{{Mag}}}}$={g_str}"


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

def build_base_output_path(output_dir: Path, target_name: str, channel_tag: str, frame_type: str, exposure: float, index: int, *, waltzer_prefix: bool = True, suffix: str = "") -> Path:
    safe = normalize_target_name(target_name)
    exposure_int = int(exposure)
    prefix = "WALTzER_" if waltzer_prefix else ""
    basename = f"{prefix}{safe}_{channel_tag}_{frame_type}_{exposure_int}s_{index:05d}"
    return output_dir / f"{basename}{suffix}"


def build_png_filename(output_dir: Path, target_name: str, channel_tag: str, frame_type: str, exposure: float, index: int | None = None, *, waltzer_prefix: bool = True) -> Path:
    if index is None:
        return output_dir
    return build_base_output_path(output_dir, target_name, channel_tag, frame_type, exposure, index, waltzer_prefix=waltzer_prefix, suffix=".png")




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
            if fmt:
                formatted = format(val, fmt)
                if "e" not in formatted and "E" not in formatted:
                    if "." in formatted:
                        left, right = formatted.split(".")
                        left = f"{int(left):,}".replace(",", " ")
                        formatted = f"{left}.{right}"
                    else:
                        formatted = f"{int(round(val)):,}".replace(",", " ")
                s = f"{key}={formatted}"
            else:
                s = f"{key}={val}"

            unit = STATS_KEY_UNIT.get(key, "")
            if unit:
                s += f" {unit}"

        parts.append(s)
        if (i + 1) % 5 == 0 and i + 1 < len(pairs):
            parts.append("\n")
        elif i + 1 < len(pairs):
            parts.append("  ")
    return "".join(parts)


