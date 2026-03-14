from domain.star import Star

def format_star_metadata(star: Star | None) -> str:
    """Format Teff, distance and Gaia G magnitude for titles. Returns empty string if star is None."""
    if star is None:
        return ""
    teff_str = f"{int(round(star.effective_temperature))}" if star.effective_temperature is not None else "—"
    dist_str = f"{int(round(star.distance_pc))}" if star.distance_pc is not None else "—"
    g_str = f"{int(round(star.gaia_magnitude))}" if star.gaia_magnitude is not None else "—"
    return f"$T_{{\\mathrm{{eff}}}}$={teff_str} K, $d$={dist_str} pc, $G$={g_str}"


def format_frame_title(target_name: str, channel_tag: str, frame_type: str, star: Star | None) -> str:
    """Build title with optional Teff and distance when star is provided."""
    base = f"{target_name} | {channel_tag} {frame_type}"
    meta = format_star_metadata(star)
    return f"{base} | {meta}" if meta else base


def normalize_target_name(name: str) -> str:
    """Normalize target/star name for filenames (spaces → underscores). Used by PNG writes and plots."""
    return str(name).replace(" ", "_")
