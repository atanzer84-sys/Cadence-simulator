import logging
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from utils.images_common import format_frame_title
from utils.images import write_one_frame_png, stats_keys_for_filetype


def write_science_frame_png(frame, header, frame_type: str, channel_tag: str, ctx: RunContext, star: Star, show_stats: bool = False, inverted: bool = False, index: int | None = None) -> None:
    """Write one science frame as PNG."""
    logging.info("PNG writing: channel=%s frame_type=%s index=%s", channel_tag, frame_type, index)

    title_base = format_frame_title(star.name, channel_tag, frame_type, star)
    axis_label_fontsize = 15 if str(channel_tag).upper() == "NIR" else None
    tick_label_fontsize = 15 if str(channel_tag).upper() == "NIR" else None

    frame_to_plot = (frame.max() - frame) if inverted else frame
    stats_values, stats_keys = _build_frame_write_context(header, show_stats)
    write_one_frame_png(frame_to_plot, ctx.output_dir, star.name, channel_tag, frame_type, title_base, stats_values, stats_keys, index=index, axis_label_fontsize=axis_label_fontsize, tick_label_fontsize=tick_label_fontsize)

def _build_frame_write_context(header, show_stats: bool) -> tuple[dict | None, list[str]]:
    if not show_stats:
        return None, []
    stats_values, stats_keys = _build_header_stats_payload(header)
    return stats_values, stats_keys

def _build_header_stats_payload(header) -> tuple[dict, list[str]]:
    stats_keys = _stats_keys_for_header(header)
    return _stats_from_header(header, stats_keys), stats_keys

def _stats_keys_for_header(header):
    ft = _header_val(header, "FILETYPE")
    filetype = str(ft).upper() if ft else "BIAS"
    return stats_keys_for_filetype(filetype, default_filetype="BIAS")

def _stats_from_header(header, keys: list[str]) -> dict:
    """Build stats dict from FITS header. Values not in header become None (filtered later)."""
    return {k: _header_val(header, k) for k in keys}


def _header_val(header, key):
    if hasattr(header, "get"):
        return header.get(key)
    for item in header:
        if len(item) >= 2 and item[0] == key:
            return item[1]
    return None

