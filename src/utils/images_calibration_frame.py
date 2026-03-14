import logging
from loaders.run_waltzer_context import RunContext
from domain.star import Star
import numpy as np
from configs.channel_config import Channel
from utils.images_common import format_frame_title
from utils.images import write_one_frame_png, stats_keys_for_filetype, stats_from_array_channel 

_FRAME_TYPE_TO_STATS_FILETYPE = (("BIAS", "BIAS"), ("DARK", "DARK"))
_DEFAULT_FRAME_STATS_FILETYPE = "SCIENCE"

def write_calibration_frame_png(array, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool = True, star: Star | None = None, index: int | None = None, inverted: bool = False) -> None:
    logging.info("write_calibration_frame_png called | frame_type=%s | channel=%s | shape=%s | index=%s", frame_type, channel.channel_name, array.shape, index)

    frame_to_plot = (array.max() - array) if inverted else array
    title, stats_values, stats_keys = _build_image_write_context(array, frame_type, ctx, channel, show_stats, star)
    write_one_frame_png(frame_to_plot, ctx.output_dir, ctx.target_name, channel.channel_name, frame_type, title, stats_values, stats_keys, index=index, waltzer_prefix=False)


def _build_image_write_context(array: np.ndarray, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool, star: Star | None) -> tuple[str, dict | None, list[str]]:
    title = format_frame_title(ctx.target_name, channel.channel_name, frame_type, star)
    if not show_stats:
        return title, None, []
    stats_values, stats_keys = _build_image_stats_payload(array, channel, frame_type)
    return title, stats_values, stats_keys

def _build_image_stats_payload(array: np.ndarray, channel: Channel, frame_type: str) -> tuple[dict, list[str]]:
    filetype = _stats_filetype_for_frame_type(frame_type)
    stats_keys = stats_keys_for_filetype(filetype)
    return stats_from_array_channel(array, channel), stats_keys

def _stats_filetype_for_frame_type(frame_type: str) -> str:
    u = frame_type.upper()
    for token, filetype in _FRAME_TYPE_TO_STATS_FILETYPE:
        if token in u:
            return filetype
    return _DEFAULT_FRAME_STATS_FILETYPE
