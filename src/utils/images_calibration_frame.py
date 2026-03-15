import logging
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from configs.channel_config import Channel
from utils.images_common import format_frame_title, build_stats_row, build_png_filename, format_stats_text
from utils.images_science_frame import save_single_frame_png_NIR, save_single_frame_png_NUV, save_single_frame_png_VIS_cropped
from configs.global_config import GlobalConfig


def write_calibration_frame_png(detector_data, frame_type: str, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, index: int) -> None:

    inverted = cfg.invert_science_frames
    channel_name = channel.channel_name
    logging.info("PNG calibration: channel=%s frame_type=%s index=%s", channel_name, frame_type, index)

    title = format_frame_title(ctx.target_name, channel_name, frame_type, star)

    frame_to_plot = (detector_data.max() - detector_data) if inverted else detector_data

    stats_values, stats_keys = build_stats_row(detector_data, channel, frame_type)
    stats_text = format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    filename = build_png_filename(ctx.output_dir, star.name, channel_name, frame_type, index, waltzer_prefix=False)

    save_fn = {"NIR": save_single_frame_png_NIR, "NUV": save_single_frame_png_NUV, "VIS": save_single_frame_png_VIS_cropped}.get(channel_name, save_single_frame_png_VIS_cropped)
    save_fn(frame_to_plot, filename, title, stats_text, channel_name=channel_name)

