import logging
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from utils.images_common import format_frame_title, build_stats_row, build_png_filename, format_stats_text, save_single_frame_png
from configs.channel_config import Channel
from configs.global_config import GlobalConfig


def write_science_frame_png(detector_data, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, index: int | None = None) -> None:

    filetype = "science"
    channel_name = channel.channel_name
    inverted = cfg.invert_science_frames
    logging.info("PNG writing: channel=%s frame_type=%s index=%s", channel_name, filetype, index)

    title = format_frame_title(star.name, channel_name, filetype, star)
    axis_label_fontsize = 15 if str(channel_name).upper() == "NIR" else None
    tick_label_fontsize = 15 if str(channel_name).upper() == "NIR" else None

    frame_to_plot = (detector_data.max() - detector_data) if inverted else detector_data

    stats_values, stats_keys = build_stats_row(detector_data, channel, filetype)
    stats_text = format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    filename = build_png_filename(ctx.output_dir, star.name, channel_name, filetype, index, waltzer_prefix=True)

    save_single_frame_png(frame_to_plot, filename, title, stats_text, axis_label_fontsize=axis_label_fontsize, tick_label_fontsize=tick_label_fontsize)
