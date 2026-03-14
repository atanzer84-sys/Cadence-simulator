# import logging
# from loaders.run_waltzer_context import RunContext
# from utils.images_common import format_frame_title
# from domain.star import Star


# def write_science_frame_png(frame, header, frame_type, channel_tag, ctx: RunContext, star: Star, index: int, inverted=False):
#     logging.info("PNG writing started: channel=%s frame_type=%s frame_index=%d", channel_tag, frame_type, index)

#     title_base = format_frame_title(star.name, channel_tag, frame_type, star)
#     axis_label_fontsize = 15 if str(channel_tag).upper() == "NIR" else None
#     tick_label_fontsize = 15 if str(channel_tag).upper() == "NIR" else None

#     if inverted:
#         frame_to_plot = frame.max() - frame
#     else:
#         frame_to_plot = frame

#     stats_values, stats_keys = _build_frame_write_context(header, show_stats)
#     _write_one_frame_png(frame_to_plot, ctx.output_dir, star.name, channel_tag, frame_type, title_base, stats_values, stats_keys, index=index, axis_label_fontsize=axis_label_fontsize, tick_label_fontsize=tick_label_fontsize)

#     logging.info("PNG writing finished: channel=%s frame_type=%s frame_index=%d", channel_tag, frame_type, index)