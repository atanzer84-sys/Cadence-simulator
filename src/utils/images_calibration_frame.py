


def write_calibration_frame_png(array, frame_type: str, ctx: RunContext, channel: Channel, show_stats: bool = True, star: Star | None = None, index: int | None = None, inverted: bool = False) -> None:
    """Write 2D array as PNG. Uses percentile scaling (1–99.9) like write_science_frames_png. Optional index for multi-frame output (e.g. frame_00042.png)."""
    logging.info("write_calibration_frame_png called | frame_type=%s | channel=%s | shape=%s | index=%s", frame_type, channel.channel_name, array.shape, index)

    frame_to_plot = (array.max() - array) if inverted else array
    title, stats_values, stats_keys = _build_image_write_context(array, frame_type, ctx, channel, show_stats, star)
    _write_one_frame_png(frame_to_plot, ctx.output_dir, ctx.target_name, channel.channel_name, frame_type, title, stats_values, stats_keys, index=index, waltzer_prefix=False)
