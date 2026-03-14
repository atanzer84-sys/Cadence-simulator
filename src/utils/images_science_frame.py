import logging
from loaders.run_waltzer_context import RunContext
from domain.star import Star
from utils.images_common import format_frame_title, build_stats_row, build_png_filename, format_stats_text, save_single_frame_png_NIR, save_single_frame_png_NUV, save_single_frame_png_VIS, save_single_frame_png_VIS_cropped
from configs.channel_config import Channel, SpectroscopyChannel
from configs.global_config import GlobalConfig
from instrument.spectrum_spread import get_target_star_detector_position

_SPECTRUM_STRIP_MARGIN_FRAC = 0.2


def write_science_frame_png(detector_data, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, index: int | None = None) -> None:

    filetype = "science"
    channel_name = channel.channel_name
    inverted = cfg.invert_science_frames
    logging.info("PNG writing: channel=%s frame_type=%s index=%s", channel_name, filetype, index)

    # Prepare data for PNG: optionally crop to spectrum strip for spectroscopy
    data_for_png = detector_data
    if cfg.science_frame_png_crop_spectrum_region and isinstance(channel, SpectroscopyChannel):
        ny = channel.y_pixels
        _, y0 = get_target_star_detector_position(channel)  # same position used when placing spectrum on detector
        half_strip_pix = int(round(channel.slit_half_length_arcsec / channel.pixel_scale))
        strip_height = 2 * half_strip_pix + 1
        margin = int(round(_SPECTRUM_STRIP_MARGIN_FRAC * strip_height))
        half_height = half_strip_pix + margin
        y_lo = max(0, y0 - half_height)
        y_hi = min(ny, y0 + half_height + 1)
        data_for_png = detector_data[y_lo:y_hi, :]

    title = format_frame_title(star.name, channel_name, filetype, star)

    frame_to_plot = (data_for_png.max() - data_for_png) if inverted else data_for_png

    stats_values, stats_keys = build_stats_row(detector_data, channel, filetype)
    stats_text = format_stats_text(stats_values, stats_keys) if (stats_values and stats_keys) else None
    filename = build_png_filename(ctx.output_dir, star.name, channel_name, filetype, index, waltzer_prefix=True)

    if cfg.science_frame_png_crop_spectrum_region and isinstance(channel, SpectroscopyChannel):
        save_single_frame_png_VIS_cropped(frame_to_plot, filename, title, stats_text, channel_name=channel_name)
    else:
        save_fn = {"NIR": save_single_frame_png_NIR, "NUV": save_single_frame_png_NUV, "VIS": save_single_frame_png_VIS}.get(channel_name, save_single_frame_png_VIS)
        save_fn(frame_to_plot, filename, title, stats_text, channel_name=channel_name)
