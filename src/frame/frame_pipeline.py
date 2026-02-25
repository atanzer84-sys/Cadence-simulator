import logging
from configs.global_config import get_global_config
from frame.bias_frame import generate_bias_frames
from frame.dark_frame import generate_dark_frames
from frame.fits_header import initialize_fits_header
from frame.science_frame import generate_science_frames
from frame.write_fits import write_fits_frames
from utils.images import write_frames_png
from configs.channel_config import SpectroscopyChannel
from loaders.run_waltzer_context import RunContext
from domain.star import Star

def generate_Frames(nuv_image, vis_image, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, ctx: RunContext, star: Star):
    global_cfg = get_global_config()
    n_non_science_frames = global_cfg.n_non_science_frames
    n_science_frames = global_cfg.n_science_frames_per_channel

    header = initialize_fits_header(star, ctx.timestamp)

    if n_non_science_frames > 0:
        bias_nuv_frames = generate_bias_frames(nuv, n_non_science_frames, header)
        bias_vis_frames = generate_bias_frames(vis, n_non_science_frames, header)
        #bias + dark = dark
        dark_nuv_frames = generate_dark_frames(nuv, n_non_science_frames, header)
        dark_vis_frames = generate_dark_frames(vis, n_non_science_frames, header)
        # dark + spectra = spectra

        non_science_list = [bias_nuv_frames, bias_vis_frames, dark_nuv_frames, dark_vis_frames]

        _write_fits_for_all(non_science_list, ctx)
        _write_png_for_all(non_science_list, ctx, star)

    else:
        logging.info("Non Science Frames: n_non_science_frames=%d \u2192 skipped.", n_non_science_frames)

    if n_science_frames > 0:
        science_nuv_frames = generate_science_frames(nuv_image, nuv, n_science_frames, header)
        science_vis_frames = generate_science_frames(vis_image, vis, n_science_frames, header)
        science_lists = [science_nuv_frames, science_vis_frames]

        # Write science FITS
        _write_fits_for_all(science_lists, ctx)

        # Write PNGs
        if global_cfg.write_science_frames_png:
            _write_png_for_all(science_lists, ctx, star)
    else:
        logging.info("SCIENCE: n_science_frames=%d \u2192 skipped.", n_science_frames)


def _write_fits_for_all(frame_lists, ctx: RunContext) -> None:
    for frames in frame_lists:
        if not frames:
            continue

        frame_type = frames[0].frame_type
        channel_tag = frames[0].channel_tag

        data_list = [frame.data for frame in frames]
        header_list = [frame.header for frame in frames]

        write_fits_frames(
            frames=data_list,
            headers=header_list,
            frame_type=frame_type,
            channel_tag=channel_tag,
            ctx=ctx,
        )


def _write_png_for_all(frame_lists, ctx: RunContext, star: Star) -> None:
    for frames in frame_lists:
        if not frames:
            continue
        frame_type = frames[0].frame_type
        channel_tag = frames[0].channel_tag
        data_list = [f.data for f in frames]
        header_list = [f.header for f in frames]
        write_frames_png(
            frames=data_list,
            headers=header_list,
            frame_type=frame_type,
            channel_tag=channel_tag,
            ctx=ctx,
            star=star,
            show_stats=True,
        )


