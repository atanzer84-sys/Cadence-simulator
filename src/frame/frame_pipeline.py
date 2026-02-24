import logging
from configs.global_config import get_global_config
from frame.bias import generate_bias_frames
from frame.dark import generate_dark_frames
from frame.fits_header import initialize_fits_header
from frame.science import generate_science_frames
from configs.channel_config import SpectroscopyChannel
from loaders.run_waltzer_context import RunContext
from domain.star import Star


def generate_Frames(counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, nuv: SpectroscopyChannel, vis: SpectroscopyChannel, ctx: RunContext, star: Star):
    global_cfg = get_global_config()
    n_bias_and_darkframes = global_cfg.n_bias_and_darkframes
    n_science_frames = global_cfg.n_science_frames_per_channel

    header = initialize_fits_header(star, ctx.timestamp)

    if n_bias_and_darkframes > 0:
        bias_nuv_frames = generate_bias_frames(nuv, n_bias_and_darkframes, header)
        bias_vis_frames = generate_bias_frames(vis, n_bias_and_darkframes, header)
        dark_nuv_frames = generate_dark_frames(nuv, n_bias_and_darkframes, header)
        dark_vis_frames = generate_dark_frames(vis, n_bias_and_darkframes, header)
        bias_dark_lists = [bias_nuv_frames, bias_vis_frames, dark_nuv_frames, dark_vis_frames]

        _write_fits_for_all(bias_dark_lists, ctx)

        if global_cfg.write_dark_and_bias_png:
            _write_png_for_all(bias_dark_lists, ctx, star)

    else:
        logging.info(
            "BIAS and DARK: n_bias_and_darkframes=%d \u2192 skipped.",
            n_bias_and_darkframes,
        )

    if n_science_frames > 0:
        science_nuv_frames = generate_science_frames(counts_s_pixel_convolved_nuv, nuv, n_science_frames, header)
        science_vis_frames = generate_science_frames(counts_s_pixel_convolved_vis, vis, n_science_frames, header)
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
        for k, frame in enumerate(frames):
            frame.write_fits(ctx, index=k)


def _write_png_for_all(frame_lists, ctx: RunContext, star: Star) -> None:
    for frames in frame_lists:
        for k, frame in enumerate(frames):
            frame.write_png(ctx=ctx, star=star, index=k)
