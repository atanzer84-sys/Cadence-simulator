import logging
from configs.global_config import get_global_config
from frame.bias import generate_bias_frames
from frame.dark import generate_dark_frames
from frame.fits_header import initialize_fits_header
from frame.fits import write_fits_frames
from frame.science import generate_science_frames
from utils.images import write_frames_png


def generate_Frames(counts_s_pixel_convolved_nuv, counts_s_pixel_convolved_vis, nuv_cfg, vis_cfg, nuv_cal, vis_cal, user_cfg, output_dir, star):

    global_cfg = get_global_config()
    n_bias_and_darkframes = global_cfg.n_bias_and_darkframes
    n_science_frames = global_cfg.n_science_frames_per_channel

    header = initialize_fits_header(star)

    if n_bias_and_darkframes > 0:
        logging.info("Creating Bias and Dark Frames.")
        print("Creating Bias and Dark Frames.")

        bias_nuv_frames, bias_nuv_headers = generate_bias_frames(nuv_cfg, n_bias_and_darkframes, header)
        bias_vis_frames, bias_vis_headers = generate_bias_frames(vis_cfg, n_bias_and_darkframes, header)
        dark_nuv_frames, dark_nuv_headers = generate_dark_frames(nuv_cfg, n_bias_and_darkframes, user_cfg.exposure_NUV_s, header)
        dark_vis_frames, dark_vis_headers = generate_dark_frames(vis_cfg, n_bias_and_darkframes, user_cfg.exposure_VIS_s, header)

        #write FITS
        write_fits_frames(bias_nuv_frames, bias_nuv_headers, "bias", nuv_cfg.channel_name, output_dir)
        write_fits_frames(bias_vis_frames, bias_vis_headers, "bias", vis_cfg.channel_name, output_dir)
        write_fits_frames(dark_nuv_frames, dark_nuv_headers, "dark", nuv_cfg.channel_name, output_dir)
        write_fits_frames(dark_vis_frames, dark_vis_headers, "dark", vis_cfg.channel_name, output_dir)

        if global_cfg.write_dark_and_bias_png:
            write_frames_png(bias_nuv_frames, bias_nuv_headers, "bias", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(bias_vis_frames, bias_vis_headers, "bias", vis_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(dark_nuv_frames, dark_nuv_headers, "dark", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(dark_vis_frames, dark_vis_headers, "dark", vis_cfg.channel_name, output_dir, show_stats=True)
    else:
        logging.info("BIAS and DARK: n_bias_and_darkframes=%d → skipped.", n_bias_and_darkframes)

    if n_science_frames > 0:
        science_nuv_frames, science_nuv_headers = generate_science_frames(counts_s_pixel_convolved_nuv, nuv_cfg, nuv_cal, n_science_frames, user_cfg.exposure_NUV_s, header)
        science_vis_frames, science_vis_headers = generate_science_frames(counts_s_pixel_convolved_vis, vis_cfg, vis_cal, n_science_frames, user_cfg.exposure_VIS_s, header)
        # write science FITS
        # write_fits_frames(science_nuv_frames, science_nuv_headers, "science", nuv_cfg.channel_name, output_dir)
        # write_fits_frames(science_vis_frames, science_vis_headers, "science", vis_cfg.channel_name, output_dir)
        # # Write PNGs
        if global_cfg.write_science_frames_png:
            write_frames_png(science_nuv_frames, science_nuv_headers, "science", nuv_cfg.channel_name, output_dir, show_stats=True)
            write_frames_png(science_vis_frames, science_vis_headers, "science", vis_cfg.channel_name, output_dir, show_stats=True)
