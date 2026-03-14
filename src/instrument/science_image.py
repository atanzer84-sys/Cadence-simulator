import logging
import numpy as np
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel, PhotometryChannel, Channel
from instrument.bias_image import generate_bias_image
from instrument.dark_image import generate_dark_image
from instrument.cosmic_image import generate_cosmic_rays
from configs.global_config import GlobalConfig, get_global_config
from instrument.background_image import generate_background_image
from domain.star import Star
from domain.star_catalog import StarCatalog
from instrument.background_star_spectroscopy import generate_background_star_spectroscopy_image
from instrument.background_star_photometry import generate_background_star_photometry_image
from instrument.photon_noise import apply_photon_noise_gauss_from_spectra2d
from frame.fits_header import initialize_fits_header
from frame.bias_frame import generate_bias_frames
from frame.dark_frame import generate_dark_frames
from frame.write_fits import write_fits_frames
from utils.images import write_science_frames_png
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header
from frame.frame_class import Frame


def build_science_images(stellar_signal, channel: Channel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog):
    cfg = get_global_config()
    header = initialize_fits_header(star, ctx.timestamp)

    ctx.plot_background_star_counts(background_stars_catalog, channel, ctx)

    _generate_channel_calibration_frames(channel, header, ctx, star, cfg)

    _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog, header)


def _generate_channel_calibration_frames(channel: Channel, header, ctx: RunContext, star: Star, cfg: GlobalConfig):
    n_calibration_frames = cfg.n_calibration_frames

    logging.info("FITS generation starting (n_calibration_frames=%d)", n_calibration_frames)
    print(f"\n==== STARTING FITS GENERATION ({channel.channel_name}) =====")

    if n_calibration_frames > 0:
        bias_frames = generate_bias_frames(channel, n_calibration_frames, header)
        dark_frames = generate_dark_frames(channel, n_calibration_frames, header)
        calibration_frame_list = [bias_frames, dark_frames]

        _write_fits_for_all(calibration_frame_list, ctx, phase="calibration frames")

        if cfg.write_calibration_frames_png:
            _write_png_for_all(calibration_frame_list, ctx, star, phase="calibration-frames", inverted=cfg.invert_calibration_science_frame_component)
        
    else:
        logging.info("Calibration Frames: n_calibration_frames=%d -> skipped.", n_calibration_frames)

def _create_channel_images(stellar_signal, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, base_header) -> None:
    print(f"\n==== STARTING SCIENCE IMAGE GENERATION ({channel.channel_name}) =====")
    logging.info("Science Image generation starting for channel %s", channel.channel_name)

    exposure = channel.exposure_s
    readout_gap_s = cfg.readout_gap_s
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    stellar_component = stellar_signal * exposure
    background_component = generate_background_image(channel, ctx, star)
    n_science_frames = channel.n_science_frames

    for frame_index in range(n_science_frames):
        time_s = frame_index * (exposure + readout_gap_s)
        roll_angle_start = 360.0 * (time_s / orbit_duration_s)
        roll_angle_end = 360.0 * ((time_s + exposure) / orbit_duration_s)

        print(f"science exposure image {frame_index + 1}/{n_science_frames} (roll_angle_start={roll_angle_start:.2f}°, roll_angle_end={roll_angle_end:.2f}°)")

        img = _create_per_exposure(stellar_component, background_component, channel, ctx, cfg, star, background_stars_catalog, frame_index, roll_angle_start, roll_angle_end)

        logging.info("SCIENCE: generating frame %d/%d for %s (%d x %d), exptime_s=%g.", frame_index + 1, n_science_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, exposure)
        print(f"Creating SCIENCE frame {frame_index + 1}/{n_science_frames} for channel {channel.channel_name}.")
        
        header = append_base_frame_header(base_header, filetype="SCIENCE", channel=channel, index0=frame_index)
        append_image_stats_header(header, img)
        append_channel_frame_header(header, channel, exptime_s=exposure, include_bias=True, include_dark=True)

        frame = Frame(data=img, header=header, frame_type="science", channel_tag=channel.channel_name)
        science_list = [frame]

        _write_fits_for_all([science_list], ctx, phase="science", start_index=frame_index)
        if cfg.write_science_frames_png:
            _write_png_for_all([science_list], ctx, star, phase="science", inverted=cfg.invert_science_frames, start_index=frame_index)
            
    logging.info("Science image generation finished: channel=%s frames=%d exposure_s=%g orbit_duration_s=%g", channel.channel_name, n_science_frames, exposure, orbit_duration_s)



def _create_per_exposure(stellar_component, background_component, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, frame_index: int, roll_angle_start: float, roll_angle_end: float) -> np.ndarray:
    ccd_gain = channel.ccd_gain

    image = _build_science_image_without_bg_stars(stellar_component, background_component, channel, ctx, cfg, star, frame_index)

    if isinstance(channel, SpectroscopyChannel):
        bg_stars, background_star_visibility = generate_background_star_spectroscopy_image(channel, ctx, star, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index)
        visibility_kwargs = {"background_star_bands": background_star_visibility}
    elif isinstance(channel, PhotometryChannel):
        bg_stars, background_star_visibility = generate_background_star_photometry_image(channel, ctx, star, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index)
        visibility_kwargs = {"background_star_arcs": background_star_visibility}
    else:
        raise TypeError(f"Unsupported channel type: {type(channel)}")

    image += bg_stars
    image *= ccd_gain

    ctx.generate_background_star_visibility_on_science_frame(image, bg_stars, "SCIENCE PANEL", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component, **visibility_kwargs)

    return image

    
def _build_science_image_without_bg_stars(target_star_component, background_component, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, frame_index: int):
    image = np.zeros((channel.y_pixels, channel.x_pixels), dtype=np.float32)

    bias = generate_bias_image(channel)
    image += bias
    if frame_index < 1:
        ctx.write_science_frame_component_png(image, "SCIENCE_BIAS_ONLY", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    dark = generate_dark_image(channel)
    image += dark
    if frame_index < 1:
        ctx.write_science_frame_component_png(image, "SCIENCE_DARK_ONLY", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    image += target_star_component
    if frame_index < 1:
        ctx.write_science_frame_component_png(target_star_component, "SCIENCE_SIGNAL_ONLY", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)
        ctx.write_science_frame_component_png(image, "SCIENCE_SPECTRA", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    photon_noise = apply_photon_noise_gauss_from_spectra2d(target_star_component, channel, ctx, star)
    image += photon_noise
    if frame_index < 1:
        ctx.write_science_frame_component_png(photon_noise, "SCIENCE_PHOTON_NOISE_ONLY", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    image += background_component
    if frame_index < 1:
        ctx.write_science_frame_component_png(background_component, "SCIENCE_BACKGROUND_ONLY", ctx, channel, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    cosmic = generate_cosmic_rays(ctx, channel, cfg, star)
    image += cosmic
    if frame_index < 1:
        ctx.write_science_frame_component_png(cosmic, "SCIENCE_COSMIC_ONLY", ctx, channel, star=star, index=frame_index, inverted=cfg.invert_calibration_science_frame_component)

    return image


def _write_fits_for_all(frame_lists, ctx: RunContext, *, phase: str = "", start_index: int = 0) -> None:
    phase_str = f" for {phase} frames" if phase else ""
    logging.info("Creating FITS files%s", phase_str)
    print(f"Creating FITS files{phase_str}")
    for frames in frame_lists:
        if not frames:
            continue

        frame_type = frames[0].frame_type
        channel_tag = frames[0].channel_tag

        data_list = [frame.data for frame in frames]
        header_list = [frame.header for frame in frames]

        write_fits_frames(frames=data_list, headers=header_list, frame_type=frame_type, channel_tag=channel_tag, ctx=ctx, start_index=start_index)


def _write_png_for_all(frame_lists, ctx: RunContext, star: Star, phase: str = "", inverted: bool = False, start_index: int = 0) -> None:
    phase_str = f" for {phase} frames" if phase else ""
    logging.info("Creating PNG files%s", phase_str)
    print(f"Creating PNG files{phase_str}")
    for frames in frame_lists:
    
        if not frames:
            continue
        
        frame_type = frames[0].frame_type
        channel_tag = frames[0].channel_tag

        data_list = [f.data for f in frames]
        header_list = [f.header for f in frames]

        write_science_frames_png(frames=data_list, headers=header_list, frame_type=frame_type, channel_tag=channel_tag, ctx=ctx, star=star, show_stats=True, inverted=inverted, start_index=start_index)

