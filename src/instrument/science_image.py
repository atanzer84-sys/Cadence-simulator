import logging
import numpy as np
from instrument.flat_image import generate_flat_image
from loaders.run_cadence_context import RunContext
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
from instrument.photon_noise import generate_photon_noise_from_spectra2d
from frame.fits_header import initialize_fits_header
from frame.bias_frame import generate_bias_frame_with_index
from frame.dark_frame import generate_dark_frame_with_index
from frame.flat_frame import generate_flat_frame_with_index
from frame.write_fits import write_fits_frame
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header, append_photometry_header
from frame.frame_class import Frame
from instrument.psf_spread import compute_aperture_photometry
from utils.images_calibration_frame import write_calibration_frame_png
from utils.images_science_frame import write_science_frame_png
from utils.debug_dumps import dump_npz_snapshot


def build_science_images(stellar_signal, channel: Channel, ctx: RunContext, star: Star, background_stars_catalog: StarCatalog):
    cfg = get_global_config()
    header = initialize_fits_header(star, ctx.timestamp)

    _generate_channel_calibration_frames(channel, header, ctx, star, cfg)

    _create_channel_images(stellar_signal, channel, ctx, cfg, star, background_stars_catalog, header)


def _generate_channel_calibration_frames(channel: Channel, header, ctx: RunContext, star: Star, cfg: GlobalConfig):
    n_calibration_frames = cfg.n_calibration_frames
    exposure = channel.exposure_s
    logging.info("FITS generation starting (n_calibration_frames=%d)", n_calibration_frames)
    print(f"\n==== STARTING FITS CALIBRATION FRAME GENERATION ({channel.channel_name}) =====")

    if n_calibration_frames <= 0:
        logging.info("Calibration Frames: n_calibration_frames=%d -> skipped.", n_calibration_frames)
        return

    for i in range(n_calibration_frames):
        bias_frame = generate_bias_frame_with_index(channel, i, header)
        write_fits_frame(bias_frame, ctx, i, exposure)
        if cfg.write_calibration_frame_png:
            write_calibration_frame_png(bias_frame.data, bias_frame.frame_type, channel, ctx, cfg, star=star, index=i)


        dark_frame = generate_dark_frame_with_index(channel, i, header)
        write_fits_frame(dark_frame, ctx, i, exposure)
        if cfg.write_calibration_frame_png:
            write_calibration_frame_png(dark_frame.data, dark_frame.frame_type, channel, ctx, cfg, star=star, index=i)

        flat_frame = generate_flat_frame_with_index(channel, i, header)
        write_fits_frame(flat_frame, ctx, i, exposure)
        if cfg.write_calibration_frame_png:
            write_calibration_frame_png(flat_frame.data, flat_frame.frame_type, channel, ctx, cfg, star=star, index=i)


def _create_channel_images(stellar_signal, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, base_header) -> None:
    print(f"\n==== STARTING SCIENCE IMAGE GENERATION ({channel.channel_name}) =====")
    logging.info("Science Image generation starting for channel %s", channel.channel_name)

    exposure = channel.exposure_s
    readout_gap_s = cfg.readout_gap_s
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    stellar_component = stellar_signal * exposure

    background_component = generate_background_image(channel, star)
    
    n_science_frames = channel.n_science_frames
    total_orbits = max(1, int(np.ceil(cfg.orbit_revolutions)))

    for frame_index in range(n_science_frames):
        time_s = frame_index * (exposure + readout_gap_s)
        roll_angle_start = 360.0 * (time_s / orbit_duration_s)
        roll_angle_end = 360.0 * ((time_s + exposure) / orbit_duration_s)

        img = _create_per_exposure(stellar_component, background_component, channel, ctx, cfg, star, background_stars_catalog, frame_index, roll_angle_start, roll_angle_end, base_header)
        phot = compute_aperture_photometry(img, channel)

        _report_science_frame_progress(channel, frame_index, total_orbits, roll_angle_start, roll_angle_end)

        header = append_base_frame_header(base_header, filetype="SCIENCE", channel=channel, index0=frame_index)
        append_image_stats_header(header, img)
        append_channel_frame_header(header, channel, exptime_s=exposure)
        append_photometry_header(header, phot)

        frame = Frame(data=img, header=header, frame_type="science", channel_tag=channel.channel_name)
        write_fits_frame(frame, ctx, frame_index, exposure)
        if cfg.write_science_frames_png:
            write_science_frame_png(frame.data, channel, ctx, cfg, star=star, index=frame_index,phot=phot)
            
    logging.info("Science image generation finished: channel=%s frames=%d exposure_s=%g orbit_duration_s=%g", channel.channel_name, n_science_frames, exposure, orbit_duration_s)

def _create_per_exposure(stellar_component, background_component, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, background_stars_catalog: StarCatalog, frame_index: int, roll_angle_start: float, roll_angle_end: float, base_header) -> np.ndarray:
    ccd_gain = channel.ccd_gain

    image = _build_science_image_without_bg_stars(stellar_component, background_component, channel, ctx, cfg, star, frame_index, base_header)

    if isinstance(channel, SpectroscopyChannel):
        bg_stars = generate_background_star_spectroscopy_image(channel, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index)
    elif isinstance(channel, PhotometryChannel):
        bg_stars = generate_background_star_photometry_image(channel, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index)
    else:
        raise TypeError(f"Unsupported channel type: {type(channel)}")

    image += bg_stars
    image *= ccd_gain

    if cfg.write_intermediate_arrays and frame_index < 1:
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_background_stars_component_full.npz", image_full=bg_stars)
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_science_image_full.npz", image_full=image)

    return image

def _build_science_image_without_bg_stars(target_star_component, background_component, channel: Channel, ctx: RunContext, cfg: GlobalConfig, star: Star, frame_index: int, base_header):

    image = generate_bias_image(channel).astype(np.float32)
    if frame_index < 1:
        if cfg.write_science_frame_component_png:
            write_calibration_frame_png(image, "SCIENCE_BIAS_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
        if cfg.write_science_frame_component_fits:
            _write_science_component_fits(image, "SCIENCE_BIAS_ONLY", channel, ctx, frame_index, base_header)
        if cfg.write_intermediate_arrays:
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_bias_full.npz", image_full=image)

    dark = generate_dark_image(channel)
    image += dark
    if frame_index < 1:
        if cfg.write_science_frame_component_png:
            write_calibration_frame_png(image, "SCIENCE_DARK_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
        if cfg.write_science_frame_component_fits:
            _write_science_component_fits(image, "SCIENCE_DARK_ONLY", channel, ctx, frame_index, base_header)
        if cfg.write_intermediate_arrays:
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_dark_full.npz", image_full=dark)


    photon_noise = generate_photon_noise_from_spectra2d(target_star_component)
    flat = generate_flat_image(channel)
    signal = target_star_component + photon_noise + background_component
    flat_total = signal * flat
    image += flat_total
    if frame_index < 1:
        if cfg.write_science_frame_component_png:
            write_calibration_frame_png(target_star_component, "SCIENCE_SIGNAL_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
            write_calibration_frame_png(background_component, "SCIENCE_BACKGROUND_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
            write_calibration_frame_png(photon_noise, "SCIENCE_PHOTON_NOISE_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
            write_calibration_frame_png(flat, "SCIENCE_FLAT_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
            write_calibration_frame_png(flat_total, "SCIENCE_SPECTRA_FLAT_PHOTONNOISE_BACKG", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)

        if cfg.write_science_frame_component_fits:
            _write_science_component_fits(target_star_component, "SCIENCE_SIGNAL_ONLY", channel, ctx, frame_index, base_header)
            _write_science_component_fits(background_component, "SCIENCE_BACKGROUND_ONLY", channel, ctx, frame_index, base_header)
            _write_science_component_fits(photon_noise, "SCIENCE_PHOTON_NOISE_ONLY", channel, ctx, frame_index, base_header)
            _write_science_component_fits(flat, "SCIENCE_FLAT_ONLY", channel, ctx, frame_index, base_header)
            _write_science_component_fits(flat_total, "SCIENCE_SPECTRA_FLAT_PHOTONNOISE_BACKG", channel, ctx, frame_index, base_header)

        if cfg.write_intermediate_arrays:
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_photon_noise_full.npz", image_full=photon_noise)
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_flat_full.npz", image_full=flat)
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_background_component_full.npz", image_full=background_component)

    cosmic = generate_cosmic_rays(channel, cfg)
    image += cosmic
    if frame_index < 1:
        if cfg.write_science_frame_component_png:
            write_calibration_frame_png(cosmic, "SCIENCE_COSMIC_ONLY", channel=channel, ctx=ctx, cfg=cfg, star=star, index=frame_index)
        if cfg.write_science_frame_component_fits:
            _write_science_component_fits(cosmic, "SCIENCE_COSMIC_ONLY", channel, ctx, frame_index, base_header)
        if cfg.write_intermediate_arrays:
            dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_cosmic_full.npz", image_full=cosmic)
    
    
    if cfg.write_intermediate_arrays and frame_index < 1:
        dump_npz_snapshot(ctx.output_dir, f"{star.name}_{channel.channel_name}_science_image_without_bg_stars_full.npz", image_full=image)

    return image

def _write_science_component_fits(data: np.ndarray, frame_type: str, channel: Channel, ctx: RunContext, frame_index: int, base_header) -> None:
    header = append_base_frame_header(base_header.copy(), filetype=frame_type, channel=channel, index0=frame_index)
    append_image_stats_header(header, data)
    append_channel_frame_header(header, channel, exptime_s=channel.exposure_s)

    frame = Frame(
        data=data.astype(np.float32, copy=False),
        header=header,
        frame_type=frame_type.lower(),
        channel_tag=channel.channel_name,
    )
    write_fits_frame(frame, ctx, frame_index, channel.exposure_s)

def _report_science_frame_progress(channel, frame_index, total_orbits, roll_angle_start, roll_angle_end):
    orbit_idx = int(np.floor(roll_angle_start / 360.0)) + 1

    start_mod = roll_angle_start % 360.0
    end_mod = roll_angle_end % 360.0

    message = (
        f"{channel.channel_name} | Orbit {orbit_idx}/{total_orbits} | "
        f"Frame {frame_index + 1}/{channel.n_science_frames} | "
        f"Roll {start_mod:.2f}° -> {end_mod:.2f}°"
    )

    print(message)