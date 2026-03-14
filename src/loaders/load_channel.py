from loaders.run_waltzer_context import get_repo_root
from pathlib import Path
import logging
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from configs.user_config import UserConfig
from configs.config_parsing import parse_simple_kv, as_int, as_float, as_optional_int
from loaders.load_channel_files_common import load_effective_area_file, load_background_file, load_zod_dist_file, load_zod_spectrum_file
from loaders.load_channel_files_spectroscopy import load_spread_profile_file_spectroscopy
from loaders.load_channel_files_photometry import load_psf_image_file
from configs.global_config import get_global_config


def load_channels_config(user_cfg: UserConfig, ctx):
    repo_root = get_repo_root()
    background = _load_background_from_global_cfg()
    cfg = get_global_config()

    nuv_channel = None
    vis_channel = None
    nir_channel = None

    if cfg.run_nuv:
        nuv_channel = load_channel_config(repo_root / "configs" / "waltzer_nuv.cfg", user_cfg.exposure_NUV_s, ctx, background)
    if cfg.run_vis:
        vis_channel = load_channel_config(repo_root / "configs" / "waltzer_vis.cfg", user_cfg.exposure_VIS_s, ctx, background)
    if cfg.run_nir:
        nir_channel = load_channel_config(repo_root / "configs" / "waltzer_nir.cfg", user_cfg.exposure_IR_s, ctx, background)
    
    return nuv_channel, vis_channel, nir_channel


def load_channel_config(path: Path, exposure_s: float, ctx, background: dict):

    logging.info("Reading channel config from %s", path)

    raw = parse_simple_kv(path)
    channel_name=str(raw["channel_name"]).strip()
    x_pixels=as_int(raw["x_pixels"], key="x_pixels")
    y_pixels=as_int(raw["y_pixels"], key="y_pixels")
    resolution_factor=as_float(raw["resolution_factor"], key="resolution_factor")
    dark_noise=as_float(raw["dark_noise"], key="dark_noise")
    dark_current_sigma=as_float(raw["dark_current_sigma"], key="dark_current_sigma")
    read_noise=as_float(raw["read_noise"], key="read_noise")
    bias_offset=as_float(raw.get("bias_offset", 0.0), key="bias_offset")
    ccd_gain=as_float(raw.get("ccd_gain", 1.0), key="ccd_gain")
    source_file=str(path)

    n_science_frames = _compute_n_science_frames(channel_name, exposure_s)

    effective_area_file=str(raw.get("effective_area_file", "")).strip()
    effective_area_wavelength, effective_area, pixel_scale = load_effective_area_file(effective_area_file)

    if channel_name == "NIR":
        psf_file = str(raw.get("psf_file", "")).strip()
        psf_image, psf_center_y, psf_center_x = load_psf_image_file(psf_file, channel_name, ctx)
        source_position_x_arcsec = as_float(raw.get("source_position_x_arcsec", 0.0), key="source_position_x_arcsec")
        source_position_y_arcsec = as_float(raw.get("source_position_y_arcsec", 0.0), key="source_position_y_arcsec")
        
        return PhotometryChannel(
            channel_name=channel_name,
            x_pixels=x_pixels,
            y_pixels=y_pixels,
            resolution_factor=resolution_factor,
            dark_noise=dark_noise,
            dark_current_sigma=dark_current_sigma,
            read_noise=read_noise,
            bias_offset=bias_offset,
            ccd_gain=ccd_gain,
            exposure_s=exposure_s,
            n_science_frames=n_science_frames,
            source_file=source_file,
            effective_area_file=effective_area_file,
            effective_area_wavelength=effective_area_wavelength,
            effective_area=effective_area,
            pixel_scale=pixel_scale,
            psf_file=psf_file,
            psf_image=psf_image,
            psf_center_x=psf_center_x,
            psf_center_y=psf_center_y,
            source_position_x_arcsec=source_position_x_arcsec,
            source_position_y_arcsec=source_position_y_arcsec,
            background_type=background["background_type"],
            background_wavelength=background["background_wavelength"],
            background_flux=background["background_flux"],
            sky_pixel_area_arcsec2=background["sky_pixel_area_arcsec2"],
            zod_dist=background["zod_dist"],
            zod_spectrum_wavelength=background["zod_spectrum_wavelength"],
            zod_spectrum_flux=background["zod_spectrum_flux"],

        )

    # Spectroscopy only:
    # effective area only matches x pixels in spectroscopy
    _ensure_effective_area_matches_x_pixels(channel_name, effective_area_file, effective_area_wavelength, x_pixels, source_file)
    
    mode=as_int(raw["mode"], key="mode")
    spread_profile_file=str(raw.get("spread_profile_file", "")).strip()
    spread_half_height_pix=as_optional_int(raw.get("spread_half_height_pix", None)) or 0
        
    spread_pos, spread_w, spread_wl_header = load_spread_profile_file_spectroscopy(spread_profile_file, channel_name)
    slit_position_x_arcsec = as_float(raw.get("slit_position_x_arcsec", 0.0), key="slit_position_x_arcsec")
    slit_position_y_arcsec = as_float(raw.get("slit_position_y_arcsec", 0.0), key="slit_position_y_arcsec")
    slope = as_float(raw.get("slope", 0.0), key="slope")
    intercept_pixels = as_float(raw.get("intercept_pixels", 0.0), key="intercept_pixels")

    slit_width_arcsec = _ensure_positive(as_float(raw["slit_width_arcsec"], key="slit_width_arcsec"), "slit_width_arcsec", channel_name)
    slit_length_arcsec = _ensure_positive(as_float(raw["slit_length_arcsec"], key="slit_length_arcsec"), "slit_length_arcsec", channel_name)

    smear_shift_pixels = slit_width_arcsec / pixel_scale

    return SpectroscopyChannel(
        channel_name=channel_name,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        resolution_factor=resolution_factor,

        dark_noise=dark_noise,
        dark_current_sigma=dark_current_sigma,
        read_noise=read_noise,
        bias_offset=bias_offset,
        ccd_gain=ccd_gain,
        
        exposure_s=exposure_s,
        
        n_science_frames=n_science_frames,
        source_file=source_file,
        
        effective_area_file=effective_area_file,
        effective_area_wavelength=effective_area_wavelength,
        effective_area=effective_area,
        pixel_scale=pixel_scale,
        
        mode=mode,
        
        spread_profile_file=spread_profile_file,
        spread_half_height_pix=spread_half_height_pix,

        spread_y_positions=spread_pos,
        spread_y_weights=spread_w,
        spread_y_wavelengths=spread_wl_header,

        slit_position_x_arcsec=slit_position_x_arcsec,
        slit_position_y_arcsec=slit_position_y_arcsec,

        slit_width_arcsec=slit_width_arcsec,
        slit_length_arcsec=slit_length_arcsec,
        slit_half_width_arcsec=0.5 * slit_width_arcsec,
        slit_half_length_arcsec=0.5 * slit_length_arcsec,
        smear_shift_pixels=smear_shift_pixels,
        
        slope=slope,
        intercept_pixels=intercept_pixels,
        background_type=background["background_type"],
        background_wavelength=background["background_wavelength"],
        background_flux=background["background_flux"],
        sky_pixel_area_arcsec2=background["sky_pixel_area_arcsec2"],
        zod_dist=background["zod_dist"],
        zod_spectrum_wavelength=background["zod_spectrum_wavelength"],
        zod_spectrum_flux=background["zod_spectrum_flux"],
    )  


def _load_background_from_global_cfg():
    cfg = get_global_config()

    background_type = (cfg.background_type or "").strip().lower()
    if background_type == "":
        background_type = None

    background_wavelength = None
    background_flux = None
    sky_pixel_area_arcsec2 = cfg.sky_pixel_area_arcsec2  # may be None
    zod_dist = None
    zod_spec_wl = None
    zod_spec_flux = None

    if background_type is None:
        return {
            "background_type": background_type,
            "background_wavelength": background_wavelength,
            "background_flux": background_flux,
            "sky_pixel_area_arcsec2": sky_pixel_area_arcsec2,
            "zod_dist": zod_dist,
            "zod_spectrum_wavelength": zod_spec_wl,
            "zod_spectrum_flux": zod_spec_flux,
        }

    if background_type == "default":
        if cfg.background_file is None:
            raise ValueError("global.cfg: background_type=default requires background_file")
        if cfg.sky_pixel_area_arcsec2 is None:
            raise ValueError("global.cfg: background_type=default requires sky_pixel_area_arcsec2")
        background_wavelength, background_flux = load_background_file(cfg.background_file)

    elif background_type == "calc":
        if cfg.zod_dist_file is None or cfg.zod_spectrum_file is None:
            raise ValueError("global.cfg: background_type=calc requires zod_dist_file and zod_spectrum_file")
        if cfg.sky_pixel_area_arcsec2 is None:
            raise ValueError("global.cfg: background_type=calc requires sky_pixel_area_arcsec2")
        zod_dist = load_zod_dist_file(cfg.zod_dist_file)
        zod_spec_wl, zod_spec_flux = load_zod_spectrum_file(cfg.zod_spectrum_file)

    else:
        raise ValueError(
            f"global.cfg: invalid background_type={background_type!r} (expected: default, calc, or empty)"
        )

    return {
        "background_type": background_type,
        "background_wavelength": background_wavelength,
        "background_flux": background_flux,
        "sky_pixel_area_arcsec2": sky_pixel_area_arcsec2,
        "zod_dist": zod_dist,
        "zod_spectrum_wavelength": zod_spec_wl,
        "zod_spectrum_flux": zod_spec_flux,
    }

def _ensure_effective_area_matches_x_pixels(channel_name: str, effective_area_file: str, effective_area_wavelength, x_pixels: int, source_file: str) -> None:

    if len(effective_area_wavelength) != x_pixels:
        logging.error("%s: effective_area_file=%s len(wavelength)=%d != x_pixels=%d source_file=%s", channel_name, effective_area_file, len(effective_area_wavelength), x_pixels, source_file)
        raise ValueError(f"{channel_name}: effective_area_file={effective_area_file} len(wavelength)={len(effective_area_wavelength)} != x_pixels={x_pixels} source_file={source_file}")

def _ensure_positive(value: float, name: str, channel_name: str) -> float:

    if value <= 0.0:
        raise ValueError(f"{channel_name}: {name} must be > 0, got {value}")
    return float(value)

def _compute_n_science_frames(channel_name: str, exposure_s: float) -> int:
    cfg = get_global_config()
    frame_time_s = exposure_s + cfg.readout_gap_s
    n_science_frames = int(cfg.orbit_total_duration_s / frame_time_s)
    logging.info("Channel %s science frame calculation: orbit_total_duration_s=%g exposure_s=%g readout_gap_s=%g frame_time=%g n_science_frames=%d", channel_name, cfg.orbit_total_duration_s, exposure_s, cfg.readout_gap_s, frame_time_s, n_science_frames)
    return n_science_frames

