from loaders.run_waltzer_context import get_repo_root
from pathlib import Path
import logging
from configs.channel_config import Channel, PhotometryChannel, SpectroscopyChannel
from configs.user_config import UserConfig
from configs.config_parsing import parse_simple_kv, as_int, as_float, as_bool, as_optional_int, ensure_non_negative
from loaders.load_channel_files_common import load_effective_area_file, load_background_from_global_cfg
from loaders.load_channel_files_spectroscopy import load_spread_profile_file_spectroscopy, load_polarization_file, validate_polarization_config
from loaders.load_channel_files_photometry import load_psf_image_file
from configs.global_config import get_global_config
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

_EFF_AREA_COLORS = {"NUV": "C0", "VIS": "C1", "NIR": "#2ca02c"}

def _eff_area_legend_label(channel: Channel):

    wl = channel.effective_area_wavelength
    wa, wb = float(np.min(wl)), float(np.max(wl))
    return f"{channel.channel_name} ({wa:g}–{wb:g} Å)"

def _plot_eff_area_line(ax, channel: Channel, lw: float = 1.0):

    cc = _EFF_AREA_COLORS.get(channel.channel_name)
    kw = {"lw": lw, "label": _eff_area_legend_label(channel)}
    if cc is not None:
        kw["color"] = cc
    ax.plot(channel.effective_area_wavelength, channel.effective_area, **kw)

def load_channels_config(user_cfg: UserConfig, run_ctx):
    repo_root = get_repo_root()
    background = load_background_from_global_cfg()
    cfg = get_global_config()

    nuv_channel = None
    vis_channel = None
    nir_channel = None

    if cfg.run_nuv:
        nuv_channel = load_channel_spectroscopy(repo_root / "configs" / "waltzer_nuv.cfg", user_cfg.exposure_NUV_s, background)
        plot_effective_area(nuv_channel, run_ctx)
    if cfg.run_vis:
        vis_channel = load_channel_spectroscopy(repo_root / "configs" / "waltzer_vis.cfg", user_cfg.exposure_VIS_s, background)
        plot_effective_area(vis_channel, run_ctx)

    if cfg.run_nir:
        nir_channel = load_channel_photometry(repo_root / "configs" / "waltzer_nir.cfg", user_cfg.exposure_NIR_s, background)
        plot_effective_area(nir_channel, run_ctx)

    loaded_channels = [c for c in (nuv_channel, vis_channel, nir_channel) if c is not None]
    if loaded_channels:
        plot_effective_area_all_channels(loaded_channels, run_ctx)

    return nuv_channel, vis_channel, nir_channel

def plot_effective_area(channel, run_ctx):

    fig, ax = plt.subplots(figsize=(8, 2.85))

    _plot_eff_area_line(ax, channel, lw=1.0)

    ax.margins(y=0.03)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="minor", direction="out", length=2.4, width=0.6)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Effective Area [$\mathrm{cm}^2$]")

    ax.set_title(_eff_area_legend_label(channel))

    fig.tight_layout()

    output_file = run_ctx.output_dir / f"{channel.channel_name}_effective_area.png"
    fig.savefig(output_file, dpi=300)

    plt.close(fig)

def _pad_axis_limits(v_min: float, v_max: float, frac: float = 0.03):

    span = v_max - v_min
    pad = (frac * span) if span > 0 else max(abs(v_max), 1.0) * frac
    return v_min - pad, v_max + pad

def _eff_area_xy_bounds(channel_group: list[Channel]):

    wl = np.concatenate([c.effective_area_wavelength for c in channel_group])
    ea = np.concatenate([c.effective_area for c in channel_group])
    return float(np.min(wl)), float(np.max(wl)), float(np.min(ea)), float(np.max(ea))

def _decorate_eff_area_ax(ax):

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="minor", direction="out", length=2.4, width=0.6)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Effective Area [$\mathrm{cm}^2$]")

def _finish_eff_area_ax(ax, x0: float, x1: float, y0: float, y1: float):

    ax.set_xlim(*_pad_axis_limits(x0, x1))
    ax.set_ylim(*_pad_axis_limits(y0, y1))
    _decorate_eff_area_ax(ax)

def plot_effective_area_all_channels(channels: list[Channel], run_ctx):

    spec = [c for c in channels if c.channel_name in ("NUV", "VIS")]
    nir = [c for c in channels if c.channel_name == "NIR"]
    if not spec and not nir:
        return

    if spec and nir:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 5.4), gridspec_kw={"height_ratios": [1, 1]})
        for c in spec:
            _plot_eff_area_line(ax_top, c)
        for c in nir:
            _plot_eff_area_line(ax_bot, c)
        ax_top.legend(loc="best", fontsize=9)
        ax_bot.legend(loc="best", fontsize=9)
        _finish_eff_area_ax(ax_top, *_eff_area_xy_bounds(spec))
        _finish_eff_area_ax(ax_bot, *_eff_area_xy_bounds(nir))
        fig.tight_layout()
    else:
        grp = spec or nir
        fig, ax = plt.subplots(figsize=(8, 2.85))
        for c in grp:
            _plot_eff_area_line(ax, c)
        ax.legend(loc="best", fontsize=9)
        _finish_eff_area_ax(ax, *_eff_area_xy_bounds(grp))
        fig.tight_layout()

    fig.savefig(run_ctx.output_dir / "effective_area_all_channels.png", dpi=300)
    plt.close(fig)

def load_common_channel(path: Path, exposure_s: float):
    raw = parse_simple_kv(path)
    channel_name=str(raw["channel_name"]).strip()
    source_file=str(path)
    x_pixels=as_int(raw["x_pixels"], key="x_pixels")
    y_pixels=as_int(raw["y_pixels"], key="y_pixels")
    resolution_factor=as_float(raw["resolution_factor"], key="resolution_factor")
    dark_current=as_float(raw["dark_current"], key="dark_current")
    dark_current_noise=as_float(raw["dark_current_noise"], key="dark_current_noise")
    read_noise=as_float(raw["read_noise"], key="read_noise")
    bias_offset=as_float(raw.get("bias_offset", 0.0), key="bias_offset")
    ccd_gain=as_float(raw.get("ccd_gain", 1.0), key="ccd_gain")
    n_science_frames = _compute_n_science_frames(channel_name, exposure_s)
    effective_area_file=str(raw.get("effective_area_file", "")).strip()
    effective_area_wavelength, effective_area, pixel_scale = load_effective_area_file(effective_area_file)

    channel_properties = raw, channel_name, source_file
    detector_common = x_pixels, y_pixels, resolution_factor, dark_current, dark_current_noise, read_noise, bias_offset, ccd_gain
    exposure_common = exposure_s, n_science_frames
    effective_area_common = effective_area_file, effective_area_wavelength, effective_area, pixel_scale
    
    return channel_properties, detector_common, exposure_common, effective_area_common

def load_channel_photometry(path: Path, exposure_s: float, background: dict):
    # Common values provided by load_common_channel()
    channel_properties, detector_common, exposure_common, effective_area_common = load_common_channel(path, exposure_s)
    raw, channel_name, source_file = channel_properties
    x_pixels, y_pixels, resolution_factor, dark_current, dark_current_noise, read_noise, bias_offset, ccd_gain = detector_common
    exposure_s, n_science_frames = exposure_common
    effective_area_file, effective_area_wavelength, effective_area, pixel_scale = effective_area_common

    # Photometry-only parsing
    psf_file = str(raw.get("psf_file", "")).strip()
    psf_image, psf_center_y, psf_center_x = load_psf_image_file(psf_file, channel_name)
    source_position_x_arcsec = as_float(raw.get("source_position_x_arcsec", 0.0), key="source_position_x_arcsec")
    source_position_y_arcsec = as_float(raw.get("source_position_y_arcsec", 0.0), key="source_position_y_arcsec")
    draw_aperture_photometry_overlay = as_bool(raw.get("draw_aperture_photometry_overlay", 0), key="draw_aperture_photometry_overlay")

    # Final channel object
    return PhotometryChannel(
        channel_name=channel_name,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        resolution_factor=resolution_factor,
        dark_current=dark_current,
        dark_current_noise=dark_current_noise,
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
        draw_aperture_photometry_overlay=draw_aperture_photometry_overlay,
        background_type=background["background_type"],
        background_wavelength=background["background_wavelength"],
        background_flux=background["background_flux"],
        sky_pixel_area_arcsec2=background["sky_pixel_area_arcsec2"],
        zod_dist=background["zod_dist"],
        zod_spectrum_wavelength=background["zod_spectrum_wavelength"],
        zod_spectrum_flux=background["zod_spectrum_flux"],
    )

def load_channel_spectroscopy(path: Path, exposure_s: float, background: dict):
    channel_properties, detector_common, exposure_common, effective_area_common = load_common_channel(path, exposure_s)
    raw, channel_name, source_file = channel_properties
    x_pixels, y_pixels, resolution_factor, dark_current, dark_current_noise, read_noise, bias_offset, ccd_gain = detector_common
    exposure_s, n_science_frames = exposure_common
    effective_area_file, effective_area_wavelength, effective_area, pixel_scale = effective_area_common

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

    slit_width_arcsec = ensure_non_negative(
        as_float(raw["slit_width_arcsec"], key="slit_width_arcsec"),
        key=f"{channel_name}: slit_width_arcsec",
    )
    slit_length_arcsec = ensure_non_negative(
        as_float(raw["slit_length_arcsec"], key="slit_length_arcsec"),
        key=f"{channel_name}: slit_length_arcsec",
    )

    smear_shift_pixels = slit_width_arcsec / pixel_scale

    observation_mode = str(raw.get("observation_mode", "spectroscopy")).strip().lower()
    polarization_file = str(raw.get("polarization_file", "")).strip()
    beam_separation_pix = as_int(raw.get("beam_separation_pix", 0), key="beam_separation_pix")
    polarization_wavelength, polarization_delta = load_polarization_file(polarization_file, channel_name)
    validate_polarization_config(channel_name, observation_mode, polarization_wavelength, polarization_delta, beam_separation_pix, y_pixels)

    return SpectroscopyChannel(
        channel_name=channel_name,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        resolution_factor=resolution_factor,

        dark_current=dark_current,
        dark_current_noise=dark_current_noise,
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
        observation_mode=observation_mode,
        polarization_file=polarization_file or None,
        polarization_wavelength=polarization_wavelength,
        polarization_delta=polarization_delta,
        beam_separation_pix=beam_separation_pix,
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

def _ensure_effective_area_matches_x_pixels(channel_name: str, effective_area_file: str, effective_area_wavelength, x_pixels: int, source_file: str) -> None:

    if len(effective_area_wavelength) != x_pixels:
        logging.error("%s: effective_area_file=%s len(wavelength)=%d != x_pixels=%d source_file=%s", channel_name, effective_area_file, len(effective_area_wavelength), x_pixels, source_file)
        raise ValueError(f"{channel_name}: effective_area_file={effective_area_file} len(wavelength)={len(effective_area_wavelength)} != x_pixels={x_pixels} source_file={source_file}")

def _compute_n_science_frames(channel_name: str, exposure_s: float) -> int:
    cfg = get_global_config()
    frame_time_s = exposure_s + cfg.readout_gap_s
    n_science_frames = int(cfg.orbit_total_duration_s / frame_time_s)
    return n_science_frames

