from loaders.run_waltzer_context import get_repo_root
from pathlib import Path
import logging
from configs.channel_config import SpectroscopyChannel, PhotometryChannel
from configs.user_config import UserConfig
from loaders.load_channel_files import load_effective_area_file, load_spread_profile_file, load_background_file

def load_channels_config(user_cfg: UserConfig):
    repo_root = get_repo_root()
    # load channel config, not cached, pass it through
    nuv_channel = load_channel_config(repo_root / "configs" / "waltzer_nuv.cfg", user_cfg.exposure_NUV_s)
    vis_channel = load_channel_config(repo_root / "configs" / "waltzer_vis.cfg", user_cfg.exposure_VIS_s)
    ir_channel  = load_channel_config(repo_root / "configs" / "waltzer_ir.cfg", user_cfg.exposure_IR_s)

    return nuv_channel, vis_channel, ir_channel



def load_channel_config(path: Path, exposure_s:float):
    logging.info("Reading channel config from %s", path)

    raw = _parse_simple_kv(path)

    channel_name=str(raw["channel_name"]).strip()
    x_pixels=_as_int(raw["x_pixels"], key="x_pixels")
    y_pixels=_as_int(raw["y_pixels"], key="y_pixels")
    resolution_factor=_as_float(raw["resolution_factor"], key="resolution_factor")
    dark_noise=_as_float(raw["dark_noise"], key="dark_noise")
    dark_current_sigma=_as_float(raw["dark_current_sigma"], key="dark_current_sigma")
    read_noise=_as_float(raw["read_noise"], key="read_noise")
    effective_area_file=str(raw.get("effective_area_file", "")).strip()
    bias_offset=_as_float(raw.get("bias_offset", 0.0), key="bias_offset")
    ccd_gain=_as_float(raw.get("ccd_gain", 1.0), key="ccd_gain")
    mode=_as_int(raw["mode"], key="mode")
    spread_profile_file=str(raw.get("spread_profile_file", "")).strip()
    spread_half_height_pix=_as_optional_int(raw.get("spread_half_height_pix", None)) or 0
    source_file=str(path)



    if channel_name == "IR":
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
            source_file=source_file,
        )

    wavelength, effective_area, pixel_scale = load_effective_area_file(effective_area_file)
    if len(wavelength) != x_pixels:
        logging.error("%s: effective_area_file=%s len(wavelength)=%d != x_pixels=%d source_file=%s", channel_name, effective_area_file, len (wavelength), x_pixels, source_file, )
        raise ValueError(
            f"{channel_name}: effective_area_file={effective_area_file} "
            f"len(wavelength)={len(wavelength)} != x_pixels={x_pixels} "
            f"source_file={source_file}"
        )
        
    spread_pos, spread_w, spread_wl_header = load_spread_profile_file(spread_profile_file, channel_name)
    slit_position_x_arcsec = _as_float(raw.get("slit_position_x_arcsec", 0.0), key="slit_position_x_arcsec")
    slit_position_y_arcsec = _as_float(raw.get("slit_position_y_arcsec", 0.0), key="slit_position_y_arcsec")
    slope = _as_float(raw.get("slope", 0.0), key="slope")
    intercept_pixels = _as_float(raw.get("intercept_pixels", 0.0), key="intercept_pixels")

    background_file = str(raw.get("background_file", "")).strip()
    background_wavelength, background_flux = load_background_file(background_file)


    return SpectroscopyChannel(
        channel_name=channel_name,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        resolution_factor=resolution_factor,
        dark_noise=dark_noise,
        dark_current_sigma=dark_current_sigma,
        read_noise=read_noise,
        bias_offset=bias_offset,
        effective_area_file=effective_area_file,
        ccd_gain=ccd_gain,
        exposure_s=exposure_s,
        mode=mode,
        spread_profile_file=spread_profile_file,
        spread_half_height_pix=spread_half_height_pix,
        effective_area_wavelength=wavelength,
        effective_area=effective_area,
        pixel_scale=pixel_scale,
        spread_y_positions=spread_pos,
        spread_y_weights=spread_w,
        spread_y_wavelengths=spread_wl_header,
        source_file=source_file,
        slit_position_x_arcsec=slit_position_x_arcsec,
        slit_position_y_arcsec=slit_position_y_arcsec,
        slope=slope,
        intercept_pixels=intercept_pixels,
        background_wavelength=background_wavelength,
        background_flux=background_flux,    
    )  


def _as_optional_int(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.casefold() == "none":
        return None
    try:
        return int(s)
    except Exception as exc:
        logging.error("Invalid int value: %r", value)
        raise ValueError(f"Invalid int value: {value!r}") from exc

def _as_int(value, *, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        logging.error("Invalid int for key '%s': %r", key, value)
        raise ValueError(f"Invalid int for key '{key}': {value!r}") from exc


def _as_float(value, *, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        logging.error("Invalid float for key '%s': %r", key, value)
        raise ValueError(f"Invalid float for key '{key}': {value!r}") from exc


def _parse_simple_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.error("Channel config file not found at %s", path)
        raise FileNotFoundError(f"Config not found: {path}")

    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if "=" not in s:
            continue
        k, v = (p.strip() for p in s.split("=", 1))
        data[k] = v
    return data