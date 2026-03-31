import numpy as np
from pathlib import Path
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
from instrument.spectrum_spread import spread_1d_spectrum_to_2d, get_spectrum_placement
from instrument.psf_spread import spread_1d_photometry_to_2d
from instrument.science_image import _build_science_image_without_bg_stars, _create_per_exposure

SNAPSHOT_BASE = Path(__file__).parent / "snapshots"

# Fixed detector geometry from configs/waltzer_*.cfg
NUV_X_PIXELS = 2048
NUV_Y_PIXELS = 515
VIS_X_PIXELS = 4096
VIS_Y_PIXELS = 2048
NIR_X_PIXELS = 640
NIR_Y_PIXELS = 512


def _load_npz(path: Path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _load_effective_area(channel: str):
    path = SNAPSHOT_BASE / f"{channel}_effective_area.txt"
    first_line = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    pixel_scale = float(first_line.split("=", 1)[1])
    table = np.loadtxt(path, dtype=np.float64)
    return table[:, 0], table[:, 1], pixel_scale


def run_snapshot_convolved_counts_full_photometry(star_name: str, channel: str, tmp_path: Path, make_photometry_channel, make_star, make_run_context):
    photons_full = _load_npz(SNAPSHOT_BASE / f"{star_name}_FluxCalc_8_photons_star_full.npz")
    convolved_counts = _load_npz(SNAPSHOT_BASE / f"{star_name}_{channel}_convolved_counts_full.npz")
    effective_area_wavelength, effective_area, pixel_scale = _load_effective_area(channel)

    channel_dims = {
        "NIR": (NIR_X_PIXELS, NIR_Y_PIXELS),
    }
    x_pixels, y_pixels = channel_dims[channel]
    channel_cfg = make_photometry_channel(channel_name=channel, effective_area_wavelength=effective_area_wavelength, effective_area=effective_area, pixel_scale=float(pixel_scale), x_pixels=x_pixels, y_pixels=y_pixels)
    star = make_star(name=star_name)
    ctx = make_run_context(output_dir=tmp_path)

    got_counts = compute_counts_per_s_px_one_channel(photons_full["photons_star"], photons_full["wavelengths"], channel_cfg, ctx, star)
    np.testing.assert_allclose(got_counts, convolved_counts["counts_s_px_convolved"], rtol=1e-7, atol=0.0)


def run_snapshot_convolved_counts_full_spectroscopy(star_name: str, channel: str, tmp_path: Path, make_spectroscopy_channel, make_star, make_run_context):
    photons_full = _load_npz(SNAPSHOT_BASE / f"{star_name}_FluxCalc_8_photons_star_full.npz")
    snap = _load_npz(SNAPSHOT_BASE / f"{star_name}_{channel}_convolved_counts_full.npz")
    effective_area_wavelength, effective_area, pixel_scale = _load_effective_area(channel)

    channel_dims = {
        "NUV": (NUV_X_PIXELS, NUV_Y_PIXELS),
        "VIS": (VIS_X_PIXELS, VIS_Y_PIXELS),
    }
    x_pixels, y_pixels = channel_dims[channel]
    channel_cfg = make_spectroscopy_channel(channel_name=channel, effective_area_wavelength=effective_area_wavelength, effective_area=effective_area, pixel_scale=float(pixel_scale), x_pixels=x_pixels, y_pixels=y_pixels)
    star = make_star(name=star_name)
    ctx = make_run_context(output_dir=tmp_path)

    got_counts = compute_counts_per_s_px_one_channel(photons_full["photons_star"], photons_full["wavelengths"], channel_cfg, ctx, star)
    np.testing.assert_allclose(got_counts, snap["counts_s_px_convolved"], rtol=1e-7, atol=0.0)


def run_snapshot_spread_image_2d_nuv_profile(star_name: str, make_spectroscopy_channel):

    convolved = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_convolved_counts_full.npz")
    spread_image = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_spread_image_2d_full.npz")
    spread_profile = _load_npz(SNAPSHOT_BASE / "NUV_spread_profile_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("NUV")

    nuv_channel = make_spectroscopy_channel(channel_name="NUV", mode=1, x_pixels=NUV_X_PIXELS, y_pixels=NUV_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=spread_profile["spread_y_positions"], spread_y_weights=spread_profile["spread_y_weights"], spread_y_wavelengths=spread_profile["spread_y_wavelengths"])

    placement = get_spectrum_placement(nuv_channel)
    image_full = spread_1d_spectrum_to_2d(convolved["counts_s_px_convolved"], nuv_channel, placement, announce_user=False)

    np.testing.assert_allclose(image_full, spread_image["image_full"], rtol=1e-3, atol=1e-2)

def run_snapshot_spread_image_2d_vis_gaussian(star_name: str, make_spectroscopy_channel):
    convolved = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_convolved_counts_full.npz")
    spread_image = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_spread_image_2d_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("VIS")

    vis_channel = make_spectroscopy_channel(channel_name="VIS", observation_mode="spectroscopy", mode=1, x_pixels=VIS_X_PIXELS, y_pixels=VIS_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=None, spread_y_weights=None, spread_y_wavelengths=None, spread_half_height_pix=10)

    placement = get_spectrum_placement(vis_channel)
    image_full = spread_1d_spectrum_to_2d(convolved["counts_s_px_convolved"], vis_channel, placement, announce_user=False)

    np.testing.assert_allclose(image_full, spread_image["image_full"], rtol=1e-6, atol=1e-6)

def run_snapshot_spread_image_2d_nir_psf(star_name: str, make_photometry_channel):
    convolved = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_convolved_counts_full.npz")
    spread_image = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_spread_image_full.npz")
    spread_profile = _load_npz(SNAPSHOT_BASE / "NIR_psf_profile_full.npz")

    nir_channel = make_photometry_channel(channel_name="NIR", x_pixels=NIR_X_PIXELS, y_pixels=NIR_Y_PIXELS, psf_image=spread_profile["psf_image"], psf_center_x=int(spread_profile["psf_center_x"]), psf_center_y=int(spread_profile["psf_center_y"]), source_position_x_arcsec=float(spread_profile["source_position_x_arcsec"]), source_position_y_arcsec=float(spread_profile["source_position_y_arcsec"]))

    image_full = spread_1d_photometry_to_2d(convolved["counts_s_px_convolved"], nir_channel, announce_user=False)

    np.testing.assert_allclose(image_full, spread_image["image_full"], rtol=1e-6, atol=1e-6)


# Tests: convolved counts full snapshots
# Behavior: NIR full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_HD2685_convolved_counts_full_NIR(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_photometry("HD 2685", "NIR", tmp_path, make_photometry_channel, make_star, make_run_context)


# Tests: convolved counts full snapshots
# Behavior: NUV full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_HD2685_convolved_counts_full_NUV(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_spectroscopy("HD 2685", "NUV", tmp_path, make_spectroscopy_channel, make_star, make_run_context)


# Tests: convolved counts full snapshots
# Behavior: VIS full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_HD2685_convolved_counts_full_VIS(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_spectroscopy("HD 2685", "VIS", tmp_path, make_spectroscopy_channel, make_star, make_run_context)


# Tests: spread image 2d snapshots
# Behavior: NUV profile spread from convolved counts matches snapshot image
def test_HD2685_spread_image_2d_NUV_profile_pipeline(make_spectroscopy_channel):
    run_snapshot_spread_image_2d_nuv_profile("HD 2685", make_spectroscopy_channel)


# Tests: spread image 2d snapshots
# Behavior: VIS Gaussian spread from convolved counts matches snapshot image
def test_HD2685_spread_image_2d_VIS_gaussian_pipeline(make_spectroscopy_channel):
    run_snapshot_spread_image_2d_vis_gaussian("HD 2685", make_spectroscopy_channel)


# Tests: spread image 2d snapshots
# Behavior: NIR photometry PSF spread from convolved counts matches snapshot image
def test_HD2685_spread_image_2d_NIR_psf_pipeline(make_photometry_channel):
    run_snapshot_spread_image_2d_nir_psf("HD 2685", make_photometry_channel)


# Tests: convolved counts full snapshots
# Behavior: NIR full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_KELT9_convolved_counts_full_NIR(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_photometry("KELT-9", "NIR", tmp_path, make_photometry_channel, make_star, make_run_context)


# Tests: convolved counts full snapshots
# Behavior: NUV full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_KELT9_convolved_counts_full_NUV(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_spectroscopy("KELT-9", "NUV", tmp_path, make_spectroscopy_channel, make_star, make_run_context)


# Tests: convolved counts full snapshots
# Behavior: VIS full photons -> Gaussian broadening -> detector convolution matches snapshot
def test_KELT9_convolved_counts_full_VIS(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_star, make_run_context):
    cfg = make_global_config()
    monkeypatch.setattr("instrument.prepare_detector_images.get_global_config", lambda: cfg)
    run_snapshot_convolved_counts_full_spectroscopy("KELT-9", "VIS", tmp_path, make_spectroscopy_channel, make_star, make_run_context)


# Tests: spread image 2d snapshots
# Behavior: NUV profile spread from convolved counts matches snapshot image
def test_KELT9_spread_image_2d_NUV_profile_pipeline(make_spectroscopy_channel):
    run_snapshot_spread_image_2d_nuv_profile("KELT-9", make_spectroscopy_channel)


# Tests: spread image 2d snapshots
# Behavior: VIS Gaussian spread from convolved counts matches snapshot image
def test_KELT9_spread_image_2d_VIS_gaussian_pipeline(make_spectroscopy_channel):
    run_snapshot_spread_image_2d_vis_gaussian("KELT-9", make_spectroscopy_channel)


# Tests: spread image 2d snapshots
# Behavior: NIR photometry PSF spread from convolved counts matches snapshot image
def test_KELT9_spread_image_2d_NIR_psf_pipeline(make_photometry_channel):
    run_snapshot_spread_image_2d_nir_psf("KELT-9", make_photometry_channel)



def run_snapshot_science_image_without_bg_stars_nuv(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_spread_image_2d_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_background_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_science_image_without_bg_stars_full.npz")

    spread_profile = _load_npz(SNAPSHOT_BASE / "NUV_spread_profile_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("NUV")

    channel = make_spectroscopy_channel(channel_name="NUV", mode=1, x_pixels=NUV_X_PIXELS, y_pixels=NUV_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=spread_profile["spread_y_positions"], spread_y_weights=spread_profile["spread_y_weights"], spread_y_wavelengths=spread_profile["spread_y_wavelengths"], exposure_s=300.0)

    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False)
    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))
    star = make_star(name=star_name)
    base_header = {}

    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])

    stellar = stellar["image_full"] * channel.exposure_s
    image_full = _build_science_image_without_bg_stars(stellar, background["image_full"], channel, ctx, cfg, star, 0, base_header)


    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)

def run_snapshot_science_image_without_bg_stars_vis(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_spread_image_2d_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_background_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_science_image_without_bg_stars_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("VIS")

    channel_cfg = make_spectroscopy_channel(channel_name="VIS", observation_mode="spectroscopy", mode=1, x_pixels=VIS_X_PIXELS, y_pixels=VIS_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=None, spread_y_weights=None, spread_y_wavelengths=None, spread_half_height_pix=10, exposure_s=60.0)

    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False)
    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))
    star = make_star(name=star_name)
    base_header = {}

    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])

    stellar = stellar["image_full"] * channel_cfg.exposure_s
    image_full = _build_science_image_without_bg_stars(stellar, background["image_full"], channel_cfg, ctx, cfg, star, 0, base_header)

    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)


def run_snapshot_science_image_without_bg_stars_nir(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_spread_image_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_background_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_science_image_without_bg_stars_full.npz")
    spread_profile = _load_npz(SNAPSHOT_BASE / "NIR_psf_profile_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("NIR")

    channel_cfg = make_photometry_channel(channel_name="NIR", x_pixels=NIR_X_PIXELS, y_pixels=NIR_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, psf_image=spread_profile["psf_image"], psf_center_x=int(spread_profile["psf_center_x"]), psf_center_y=int(spread_profile["psf_center_y"]), source_position_x_arcsec=float(spread_profile["source_position_x_arcsec"]), source_position_y_arcsec=float(spread_profile["source_position_y_arcsec"]), exposure_s=60.0)

    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False)
    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))
    star = make_star(name=star_name)
    base_header = {}

    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])


    stellar = stellar["image_full"] * channel_cfg.exposure_s
    image_full = _build_science_image_without_bg_stars(stellar, background["image_full"], channel_cfg, ctx, cfg, star, 0, base_header)

    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)



def test_HD2685_science_image_without_bg_stars_NUV_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_nuv("HD 2685", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_HD2685_science_image_without_bg_stars_VIS_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_vis("HD 2685", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_HD2685_science_image_without_bg_stars_NIR_pipeline(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_nir("HD 2685", tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star)


def test_KELT9_science_image_without_bg_stars_NUV_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_nuv("KELT-9", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_KELT9_science_image_without_bg_stars_VIS_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_vis("KELT-9", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_KELT9_science_image_without_bg_stars_NIR_pipeline(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    run_snapshot_science_image_without_bg_stars_nir("KELT-9", tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star)



def run_snapshot_science_image_with_bg_stars_nuv(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_spread_image_2d_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_background_component_full.npz")
    background_stars = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_background_stars_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_NUV_science_image_full.npz")
    spread_profile = _load_npz(SNAPSHOT_BASE / "NUV_spread_profile_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("NUV")

    channel = make_spectroscopy_channel(channel_name="NUV", mode=1, x_pixels=NUV_X_PIXELS, y_pixels=NUV_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=spread_profile["spread_y_positions"], spread_y_weights=spread_profile["spread_y_weights"], spread_y_wavelengths=spread_profile["spread_y_wavelengths"], exposure_s=300.0)

    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False, write_background_star_footprint_on_science_frame=False)

    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))

    star = make_star(name=star_name)

    base_header = {}
    frame_index = 0
    exposure = channel.exposure_s
    readout_gap_s = cfg.readout_gap_s
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    time_s = frame_index * (exposure + readout_gap_s)
    roll_angle_start = 360.0 * (time_s / orbit_duration_s)
    roll_angle_end = 360.0 * ((time_s + exposure) / orbit_duration_s)

    monkeypatch.setattr("instrument.science_image.generate_background_star_visibility_on_science_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_background_star_spectroscopy_image", lambda channel, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index: (background_stars["image_full"], {}))

    stellar_component = stellar["image_full"] * exposure
    image_full = _create_per_exposure(stellar_component, background["image_full"], channel, ctx, cfg, star, None, frame_index, roll_angle_start, roll_angle_end, base_header)

    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)


def run_snapshot_science_image_with_bg_stars_vis(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_spread_image_2d_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_background_component_full.npz")
    background_stars = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_background_stars_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_VIS_science_image_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("VIS")

    channel = make_spectroscopy_channel(channel_name="VIS", observation_mode="spectroscopy", mode=1, x_pixels=VIS_X_PIXELS, y_pixels=VIS_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, spread_y_positions=None, spread_y_weights=None, spread_y_wavelengths=None, spread_half_height_pix=10, exposure_s=60.0)
    
    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False, write_background_star_footprint_on_science_frame=False)
    
    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))
    
    star = make_star(name=star_name)
    base_header = {}
    frame_index = 0
    exposure = channel.exposure_s
    readout_gap_s = cfg.readout_gap_s
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    time_s = frame_index * (exposure + readout_gap_s)
    roll_angle_start = 360.0 * (time_s / orbit_duration_s)
    roll_angle_end = 360.0 * ((time_s + exposure) / orbit_duration_s)

    monkeypatch.setattr("instrument.science_image.generate_background_star_visibility_on_science_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_background_star_spectroscopy_image", lambda channel, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index: (background_stars["image_full"], {}))

    stellar_component = stellar["image_full"] * exposure
    image_full = _create_per_exposure(stellar_component, background["image_full"], channel, ctx, cfg, star, None, frame_index, roll_angle_start, roll_angle_end, base_header)

    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)


def run_snapshot_science_image_with_bg_stars_nir(star_name: str, tmp_path: Path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    stellar = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_spread_image_full.npz")
    background = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_background_component_full.npz")
    background_stars = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_background_stars_component_full.npz")
    bias = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_bias_full.npz")
    dark = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_dark_full.npz")
    photon_noise = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_photon_noise_full.npz")
    flat = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_flat_full.npz")
    cosmic = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_cosmic_full.npz")
    snapshot = _load_npz(SNAPSHOT_BASE / f"{star_name}_NIR_science_image_full.npz")
    spread_profile = _load_npz(SNAPSHOT_BASE / "NIR_psf_profile_full.npz")
    effective_area_wavelength, _, pixel_scale = _load_effective_area("NIR")

    channel = make_photometry_channel(channel_name="NIR", x_pixels=NIR_X_PIXELS, y_pixels=NIR_Y_PIXELS, pixel_scale=float(pixel_scale), effective_area_wavelength=effective_area_wavelength, psf_image=spread_profile["psf_image"], psf_center_x=int(spread_profile["psf_center_x"]), psf_center_y=int(spread_profile["psf_center_y"]), source_position_x_arcsec=float(spread_profile["source_position_x_arcsec"]), source_position_y_arcsec=float(spread_profile["source_position_y_arcsec"]), exposure_s=60.0)

    cfg = make_global_config(write_science_frame_component_png=False, write_science_frame_component_fits=False, write_background_star_footprint_on_science_frame=False)

    ctx = make_run_context(output_dir=tmp_path, target_name=star_name.replace(" ", "_"))

    star = make_star(name=star_name)

    base_header = {}
    frame_index = 0
    exposure = channel.exposure_s
    readout_gap_s = cfg.readout_gap_s
    orbit_duration_s = cfg.orbit_duration_minutes * 60.0
    time_s = frame_index * (exposure + readout_gap_s)
    roll_angle_start = 360.0 * (time_s / orbit_duration_s)
    roll_angle_end = 360.0 * ((time_s + exposure) / orbit_duration_s)

    monkeypatch.setattr("instrument.science_image.generate_background_star_visibility_on_science_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr("instrument.science_image.generate_bias_image", lambda channel: bias["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_dark_image", lambda channel: dark["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_photon_noise_from_spectra2d", lambda target_star_component: photon_noise["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_flat_image", lambda channel: flat["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_cosmic_rays", lambda channel, cfg: cosmic["image_full"])
    monkeypatch.setattr("instrument.science_image.generate_background_star_photometry_image", lambda channel, background_stars_catalog, roll_angle_start, roll_angle_end, frame_index: (background_stars["image_full"], {}))

    stellar_component = stellar["image_full"] * exposure
    image_full = _create_per_exposure(stellar_component, background["image_full"], channel, ctx, cfg, star, None, frame_index, roll_angle_start, roll_angle_end, base_header)

    np.testing.assert_allclose(image_full, snapshot["image_full"], rtol=1e-6, atol=1e-6)

def test_HD2685_science_image_with_bg_stars_NUV_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_nuv("HD 2685", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_HD2685_science_image_with_bg_stars_VIS_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_vis("HD 2685", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_HD2685_science_image_with_bg_stars_NIR_pipeline(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_nir("HD 2685", tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star)


def test_KELT9_science_image_with_bg_stars_NUV_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_nuv("KELT-9", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_KELT9_science_image_with_bg_stars_VIS_pipeline(tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_vis("KELT-9", tmp_path, monkeypatch, make_global_config, make_spectroscopy_channel, make_run_context, make_star)


def test_KELT9_science_image_with_bg_stars_NIR_pipeline(tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star):
    run_snapshot_science_image_with_bg_stars_nir("KELT-9", tmp_path, monkeypatch, make_global_config, make_photometry_channel, make_run_context, make_star)