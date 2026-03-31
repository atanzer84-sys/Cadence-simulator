import numpy as np
from pathlib import Path
from instrument.prepare_detector_images import compute_counts_per_s_px_one_channel
from instrument.spectrum_spread import spread_1d_spectrum_to_2d, get_spectrum_placement
from instrument.psf_spread import spread_1d_photometry_to_2d

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

