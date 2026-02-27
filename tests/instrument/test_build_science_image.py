import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from instrument.build_science_image import build_science_image
from configs.global_config import GlobalConfig


def _channel():
    # Minimal SpectroscopyChannel-like object for build_science_image tests
    return SimpleNamespace(
        channel_name="NUV",
        x_pixels=4,
        y_pixels=3,
        bias_offset=10.0,
        read_noise=1.0,
        dark_noise=0.5,
        dark_current_sigma=0.1,
        exposure_s=2.0,
        ccd_gain=3.0,
    )


class _NoOpWriteImage:
    def write_image(self, *args, **kwargs):
        # Do nothing – we only care that build_science_image runs without error
        pass


def _ctx(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path,
        write_image_png=_NoOpWriteImage(),
    )


def _star():
    # Minimal Star-like object; build_science_image only passes it through to background generation,
    # which we stub out in these tests.
    return SimpleNamespace(name="TESTSTAR")

def _cfg() -> GlobalConfig:
    # Minimal config for tests: disable cosmic rays and plotting, keep other fields valid
    return GlobalConfig(
        line_core_emission=False,
        interstellar_absorption=False,
        mg2_col=None,
        mg1_col=None,
        fe2_col=None,
        sigmaMg22=0.257,
        sigmaMg21=0.288,
        enable_log_r_fallback=False,
        log_r_teff_threshold=0.0,
        log_r_hot_value=0.0,
        log_r_cool_value=0.0,
        n_non_science_frames=0,
        write_non_science_frames_png=False,
        n_science_frames_per_channel=1,
        write_science_frames_png=False,
        cosmic_rays_min=0,
        cosmic_rays_max=0,
        cosmic_ray_signal_electrons=0,
        cosmic_ray_length_min_px=1,
        cosmic_ray_length_max_px=1,
        test_mode=True,
        produce_Plots=False,
    )


def test_build_science_image_shape_and_gain(tmp_path):
    np.random.seed(0)
    ch = _channel()
    ctx = _ctx(tmp_path)

    # Simple input spectra: all ones
    spectra_2d = np.ones((ch.y_pixels, ch.x_pixels), dtype=float)

    # Background image uses full SpectroscopyChannel + Star; stub it out for this minimal test channel.
    with patch(
        "instrument.build_science_image.generate_Background_Image",
        return_value=np.zeros((ch.y_pixels, ch.x_pixels), dtype=float),
    ):
        image = build_science_image(spectra_2d, ch, ctx, _cfg(), _star())

    assert image.shape == (ch.y_pixels, ch.x_pixels)
    # Final image is scaled by CCD gain
    # We don't assert exact values (noise, bias, dark, photon noise), but mean must be > 0
    assert float(image.mean()) > 0.0


def test_build_science_image_adds_photon_noise(tmp_path):
    """Photon noise introduces stochastic variation on top of the deterministic spectra."""
    np.random.seed(0)
    ch = _channel()
    # Disable other noise sources to isolate photon noise behaviour
    ch.bias_offset = 0.0
    ch.read_noise = 0.0
    ch.dark_noise = 0.0
    ch.dark_current_sigma = 0.0
    ch.ccd_gain = 1.0
    ch.exposure_s = 5.0
    ctx = _ctx(tmp_path)

    spectra_2d = np.ones((ch.y_pixels, ch.x_pixels), dtype=float) * 100.0

    # Stub out background so this test isolates spectra + noise behaviour on a minimal channel.
    with patch(
        "instrument.build_science_image.generate_Background_Image",
        return_value=np.zeros((ch.y_pixels, ch.x_pixels), dtype=float),
    ):
        image1 = build_science_image(spectra_2d, ch, ctx, _cfg(), _star())
        image2 = build_science_image(spectra_2d, ch, ctx, _cfg(), _star())

    # Same input & channel but different random draws → images should differ
    assert image1.shape == spectra_2d.shape
    assert image2.shape == spectra_2d.shape
    assert not np.allclose(image1, image2)

    # Mean should remain close to the deterministic exposure*spectra level
    base_mean = float((spectra_2d * ch.exposure_s).mean())
    img_mean = float(image1.mean())
    assert abs(img_mean - base_mean) / base_mean < 0.2

