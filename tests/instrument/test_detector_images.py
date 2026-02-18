import numpy as np
from types import SimpleNamespace
from instrument import detector_images, detector
from astropy.io import fits
from datetime import datetime

def test_generate_bias_frame_zero_noise_returns_constant_frame():
    # Verifies bias frame equals bias_offset when read_noise is zero.

    cfg = SimpleNamespace(
        x_pixels=3,
        y_pixels=2,
        bias_offset=100.0,
        read_noise=0.0,
        channel_name="NUV"
    )

    frame, _ = detector_images.generate_bias_frame(cfg, header=None)

    assert frame.shape == (2, 3)
    assert np.allclose(frame, 100.0)

def test_generate_dark_frame_adds_dark_current(monkeypatch):
    # Verifies dark frame equals bias + dark_current * exptime.

    cfg = SimpleNamespace(
        x_pixels=2,
        y_pixels=2,
        bias_offset=0.0,
        read_noise=0.0,
        dark_noise=5.0,
        channel_name="VIS",
    )

    # Force bias to zero frame
    monkeypatch.setattr(
        detector_images,
        "generate_bias_frame",
        lambda c, header=None: (np.zeros((2, 2)), None),
    )

    dark, _ = detector_images.generate_dark_frame(cfg, exptime_s=10.0, header=None)

    assert np.allclose(dark, 50.0)  # 5 * 10

def test_initialize_fits_header_contains_basic_keys(monkeypatch):
    # Verifies initialize_fits_header sets telescope, RA, DEC, and target metadata.

    from types import SimpleNamespace
    from datetime import datetime

    star = SimpleNamespace(
        name="TEST",
        right_ascension=10.0,
        declination=20.0,
        distance_pc=5.0,
        mass_sun_kg=1.0,
    )

    monkeypatch.setattr(
        detector_images.loaders.run_setup,
        "GLOBAL_TIMESTAMP",
        datetime(2025, 1, 1, 12, 0, 0),
    )

    header = detector_images.initialize_fits_header(star)

    assert header["TELESCOP"] == "WALTzER"
    assert header["TARGT_ID"] == "TEST"
    assert "RA_HEX" in header
    assert "DEC_HEX" in header


# Ensures generate_bias_frame populates header statistics when header is provided.
def test_generate_bias_frame_populates_header(monkeypatch):

    monkeypatch.setattr(detector_images.np.random, "normal",
                        lambda *_a, size=None, **_k: np.zeros(size))

    cfg = SimpleNamespace(channel_name="NUV",
                          x_pixels=3, y_pixels=2,
                          bias_offset=5.0, read_noise=1.0)

    hdr = fits.Header()

    frame, out_hdr = detector_images.generate_bias_frame(cfg, hdr)

    assert frame.shape == (2, 3)
    for key in ("MEAN", "MEDIAN", "MAX", "MIN",
                "B_OFFSET", "RNOISE", "EXPTIME"):
        assert key in out_hdr


# Ensures generate_dark_frame populates header statistics when header is provided.
def test_generate_dark_frame_populates_header(monkeypatch):

    monkeypatch.setattr(detector_images.np.random, "normal",
                        lambda *_a, size=None, **_k: np.zeros(size))

    cfg = SimpleNamespace(channel_name="NUV",
                          x_pixels=3, y_pixels=2,
                          bias_offset=5.0,
                          read_noise=1.0,
                          dark_noise=2.0)

    hdr = fits.Header()

    frame, out_hdr = detector_images.generate_dark_frame(
        cfg, exptime_s=10.0, header=hdr
    )

    assert frame.shape == (2, 3)
    for key in ("MEAN", "MEDIAN", "MAX", "MIN",
                "DARKVAL", "EXPTIME",
                "B_OFFSET", "RNOISE"):
        assert key in out_hdr


# Ensures gaussbroad returns flat mean array when hwhm is extremely large.
def test_gaussbroad_large_hwhm_returns_flat():

    w = np.linspace(1000.0, 1010.0, 10)
    s = np.arange(10.0)
    hwhm = 1000.0

    out = detector.gaussbroad(w, s, hwhm)

    assert np.allclose(out, np.full(len(w), np.sum(s) / len(w)))


# Ensures test_mode triggers dump in convolution path.
def test_counts_conv_triggers_dump(monkeypatch, tmp_path):

    called = {"dump": False}

    monkeypatch.setattr(detector, "dump_1d_array",
                        lambda *a, **k: called.__setitem__("dump", True))

    monkeypatch.setattr(detector, "gaussbroad",
                        lambda _w, s, _h: s)

    cfg = SimpleNamespace(test_mode=True, produce_Plots=False)

    cal = SimpleNamespace(name="NUV",
                          wavelength=np.array([1.0, 2.0]),
                          effective_area=np.array([1.0, 1.0]),
                          pixel_scale=1.0)

    star = SimpleNamespace(name="Star")

    detector.counts_per_s_px_conv_all_channels_per_channel(
        photon_flux_at_earth=np.array([10.0, 20.0]),
        wavelengths_total=np.array([1.0, 2.0]),
        cal=cal,
        output_dir=tmp_path,
        cfg=cfg,
        star=star,
    )

    assert called["dump"] is True


# Ensures produce_Plots triggers plotting in convolution path.
def test_counts_conv_triggers_plot(monkeypatch, tmp_path):

    called = {"plot": False}

    monkeypatch.setattr(detector, "plot_flux_and_photons_windows",
                        lambda *a, **k: called.__setitem__("plot", True))

    monkeypatch.setattr(detector, "gaussbroad",
                        lambda _w, s, _h: s)

    cfg = SimpleNamespace(test_mode=False, produce_Plots=True)

    cal = SimpleNamespace(name="NUV",
                          wavelength=np.array([1.0, 2.0]),
                          effective_area=np.array([1.0, 1.0]),
                          pixel_scale=1.0)

    star = SimpleNamespace(name="Star")

    detector.counts_per_s_px_conv_all_channels_per_channel(
        photon_flux_at_earth=np.array([10.0, 20.0]),
        wavelengths_total=np.array([1.0, 2.0]),
        cal=cal,
        output_dir=tmp_path,
        cfg=cfg,
        star=star,
    )

    assert called["plot"] is True


