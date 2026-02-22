import numpy as np
import pytest
from types import SimpleNamespace

from instrument.detector import counts_per_s_px_conv_per_channel
from instrument import detector


def _cfg(test_mode=False, produce_Plots=False):
    return SimpleNamespace(test_mode=test_mode, produce_Plots=produce_Plots)


def _dummy_star(name="TESTSTAR"):
    return SimpleNamespace(name=name)


def _channel(**kwargs):
    """Minimal channel-like object for detector tests (channel_name, wavelength, effective_area, pixel_scale)."""
    d = {"channel_name": "NUV", "wavelength": None, "effective_area": None, "pixel_scale": 0.01}
    d.update(kwargs)
    return SimpleNamespace(**d)


def test_single_channel_counts_identity_gaussbroad():
    """
    Verifies per-channel pipeline: interp onto channel.wavelength, multiply by pixel_scale and effective_area.
    """
    wavelength = np.array([100.0, 101.0, 102.0], dtype=float)
    broadened_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    channel = _channel(
        wavelength=np.array([100.0, 100.5, 101.0], dtype=float),
        effective_area=np.array([2.0, 2.0, 2.0], dtype=float),
        pixel_scale=0.01,
    )

    counts = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, channel, output_dir="OUTDIR", cfg=_cfg(), star=_dummy_star()
    )

    # interp: [10, 15, 20] -> *0.01 -> [0.10, 0.15, 0.20] -> *2 -> [0.20, 0.30, 0.40]
    expected = np.array([0.20, 0.30, 0.40], dtype=float)
    assert np.allclose(counts, expected)


def test_all_channels_counts_identity_gaussbroad():
    """Same as single-channel, but for NUV and VIS using counts_per_s_px_conv_all_channels."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(detector, "get_global_config", lambda: _cfg())

        wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
        photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)

        nuv = _channel(
            channel_name="NUV",
            wavelength=np.array([100.0, 100.5], dtype=float),
            effective_area=np.array([2.0, 2.0], dtype=float),
            pixel_scale=0.01,
        )
        vis = _channel(
            channel_name="VIS",
            wavelength=np.array([101.0, 102.0], dtype=float),
            effective_area=np.array([1.0, 1.0], dtype=float),
            pixel_scale=0.01,
        )
        ctx = SimpleNamespace(output_dir="OUTDIR")

        nuv_counts, vis_counts = detector.counts_per_s_px_conv_all_channels(
            photon_flux, wavelengths_total, nuv, vis, ctx, _dummy_star()
        )

        # NUV: interp [10, 15] -> *0.01 -> [0.10,0.15] -> *2 -> [0.20,0.30]
        assert np.allclose(nuv_counts, np.array([0.20, 0.30]))

        # VIS: interp [20, 30] -> *0.01 -> [0.20,0.30] -> *1 -> [0.20,0.30]
        assert np.allclose(vis_counts, np.array([0.20, 0.30]))

    finally:
        monkeypatch.undo()


def test_counts_per_channel_uses_channel_wavelength_grid_and_returns_same_length():
    """Verifies the function interpolates onto channel.wavelength and returns an array of identical length."""
    wavelength = np.array([100.0, 200.0, 300.0], dtype=float)
    broadened_flux = np.array([10.0, 20.0, 30.0], dtype=float)

    channel = _channel(
        wavelength=np.array([150.0, 250.0], dtype=float),
        effective_area=np.array([1.0, 1.0], dtype=float),
        pixel_scale=0.01,
    )

    out = counts_per_s_px_conv_per_channel(
        broadened_flux, wavelength, channel, output_dir="OUTDIR", cfg=_cfg(), star=_dummy_star()
    )

    # interp at 150 -> 15, at 250 -> 25; *0.01*1 -> [0.15, 0.25]
    expected = np.array([15.0, 25.0], dtype=float) * 0.01 * 1.0
    assert out.shape == channel.wavelength.shape
    assert np.allclose(out, expected)


def test_cut_wavelength_window_with_margin_basic_slice_no_margin():
    """Returns the exact expected flux and wavelength slice when margin is zero and window is fully inside bounds."""
    wavelengths_total = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

    channel = _channel(wavelength=np.array([101.0, 103.0], dtype=float))

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, channel, output_dir="OUT", cfg=_cfg(), star=_dummy_star("S"), margin_A=0.0
    )

    wl_min = channel.wavelength[0]
    wl_max = channel.wavelength[-1]
    i0 = max(np.searchsorted(wavelengths_total, wl_min), 0)
    i1 = min(np.searchsorted(wavelengths_total, wl_max), len(wavelengths_total))

    np.testing.assert_allclose(w_cut, wavelengths_total[i0:i1])
    np.testing.assert_allclose(f_cut, photon_flux[i0:i1])


@pytest.mark.parametrize("bound,index", [
    ("low", 0),
    ("high", -1),
])
def test_cut_wavelength_window_with_margin_clamps_bounds(bound, index):
    """Lower/upper index is clamped when margin extends beyond available wavelength range."""
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)
    channel = _channel(wavelength=np.array([100.5, 101.5], dtype=float))

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, channel, output_dir="OUT", cfg=_cfg(), star=_dummy_star("S"), margin_A=999.0
    )

    assert w_cut[index] == wavelengths_total[index]
    assert f_cut[index] == photon_flux[index]


def test_cut_wavelength_window_with_margin_calls_dump_with_x_then_y(monkeypatch):
    """In test_mode, dump_1d_for_channel is called with wavelength as x and flux as y."""
    calls = []
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)
    channel = _channel(wavelength=np.array([100.0, 102.0], dtype=float))

    def _fake_dump(wave, array, output_dir, star_name, tag, channel_name=None, full=True, zoom=False, **kwargs):
        calls.append((wave.copy(), array.copy(), output_dir, star_name, tag, channel_name, full, zoom))

    monkeypatch.setattr(detector, "dump_1d_for_channel", _fake_dump)

    f_cut, w_cut = detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, channel, output_dir="OUT", cfg=_cfg(test_mode=True), star=_dummy_star("S"), margin_A=0.0
    )

    assert len(calls) == 1
    x, y, outdir, star_name, tag, channel_name, full, zoom = calls[0]
    np.testing.assert_allclose(x, w_cut)
    np.testing.assert_allclose(y, f_cut)
    assert outdir == "OUT"
    assert star_name == "S"
    assert "cut" in tag and "wavelength" in tag
    assert channel_name == "NUV"
    assert full is True
    assert zoom is True


def test_compute_broadened_channel_flux_calls_cut_and_gaussbroad_in_order(monkeypatch):
    """compute_broadened_channel_flux passes flux and wavelength in correct order to gaussbroad."""
    channel = _channel(pixel_scale=0.25)

    flux_cut = np.array([1.0, 2.0, 3.0], dtype=float)
    wl_cut = np.array([10.0, 11.0, 12.0], dtype=float)
    smoothed = np.array([9.0, 8.0, 7.0], dtype=float)

    def _fake_cut(photon_flux_at_earth, wavelengths_total, ch, output_dir, cfg_arg, star_arg, margin_A=100.0):
        return flux_cut, wl_cut

    def _fake_gaussbroad(w, s, hwhm):
        np.testing.assert_allclose(w, wl_cut)
        np.testing.assert_allclose(s, flux_cut)
        assert hwhm == pytest.approx(channel.pixel_scale)
        return smoothed

    monkeypatch.setattr(detector, "cut_wavelength_window_with_margin", _fake_cut)
    monkeypatch.setattr(detector, "gaussbroad", _fake_gaussbroad)

    out_flux, out_wl = detector.compute_broadened_channel_flux(
        photon_flux_at_earth=np.array([0.0]),
        wavelengths_total=np.array([0.0]),
        channel=channel,
        output_dir="OUT",
        cfg=_cfg(),
        star=_dummy_star("S"),
    )

    np.testing.assert_allclose(out_flux, smoothed)
    np.testing.assert_allclose(out_wl, wl_cut)


def test_counts_per_s_px_conv_all_channels_calls_broaden_then_convert_for_each_channel(monkeypatch):
    """Wrapper calls broaden and conversion once per channel and returns results in the correct order."""
    ctx = SimpleNamespace(output_dir="OUT")

    nuv = SimpleNamespace(channel_name="NUV")
    vis = SimpleNamespace(channel_name="VIS")

    call_sequence = []

    def _fake_get_global_config():
        return _cfg()

    def _fake_broaden(photon_flux_at_earth, wavelengths_total, ch, output_dir, cfg_arg, star_arg):
        call_sequence.append(("broaden", ch.channel_name))
        return np.array([1.0]), np.array([2.0])

    def _fake_conv(broadened_photon_flux, wavelength, ch, output_dir, cfg_arg, star_arg):
        call_sequence.append(("convert", ch.channel_name))
        return np.array([ch.channel_name], dtype=object)

    monkeypatch.setattr(detector, "get_global_config", _fake_get_global_config)
    monkeypatch.setattr(detector, "compute_broadened_channel_flux", _fake_broaden)
    monkeypatch.setattr(detector, "counts_per_s_px_conv_per_channel", _fake_conv)

    out_nuv, out_vis = detector.counts_per_s_px_conv_all_channels(
        photon_flux_at_earth=np.array([0.0]),
        wavelengths_total=np.array([0.0]),
        nuv=nuv,
        vis=vis,
        ctx=ctx,
        star=_dummy_star("S"),
    )

    assert call_sequence == [
        ("broaden", "NUV"),
        ("broaden", "VIS"),
        ("convert", "NUV"),
        ("convert", "VIS"),
    ]

    assert out_nuv.tolist() == ["NUV"]
    assert out_vis.tolist() == ["VIS"]




def test_counts_per_channel_identity_scaling():
    """With identical wavelength grids, reduces to simple pixel_scale and effective_area scaling."""
    wavelength = np.array([1.0, 2.0, 3.0, 4.0])
    broadened_flux = np.array([10.0, 20.0, 30.0, 40.0])

    channel = _channel(
        channel_name="TEST",
        wavelength=wavelength.copy(),
        pixel_scale=2.0,
        effective_area=np.array([5.0, 5.0, 5.0, 5.0]),
    )

    out = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(), _dummy_star("S"))

    expected = broadened_flux * 2.0 * 5.0
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_interpolation_linear():
    """Linear interpolation is applied correctly before scaling."""
    wavelength = np.array([0.0, 1.0, 2.0, 3.0])
    broadened_flux = np.array([0.0, 10.0, 20.0, 30.0])

    channel = _channel(
        channel_name="TEST",
        wavelength=np.array([0.5, 1.5, 2.5]),
        pixel_scale=1.0,
        effective_area=np.ones(3),
    )

    out = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(), _dummy_star("S"))

    expected = np.array([5.0, 15.0, 25.0])
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_interpolation_clamping():
    """Interpolation outside the wavelength range clamps to boundary flux values."""
    wavelength = np.array([0.0, 1.0, 2.0, 3.0])
    broadened_flux = np.array([0.0, 10.0, 20.0, 30.0])

    channel = _channel(
        channel_name="TEST",
        wavelength=np.array([-1.0, 0.0, 3.0, 5.0]),
        pixel_scale=1.0,
        effective_area=np.ones(4),
    )

    out = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(), _dummy_star("S"))

    expected = np.array([0.0, 0.0, 30.0, 30.0])
    np.testing.assert_allclose(out, expected)


def test_counts_per_channel_calls_dump_in_test_mode(monkeypatch):
    """dump_1d_for_channel is called with channel wavelength and output values when test_mode is enabled."""
    calls = []

    def fake_dump(wave, array, output_dir, star_name, tag, channel_name=None, full=True, zoom=True, **kwargs):
        calls.append((wave.copy(), array.copy(), tag, channel_name, full, zoom))

    wavelength = np.array([1.0, 2.0, 3.0])
    broadened_flux = np.array([10.0, 20.0, 30.0])

    channel = _channel(
        channel_name="TEST",
        wavelength=wavelength.copy(),
        pixel_scale=1.0,
        effective_area=np.ones(3),
    )

    monkeypatch.setattr(detector, "dump_1d_for_channel", fake_dump)

    out = counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(test_mode=True), _dummy_star("S"))

    assert len(calls) == 1
    x, y, tag, channel_name, full, zoom = calls[0]
    np.testing.assert_allclose(x, channel.wavelength)
    np.testing.assert_allclose(y, out)
    assert "counts" in tag
    assert channel_name == "TEST"
    assert full is True
    assert zoom is True


def test_cut_wavelength_window_with_margin_empty_slice_raises():
    """Raises ValueError when channel wavelength range does not overlap wavelengths_total."""
    wavelengths_total = np.array([100.0, 200.0, 300.0], dtype=float)
    photon_flux = np.array([10.0, 20.0, 30.0], dtype=float)
    channel = _channel(wavelength=np.array([500.0, 600.0], dtype=float))

    with pytest.raises(ValueError, match="does not overlap wavelengths_total"):
        detector.cut_wavelength_window_with_margin(
            photon_flux, wavelengths_total, channel, output_dir="OUT", cfg=_cfg(), star=_dummy_star("S"), margin_A=0.0
        )


def test_cut_wavelength_window_with_margin_calls_plot_when_produce_Plots(monkeypatch):
    """plot_1d_for_channel is called when produce_Plots=True."""
    calls = []
    wavelengths_total = np.array([100.0, 101.0, 102.0], dtype=float)
    photon_flux = np.array([10.0, 11.0, 12.0], dtype=float)
    channel = _channel(wavelength=np.array([100.0, 102.0], dtype=float))

    def _fake_plot(wave, flux, output_dir, star, filename_tag=None, **kwargs):
        calls.append((wave.copy(), flux.copy(), output_dir, filename_tag))

    monkeypatch.setattr(detector, "plot_1d_for_channel", _fake_plot)

    detector.cut_wavelength_window_with_margin(
        photon_flux, wavelengths_total, channel, output_dir="OUT", cfg=_cfg(produce_Plots=True), star=_dummy_star("S"), margin_A=0.0
    )

    assert len(calls) == 1
    assert "cut" in calls[0][3] and "wavelength" in calls[0][3]


def test_compute_broadened_channel_flux_calls_plot_when_produce_Plots(monkeypatch):
    """plot_1d_for_channel is called when produce_Plots=True."""
    calls = []
    channel = _channel(wavelength=np.array([100.0, 102.0], dtype=float), pixel_scale=0.1)

    def _fake_cut(*args, **kwargs):
        return np.array([1.0, 2.0]), np.array([100.0, 102.0])

    def _fake_plot(wave, flux, output_dir, star, filename_tag=None, **kwargs):
        calls.append(filename_tag)

    monkeypatch.setattr(detector, "cut_wavelength_window_with_margin", _fake_cut)
    monkeypatch.setattr(detector, "plot_1d_for_channel", _fake_plot)

    detector.compute_broadened_channel_flux(
        np.array([0.0]), np.array([0.0]), channel, "OUT", _cfg(produce_Plots=True), _dummy_star("S")
    )

    assert any("gaussbroad" in str(t) for t in calls)


def test_counts_per_channel_calls_plot_when_produce_Plots(monkeypatch):
    """plot_1d_for_channel is called when produce_Plots=True."""
    calls = []
    wavelength = np.array([1.0, 2.0, 3.0])
    broadened_flux = np.array([10.0, 20.0, 30.0])
    channel = _channel(wavelength=wavelength.copy(), effective_area=np.ones(3))

    def _fake_plot(wave, flux, output_dir, star, filename_tag=None, **kwargs):
        calls.append(filename_tag)

    monkeypatch.setattr(detector, "plot_1d_for_channel", _fake_plot)

    counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(produce_Plots=True), _dummy_star("S"))

    assert any("counts" in str(t) for t in calls)


def test_counts_per_channel_mismatched_wavelength_effective_area_raises():
    """Raises when channel.wavelength and channel.effective_area have different lengths."""
    wavelength = np.array([1.0, 2.0, 3.0])
    broadened_flux = np.array([10.0, 20.0, 30.0])
    channel = _channel(wavelength=np.array([1.0, 2.0]), effective_area=np.array([1.0, 1.0, 1.0]))  # 2 vs 3

    with pytest.raises(ValueError):
        counts_per_s_px_conv_per_channel(broadened_flux, wavelength, channel, "OUT", _cfg(), _dummy_star("S"))

