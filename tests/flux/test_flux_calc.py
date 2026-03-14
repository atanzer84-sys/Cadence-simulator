import numpy as np
from types import SimpleNamespace

from flux.flux_calc import (
    convertStellarModelToFlux,
    compute_flux_at_earth,
    apply_unred,
    calculateFluxOnEarth,
)
from instrument.prepare_detector_images import convert_flux_to_photons
from utils.constants import C_LIGHT_Angst, PARSEC_CM


def _noop(*args, **kwargs):
    pass


def test_convertStellarModelToFlux_shape_and_wavelength():
    """convertStellarModelToFlux preserves input shape and keeps wavelength column unchanged."""
    model_data = np.array([
        [1000.0, 1.0, 2.0],
        [2000.0, 3.0, 4.0],
    ])
    r_star = 1.0

    out = convertStellarModelToFlux(model_data, r_star)

    assert out.shape == model_data.shape
    np.testing.assert_allclose(out[:,0], model_data[:,0])

def test_frequency_to_wavelength_conversion():
    """convertStellarModelToFlux applies the c/lambda^2 conversion and geometric scaling consistently."""
    model_data = np.array([[1000.0, 1.0, 0.0]])
    r_star = 1.0

    out = convertStellarModelToFlux(model_data, r_star)

    geometry = 4 * np.pi * r_star**2 * 4 * np.pi
    recovered_intensity = out[0,1] / geometry

    expected_intensity = C_LIGHT_Angst / (1000.0**2)
    np.testing.assert_allclose(recovered_intensity, expected_intensity)


def test_compute_flux_at_earth_simple():
    """compute_flux_at_earth divides by 4*pi*(d*pc)^2 for a simple input."""
    data = np.array([[100.0, 2.0], [200.0, 4.0]])
    out = compute_flux_at_earth(data, distance_pc=10.0)
    expected = np.array([2.0, 4.0]) / (4.0 * np.pi * (10.0 * PARSEC_CM) ** 2)
    assert np.allclose(out, expected)

def test_convert_flux_to_photons():
    """convert_flux_to_photons uses the same conversion constant as the production code."""
    from utils.constants import PHOTON_ENERGY_CONVERSION_A

    flux = np.array([1.0, 2.0])
    wavelengths = np.array([100.0, 200.0])
    out = convert_flux_to_photons(flux, wavelengths)
    assert np.allclose(out, flux * PHOTON_ENERGY_CONVERSION_A * wavelengths)

def test_apply_unred_flips_ebv(monkeypatch):
    """apply_unred flips the sign of EBV before calling unred."""
    called = {}

    def fake_unred(w, f, ebv, R_V):
        called["ebv"] = ebv
        return f

    monkeypatch.setattr("flux.flux_calc.unred", fake_unred)

    wavelengths = np.array([100.0])
    flux = np.array([1.0])
    ebv = 0.2

    out = apply_unred(wavelengths, flux, ebv)

    assert called["ebv"] == -0.2
    assert np.allclose(out, flux)

def test_calculateFluxOnEarth_no_optional_steps_called(monkeypatch, tmp_path):
    """calculateFluxOnEarth does not call line core emission or ISM absorption when flags are False."""
    called = {"lce": False, "ism": False}

    def fake_lce(data, *args, **kwargs):
        called["lce"] = True
        return data

    def fake_ism(data, *args, **kwargs):
        called["ism"] = True
        return data

    cfg = SimpleNamespace(
        line_core_emission=False,
        interstellar_absorption=False,
        write_intermediate_arrays=False,
        produce_flux_convolution_plots=False,
    )

    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)
    monkeypatch.setattr("flux.flux_calc.apply_line_core_emission", fake_lce)
    monkeypatch.setattr("flux.flux_calc.apply_ism_absorption", fake_ism)

    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature",
                        lambda _, announce_user=False: np.array([[5000.0, 1.0, 1.0]]))
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux",
                        lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av",
                        lambda *a: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth",
                        lambda d, _, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred",
                        lambda w, f, e, announce_user=False: f)
    monkeypatch.setattr("instrument.prepare_detector_images.convert_flux_to_photons",
                        lambda f, w: f)

    star = SimpleNamespace(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
        name="TEST",
    )
    ctx = SimpleNamespace(output_dir=tmp_path, dump_3d_array=_noop, dump_1d_array=_noop, dump_1d_for_channel=_noop, plot_1d_for_channel=_noop, plot_flux_and_photons_windows=_noop)

    calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert called["lce"] is False
    assert called["ism"] is False


def test_calculateFluxOnEarth_optional_steps_called(monkeypatch, tmp_path):
    """calculateFluxOnEarth calls line core emission and ISM absorption when flags are True."""
    called = {"lce": False, "ism": False}

    def fake_lce(data, *args, **kwargs):
        called["lce"] = True
        return data

    def fake_ism(data, *args, **kwargs):
        called["ism"] = True
        return data

    cfg = SimpleNamespace(
        line_core_emission=True,
        interstellar_absorption=True,
        write_intermediate_arrays=False,
        produce_flux_convolution_plots=False,
        sigmaMg22=1.0,
        sigmaMg21=1.0,
        mg2_col=None,
        mg1_col=None,
        fe2_col=None,
    )

    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)
    monkeypatch.setattr("flux.flux_calc.apply_line_core_emission", fake_lce)
    monkeypatch.setattr("flux.flux_calc.apply_ism_absorption", fake_ism)

    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature",
                        lambda _, announce_user=False: np.array([[100.0, 1.0, 1.0]]))
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux",
                        lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av",
                        lambda *a: (0.1, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth",
                        lambda d, _, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred",
                        lambda w, f, e, announce_user=False: f)
    monkeypatch.setattr("instrument.prepare_detector_images.convert_flux_to_photons",
                        lambda f, w: f)

    star = SimpleNamespace(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
        name="TEST",
    )
    ctx = SimpleNamespace(output_dir=tmp_path, dump_3d_array=_noop, dump_1d_array=_noop, dump_1d_for_channel=_noop, plot_1d_for_channel=_noop, plot_flux_and_photons_windows=_noop)

    calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert called["lce"] is True
    assert called["ism"] is True


def test_calculateFluxOnEarth_returns_photons_and_wavelengths_same_length(monkeypatch, tmp_path):
    """calculateFluxOnEarth returns photons and wavelength arrays of identical length and finite."""
    cfg = SimpleNamespace(
        line_core_emission=False,
        interstellar_absorption=False,
        write_intermediate_arrays=False,
        produce_flux_convolution_plots=False,
    )

    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)
    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature", lambda _, announce_user=False: np.array([[100.0, 1.0, 1.0]]))
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *a: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, _, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, announce_user=False: f)
    monkeypatch.setattr("instrument.prepare_detector_images.convert_flux_to_photons", lambda f, w: f)

    star = SimpleNamespace(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
        name="TEST",
    )
    ctx = SimpleNamespace(output_dir=tmp_path, dump_3d_array=_noop, dump_1d_array=_noop, dump_1d_for_channel=_noop, plot_1d_for_channel=_noop, plot_flux_and_photons_windows=_noop)

    photons, wavelengths = calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert len(photons) == len(wavelengths)
    assert np.all(np.isfinite(photons))
    assert np.all(np.isfinite(wavelengths))


def test_calculateFluxOnEarth_executes_write_intermediate_arrays_instrumentation(monkeypatch, tmp_path):
    """ctx.write_intermediate_arrays with real dumps triggers debug dump instrumentation (dump_3d_array called)."""
    called = {"dumped": False}

    def fake_dump_3d_array(*args, **kwargs):
        called["dumped"] = True

    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda _t, announce_user=False: np.column_stack((np.array([5000.0, 5100.0]), np.array([1.0, 1.0])))
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda model, _r: model)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_a: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda data, _d, announce_user=False: data[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda _w, f, _e, announce_user=False: f)
    monkeypatch.setattr("instrument.prepare_detector_images.convert_flux_to_photons", lambda f, _w: f)

    cfg = SimpleNamespace(
        line_core_emission=False,
        interstellar_absorption=False,
        produce_flux_convolution_plots=False,
        sigmaMg22=0.0,
        sigmaMg21=0.0,
    )
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = SimpleNamespace(
        name="Star",
        effective_temperature=5000.0,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=-4.8,
        spectral_type="G2V",
        mass=1.0,
    )
    ctx = SimpleNamespace(output_dir=tmp_path, dump_3d_array=fake_dump_3d_array, dump_1d_array=_noop, dump_1d_for_channel=_noop, plot_1d_for_channel=_noop, plot_flux_and_photons_windows=_noop)

    calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert called["dumped"] is True
