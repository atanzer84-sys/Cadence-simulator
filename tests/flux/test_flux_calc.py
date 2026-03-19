import numpy as np
from types import SimpleNamespace

from flux.flux_calc import (
    convertStellarModelToFlux,
    compute_flux_at_earth,
    apply_unred,
    calculateFluxOnEarth,
)
from utils.constants import C_LIGHT_Angst, PARSEC_CM


# Tests: convertStellarModelToFlux
# Behavior: preserves shape and wavelength column
def test_convertStellarModelToFlux_shape_and_wavelength():
    model_data = np.array([
        [1000.0, 1.0, 2.0],
        [2000.0, 3.0, 4.0],
    ])
    r_star = 1.0

    out = convertStellarModelToFlux(model_data, r_star)

    assert out.shape == model_data.shape
    np.testing.assert_allclose(out[:, 0], model_data[:, 0])


# Tests: convertStellarModelToFlux
# Behavior: verifies frequency→wavelength conversion and geometric scaling
def test_frequency_to_wavelength_conversion():
    model_data = np.array([[1000.0, 1.0, 0.0]])
    r_star = 1.0

    out = convertStellarModelToFlux(model_data, r_star)

    geometry = 4 * np.pi * r_star**2 * 4 * np.pi
    recovered_intensity = out[0, 1] / geometry
    expected_intensity = C_LIGHT_Angst / (1000.0**2)

    np.testing.assert_allclose(recovered_intensity, expected_intensity)


# Tests: compute_flux_at_earth
# Behavior: divides by 4*pi*(d*pc)^2
def test_compute_flux_at_earth_simple():
    data = np.array([[100.0, 2.0], [200.0, 4.0]])
    out = compute_flux_at_earth(data, distance_pc=10.0)
    expected = np.array([2.0, 4.0]) / (4.0 * np.pi * (10.0 * PARSEC_CM) ** 2)
    assert np.allclose(out, expected)


# Tests: apply_unred
# Behavior: flips EBV sign before calling unred
def test_apply_unred_flips_ebv(monkeypatch):
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


# Tests: calculateFluxOnEarth
# Behavior: optional steps skipped when flags are False
def test_calculateFluxOnEarth_no_optional_steps_called(make_star, make_config, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda _t, announce_user=False: np.array([[5000.0, 1.0, 1.0]])
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, _dist, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, announce_user=False: f)

    cfg = make_config(line_core_emission=0, interstellar_absorption=0)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = make_star(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
    )

    ctx = SimpleNamespace(
        output_dir=tmp_path,
        dump_3d_array=lambda *a, **k: None,
        dump_1d_array=lambda *a, **k: None,
        dump_1d_for_channel=lambda *a, **k: None,
        plot_1d_for_channel=lambda *a, **k: None,
        plot_flux_and_photons_windows=lambda *a, **k: None,
    )

    flux, wavelengths = calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert flux[0] == 1.0
    assert wavelengths[0] == 5000.0


# Tests: calculateFluxOnEarth
# Behavior: optional steps applied when flags are True
def test_calculateFluxOnEarth_optional_steps_called(make_star, make_config, monkeypatch, tmp_path):
    # wavelength must be inside 3400–18000 Å
    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda _t, announce_user=False: np.array([[5000.0, 1.0, 1.0]])
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_: (0.1, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, _dist, announce_user=False: d[:, 1])

    # optional steps
    monkeypatch.setattr("flux.flux_calc.apply_line_core_emission", lambda d, *a, **k: d * 2.0)
    monkeypatch.setattr("flux.flux_calc.apply_ism_absorption", lambda d, *a, **k: d * 3.0)
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, announce_user=False: f)

    cfg = make_config(line_core_emission=1, interstellar_absorption=1)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = make_star(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
    )

    ctx = SimpleNamespace(
        output_dir=tmp_path,
        dump_3d_array=lambda *a, **k: None,
        dump_1d_array=lambda *a, **k: None,
        dump_1d_for_channel=lambda *a, **k: None,
        plot_1d_for_channel=lambda *a, **k: None,
        plot_flux_and_photons_windows=lambda *a, **k: None,
    )

    flux, wavelengths = calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    # 1 → LCE doubles → ISM triples → 1 * 2 * 3 = 6
    assert flux[0] == 6.0
    assert wavelengths[0] == 5000.0



# Tests: calculateFluxOnEarth
# Behavior: output arrays have same length and finite values
def test_calculateFluxOnEarth_returns_photons_and_wavelengths_same_length(make_star, make_config, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda _t, announce_user=False: np.array([[100.0, 1.0, 1.0]])
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, _dist, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, announce_user=False: f)

    cfg = make_config(line_core_emission=0, interstellar_absorption=0)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = make_star(
        effective_temperature=5000,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
    )

    ctx = SimpleNamespace(
        output_dir=tmp_path,
        dump_3d_array=lambda *a, **k: None,
        dump_1d_array=lambda *a, **k: None,
        dump_1d_for_channel=lambda *a, **k: None,
        plot_1d_for_channel=lambda *a, **k: None,
        plot_flux_and_photons_windows=lambda *a, **k: None,
    )

    flux, wavelengths = calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert len(flux) == len(wavelengths)
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(wavelengths))


# Tests: calculateFluxOnEarth
# Behavior: instrumentation triggers at least one dump
def test_calculateFluxOnEarth_executes_write_intermediate_arrays_instrumentation(make_star, make_config, monkeypatch, tmp_path):
    called = {"dumped": 0}

    def fake_dump_3d_array(*args, **kwargs):
        called["dumped"] += 1

    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda _t, announce_user=False: np.array([[5000.0, 1.0, 1.0]])
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, _dist, announce_user=False: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, announce_user=False: f)

    cfg = make_config(write_intermediate_arrays=True)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = make_star(
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

    ctx = SimpleNamespace(
        output_dir=tmp_path,
        dump_3d_array=fake_dump_3d_array,
        dump_1d_array=lambda *a, **k: None,
        dump_1d_for_channel=lambda *a, **k: None,
        plot_1d_for_channel=lambda *a, **k: None,
        plot_flux_and_photons_windows=lambda *a, **k: None,
    )

    calculateFluxOnEarth(star, ctx, 3400.0, 18000.0)

    assert called["dumped"] > 0

# Tests: calculateFluxOnEarth
# Behavior: enforces wavelength cut via cut_model_wavelength_range
def test_calculateFluxOnEarth_applies_wavelength_cut(make_star, make_config, monkeypatch, tmp_path):
    # Model contains wavelengths outside the requested range
    model = np.array([
        [3000.0, 1.0, 1.0],   # below range
        [5000.0, 2.0, 1.0],   # inside
        [20000.0, 3.0, 1.0],  # above range
    ])

    monkeypatch.setattr(
        "flux.flux_calc.load_model_for_temperature",
        lambda t, announce_user=False: model
    )
    monkeypatch.setattr("flux.flux_calc.convertStellarModelToFlux", lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *_: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth", lambda d, dist, announce_user=False: d[:, 1])

    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e, **k: f)

    cfg = make_config()
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    star = make_star(effective_temperature=5000, distance_pc=10)

    ctx = SimpleNamespace(
        output_dir=tmp_path,
        dump_3d_array=lambda *a, **k: None,
        dump_1d_array=lambda *a, **k: None,
        dump_1d_for_channel=lambda *a, **k: None,
        plot_1d_for_channel=lambda *a, **k: None,
        plot_flux_and_photons_windows=lambda *a, **k: None,
    )

    flux, wavelengths = calculateFluxOnEarth(star, ctx, 4000.0, 18000.0)

    # Only the 5000 Å row should survive
    assert np.allclose(wavelengths, [5000.0])
    assert np.allclose(flux, [2.0])


# Tests: apply_ism_absorption
# Behavior: uses derived column densities when cfg.mg*_col is None
def test_apply_ism_absorption_derived_columns(make_config, monkeypatch):
    called = {}

    def fake_cute_ism_abs_all(data, nmg2, nmg1, nfe2):
        called["mg2"] = nmg2
        called["mg1"] = nmg1
        called["fe2"] = nfe2
        out = data.copy()
        out[:, 1] *= 10
        return out

    monkeypatch.setattr("flux.flux_calc.cute_ism_abs_all", fake_cute_ism_abs_all)

    cfg = make_config(mg2_col=None, mg1_col=None, fe2_col=None)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    flux_lambda = np.array([[5000.0, 1.0]])

    from flux.flux_calc import apply_ism_absorption
    out = apply_ism_absorption(flux_lambda, 0.0, cfg)

    assert isinstance(called["mg2"], float)
    assert isinstance(called["mg1"], float)
    assert isinstance(called["fe2"], float)
    assert out[0, 1] == 10.0

# Tests: apply_ism_absorption
# Behavior: uses explicit cfg.mg*_col values when provided
def test_apply_ism_absorption_explicit_columns(make_config, monkeypatch):
    called = {}

    # MUST match production signature: cute_ism_abs_all(data, nmg2, nmg1, nfe2)
    def fake_cute_ism_abs_all(data, nmg2, nmg1, nfe2):
        called["mg2"] = nmg2
        called["mg1"] = nmg1
        called["fe2"] = nfe2
        out = data.copy()
        out[:, 1] *= 5
        return out

    monkeypatch.setattr("flux.flux_calc.cute_ism_abs_all", fake_cute_ism_abs_all)

    cfg = make_config(mg2_col=12.5, mg1_col=3.0, fe2_col=0.001)
    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)

    flux_lambda = np.array([[5000.0, 1.0]])

    from flux.flux_calc import apply_ism_absorption
    out = apply_ism_absorption(flux_lambda, 0.0, cfg)

    assert called["mg2"] == 12.5
    assert called["mg1"] == 3.0
    assert called["fe2"] == 0.001
    assert out[0, 1] == 5.0


