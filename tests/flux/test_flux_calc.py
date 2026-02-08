import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from utils.constants import C_LIGHT_ROUNDED_m_s, R_SUN, PARSEC_CM
from flux.flux_calc import load_model_for_temperature, convertIntensityToFlux, compute_flux_at_earth, convert_flux_to_photons, apply_unred, calculateFluxOnEarth
import numpy as np
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from configs import global_config
import numpy as np
# from configs import global_config
# from types import SimpleNamespace

def test_load_model_exact_match():
    fake_data = np.array([[1.0, 2.0]])

    with patch("flux.flux_calc.get_repo_root") as mock_root, \
         patch("flux.flux_calc.np.loadtxt", return_value=fake_data) as mock_load, \
         patch("pathlib.Path.is_file") as mock_is_file:

        mock_root.return_value = Path("/fake/repo")

        # exact match exists
        mock_is_file.side_effect = [True]

        result = load_model_for_temperature(5756.0)

        assert result is fake_data
        mock_load.assert_called_once()

def test_load_model_fallback_used():
    fake_data = np.array([[3.0, 4.0]])

    with patch("flux.flux_calc.get_repo_root") as mock_root, \
         patch("flux.flux_calc.np.loadtxt", return_value=fake_data) as mock_load, \
         patch("pathlib.Path.is_file") as mock_is_file:

        mock_root.return_value = Path("/fake/repo")

        # exact → False, fallback → True
        mock_is_file.side_effect = [False, True]

        result = load_model_for_temperature(5756.0)

        assert result is fake_data
        mock_load.assert_called_once()

def test_load_model_not_found():
    with patch("flux.flux_calc.get_repo_root") as mock_root, \
         patch("pathlib.Path.is_file") as mock_is_file:

        mock_root.return_value = Path("/fake/repo")

        # exact → False, fallback → False
        mock_is_file.side_effect = [False, False]

        with pytest.raises(FileNotFoundError):
            load_model_for_temperature(5756.0)

def test_convertIntensityToLuminosity_shape_and_wavelength():
    model_data = np.array([
        [1000.0, 1.0, 2.0],
        [2000.0, 3.0, 4.0],
    ])
    r_star = 1.0

    out = convertIntensityToFlux(model_data, r_star)

    assert out.shape == model_data.shape
    np.testing.assert_allclose(out[:,0], model_data[:,0])

def test_frequency_to_wavelength_conversion():
    model_data = np.array([[1000.0, 1.0, 0.0]])
    r_star = 1.0

    out = convertIntensityToFlux(model_data, r_star)

    geometry = 4 * np.pi * r_star**2 * 4 * np.pi
    recovered_intensity = out[0,1] / geometry

    expected_intensity = C_LIGHT_ROUNDED_m_s / (1000.0**2)
    np.testing.assert_allclose(recovered_intensity, expected_intensity)

def test_convertIntensityToLuminosity_snapshot_WASP69():
    tests_dir = Path(__file__).resolve().parents[1]
    snap_dir = tests_dir / "snapshots"

    input_file = snap_dir / "WASP-69_model_input.txt"
    expected_file = snap_dir / "WASP-69_convertIntensityToLuminosity_snapshot.txt"

    assert input_file.exists(), f"Missing input snapshot: {input_file}"
    assert expected_file.exists(), f"Missing output snapshot: {expected_file}"

    model_data = np.loadtxt(input_file, dtype=np.float64)
    expected = np.loadtxt(expected_file, dtype=np.float64)

    r_star_cm = 0.801 * R_SUN

    got = convertIntensityToFlux(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def test_convertIntensityToLuminosity_snapshot_TIC393818343():
    tests_dir = Path(__file__).resolve().parents[1]
    snap_dir = tests_dir / "snapshots"

    input_file = snap_dir / "TIC 393818343_model_input.txt"
    expected_file = snap_dir / "TIC 393818343_convertIntensityToLuminosity_snapshot.txt"

    assert input_file.exists(), f"Missing input snapshot: {input_file}"
    assert expected_file.exists(), f"Missing output snapshot: {expected_file}"

    model_data = np.loadtxt(input_file, dtype=np.float64)
    expected = np.loadtxt(expected_file, dtype=np.float64)

    r_star_cm = 1.086 * R_SUN

    got = convertIntensityToFlux(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def test_convertIntensityToLuminosity_snapshot_WASP189():
    tests_dir = Path(__file__).resolve().parents[1]
    snap_dir = tests_dir / "snapshots"

    input_file = snap_dir / "WASP-189_model_input.txt"
    expected_file = snap_dir / "WASP-189_convertIntensityToLuminosity_snapshot.txt"

    assert input_file.exists(), f"Missing input snapshot: {input_file}"
    assert expected_file.exists(), f"Missing output snapshot: {expected_file}"

    model_data = np.loadtxt(input_file, dtype=np.float64)
    expected = np.loadtxt(expected_file, dtype=np.float64)

    r_star_cm = 2.36 * R_SUN

    got = convertIntensityToFlux(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def test_convertIntensityToLuminosity_snapshot_Kelt9():
    tests_dir = Path(__file__).resolve().parents[1]
    snap_dir = tests_dir / "snapshots"

    input_file = snap_dir / "KELT-9_model_input.txt"
    expected_file = snap_dir / "KELT-9_convertIntensityToLuminosity_snapshot.txt"

    assert input_file.exists(), f"Missing input snapshot: {input_file}"
    assert expected_file.exists(), f"Missing output snapshot: {expected_file}"

    model_data = np.loadtxt(input_file, dtype=np.float64)
    expected = np.loadtxt(expected_file, dtype=np.float64)

    r_star_cm = 2.362 * R_SUN

    got = convertIntensityToFlux(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def test_convertIntensityToLuminosity_snapshot_HD2685():
    tests_dir = Path(__file__).resolve().parents[1]
    snap_dir = tests_dir / "snapshots"

    input_file = snap_dir / "HD 2685_model_input.txt"
    expected_file = snap_dir / "HD 2685_convertIntensityToLuminosity_snapshot.txt"

    assert input_file.exists(), f"Missing input snapshot: {input_file}"
    assert expected_file.exists(), f"Missing output snapshot: {expected_file}"

    model_data = np.loadtxt(input_file, dtype=np.float64)
    expected = np.loadtxt(expected_file, dtype=np.float64)

    r_star_cm = 1.56 * R_SUN

    got = convertIntensityToFlux(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

def test_compute_flux_at_earth_simple():
    data = np.array([[100.0, 2.0], [200.0, 4.0]])
    out = compute_flux_at_earth(data, distance_pc=10.0)
    expected = np.array([2.0, 4.0]) / (4.0 * np.pi * (10.0 * PARSEC_CM) ** 2)
    assert np.allclose(out, expected)

def test_convert_flux_to_photons():
    flux = np.array([1.0, 2.0])
    wavelengths = np.array([100.0, 200.0])
    out = convert_flux_to_photons(flux, wavelengths)
    assert np.allclose(out, flux * 5.03e7 * wavelengths)

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

def test_calculateFluxOnEarth_no_optional_steps_called(monkeypatch, tmp_path):
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
        test_mode=False,
        produce_Plots=False,
    )

    monkeypatch.setattr("flux.flux_calc.get_global_config", lambda: cfg)
    monkeypatch.setattr("flux.flux_calc.apply_line_core_emission", fake_lce)
    monkeypatch.setattr("flux.flux_calc.apply_ism_absorption", fake_ism)

    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature",
                        lambda _: np.array([[100.0, 1.0, 1.0]]))
    monkeypatch.setattr("flux.flux_calc.convertIntensityToFlux",
                        lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av",
                        lambda *a: (0.0, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth",
                        lambda d, _: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred",
                        lambda w, f, e: f)
    monkeypatch.setattr("flux.flux_calc.convert_flux_to_photons",
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

    calculateFluxOnEarth(star, tmp_path)

    assert called["lce"] is False
    assert called["ism"] is False


def test_calculateFluxOnEarth_optional_steps_called(monkeypatch, tmp_path):
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
        test_mode=False,
        produce_Plots=False,
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
                        lambda _: np.array([[100.0, 1.0, 1.0]]))
    monkeypatch.setattr("flux.flux_calc.convertIntensityToFlux",
                        lambda d, _: d)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av",
                        lambda *a: (0.1, 0.0))
    monkeypatch.setattr("flux.flux_calc.compute_flux_at_earth",
                        lambda d, _: d[:, 1])
    monkeypatch.setattr("flux.flux_calc.apply_unred",
                        lambda w, f, e: f)
    monkeypatch.setattr("flux.flux_calc.convert_flux_to_photons",
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

    calculateFluxOnEarth(star, tmp_path)

    assert called["lce"] is True
    assert called["ism"] is True
