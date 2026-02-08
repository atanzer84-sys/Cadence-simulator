import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from utils.constants import C_LIGHT_ROUNDED_m_s, R_SUN, PARSEC_CM
from flux.flux_calc import load_model_for_temperature, convertIntensityToFlux, compute_flux_at_earth, convert_flux_to_photons, apply_unred, calculateFluxOnEarth
import numpy as np
from pathlib import Path
from unittest.mock import patch
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

def test_calculateFluxOnEarth_wiring_no_optional_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from configs import global_config
    import numpy as np
    global_config._GLOBAL_CONFIG = None
    cfg_path = tmp_path / "global.cfg"
    cfg_path.write_text(
        """
    line_core_emission = 0
    interstellar_absorption = 0
    test_mode = 0
    produce_Plots = 0
    """,
        encoding="utf-8",
    )
    global_config.load_global_config(cfg_path)

    model_data = np.array([[100.0, 1.0, 0.0]])
    flux = np.array([[100.0, 1.0, 0.0]])

    star = SimpleNamespace(
        name="TEST",
        effective_temperature=5000.0,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
    )

    # --- stubs to avoid physics / I/O ---
    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature", lambda t: model_data)
    monkeypatch.setattr("flux.flux_calc.convertIntensityToFlux", lambda m, r: flux)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *a: (0.1, None))
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e: f)
    monkeypatch.setattr("flux.flux_calc.convert_flux_to_photons", lambda f, w: f)

    # --- assert optional steps are NOT called ---
    monkeypatch.setattr(
        "flux.flux_calc.apply_line_core_emission",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LCE should not be called")),
    )
    monkeypatch.setattr(
        "flux.flux_calc.apply_ism_absorption",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ISM should not be called")),
    )

    out = calculateFluxOnEarth(star, tmp_path)

    assert out.shape == (1,)

def test_calculateFluxOnEarth_wiring_all_optional_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from configs import global_config
    import numpy as np

    # reset global singleton
    global_config._GLOBAL_CONFIG = None

    cfg_path = tmp_path / "global.cfg"
    cfg_path.write_text(
"""line_core_emission = True
interstellar_absorption = True
test_mode = False
produce_Plots = False
sigmaMg22 = 0.1
sigmaMg21 = 0.1
mg2_col = None
mg1_col = None
fe2_col = None
""",
        encoding="utf-8",
    )
    global_config.load_global_config(cfg_path)

    model_data = np.array([[100.0, 1.0, 0.0]])
    flux = np.array([[100.0, 1.0, 0.0]])

    star = SimpleNamespace(
        name="TEST",
        effective_temperature=5000.0,
        radius_sun_cm=1.0,
        distance_pc=10.0,
        right_ascension=0.0,
        declination=0.0,
        log_r=0.0,
        spectral_type="G",
    )

    monkeypatch.setattr("flux.flux_calc.load_model_for_temperature", lambda t: model_data)
    monkeypatch.setattr("flux.flux_calc.convertIntensityToFlux", lambda m, r: flux)
    monkeypatch.setattr("flux.flux_calc.compute_ebv_av", lambda *a: (0.1, None))
    monkeypatch.setattr("flux.flux_calc.apply_unred", lambda w, f, e: f)
    monkeypatch.setattr("flux.flux_calc.convert_flux_to_photons", lambda f, w: f)

    called = {"lce": 0, "ism": 0}

    def fake_lce(data, *a):
        called["lce"] += 1
        return data

    def fake_ism(data, ebv, cfg):
        called["ism"] += 1
        return data

    monkeypatch.setattr("flux.flux_calc.apply_line_core_emission", fake_lce)
    monkeypatch.setattr("flux.flux_calc.apply_ism_absorption", fake_ism)

    out = calculateFluxOnEarth(star, tmp_path)

    assert called["lce"] == 1
    assert called["ism"] == 1
    assert out.shape == (1,)
