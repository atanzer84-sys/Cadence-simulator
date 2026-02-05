import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from domain.constants import C_LIGHT, R_SUN
from flux.flux_calc import load_model_for_temperature, convertIntensityToLuminosity

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

    out = convertIntensityToLuminosity(model_data, r_star)

    assert out.shape == model_data.shape
    np.testing.assert_allclose(out[:,0], model_data[:,0])

def test_frequency_to_wavelength_conversion():
    model_data = np.array([[1000.0, 1.0, 0.0]])
    r_star = 1.0

    out = convertIntensityToLuminosity(model_data, r_star)

    geometry = 4 * np.pi * r_star**2 * 4 * np.pi
    recovered_intensity = out[0,1] / geometry

    expected_intensity = C_LIGHT / (1000.0**2)
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

    got = convertIntensityToLuminosity(model_data, r_star_cm)

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

    got = convertIntensityToLuminosity(model_data, r_star_cm)

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

    got = convertIntensityToLuminosity(model_data, r_star_cm)

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

    got = convertIntensityToLuminosity(model_data, r_star_cm)

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

    got = convertIntensityToLuminosity(model_data, r_star_cm)

    assert got.shape == expected.shape

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)

