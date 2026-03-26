import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
import loaders.load_model_temperature as load_model_temperature_module

from loaders.load_model_temperature import (
    load_model_for_temperature,
    _get_available_models,
    _cut_model_wavelength_range,
)

def _find_tests_dir() -> Path:
    """Find tests/ by walking up from this file; works no matter where the test lives under tests/."""
    p = Path(__file__).resolve().parent
    while p.name != "tests" and p.parent != p:
        p = p.parent
    return p if p.name == "tests" else Path(__file__).resolve().parent.parent


# Fixture root: tests/fixtures (never touches real data/models in repo root)
FIXTURES_ROOT = _find_tests_dir() / "fixtures"
MODELS_DIR = FIXTURES_ROOT / "data" / "models"

@pytest.fixture(autouse=True)
def reset_model_temperature_caches():
    """Reset module-level caches to keep tests independent."""
    load_model_temperature_module._MODEL_CACHE.clear()
    load_model_temperature_module._CUT_MODEL_CACHE.clear()
    load_model_temperature_module._AVAILABLE_MODELS = None
    yield
    load_model_temperature_module._MODEL_CACHE.clear()
    load_model_temperature_module._CUT_MODEL_CACHE.clear()
    load_model_temperature_module._AVAILABLE_MODELS = None


# Tests: _get_available_models
# Behavior: returns sorted temperatures from matching directory names
def test_get_available_model_temps_returns_sorted_matching_dirs(tmp_path):
    """_get_available_models returns sorted temperatures from matching model dir names."""
    (tmp_path / "t05700g4.4").mkdir()
    (tmp_path / "t05800g4.4").mkdir()
    (tmp_path / "t05600g4.4").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "t12345g4.4").mkdir()

    temps = [t for t, _ in _get_available_models(tmp_path)]

    assert temps == [5600, 5700, 5800, 12345]


# Tests: _get_available_models
# Behavior: ignores files and non-matching directory names
def test_get_available_model_temps_ignores_files(tmp_path):
    """_get_available_models ignores files and non-matching names."""
    (tmp_path / "t05700g4.4").mkdir()
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "t05700g44").mkdir()  # wrong pattern (no dot)

    temps = [t for t, _ in _get_available_models(tmp_path)]

    assert temps == [5700]


# Tests: _get_available_models
# Behavior: accepts both g4.4 and g4,4 naming styles
def test_get_available_model_temps_accepts_comma_gravity_separator(tmp_path):
    """_get_available_models accepts both g4.4 and g4,4 naming."""
    (tmp_path / "t05700g4,4").mkdir()
    (tmp_path / "t05800g4.4").mkdir()

    temps = [t for t, _ in _get_available_models(tmp_path)]

    assert temps == [5700, 5800]


# Tests: load_model_for_temperature
# Behavior: picks the closest available model temperature
def test_load_model_picks_closest_temp():
    """load_model_for_temperature picks the available temp closest to the requested one (5756 → 5800)."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT):
        result = load_model_for_temperature(5756.0, 900.0, 2100.0)

    # Fixtures have t05600, t05700, t05800. Closest to 5756 is 5800 → loads t05800g4.4/model.flx.
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [1000.0, 2000.0])


# Tests: load_model_for_temperature
# Behavior: uses exact-match model when available
def test_load_model_exact_match():
    """load_model_for_temperature loads from exact temp dir when it exists in available temps."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT):
        result = load_model_for_temperature(5700.0, 900.0, 2100.0)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [1000.0, 2000.0])


# Tests: load_model_for_temperature
# Behavior: announces selected model on successful load
def test_load_model_success_announces_loaded_model():
    """load_model_for_temperature announces loaded model path on success."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT), \
         patch("loaders.load_model_temperature.announce") as mock_announce:
        load_model_for_temperature(5700.0, 900.0, 2100.0, announce_user=True)

    mock_announce.assert_called_once()
    message, announce_user = mock_announce.call_args.args
    assert "Loaded stellar model" in message
    assert "t05700g4.4/model.flx" in message
    assert announce_user is True


# Tests: load_model_for_temperature
# Behavior: warns and prints when selected model is far from requested temperature
def test_load_model_large_delta_warns_and_prints():
    """load_model_for_temperature emits warning path and still loads successfully for large delta."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT), \
         patch("loaders.load_model_temperature.logging.warning") as mock_warning, \
         patch("loaders.load_model_temperature.announce") as mock_print:
        result = load_model_for_temperature(6201.0, 900.0, 2100.0, announce_user=True)

    assert result.shape == (2, 3)
    mock_warning.assert_called_once()
    assert mock_print.call_count == 2
    assert all(call.args[1] is True for call in mock_print.call_args_list)


# Tests: load_model_for_temperature
# Behavior: raises when no model directories are available
def test_load_model_no_models_raises():
    """load_model_for_temperature raises FileNotFoundError when no stellar model dirs exist."""
    with patch("loaders.load_model_temperature.get_repo_root") as mock_root, \
         patch("loaders.load_model_temperature._get_available_models") as mock_models:

        mock_root.return_value = Path("/fake/repo")
        mock_models.return_value = []

        with pytest.raises(FileNotFoundError, match="No stellar models found"):
            load_model_for_temperature(5756.0, 900.0, 2100.0)


# Tests: load_model_for_temperature
# Behavior: raises when data/models directory is missing
def test_load_model_missing_models_directory_raises(tmp_path):
    """load_model_for_temperature raises the same FileNotFoundError when data/models directory is missing."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=tmp_path):
        with pytest.raises(FileNotFoundError, match="No stellar models found"):
            load_model_for_temperature(5756.0, 900.0, 2100.0)


# Tests: load_model_for_temperature
# Behavior: raises when selected model directory has no model.flx
def test_load_model_file_missing_raises():
    """load_model_for_temperature raises when model dir exists but model.flx is missing."""
    with patch("loaders.load_model_temperature.get_repo_root") as mock_root, \
         patch("loaders.load_model_temperature._get_available_models") as mock_models, \
         patch.object(Path, "is_file", return_value=False):

        mock_root.return_value = Path("/fake/repo")
        mock_models.return_value = [(5700, Path("/fake/repo/data/models/t05700g4.4"))]

        with pytest.raises(FileNotFoundError, match="model.flx missing"):
            load_model_for_temperature(5756.0, 900.0, 2100.0)


# Tests: _cut_model_wavelength_range
# Behavior: keeps rows within inclusive wavelength bounds
def test__cut_model_wavelength_range_inclusive_bounds():
    """_cut_model_wavelength_range keeps rows within inclusive wavelength limits."""
    model_data = np.array(
        [
            [900.0, 1.0, 2.0],
            [1000.0, 3.0, 4.0],
            [1500.0, 5.0, 6.0],
            [2000.0, 7.0, 8.0],
            [2100.0, 9.0, 10.0],
        ],
        dtype=float,
    )

    cut = _cut_model_wavelength_range(model_data, wl_min_A=1000.0, wl_max_A=2000.0)

    assert cut.shape == (3, 3)
    np.testing.assert_allclose(cut[:, 0], [1000.0, 1500.0, 2000.0])
