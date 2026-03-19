import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from loaders.load_model_temperature import (
    load_model_for_temperature,
    _get_available_models,
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


def test_get_available_model_temps_returns_sorted_matching_dirs(tmp_path):
    """_get_available_models returns sorted temperatures from matching model dir names."""
    (tmp_path / "t05700g4.4").mkdir()
    (tmp_path / "t05800g4.4").mkdir()
    (tmp_path / "t05600g4.4").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "t12345g4.4").mkdir()

    temps = [t for t, _ in _get_available_models(tmp_path)]

    assert temps == [5600, 5700, 5800, 12345]


def test_get_available_model_temps_ignores_files(tmp_path):
    """_get_available_models ignores files and non-matching names."""
    (tmp_path / "t05700g4.4").mkdir()
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "t05700g44").mkdir()  # wrong pattern (no dot)

    temps = [t for t, _ in _get_available_models(tmp_path)]

    assert temps == [5700]


def test_load_model_picks_closest_temp():
    """load_model_for_temperature picks the available temp closest to the requested one (5756 → 5800)."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT):
        result = load_model_for_temperature(5756.0)

    # Fixtures have t05600, t05700, t05800. Closest to 5756 is 5800 → loads t05800g4.4/model.flx.
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [1000.0, 2000.0])


def test_load_model_exact_match():
    """load_model_for_temperature loads from exact temp dir when it exists in available temps."""
    with patch("loaders.load_model_temperature.get_repo_root", return_value=FIXTURES_ROOT):
        result = load_model_for_temperature(5700.0)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [1000.0, 2000.0])


def test_load_model_no_models_raises():
    """load_model_for_temperature raises FileNotFoundError when no stellar model dirs exist."""
    with patch("loaders.load_model_temperature.get_repo_root") as mock_root, \
         patch("loaders.load_model_temperature._get_available_models") as mock_models:

        mock_root.return_value = Path("/fake/repo")
        mock_models.return_value = []

        with pytest.raises(FileNotFoundError, match="No stellar models found"):
            load_model_for_temperature(5756.0)


def test_load_model_file_missing_raises():
    """load_model_for_temperature raises when model dir exists but model.flx is missing."""
    with patch("loaders.load_model_temperature.get_repo_root") as mock_root, \
         patch("loaders.load_model_temperature._get_available_models") as mock_models, \
         patch.object(Path, "is_file", return_value=False):

        mock_root.return_value = Path("/fake/repo")
        mock_models.return_value = [(5700, Path("/fake/repo/data/models/t05700g4.4"))]

        with pytest.raises(FileNotFoundError, match="model.flx missing"):
            load_model_for_temperature(5756.0)
