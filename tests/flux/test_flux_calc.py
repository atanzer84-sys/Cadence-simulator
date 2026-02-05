import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from flux.flux_calc import load_model_for_temperature

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

