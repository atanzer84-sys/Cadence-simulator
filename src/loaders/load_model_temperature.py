import logging
import numpy as np
from loaders.run_waltzer_context import get_repo_root
import re
from pathlib import Path
from utils.helpers import announce, print_if_enabled


def _get_available_models(models_dir: Path) -> list[tuple[int, Path]]:
    models = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"t(\d{5})g4[.,]4", p.name)
        if m:
            models.append((int(m.group(1)), p))
    models.sort(key=lambda item: item[0])
    return models

def load_model_for_temperature(t_star, announce_user: bool = False):
    """
    Load stellar model spectrum for given effective temperature.
    Mirrors legacy selection logic exactly.
    """
    repo_root = get_repo_root()
    models_dir = repo_root / "data" / "models"

    models = _get_available_models(models_dir)
    if not models:
        raise FileNotFoundError("No stellar models found in data/models")

    t_target = float(t_star)
    t_pick, model_dir = min(models, key=lambda item: abs(item[0] - t_target))
    delta = abs(t_pick - t_target)

    if delta > 300:
        msg = f"MODEL_TEMP_LARGE_DELTA: requested={t_target:.0f} K, picked={t_pick} K, delta={delta:.0f} K"
        logging.warning(msg)
        print_if_enabled(msg, announce_user)

    model_file = model_dir / "model.flx"

    if model_file.is_file():
        model_data = np.loadtxt(model_file)
        msg = f"Loaded stellar model {model_file.relative_to(models_dir)} for Teff={t_target:.0f} K (picked={t_pick} K)"
        announce(msg, announce_user)
        return model_data

    raise FileNotFoundError(
        f"Model directory exists but model.flx missing for picked={t_pick} K (dir={model_dir.name})"
    )

