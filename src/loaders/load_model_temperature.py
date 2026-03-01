import logging
import numpy as np
from loaders.run_waltzer_context import get_repo_root
import re
from pathlib import Path


def _get_available_model_temps(models_dir: Path) -> list[int]:
    temps = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"t(\d{5})g4\.4", p.name)
        if m:
            temps.append(int(m.group(1)))
    temps.sort()
    return temps

def load_model_for_temperature(t_star):
    """
    Load stellar model spectrum for given effective temperature.
    Mirrors legacy selection logic exactly.
    """
    repo_root = get_repo_root()
    models_dir = repo_root / "data" / "models"

    temps = _get_available_model_temps(models_dir)
    if not temps:
        raise FileNotFoundError("No stellar models found in data/models")

    t_target = float(t_star)
    t_pick = min(temps, key=lambda t: abs(t - t_target))
    delta = abs(t_pick - t_target)

    if delta > 300:
        msg = f"MODEL_TEMP_LARGE_DELTA: requested={t_target:.0f} K, picked={t_pick} K, delta={delta:.0f} K"
        logging.warning(msg)
        print(msg)

    subdir = f"t{t_pick:05d}g4.4"
    model_file = models_dir / subdir / "model.flx"

    if model_file.is_file():
        model_data = np.loadtxt(model_file)
        logging.info("Loaded stellar model %s for Teff=%s K (picked=%s K)", model_file.relative_to(models_dir), t_star, t_pick)
        return model_data

    raise FileNotFoundError(f"Model directory exists but model.flx missing for picked={t_pick} K (dir={subdir})")

