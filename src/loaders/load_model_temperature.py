import logging
import numpy as np
from loaders.run_cadence_context import get_repo_root
import re
from pathlib import Path
from utils.helpers import announce

# generate a model file cache, because with a lot of background stars, we read a lot of models. we want to use a cache for models that have already been loaded.
_MODEL_CACHE = {}
_AVAILABLE_MODELS = None
_CUT_MODEL_CACHE = {}


def load_model_for_temperature(t_star, wl_min_A: float, wl_max_A: float, announce_user: bool = False):
    repo_root = get_repo_root()
    models_dir = repo_root / "data" / "models"

    global _AVAILABLE_MODELS

    if _AVAILABLE_MODELS is None:
        _AVAILABLE_MODELS = _get_available_models(models_dir)
        logging.info("MODEL_CACHE available_models_loaded count=%d", len(_AVAILABLE_MODELS))

    models = _AVAILABLE_MODELS
    if not models:
        raise FileNotFoundError("No stellar models found in data/models")

    t_target = float(t_star)
    t_pick, model_dir = min(models, key=lambda item: abs(item[0] - t_target))
    delta = abs(t_pick - t_target)

    if delta > 300:
        msg = f"MODEL_TEMP_LARGE_DELTA: requested={t_target:.0f} K, picked={t_pick} K, delta={delta:.0f} K"
        logging.warning(msg)
        announce(msg, announce_user)

    model_file = model_dir / "model.flx"

    if model_file.is_file():
        # use the cache for models
        model_file_str = str(model_file)
        is_cached = model_file_str in _CUT_MODEL_CACHE

        if not is_cached:
            full_model_data = np.loadtxt(model_file)
            cut_model_data = _cut_model_wavelength_range(full_model_data, wl_min_A, wl_max_A)
            cut_model_data.setflags(write=False)
            _CUT_MODEL_CACHE[model_file_str] = cut_model_data


        logging.info("MODEL_CACHE %s | %s | Teff_req=%d Teff_pick=%d delta=%d | wl=%.0f-%.0f | cache=%d", "HIT" if is_cached else "MISS", model_dir.name, int(t_target), t_pick, int(delta), wl_min_A, wl_max_A, len(_CUT_MODEL_CACHE))
        
        model_data = _CUT_MODEL_CACHE[model_file_str]
        msg = f"Loaded stellar model {model_file.relative_to(models_dir)} for Teff={t_target:.0f} K (picked={t_pick} K)"
        announce(msg, announce_user)
        return model_data

    raise FileNotFoundError(
        f"Model directory exists but model.flx missing for picked={t_pick} K (dir={model_dir.name})"
    )

def _cut_model_wavelength_range(model_data: np.ndarray, wl_min_A: float, wl_max_A: float) -> np.ndarray:
    mask = (model_data[:, 0] >= wl_min_A) & (model_data[:, 0] <= wl_max_A)
    return model_data[mask]
    
def _get_available_models(models_dir: Path) -> list[tuple[int, Path]]:
    if not models_dir.is_dir():
        return []

    models = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"t(\d{5})g4[.,]4", p.name)
        if m:
            models.append((int(m.group(1)), p))
    models.sort(key=lambda item: item[0])
    return models
