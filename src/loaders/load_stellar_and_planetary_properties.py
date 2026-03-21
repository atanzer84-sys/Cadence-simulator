import numpy as np
import logging
from pathlib import Path
from domain.star import Star
from domain.planet import Planet
from loaders.load_excel import load_matching_excel_row_from_excel, load_excel_cfg, map_to_planet_or_star_dictionary
from loaders.parameter_preprocessing import get_missing_properties, clean_and_cast_parameters
from loaders.load_gaia import lookup_target_star_gaia, GAIA_PROVIDES
from astropy.io import ascii
from configs.global_config import GlobalConfig, get_global_config
from loaders.run_waltzer_context import get_repo_root
from utils.constants import MAG_G_SUN, TEMP_SUN


# Module-level cache (process lifetime) for Mamajek table arrays keyed by path.
_MAMAJEK_CACHE: tuple[str, np.ndarray, np.ndarray] | None = None

def load_stellar_and_planetary_properties(target_name_user_input):
    cfg = get_global_config()

    try:
        repo_root = get_repo_root()

        # find the excel file
        excel_path = _find_excel_file(repo_root / "input")
        
        # load the star's matching row in a mixed dictionary
        planet_star_dictionary, target_name = load_matching_excel_row_from_excel(excel_path, target_name_user_input)

        # load the excel config, because it defines what is a star's and planet's property. easily expandable.
        mapping = load_excel_mapping()

        # getting (still dirty) separate dictionaries for planetary and stellar properties
        planet_params, star_params = map_to_planet_or_star_dictionary(planet_star_dictionary, mapping, target_name)

        # TODO: MISSING PLANET WHEN NEEDED 
        # list all properties that are required by config, empty in excel to prep for gaia lookup
        missing_star = get_missing_properties(star_params, mapping["required_stellar_parameters"])
        missing_for_gaia = [k for k in missing_star if k in GAIA_PROVIDES]

        if missing_for_gaia:
            gaia_star_params = lookup_target_star_gaia(star_params, missing_for_gaia, cfg)
            # merge missing only (Excel wins)
            star_params = merge_gaia_into_star_params(star_params, gaia_star_params)

        # distance and radius fallbacks using Gaia parallax/magnitude if present
        star_params = apply_distance_from_parallax_if_missing(star_params)
        star_params = apply_radius_from_teff_mag_distance_if_missing(star_params)
        # getting spectral type from mamjeck table and applying logR fallback.
        star_params = infer_mamajek(star_params)
        star_params = apply_log_r_fallback(star_params, cfg=cfg)
        # now we finally have a list on missing parameters and can throw exceptions, because with missing parameters we can not do our simulation run.
        missing_star_final = get_missing_properties(star_params, mapping["required_stellar_parameters"])

        if missing_star_final:
            raise ValueError(f"Missing required star parameters after Excel, GAIA, Mamjeck and LogR lookup: {missing_star_final}")

        star_params = clean_and_cast_parameters(star_params, Star)
        planet_params = clean_and_cast_parameters(planet_params, Planet)
        msg = "Configurations loaded, checked and parsed."
        print(msg)
        logging.info(msg)
        return (planet_params, star_params, mapping["required_planetary_parameters"], mapping["required_stellar_parameters"]
        )
    except Exception:
        logging.exception("Failed to load all required properties after Excel/Mamjeck/Gaia lookup for target_name_user_input=%r", target_name_user_input)
        raise

def load_excel_mapping():
    repo_root = get_repo_root()
    mapping_path = repo_root / "configs" / "excel_mapping.cfg"
    return load_excel_cfg(mapping_path)

def infer_mamajek(star_params, log_output: bool = True):
    repo_root = get_repo_root()
    mamajek_path = repo_root / "data" / "stellar_param_mamjeck.txt"
    return infer_mamajek_spectral_type(star_params, mamajek_path, log_output)

def apply_radius_from_teff_mag_distance_if_missing(star_params: dict) -> dict:
    """
    If radius is missing, estimate it from Gaia G magnitude, distance, and Teff.

    Uses:
        M_G = m_G - 5 * log10(d / 10)
        L/Lsun = 10^(-0.4 * (M_G - M_G_sun))
        R/Rsun = sqrt(L/Lsun) * (T_sun / Teff)^2

    """
    if star_params.get("radius") is not None:
        return star_params

    teff = star_params.get("effective_temperature")
    distance_pc = star_params.get("distance")
    gaia_mag = star_params.get("gaia_magnitude")

    if teff is None or distance_pc is None or gaia_mag is None:
        return star_params

    try:
        teff = float(teff)
        distance_pc = float(distance_pc)
        gaia_mag = float(gaia_mag)
    except Exception:
        logging.exception("Radius fallback failed: invalid inputs teff=%r distance=%r gaia_mag=%r", teff, distance_pc, gaia_mag)
        return star_params

    if teff <= 0.0 or distance_pc <= 0.0:
        return star_params

    abs_g_mag = gaia_mag - 5.0 * np.log10(distance_pc / 10.0)
    luminosity_lsun = 10.0 ** (-0.4 * (abs_g_mag - MAG_G_SUN))
    radius_rsun = np.sqrt(luminosity_lsun) * (TEMP_SUN / teff) ** 2

    if not np.isfinite(radius_rsun) or radius_rsun <= 0.0:
        return star_params

    star_params["radius"] = float(radius_rsun)

    logging.info("Radius fallback applied for %s: G=%.3f distance_pc=%.3f Teff=%.1f -> M_G=%.3f L/Lsun=%.6f R/Rsun=%.6f", star_params.get("name"), gaia_mag, distance_pc, teff, abs_g_mag, luminosity_lsun, radius_rsun)
    return star_params


def apply_distance_from_parallax_if_missing(star_params: dict) -> dict:
    """
    If distance is missing, set it from parallax in star_params
    using distance_pc = 1000 / parallax_mas when valid.
    """
    if star_params.get("distance") is not None:
        return star_params

    par = star_params.get("parallax")
    if par is None or np.ma.is_masked(par):
        return star_params

    try:
        par = float(par)
    except Exception:
        return star_params

    if not np.isfinite(par) or par <= 0.0:
        return star_params

    star_params["distance"] = 1000.0 / par
    return star_params


def _find_excel_file(repo_root: Path):
    try:
        # Ignore temporary Excel lock files (e.g. "~$Targets_V10p1.xlsx")
        excel_files = [
            f for f in repo_root.glob("*.xlsx") if not f.name.startswith("~$")
        ]

        if len(excel_files) == 0:
            raise FileNotFoundError(f"No Excel file found in repo root ({repo_root})")

        if len(excel_files) > 1:
            names = [f.name for f in excel_files]
            raise ValueError(f"Multiple Excel files found in repo root: {names}")

        logging.info("Using Excel file to look up target: '%s'", excel_files[0])

        return excel_files[0]

    except Exception:
        logging.exception("Failed to locate Excel file in repo root: %s", repo_root)
        raise

def merge_gaia_into_star_params(star_params, gaia_star_params):
    if not gaia_star_params:
        logging.info("No Gaia parameters to merge.")
        return star_params

    for k, v in gaia_star_params.items():
        current = star_params.get(k)

        if current is None or (isinstance(current, str) and current.strip() == ""):
            logging.info("Gaia merge: setting %s = %r", k, v)
            star_params[k] = v
        else:
            logging.info("Gaia merge: keeping existing %s = %r", k, current)

    return star_params

def infer_mamajek_spectral_type(star_params, mamajek_path, log_output: bool = True):
    mamajek_path = str(mamajek_path)
    global _MAMAJEK_CACHE

    if _MAMAJEK_CACHE is None or _MAMAJEK_CACHE[0] != mamajek_path:
        if log_output:
            logging.info("Loading Mamajek table from %s", mamajek_path)
        data = ascii.read(mamajek_path, comment="#")
        Sp = np.array(data["col1"])
        T_book = np.array(data["col2"], dtype=float)
        _MAMAJEK_CACHE = (mamajek_path, Sp, T_book)
        if log_output:
            logging.info("Mamajek table loaded successfully (%d rows)", len(data))
    else:
        _, Sp, T_book = _MAMAJEK_CACHE
        if log_output:
            logging.info("Using cached Mamajek table")

    Teff_raw = star_params.get("effective_temperature")

    if Teff_raw is None:
        logging.error("Mamajek inference aborted: 'effective_temperature' is missing. Current star_params keys: %s", list(star_params.keys()))
        raise ValueError("Cannot infer spectral type: effective_temperature is None.")
    try:
        Teff = float(Teff_raw)

    except (TypeError, ValueError) as e:
        logging.error("Mamajek inference failed: effective_temperature=%r is not convertible to float.", Teff_raw)
        raise ValueError(f"Cannot infer spectral type: invalid effective_temperature value {Teff_raw!r}") from e

    idx = int(np.abs(T_book - Teff).argmin())
    spectral_type = str(Sp[idx])

    star_params["spectral_type"] = spectral_type

    if log_output:
        logging.info("Set spectral_type=%s inferred from Teff=%s K",spectral_type,Teff)

    return star_params

def apply_log_r_fallback(star_params: dict, cfg: GlobalConfig, log_output: bool = True) -> dict:
    if not cfg.enable_log_r_fallback:
        logging.info("log_r fallback skipped: enable_log_r_fallback=%s", cfg.enable_log_r_fallback)
        return star_params

    if star_params.get("log_r") is not None:
        return star_params

    Teff = star_params.get("effective_temperature")
    if Teff is None:
        logging.info("log_r fallback skipped: effective_temperature missing.")
        return star_params

    try:
        Teff = float(Teff)
    except Exception:
        logging.exception("log_r fallback failed: invalid Teff=%r", Teff)
        return star_params

    threshold = cfg.log_r_teff_threshold
    hot_val = cfg.log_r_hot_value
    cool_val = cfg.log_r_cool_value

    if Teff > threshold:
        star_params["log_r"] = hot_val
    else:
        star_params["log_r"] = cool_val
    if log_output:
        logging.info("log_r fallback applied: Teff=%s threshold=%s -> log_r=%s", Teff, threshold, star_params["log_r"])

    return star_params
