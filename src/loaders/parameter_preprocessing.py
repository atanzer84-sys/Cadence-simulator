import logging
from typing import Any, Dict

def process_stellar_parameter_values(stellar_parameters_excel: dict):
    cleaned_stellar_parameters: Dict[str, Any] = {}
    for key, value in stellar_parameters_excel.items():
        if isinstance(value, str):
            v = value.strip()
            cleaned_stellar_parameters[key] = None if v == "" else v
        else:
            cleaned_stellar_parameters[key] = value

    if "st_name" not in cleaned_stellar_parameters or cleaned_stellar_parameters["st_name"] is None:
        raise ValueError("Missing required star parameter: st_name")
    
    star_numeric_keys = {
        "st_teff",
        "st_rad",
        "st_mass",
        "st_logg",
        "st_met",
        "st_dist",
        "st_vsin",
        "st_lum",
    }

    for key in star_numeric_keys:
        if key not in cleaned_stellar_parameters:
            continue

        value = cleaned_stellar_parameters[key]
        if value is None:
            continue

        try:
            cleaned_stellar_parameters[key] = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Star parameter '{key}' must be numeric, got {value!r}"
            ) from e

    # Lookup hook (Part 2): do nothing for now
    # cleaned_stellar_parameters = lookup_missing_star_parameters(cleaned_stellar_parameters)

    logging.info("Processed stellar parameters:")
    for key, value in cleaned_stellar_parameters.items():
        logging.info("  %s = %r", key, value)

    return cleaned_stellar_parameters


def process_planetary_parameter_values(planetary_parameters_excel: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_planetary_parameters: Dict[str, Any] = {}

    for key, value in planetary_parameters_excel.items():
        if isinstance(value, str):
            v = value.strip()
            cleaned_planetary_parameters[key] = None if v == "" else v
        else:
            cleaned_planetary_parameters[key] = value

    if "pl_name" not in cleaned_planetary_parameters or cleaned_planetary_parameters["pl_name"] is None:
        raise ValueError("Missing required planet parameter: pl_name")

    planet_numeric_keys = {
        "pl_orbper",
        "pl_orbsmax",
        "pl_radj",
        "pl_bmassj",
        "pl_eqt",
        "scale_height_km",
    }

    for key in planet_numeric_keys:
        if key not in cleaned_planetary_parameters:
            continue

        value = cleaned_planetary_parameters[key]
        if value is None:
            continue

        try:
            cleaned_planetary_parameters[key] = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Planet parameter '{key}' must be numeric, got {value!r}"
            ) from e

    logging.info("Processed planetary parameters:")
    for key, value in cleaned_planetary_parameters.items():
        logging.info("  %s = %r", key, value)

    return cleaned_planetary_parameters
