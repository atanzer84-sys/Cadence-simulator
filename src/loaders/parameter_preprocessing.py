import logging
from dataclasses import fields
from typing import Any, get_args, get_origin, Union

STELLAR_NUMERIC_KEYS = {
    "st_teff",
    "st_rad",
    "st_mass",
    "st_logg",
    "st_met",
    "st_dist",
    "st_vsin",
    "st_lum",
}

PLANETARY_NUMERIC_KEYS = {
    "pl_orbper",
    "pl_orbsmax",
    "pl_radj",
    "pl_bmassj",
    "pl_eqt",
    "scale_height_km",
}

def get_missing_properties(parameters: dict, required_keys: list[str]) -> list[str]:
    """
    Return list of required keys that are missing in parameters.

    Missing means:
      • key not present, or
      • value is None, or
      • value is an empty/whitespace-only string
    """
    def is_missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    return [k for k in required_keys if k not in parameters or is_missing(parameters.get(k))]

def clean_and_cast_parameters(parameters: dict[str, Any], domain_class: type) -> dict[str, Any]:
    """
    Normalize parameters according to the dataclass schema of model_cls.

    Rules:
    - strip strings, empty -> None
    - cast to float if the dataclass field type is float or float | None
    - cast to str if the dataclass field type is str or str | None
    - leave keys untouched if they are not dataclass fields
    - never decide requiredness here
    """
    # build a mapping based on the definition of our classes
    field_types = {f.name: f.type for f in fields(domain_class)}

    # force it to use the real type of a class property
    def base_type(tp: Any) -> Any:
        origin = get_origin(tp)
        if origin is Union:
            args = [a for a in get_args(tp) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return tp

    cleaned: dict[str, Any] = {}

    for key, value in parameters.items():
        # normalize strings
        if isinstance(value, str):
            v = value.strip()
            value = None if v == "" else v

        # Only fields that actually exist on Star or Planet are cast
        # Extra config keys or metadata are left alone
        # None is always allowed (optional fields)
        if key in field_types and value is not None:
            bt = base_type(field_types[key])

            if bt is float:
                try:
                    value = float(value)
                except (TypeError, ValueError) as e:
                    logging.error(
                        "Invalid numeric value for parameter '%s': %r (expected float)",
                        key,
                        value,
                    )
                    raise ValueError(
                        f"Parameter '{key}' must be a float, got {value!r}"
                    ) from e

            elif bt is str:
                value = str(value)

        cleaned[key] = value

    return cleaned


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            stripped = value.strip()
            value = None if stripped == "" else stripped
        normalized[key] = value
    return normalized


def _require_name(params: dict[str, Any], name_key: str) -> None:
    value = params.get(name_key)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"Missing required parameter: {name_key}")


def _cast_numeric(params: dict[str, Any], numeric_keys: set[str]) -> dict[str, Any]:
    casted = dict(params)
    for key in numeric_keys:
        if key in casted and casted[key] is not None:
            try:
                casted[key] = float(casted[key])
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid numeric value for {key}") from e
    return casted


def process_stellar_parameter_values(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize stellar parameters and validate required fields."""
    normalized = _normalize_params(params)
    _require_name(normalized, "st_name")
    return _cast_numeric(normalized, STELLAR_NUMERIC_KEYS)


def process_planetary_parameter_values(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize planetary parameters and validate required fields."""
    normalized = _normalize_params(params)
    _require_name(normalized, "pl_name")
    return _cast_numeric(normalized, PLANETARY_NUMERIC_KEYS)
