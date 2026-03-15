import logging
import types
from dataclasses import fields
from typing import Any, get_args, get_origin, Union


def is_missing(v: Any) -> bool:
    """Return True if value is considered missing (None or empty/whitespace-only string)."""
    return v is None or (isinstance(v, str) and v.strip() == "")


def get_missing_properties(parameters: dict, required_keys: list[str], log_output: bool = True) -> list[str]:
    """
    Return list of required keys that are missing in parameters.

    Missing means:
      • key not present, or
      • value is None, or
      • value is an empty/whitespace-only string
    """
    if log_output:
        logging.info("Checking missing properties...")
        logging.info("Required keys: %s", required_keys)
        logging.info("Current parameters: %s", parameters)
        logging.info("Missing properties result: %s", [k for k in required_keys if k not in parameters or is_missing(parameters.get(k))])

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

        # Handle both typing.Union[...] and PEP604 unions (X | Y)
        if origin is Union or origin is types.UnionType:
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
                    logging.error("Invalid numeric value for parameter '%s': %r (expected float)", key, value)
                    raise ValueError(
                        f"Parameter '{key}' must be a float, got {value!r}"
                    ) from e

            elif bt is str:
                value = str(value)

        cleaned[key] = value

    return cleaned