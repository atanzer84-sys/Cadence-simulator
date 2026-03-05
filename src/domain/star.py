from dataclasses import dataclass
from utils.constants import R_SUN_cm, M_SUN_kg
from typing import Any, Dict
import logging

from loaders.parameter_preprocessing import is_missing

@dataclass(frozen=True)
class Star:
    name: str
    spectral_type: str | None
    effective_temperature: float | None
    radius: float | None
    mass: float | None
    metallicity: float | None
    surface_gravity: float | None
    right_ascension: float | None
    declination: float | None
    distance_pc: float | None
    v_magnitude: float | None
    gaia_magnitude: float | None
    log_r: float | None
    radius_sun_cm: float | None
    mass_sun_kg: float | None

    @classmethod
    def from_params(
        cls,
        star_params: Dict[str, Any],
        required_keys: list[str],
        log_output: bool = True,
    ) -> "Star":
        missing = [k for k in required_keys if k not in star_params or is_missing(star_params.get(k))]
        if missing:
            raise ValueError(f"Star missing required keys at construction: {missing}")
        
        if log_output:
            msg1 = "\n==== Target STAR Created ===="
            msg2 = f"Star created: {star_params['name']}"
            print(msg1)
            print(msg2)
            logging.info(msg1.strip())
            logging.info(msg2)


        radius = star_params.get("radius")
        if radius is None:
            raise ValueError("Star missing required key at construction: ['radius']")
        radius_sun_cm = radius * R_SUN_cm

        mass = star_params.get("mass")
        mass_sun_kg = mass * M_SUN_kg if mass is not None else None

        star = cls(
            name=star_params["name"],
            spectral_type=star_params.get("spectral_type"),
            effective_temperature=star_params.get("effective_temperature"),
            radius=radius,
            mass=mass,
            metallicity=star_params.get("metallicity"),
            surface_gravity=star_params.get("surface_gravity"),
            right_ascension=star_params.get("right_ascension"),
            declination=star_params.get("declination"),
            distance_pc=star_params.get("distance"),
            v_magnitude=star_params.get("v_magnitude"),
            gaia_magnitude=star_params.get("gaia_magnitude"),
            log_r=star_params.get("log_r"),
            radius_sun_cm=radius_sun_cm,
            mass_sun_kg=mass_sun_kg,
        )

        logging.info("Created Star object: %s", star)

        return star
