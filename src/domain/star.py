from dataclasses import dataclass
from utils.constants import R_SUN, M_SUN
from typing import Any, Dict
import logging


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
    radius_sun_cm: float
    mass_sun_kg: float | None

    @classmethod
    def from_params(
        cls,
        star_params: Dict[str, Any],
        required_keys: list[str],
    ) -> "Star":
        def is_missing(v: Any) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "")

        missing = [k for k in required_keys if k not in star_params or is_missing(star_params.get(k))]
        if missing:
            raise ValueError(f"Star missing required keys at construction: {missing}")
        
        print("==== STAR Created ====")
        print(f"Star created: {star_params['name']}")

        radius = star_params["radius"]
        mass = star_params.get("mass")

        radius_sun_cm = radius * R_SUN
        mass_sun_kg = mass * M_SUN

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