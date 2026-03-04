from dataclasses import dataclass
from typing import Any, Dict
import logging


@dataclass(frozen=True)
class Planet:
    name: str
    discoverymethod: str | None
    orbital_period: float | None
    orbital_semi_major_axis: float | None
    radius_jupiter: float | None
    mass_jupiter: float | None
    equilibrium_temperature: float | None
    scale_height: float | None

    @classmethod
    def from_params(
        cls,
        planet_params: Dict[str, Any],
        required_keys: list[str],
    ) -> "Planet":
        def is_missing(v: Any) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "")

        missing = [k for k in required_keys if k not in planet_params or is_missing(planet_params.get(k))]
        if missing:
            raise ValueError(f"Planet missing required keys at construction: {missing}")

        msg1 = "\n==== PLANET Created ===="
        msg2 = f"Planet created: {planet_params['name']}"
        print(msg1)
        print(msg2)
        logging.info(msg1.strip())
        logging.info(msg2)

        planet = cls(
            name=planet_params["name"],
            discoverymethod=planet_params.get("discoverymethod"),
            orbital_period=planet_params.get("orbital_period"),
            orbital_semi_major_axis=planet_params.get("orbital_semi_major_axis"),
            radius_jupiter=planet_params.get("radius_jupiter"),
            mass_jupiter=planet_params.get("mass_jupiter"),
            equilibrium_temperature=planet_params.get("equilibrium_temperature"),
            scale_height=planet_params.get("scale_height"),
        )
        logging.info("Created Planet object: %s", planet)

        return planet
