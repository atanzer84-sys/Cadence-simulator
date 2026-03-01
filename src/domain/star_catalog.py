# src/domain/star_catalog.py

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

from domain.star import Star


FluxEarth = Tuple[np.ndarray, np.ndarray]  # (wavelength, flux_earth)
CountsPerPixel = np.ndarray


@dataclass
class StarCatalog:
    stars_by_id: Dict[str, Star] = field(default_factory=dict)
    flux_earth_by_id: Dict[str, FluxEarth] = field(default_factory=dict)
    counts_by_id_and_band: Dict[Tuple[str, str], CountsPerPixel] = field(default_factory=dict)

    def add_star(self, star_id: str, star: Star) -> None:
        self.stars_by_id[star_id] = star

    def get_star(self, star_id: str) -> Star:
        return self.stars_by_id[star_id]

    def get_flux_earth(self, star_id: str, compute: Callable[[Star], FluxEarth]) -> FluxEarth:
        if star_id in self.flux_earth_by_id:
            return self.flux_earth_by_id[star_id]
        star = self.get_star(star_id)
        flux_earth = compute(star)
        self.flux_earth_by_id[star_id] = flux_earth
        return flux_earth

    # def get_counts(self, star_id: str, band: str, compute: Callable[[Star, FluxEarth], CountsPerPixel]) -> CountsPerPixel:
    #     key = (star_id, band)
    #     if key in self.counts_by_id_and_band:
    #         return self.counts_by_id_and_band[key]
    #     flux_earth = self.get_flux_earth(star_id, compute=lambda s: compute_flux_earth(s))
    #     star = self.get_star(star_id)
    #     counts = compute(star, flux_earth)
    #     self.counts_by_id_and_band[key] = counts
    #     return counts

# You will provide these functions from your existing pipeline code.
def compute_flux_earth(star: Star) -> FluxEarth:
    raise NotImplementedError