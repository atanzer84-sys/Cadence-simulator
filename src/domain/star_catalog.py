from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from domain.star import Star

CountsPerPixel = np.ndarray

@dataclass
class StarCatalog:
    stars_by_id: Dict[str, Star] = field(default_factory=dict)
    
    # spectroscopy: store per-pixel counts array
    # photometry: store total counts as scalar
    counts_by_id_and_band: Dict[Tuple[str, str], CountsPerPixel] = field(default_factory=dict)
    
    offsets_arcsec_by_id: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    
    def add_star(self, star_id: str, star: Star) -> None:
        self.stars_by_id[star_id] = star

    def get_star(self, star_id: str) -> Star:
        return self.stars_by_id[star_id]

    def set_offset_arcsec(self, star_id: str, dx_arcsec: float, dy_arcsec: float) -> None:
        dx = float(dx_arcsec)
        dy = float(dy_arcsec)
        sep = (dx * dx + dy * dy) ** 0.5
        self.offsets_arcsec_by_id[star_id] = (dx, dy, sep)

    def get_offset_arcsec(self, star_id: str) -> Tuple[float, float]:
        dx, dy, _ = self.offsets_arcsec_by_id[star_id]
        return dx, dy

    def get_separation_arcsec(self, star_id: str) -> float:
        return self.offsets_arcsec_by_id[star_id][2]