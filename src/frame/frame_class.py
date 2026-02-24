from dataclasses import dataclass
import numpy as np
from astropy.io import fits


@dataclass
class Frame:
    data: np.ndarray
    header: fits.Header | None
    frame_type: str
    channel_tag: str

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 2:
            raise ValueError("data must be 2D numpy array")

        if self.header is not None and not isinstance(self.header, fits.Header):
            raise TypeError("header must be astropy.io.fits.Header or None")

    @property
    def nx(self):
        return self.data.shape[1]

    @property
    def ny(self):
        return self.data.shape[0]
