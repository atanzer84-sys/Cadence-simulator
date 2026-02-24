from dataclasses import dataclass
import numpy as np
from astropy.io import fits
from frame.write_fits import write_fits_frame


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

    def write_fits(self, ctx, index=0):
        """Write this frame to FITS. index is used in the filename (..._00000.fits). For series, pass index=k."""
        if self.header is None:
            raise RuntimeError("Cannot write FITS without header")

        write_fits_frame(
            self.data,
            self.header,
            frame_type=self.frame_type,
            channel_tag=self.channel_tag,
            ctx=ctx,
            index=index,
        )

    def write_png(self, ctx, star, show_stats=True, index=0):
        """Write this frame to PNG. index is used in the filename (..._00000.png). For series, pass index=k."""
        from utils.images import write_frame_png
        write_frame_png(
            self.data,
            self.header,
            frame_type=self.frame_type,
            channel_tag=self.channel_tag,
            ctx=ctx,
            star=star,
            show_stats=show_stats,
            index=index,
        )