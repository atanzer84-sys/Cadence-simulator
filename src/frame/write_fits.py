import logging
from astropy.io import fits
from loaders.run_waltzer_context import RunContext
from frame.frame_class import Frame
from utils.images_common import build_frame_path

def write_fits_frame(frame: Frame, ctx: RunContext, index: int, exposure: float):
    filename = build_frame_path(ctx.output_dir, ctx.target_name, frame.channel_tag, frame.frame_type, exposure, index, suffix=".fits")
    fname = filename.name
    frame.header.append(("FILENAME", fname, "Output FITS filename"))
    # _log_oversized_header_cards(frame.header, frame.data.shape)

    fits.PrimaryHDU(data=frame.data, header=frame.header).writeto(filename, overwrite=True)
    logging.info("FITS writing finished: channel=%s frame_type=%s frame=%d", frame.channel_tag, frame.frame_type, index)



# def _log_oversized_header_cards(header, data_shape):
#     """Print any FITS header cards whose raw image is at the 80-char limit, plus image size."""
#     if header is None:
#         return
#     ny, nx = data_shape if data_shape is not None and len(data_shape) == 2 else ("?", "?")
#     for card in header.cards:
#         raw = card.image
#         # astropy already truncates to 80 chars, so a card that *triggered*
#         # the warning will typically have len(raw) == 80 here.
#         if raw is not None and len(raw) >= 80:
#             print(f"[FITS header too long] key={card.keyword!r} len={len(raw)} size={nx}x{ny}: {raw}")

