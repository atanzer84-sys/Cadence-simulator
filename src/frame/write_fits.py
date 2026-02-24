import logging
from astropy.io import fits
from loaders.run_waltzer_context import RunContext


def write_fits_frame(frame, header, frame_type, channel_tag, ctx: RunContext, index=0):
    """
    Write a single frame to FITS. index is used in the filename (..._00000.fits).
    Works on raw numpy array and FITS header; knows nothing about the Frame class.
    """
    star_name = str(ctx.target_name).replace(" ", "_")
    fname = f"WALTzER_{star_name}_{channel_tag}_{frame_type}_{index:05d}.fits"
    filename = ctx.output_dir / fname

    logging.info("Writing %s %s frame for channel %s to %s", frame_type, index, channel_tag, ctx.output_dir)
    header = header.copy()
    header.append(("FILENAME", fname, "Output FITS filename"))
    fits.PrimaryHDU(data=frame, header=header).writeto(filename, overwrite=True)
    logging.debug("Wrote %s", filename)
