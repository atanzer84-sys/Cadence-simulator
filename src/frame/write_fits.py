import logging
from astropy.io import fits
from loaders.run_waltzer_context import RunContext
from frame.frame_class import Frame

def write_fits_frame(frame: Frame, ctx: RunContext, index: int, exposure: float):
    star_name = str(ctx.target_name).replace(" ", "_")
    fname = f"WALTzER_{star_name}_{frame.channel_tag}_{frame.frame_type}_{index:05d}.fits"
    filename = ctx.output_dir / fname
    frame.header.append(("FILENAME", fname, "Output FITS filename"))
    fits.PrimaryHDU(data=frame.data, header=frame.header).writeto(filename, overwrite=True)
    logging.info("FITS writing finished: channel=%s frame_type=%s frame=%d", frame.channel_tag, frame.frame_type, index)