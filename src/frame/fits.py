import logging
from astropy.io import fits
from loaders.run_waltzer_context import RunContext

def write_fits_frames(frames, headers, frame_type, channel_tag, ctx: RunContext):

    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write FITS: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, ctx.output_dir)

    star_name = str(ctx.target_name).replace(" ", "_")
    for k, (frame, header) in enumerate(zip(frames, headers)):
        fname = f"WALTzER_{star_name}_{channel_tag}_{frame_type}_{k:05d}.fits"
        filename = ctx.output_dir / fname
        header.append(("FILENAME", fname, "Output FITS filename"))
        fits.PrimaryHDU(data=frame, header=header).writeto(filename, overwrite=True)
        logging.debug("Wrote %s", filename)

    logging.info("Finished writing %d FITS file(s)", n_frames)