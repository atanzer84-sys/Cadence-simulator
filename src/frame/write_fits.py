import logging
from astropy.io import fits
from loaders.run_waltzer_context import RunContext

def write_fits_frames(frames, headers, frame_type, channel_tag, ctx: RunContext, start_index: int = 0):
    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write FITS: no frames for %s channel %s", frame_type, channel_tag)
        return

    star_name = str(ctx.target_name).replace(" ", "_")
    for k, (frame, header) in enumerate(zip(frames, headers), start=start_index):
        fname = f"WALTzER_{star_name}_{channel_tag}_{frame_type}_{k:05d}.fits"
        filename = ctx.output_dir / fname
        header.append(("FILENAME", fname, "Output FITS filename"))
        fits.PrimaryHDU(data=frame, header=header).writeto(filename, overwrite=True)

    logging.info("FITS writing finished: channel=%s frame_type=%s frames=%d", channel_tag, frame_type, n_frames)
