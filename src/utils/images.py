import logging
import matplotlib.pyplot as plt
from astropy.io import fits


def write_frames_png(frames, frame_type, channel_tag, output_dir):
    
    n_frames = len(frames)
    if not frames:
        logging.info("Write PNG: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, output_dir)
    for k, frame in enumerate(frames):
        filename = output_dir / f"{frame_type}_{channel_tag}_{k:05d}.png"
        plt.imsave(filename, frame, cmap="gray")
        logging.debug("Wrote %s", filename)
    logging.info("Finished writing %d PNG(s)", n_frames)


def write_frames_fits(frames, headers, frame_type, channel_tag, output_dir):

    n_frames = len(frames)
    if n_frames == 0:
        logging.info("Write FITS: no frames for %s channel %s", frame_type, channel_tag)
        return

    logging.info("Writing %d %s frame(s) for channel %s to %s", n_frames, frame_type, channel_tag, output_dir)

    for k, (frame, header) in enumerate(zip(frames, headers)):
        filename = output_dir / f"WALTzER_{channel_tag}_{frame_type}_{k:05d}.fits"
        fits.PrimaryHDU(data=frame, header=header).writeto(filename, overwrite=True)
        logging.debug("Wrote %s", filename)

    logging.info("Finished writing %d FITS file(s)", n_frames)
