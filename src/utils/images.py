import logging
import matplotlib.pyplot as plt
import numpy as np


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

