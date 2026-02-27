
import logging
from frame.frame_class import Frame
from frame.bias_frame import generate_bias_frame
from configs.channel_config import SpectroscopyChannel
from instrument.dark_image import generate_dark_image
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header

def generate_dark_frames(channel: SpectroscopyChannel, n_frames, base_header):

    logging.info("DARK: generating %d dark frames for %s (%d x %d), exptime_s=%g.", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, channel.exposure_s)
    print(f"Creating DARK Frames for channel {channel.channel_name}.")

    dark_frames = []

    for i in range(n_frames):
        header = append_base_frame_header(base_header, filetype="DARK", channel=channel, index0=i)
        dark = generate_dark_frame(channel, header)
        dark_frames.append(dark)

    return dark_frames

def generate_dark_frame(channel: SpectroscopyChannel, header=None):
    '''
    Dark = Bias + (dark_current * exptime)
    '''
    ccd_gain = channel.ccd_gain
    exptime_s = channel.exposure_s
    
    bias_frame = generate_bias_frame(channel, header=None)
    bias = bias_frame.data
    dark = generate_dark_image(channel)
    dark = bias + (dark * ccd_gain)

    logging.info("DARK STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, dark.mean(), dark.std(), dark.min(), dark.max())

    if header is not None:
        append_image_stats_header(header, dark)
        append_channel_frame_header(header, channel, exptime_s=exptime_s)

    return Frame(data=dark, header=header, frame_type="dark", channel_tag=channel.channel_name)
