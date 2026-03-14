
import logging
from frame.frame_class import Frame
from frame.bias_frame import generate_bias_frame
from configs.channel_config import Channel
from instrument.dark_image import generate_dark_image
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header

def generate_dark_frame(channel: Channel, header=None):
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


def generate_dark_frame_with_index(channel: Channel, index0: int, base_header):
    header = append_base_frame_header(base_header, filetype="DARK", channel=channel, index0=index0)
    return generate_dark_frame(channel, header)