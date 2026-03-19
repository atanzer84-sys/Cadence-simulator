
import logging
from frame.frame_class import Frame
from configs.channel_config import Channel
from instrument.bias_image import generate_bias_image
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header


def generate_bias_frame_with_index(channel: Channel, index0: int, base_header):
    header = append_base_frame_header(base_header, filetype="BIAS", channel=channel, index0=index0)
    return generate_bias_frame(channel, header)
    
def generate_bias_frame(channel: Channel, header=None):
    '''
    Bias = Offset (bias_offset) + Gaussian Noise (read_noise)
    '''
    ccd_gain = channel.ccd_gain

    bias = generate_bias_image(channel) * ccd_gain


    if header is not None:
        append_image_stats_header(header, bias)
        append_channel_frame_header(header, channel, exptime_s=0.0, include_bias=True, include_dark=False)

    logging.info("BIAS STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, bias.mean(), bias.std(), bias.min(), bias.max())
    return Frame(data=bias, header=header, frame_type="bias", channel_tag=channel.channel_name)

