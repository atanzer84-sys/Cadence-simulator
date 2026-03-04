
import logging
from frame.frame_class import Frame
from configs.channel_config import Channel
from instrument.bias_image import generate_bias_image
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header


def generate_bias_frames(channel: Channel, n_frames, base_header):

    logging.info("BIAS: generating %d bias frames for %s (%d x %d).", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels)
    print(f"Creating BIAS Frames for channel {channel.channel_name}.")

    bias_frames = []

    for i in range(n_frames):
        header = append_base_frame_header(base_header, filetype="BIAS", channel=channel, index0=i)
        bias = generate_bias_frame(channel, header)
        bias_frames.append(bias)

    return bias_frames

def generate_bias_frame(channel: Channel, header=None):
    '''
    Bias = Offset (bias_offset) + Gaussian Noise (read_noise)
    '''
    ccd_gain = channel.ccd_gain

    bias = generate_bias_image(channel) * ccd_gain

    logging.info("BIAS STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, bias.mean(), bias.std(), bias.min(), bias.max())

    if header is not None:
        append_image_stats_header(header, bias)
        append_channel_frame_header(header, channel, exptime_s=0.0, include_bias=True, include_dark=False)

    return Frame(data=bias, header=header, frame_type="bias", channel_tag=channel.channel_name)