
import logging
from frame.frame_class import Frame
from configs.channel_config import Channel
from instrument.flat_image import generate_flat_image
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header


def generate_flat_frame_with_index(channel: Channel, index0: int, base_header):
    header = append_base_frame_header(base_header, filetype="FLAT", channel=channel, index0=index0)
    return generate_flat_frame(channel, header)
    
def generate_flat_frame(channel: Channel, header=None):
    ccd_gain = channel.ccd_gain

    flat = generate_flat_image(channel) * ccd_gain

    if header is not None:
        append_image_stats_header(header, flat)
        append_channel_frame_header(header, channel, exptime_s=0.0, include_dark=False)

    logging.info("FLAT STATS %s mean=%g std=%g min=%g max=%g", channel.channel_name, flat.mean(), flat.std(), flat.min(), flat.max())
    return Frame(data=flat, header=header, frame_type="flat", channel_tag=channel.channel_name)

