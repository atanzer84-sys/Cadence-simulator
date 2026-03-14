import logging
from configs.channel_config import Channel
from frame.frame_class import Frame
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header


def generate_science_frame(images, channel: Channel, base_header):
    """Generate science frame(s). Accepts single image or list of images. Returns list of frames."""
    
    if not isinstance(images, list):
        images = [images]
    exposure_time_s = channel.exposure_s
    
    logging.info("SCIENCE: generating %d science frame(s) for %s (%d x %d), exptime_s=%g.", len(images), channel.channel_name, channel.x_pixels, channel.y_pixels, exposure_time_s)
    print(f"Creating SCIENCE Frames for channel {channel.channel_name} ({len(images)} frame(s)).")

    frames = []
    for i, image in enumerate(images):
        header = append_base_frame_header(base_header, filetype="SCIENCE", channel=channel, index0=i)
        append_image_stats_header(header, image)
        append_channel_frame_header(header, channel, exptime_s=exposure_time_s, include_bias=True, include_dark=True)
        frame = Frame(data=image, header=header, frame_type="science", channel_tag=channel.channel_name)
        frames.append(frame)
    return frames

