import logging
from configs.channel_config import Channel
from frame.frame_class import Frame
from frame.fits_header import append_image_stats_header, append_channel_frame_header, append_base_frame_header


def generate_science_frames(image, channel: Channel, n_frames, base_header):
    exposure_time_s = channel.exposure_s
    logging.info("SCIENCE: generating %d science frames for %s (%d x %d), exptime_s=%g.", n_frames, channel.channel_name, channel.x_pixels, channel.y_pixels, exposure_time_s)
    print(f"Creating SCIENCE Frames for channel {channel.channel_name}.")

    frames = []

    for i in range(n_frames):
        header = append_base_frame_header(base_header, filetype="SCIENCE", channel=channel, index0=i)
        frame = generate_science_frame(image, channel, header)
        frames.append(frame)

    return frames


def generate_science_frame(image, channel: Channel, header):

    science = image
    append_image_stats_header(header, science)
    append_channel_frame_header(header, channel, exptime_s=channel.exposure_s, include_bias=True, include_dark=True)

    return Frame(data=science, header=header, frame_type="science", channel_tag=channel.channel_name)

