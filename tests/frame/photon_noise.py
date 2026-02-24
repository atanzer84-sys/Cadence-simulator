import numpy as np
from frame.frame_class import Frame
from configs.channel_config import SpectroscopyChannel

def generate_photon_noise_frame(image, channel: SpectroscopyChannel, header):
    """
    Apply Gaussian photon noise approximation: photon_noise = image + N(0,1) * sqrt(image)

    Parameters
    ----------
    image : 2D numpy array
        Signal image before noise (same as IDL im_jls)
    channel : SpectroscopyChannel
        Channel information (for tag)

    Returns
    -------
    Frame
        Photon noise frame (no header)
    """

    # Gaussian random array same shape as image
    distribution = np.random.normal(loc=0.0, scale=1.0, size=image.shape)

    # sqrt of signal (protect against negative values)
    sqrt_signal = np.sqrt(np.clip(image, 0, None))

    # IDL equivalent: im_n = im_jls + distr * sqrt(im_jls)
    photon_noise = image + distribution * sqrt_signal

    header.append(("MEAN",   float(photon_noise.mean()),     "Mean value of the frame"))
    header.append(("MEDIAN", float(np.median(photon_noise)), "Median value of the frame"))
    header.append(("STDDEV", float(photon_noise.std()),      "Standard deviation of the frame"))
    header.append(("MAX",    float(photon_noise.max()),      "Maximum value of the frame"))
    header.append(("MIN",    float(photon_noise.min()),      "Minimum value of the frame"))

    return Frame(data=photon_noise, header=header, frame_type="photon_noise", channel_tag=channel.channel_name)