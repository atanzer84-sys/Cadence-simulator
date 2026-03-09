import numpy as np
from configs.channel_config import Channel
from loaders.run_waltzer_context import RunContext
from domain.star import Star


def apply_photon_noise_gauss_from_spectra2d(spectra_2d_exposure, channel: Channel, ctx: RunContext, star: Star):
    distr = np.random.normal(loc=0.0, scale=1.0, size=spectra_2d_exposure.shape).astype(np.float32)
    sigma = np.sqrt(np.clip(spectra_2d_exposure, 0, None))
    noise = distr * sigma
    return noise