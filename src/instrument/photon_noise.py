import numpy as np

_rng = np.random.default_rng()

def generate_photon_noise_from_spectra2d(stellar_component_2d):
    distr = _rng.normal(loc=0.0, scale=1.0, size=stellar_component_2d.shape).astype(np.float32)
    sigma = np.sqrt(np.clip(stellar_component_2d, 0, None))
    noise = distr * sigma
    return noise