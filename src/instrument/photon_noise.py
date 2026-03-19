import numpy as np

_rng = np.random.default_rng()

def generate_photon_noise_from_spectra2d(spectra_2d_exposure):
    distr = _rng.normal(loc=0.0, scale=1.0, size=spectra_2d_exposure.shape).astype(np.float32)
    sigma = np.sqrt(np.clip(spectra_2d_exposure, 0, None))
    noise = distr * sigma
    return noise