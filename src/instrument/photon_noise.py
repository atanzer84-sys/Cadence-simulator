import numpy as np

def apply_photon_noise_gauss_from_spectra2d(spectra_2d):

    distr = np.random.normal(loc=0.0, scale=1.0, size=spectra_2d.shape)
    noisy = spectra_2d + distr * np.sqrt(spectra_2d)

    return noisy