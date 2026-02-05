import logging
import numpy as np
import astropy.units as u
from pathlib import Path
from loaders.run_setup import get_repo_root
from domain.star import Star
from domain.constants import C_LIGHT

def calculateFluxOnEarth(star: Star, output_dir):
    print("Starting to calculate Flux on Earth")
    model_data = load_model_for_temperature(star.effective_temperature)
    print(np.shape(model_data))
    luminosity_lambda = convertIntensityToLuminosity(model_data, star.radius_sun_cm)
    dump_spectrum_txt(luminosity_lambda, output_dir, filename="convertIntensityToLuminosity_TIC393818343.txt")


def load_model_for_temperature(t_star):
    """
    Load stellar model spectrum for given effective temperature.
    Mirrors legacy selection logic exactly.
    """

    repo_root = get_repo_root()
    models_dir = repo_root / "data" / "models"

    t_rounded = int(round(t_star, -2))
    subdir = f"t{t_rounded:05d}g4.4"
    model_file = models_dir / subdir / "model.flx"

    if model_file.is_file():
        model_data = np.loadtxt(model_file)
        logging.info(
            "Loaded stellar model %s for Teff=%s K",
            model_file.relative_to(models_dir),
            t_star,
        )
        return model_data


    # legacy fallback: +100 K
    t_rounded_fallback = t_rounded + 100
    subdir_fb = f"t{t_rounded_fallback:05d}g4.4"
    model_file_fb = models_dir / subdir_fb / "model.flx"

    if model_file_fb.is_file():
        model_data = np.loadtxt(model_file_fb)
        logging.info(
            "Loaded stellar model %s for Teff=%s K",
            model_file_fb.relative_to(models_dir),
            t_star,
        )
        return model_data
    raise FileNotFoundError(
        f"No model.flx found for T={t_star} K "
        f"(tried {subdir} and {subdir_fb})"
    )

def convertIntensityToLuminosity(model_data, r_star):
    '''
    Legacy model flux:
    frequency-based stellar model quantity, converted to per-wavelength
    and integrated over stellar surface and solid angle.
    Resulting quantity is stellar spectral luminosity (erg/s/A),
    later converted to flux at Earth by geometric dilution.
    '''
    intensity_lambda        = np.zeros(np.shape(model_data))
    luminosity_lambda  = np.zeros(np.shape(model_data))
    print("c: ", C_LIGHT)
    print("r_star: ", r_star)
    # we convert from frq to wavelength using lambda in Angstrom: F_lambda = F_nu * c / lambda^2
    # Unit before: erg/cm2/s/Hz, After:  ergs/cm2/s/A
    intensity_lambda[:,1] = (C_LIGHT * model_data[:,1])/(model_data[:,0]**2)    
    intensity_lambda[:,2] = (C_LIGHT * model_data[:,2])/(model_data[:,0]**2)

    # Integrate over stellar surface area (4*pi*R^2) and over solid angle (4*pi)
    # multiply with surface area of star -> ergs/cm2/s/A to ergs/s/A
    # then multiply with 4*!pi for steradian conversion
    luminosity_lambda[:,0]  = model_data[:,0]
    luminosity_lambda[:,1]  = intensity_lambda[:,1] * 4 * np.pi * (r_star**2) * 4 * np.pi
    luminosity_lambda[:,2]  = intensity_lambda[:,2] * 4 * np.pi * (r_star**2) * 4 * np.pi

    return luminosity_lambda

def dump_spectrum_txt(spectrum, output_dir, filename):
    """
    Writes a sampled version of the spectrum focusing on specific 
    wavelength windows: NUV, Visible, and IR.
    """
    outpath = Path(output_dir) / filename
    data = np.asarray(spectrum)
    
    # The first column (index 0) is Wavelength in Angstroms
    wl = data[:, 0]

    # Define our three masks (Boolean filters)
    mask_nuv = (wl >= 3500) & (wl <= 3700)
    mask_vis = (wl >= 4500) & (wl <= 4700)
    mask_ir  = (wl >= 7800) & (wl <= 8000)

    # Combine the data chunks
    # We use data[mask] to grab only the rows where the wavelength matches
    sampled_data = np.vstack((
        data[mask_nuv],
        data[mask_vis],
        data[mask_ir]
    ))

    # Safety check: if the file didn't cover these ranges, 
    # sampled_data might be smaller than expected.
    if sampled_data.size == 0:
        print("Warning: No data found in the specified wavelength ranges!")
        return

    # Save to file
    np.savetxt(
        outpath,
        sampled_data,
        fmt=["%.4f", "%.6E", "%.6E"],
    )
    
    print(f"Saved sampled NUV, VIS, and IR windows to {filename}")