import logging
import numpy as np
from pathlib import Path
from loaders.run_setup import get_repo_root
from domain.star import Star
from domain.constants import C_LIGHT_ROUNDED_m_s, PARSEC_CM
from configs.global_config import get_global_config
from flux.line_core_emission import apply_line_core_emission

def calculateFluxOnEarth(star: Star, output_dir):
    print("Starting to calculate Flux on Earth")
    cfg = get_global_config()

    model_data = load_model_for_temperature(star.effective_temperature)

    if cfg.test_mode:
        logging.info("test_mode=1 -> dumping model data + flux snapshots (legacy debug mode)")
        dump_array(
            model_data,
            output_dir,
            filename=f"{star.name}_model_input.txt"
    )

    flux_lambda_original = convertIntensityToFlux(model_data, star.radius_sun_cm)
    # keep undiluted flux
    flux_lambda_diluted = flux_lambda_original
    wavelengths = flux_lambda_original[:,0]

    if cfg.test_mode:
        logging.info("test_mode=1 -> dumping flux snapshots (convertIntensityToLuminosity)")
        dump_array(
            flux_lambda_original,
            output_dir,
            filename=f"{star.name}_convertIntensityToLuminosity_snapshot.txt"
        )

    if cfg.line_core_emission:
        flux_lambda_diluted = apply_line_core_emission(flux_lambda_diluted,cfg.sigmaMg22, 
                                        cfg.sigmaMg21, star.log_r, star.spectral_type)

    if cfg.test_mode:
        logging.info("test_mode=1 -> dumping flux snapshots (apply_line_core_emission_snapshot)")
        dump_array(
            flux_lambda_diluted,
            output_dir,
            filename=f"{star.name}_apply_line_core_emission_snapshot.txt",
            cut = False
    )

    # if cfg.add_ism_abs:
    #     flux_lambda_diluted = apply_ism_abs(...)

    # if cfg.apply_extinction:
    #     flux_lambda_diluted = apply_extinction(...)



    flux = flux_lambda_original[:,1]
    flux_at_earth    = flux/(4.*np.pi*(star.distance_pc*(PARSEC_CM))**2)


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

def convertIntensityToFlux(model_data, r_star):
    '''
    Legacy model flux:
    frequency-based stellar model quantity, converted to per-wavelength
    and integrated over stellar surface and solid angle.
    Resulting quantity is stellar spectral luminosity (erg/s/A),
    later converted to flux at Earth by geometric dilution.
    '''
    intensity_lambda        = np.zeros(np.shape(model_data))
    flux_lambda  = np.zeros(np.shape(model_data))
    # we convert from frq to wavelength using lambda in Angstrom: F_lambda = F_nu * c / lambda^2
    # Unit before: erg/cm2/s/Hz, After:  ergs/cm2/s/A
    intensity_lambda[:,1] = (C_LIGHT_ROUNDED_m_s * model_data[:,1])/(model_data[:,0]**2)    
    intensity_lambda[:,2] = (C_LIGHT_ROUNDED_m_s * model_data[:,2])/(model_data[:,0]**2)

    # Integrate over stellar surface area (4*pi*R^2) and over solid angle (4*pi)
    # multiply with surface area of star -> ergs/cm2/s/A to ergs/s/A
    # then multiply with 4*!pi for steradian conversion
    flux_lambda[:,0]  = model_data[:,0]
    flux_lambda[:,1]  = intensity_lambda[:,1] * 4 * np.pi * (r_star**2) * 4 * np.pi
    flux_lambda[:,2]  = intensity_lambda[:,2] * 4 * np.pi * (r_star**2) * 4 * np.pi
    logging.info(
        "Converting intensity to luminosity for r_star=%.6e cm with %d wavelength points",
        r_star,
        model_data.shape[0]
    )

    return flux_lambda

def dump_array(array, output_dir, filename, cut=True, fmt="%.18e"):
    """
    Dump a spectrum array to disk.

    If cut=True (default), the array is cut to the NUV, VIS, IR wavelength windows.
    If cut=False, the full array is dumped unchanged.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cut:
        wl = array[:, 0]

        mask_nuv = (wl >= 2600) & (wl <= 2900)
        mask_vis = (wl >= 5600) & (wl <= 5800)
        mask_ir  = (wl >= 10000) & (wl <= 10200)

        out_array = np.vstack((
            array[mask_nuv],
            array[mask_vis],
            array[mask_ir],
        ))
    else:
        out_array = array

    np.savetxt(output_dir / filename, out_array, fmt=fmt)

    return out_array