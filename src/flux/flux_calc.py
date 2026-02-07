import logging
import numpy as np
from loaders.run_setup import get_repo_root
from domain.star import Star
from utils.constants import C_LIGHT_ROUNDED_m_s
from configs.global_config import get_global_config
from flux.cute_line_core_emission import apply_line_core_emission
from flux.cute_extinction import extinction_amores
from flux.cute_ism_abs_all import cute_ism_abs_all
from utils.debug_dumps import dump_spectrum_snapshots, dump_diff_windows_3d
# , dump_spectrum_snapshots_1d, dump_diff_windows_1d
from astropy.coordinates import SkyCoord
from astropy import units as u

def calculateFluxOnEarth(star: Star, output_dir):
    print("Starting to calculate Flux on Earth")
    cfg = get_global_config()

    model_data = load_model_for_temperature(star.effective_temperature)
    flux_lambda_original = convertIntensityToFlux(model_data, star.radius_sun_cm)
    # keep undiluted flux
    flux_lambda_diluted = flux_lambda_original.copy()
    # wavelengths = flux_lambda_original[:,0]

    if cfg.test_mode:
        logging.info("test_mode=1 -> dumping model data (legacy debug mode)")
        dump_spectrum_snapshots(model_data, output_dir, star.name, "model_input")
        logging.info("test_mode=1 -> dumping flux snapshots (convertIntensityToLuminosity_snapshot)")
        dump_spectrum_snapshots(flux_lambda_original, output_dir, star.name, "convertIntensityToLuminosity_snapshot")
        logging.info("test_mode=1 -> dumping flux diffs (after_convertIntensityToLuminosity)")
        dump_diff_windows_3d(model_data, flux_lambda_original, output_dir, star.name, tag="after_convertIntensityToLuminosity")


    if cfg.line_core_emission:
        if cfg.test_mode:
            spectrum_before_lce = flux_lambda_diluted.copy()
            logging.info("test_mode=1 -> dumping flux snapshots (lca_original)")
            dump_spectrum_snapshots(spectrum_before_lce, output_dir, star.name, "lca_original")

        # ACTUAL LCA EMISSION 
        flux_lambda_diluted = apply_line_core_emission(flux_lambda_diluted, cfg.sigmaMg22, cfg.sigmaMg21, star.log_r, star.spectral_type)

        if cfg.test_mode:
            logging.info("test_mode=1 -> dumping flux snapshots (after_line_core_emission)")
            dump_spectrum_snapshots(flux_lambda_diluted, output_dir, star.name, "after_line_core_emission")
            logging.info("test_mode=1 -> dumping flux diffs (after_line_core_emission)")
            dump_diff_windows_3d(spectrum_before_lce, flux_lambda_diluted, output_dir, star.name, tag="after_line_core_emission")
    else:
        logging.info("Line Core Emission not applied!")


    # Starting extinction
    distance_kpc         = star.distance_pc/1000.0

    #get extinction based on coordinates and distance
    c      = SkyCoord(ra=star.right_ascension, dec=star.declination, unit=(u.degree, u.degree))
    # print("c: ", c)
    glon   = c.galactic.l.deg
    glat   = c.galactic.b.deg
    ebv,av = extinction_amores(glon,glat,distance_kpc) 

    if cfg.add_ism_abs:
        if cfg.test_mode:
            spectrum_before_ism = flux_lambda_diluted.copy()
            logging.info("test_mode=1 -> dumping flux snapshots (ism_snapshot)")
            dump_spectrum_snapshots(spectrum_before_ism, output_dir, star.name, "ism_snapshot")

        # ACTUAL ISM CALL
        flux_lambda_diluted = apply_ism_absorption(flux_lambda_diluted, ebv, cfg)

        if cfg.test_mode:
            logging.info("test_mode=1 -> dumping flux snapshots (after_ISM)")
            dump_spectrum_snapshots(flux_lambda_diluted, output_dir, star.name, "after_ISM")

            logging.info("test_mode=1 -> dumping flux diffs (after_ISM)")
            dump_diff_windows_3d(spectrum_before_ism, flux_lambda_diluted, output_dir, star.name, tag="after_ISM")
    else:
        logging.info("Interstellar Medium absorption not applied!")


    # flux = flux_lambda_original[:,1]
    # flux_at_earth    = flux/(4.*np.pi*(star.distance_pc*(PARSEC_CM))**2)


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

def apply_ism_absorption(data, ebv, cfg):
    print("Starting ISM absorption")
    logging.info("Starting ISM absorption")
    logging.info("ISM input: E(B-V)=%s", ebv)

    if cfg.mg2_col is None:
        nh = 5.8e21 * ebv  # The Mg2 column density is
        fractionMg2 = 0.825  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        nmg2 = np.log10(nh * fractionMg2 * 10.0**Mg_abn)
        logging.info("ISM MgII column computed: nmg2=%s", nmg2)
    else:
        nmg2 = float(cfg.mg2_col)
        logging.info("ISM MgII column from cfg: nmg2=%s", nmg2)

    if cfg.mg1_col is None:
        nh = 5.8e21 * ebv  # The Mg1 column density is
        fractionMg1 = 0.00214  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        nmg1 = np.log10(nh * fractionMg1 * 10.0**Mg_abn)
        logging.info("ISM MgI column computed: nmg1=%s", nmg1)
    else:
        nmg1 = float(cfg.mg1_col)
        logging.info("ISM MgI column from cfg: nmg1=%s", nmg1)

    if cfg.fe2_col is None:
        nh = 5.8e21 * ebv  # The Fe2 column density is
        fractionFe2 = 0.967  # (Frisch & Slavin 2003; this is the fraction of Fe in the ISM that is singly ionised)
        Fe_abn = -5.73  # (Frisch & Slavin 2003; this is the ISM abundance of Fe)
        nfe2 = np.log10(nh * fractionFe2 * 10.0**Fe_abn)
        logging.info("ISM FeII column computed: nfe2=%s", nfe2)
    else:
        nfe2 = float(cfg.fe2_col)
        logging.info("ISM FeII column from cfg: nfe2=%s", nfe2)

    flux_data = cute_ism_abs_all(data, nmg2, nmg1, nfe2)
    logging.info("ISM absorption applied")

    return flux_data
