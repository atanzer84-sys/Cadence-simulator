import logging
import numpy as np
from domain.star import Star
from utils.constants import C_LIGHT_Angst, PARSEC_CM, R_V
from configs.global_config import get_global_config
from flux.cute_line_core_emission import apply_line_core_emission
from flux.cute_extinction import extinction_amores
from flux.cute_ism_abs_all import cute_ism_abs_all
from flux.cute_unred import unred
from astropy.coordinates import SkyCoord
from astropy import units as u
from loaders.run_waltzer_context import RunContext
from loaders.load_model_temperature import load_model_for_temperature, cut_model_wavelength_range
from utils.debug_dumps import dump_1d_array, dump_3d_array
from utils.helpers import announce
from utils.flux_image_array import plot_flux_and_photons_windows

def calculateFluxOnEarth(star: Star, ctx: RunContext, wl_min_A: float, wl_max_A: float, announce_user: bool = False):
    announce(f"Starting to calculate Flux on Earth for target star {star.name}", announce_user)
    cfg = get_global_config()

    model_data = load_model_for_temperature(star.effective_temperature, announce_user=announce_user)
    model_data = cut_model_wavelength_range(model_data, wl_min_A, wl_max_A)

    if cfg.write_intermediate_arrays:
        dump_3d_array(model_data, ctx.output_dir, star.name, "FluxCalc_1_model_input", perChannel=True, zoom=True)

    flux_lambda_original = convertStellarModelToFlux(model_data, star.radius_sun_cm)
    # keep undiluted flux
    flux_lambda_diluted = flux_lambda_original
    wavelengths = flux_lambda_original[:, 0]

    if cfg.write_intermediate_arrays:
        dump_3d_array(flux_lambda_original, ctx.output_dir, star.name, "FluxCalc_2_convertIntensityToLuminosity_snapshot", perChannel=True, zoom=True)

    if cfg.line_core_emission:
        # ACTUAL LCA EMISSION
        flux_lambda_diluted = apply_line_core_emission(flux_lambda_diluted, cfg.sigmaMg22, cfg.sigmaMg21, star.log_r, star.spectral_type, announce_user=announce_user)

        if cfg.write_intermediate_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_3_after_line_core_emission", perChannel=True, zoom=True)
    else:
        logging.info("Line Core Emission not applied!")

    # Starting extinction
    ebv, _ = compute_ebv_av(star.right_ascension, star.declination, star.distance_pc)

    if cfg.interstellar_absorption:
        # ACTUAL ISM_ABS CALL
        flux_lambda_diluted = apply_ism_absorption(flux_lambda_diluted, ebv, cfg, announce_user=announce_user)

        if cfg.write_intermediate_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_4_after_ISM", perChannel=True, zoom=True)
    else:
        logging.info("Interstellar Medium absorption not applied!")


    flux_di_before = flux_lambda_diluted[:, 1].copy()
    if cfg.write_intermediate_arrays:
        dump_1d_array(wavelengths, flux_di_before, ctx.output_dir, star.name, "FluxCalc_5_before_flux_at_earth", perChannel=True, zoom=True)

    # FINALLY FLUX ON EARTH
    flux_at_earth = compute_flux_at_earth(flux_lambda_diluted, star.distance_pc, announce_user=announce_user)

    if cfg.write_intermediate_arrays:
        dump_1d_array(wavelengths, flux_at_earth, ctx.output_dir, star.name, "FluxCalc_6_after_flux_at_earth", perChannel=True, zoom=True)

    # UNRED FLUX
    flux_unred = apply_unred(wavelengths, flux_at_earth, ebv, announce_user=announce_user)

    if cfg.write_intermediate_arrays:
        dump_1d_array(wavelengths, flux_unred, ctx.output_dir, star.name, "FluxCalc_7_after_unred", perChannel=True, zoom=True)
    if cfg.produce_flux_convolution_plots:
        plot_flux_and_photons_windows(wavelengths, flux_unred, ctx.output_dir, star, "FluxCalc_1_Flux", "Flux", "Flux [erg s⁻¹ cm⁻² Å⁻¹]")

    return flux_unred, wavelengths


def convertStellarModelToFlux(model_data, r_star):
    '''
    Legacy model flux:
    frequency-based stellar model quantity, converted to per-wavelength
    and integrated over stellar surface and solid angle.
    Resulting quantity is stellar spectral luminosity (erg/s/A),
    later converted to flux at Earth by geometric dilution.
    '''
    intensity_lambda = np.zeros(np.shape(model_data))
    flux_lambda  = np.zeros(np.shape(model_data))
    # we convert from frq to wavelength using lambda in Angstrom: F_lambda = F_nu * c / lambda^2
    # Unit before: erg/cm2/s/Hz, After:  ergs/cm2/s/A
    intensity_lambda[:,1] = (C_LIGHT_Angst * model_data[:,1])/(model_data[:,0]**2)    
    intensity_lambda[:,2] = (C_LIGHT_Angst * model_data[:,2])/(model_data[:,0]**2)

    # Integrate over stellar surface area (4*pi*R^2) and over solid angle (4*pi)
    # multiply with surface area of star -> ergs/cm2/s/A to ergs/s/A
    # then multiply with 4*!pi for steradian conversion
    flux_lambda[:,0]  = model_data[:,0]
    flux_lambda[:,1]  = intensity_lambda[:,1] * 4 * np.pi * (r_star**2) * 4 * np.pi
    flux_lambda[:,2]  = intensity_lambda[:,2] * 4 * np.pi * (r_star**2) * 4 * np.pi
    logging.info("Converting intensity to luminosity for r_star=%.6e cm with %d wavelength points", r_star, model_data.shape[0])

    return flux_lambda

def apply_ism_absorption(data, ebv, cfg, announce_user: bool = False):
    announce("Starting ISM absorption", announce_user)
    logging.info("ISM input: E(B-V)=%s", ebv)

    if cfg.mg2_col is None:
        nh = 5.8e21 * ebv  # The Mg2 column density is
        fractionMg2 = 0.825  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        # Safe runtime path: keep physically correct -inf for zero columns without triggering log10(0) warnings.
        mg2_col_linear = nh * fractionMg2 * 10.0**Mg_abn
        nmg2 = -np.inf if mg2_col_linear <= 0.0 else np.log10(mg2_col_linear)
        logging.info("ISM MgII column computed: nmg2=%s", nmg2)
    else:
        nmg2 = float(cfg.mg2_col)
        logging.info("ISM MgII column from cfg: nmg2=%s", nmg2)

    if cfg.mg1_col is None:
        nh = 5.8e21 * ebv  # The Mg1 column density is
        fractionMg1 = 0.00214  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        mg1_col_linear = nh * fractionMg1 * 10.0**Mg_abn
        nmg1 = -np.inf if mg1_col_linear <= 0.0 else np.log10(mg1_col_linear)
        logging.info("ISM MgI column computed: nmg1=%s", nmg1)
    else:
        nmg1 = float(cfg.mg1_col)
        logging.info("ISM MgI column from cfg: nmg1=%s", nmg1)

    if cfg.fe2_col is None:
        nh = 5.8e21 * ebv  # The Fe2 column density is
        fractionFe2 = 0.967  # (Frisch & Slavin 2003; this is the fraction of Fe in the ISM that is singly ionised)
        Fe_abn = -5.73  # (Frisch & Slavin 2003; this is the ISM abundance of Fe)
        fe2_col_linear = nh * fractionFe2 * 10.0**Fe_abn
        nfe2 = -np.inf if fe2_col_linear <= 0.0 else np.log10(fe2_col_linear)
        logging.info("ISM FeII column computed: nfe2=%s", nfe2)
    else:
        nfe2 = float(cfg.fe2_col)
        logging.info("ISM FeII column from cfg: nfe2=%s", nfe2)

    flux_data = cute_ism_abs_all(data, nmg2, nmg1, nfe2)
    logging.info("ISM absorption applied")

    return flux_data

def compute_ebv_av(right_ascension, declination, distance_pc):
    distance_kpc = distance_pc / 1000.0
    glon, glat = calculate_glon_glat(right_ascension, declination)
    ebv, av = extinction_amores(glon, glat, distance_kpc)
    logging.info("EBV=%s AV=%s (glon=%s glat=%s dist_kpc=%s)", ebv, av, glon, glat, distance_kpc)

    return ebv, av

def calculate_glon_glat(right_ascension, declination):
    c = SkyCoord(ra=right_ascension, dec=declination, unit=(u.degree, u.degree))
    glon = c.galactic.l.deg
    glat = c.galactic.b.deg
    return glon, glat

def compute_flux_at_earth(flux_lambda_diluted, distance_pc, announce_user: bool = False):
    announce("Starting Flux at Earth calculation", announce_user)
    flux_di = flux_lambda_diluted[:,1]
    flux_at_earth = flux_di / (4.0 * np.pi * (distance_pc * PARSEC_CM) ** 2)
    logging.info("Flux at Earth calculation finished")
    return flux_at_earth

def apply_unred(wavelengths, flux_at_earth, ebv, announce_user: bool = False):
    announce("Starting to apply UNRED extinction correction", announce_user)
    ebv = -1.0 * ebv
    flux_unred = unred(wavelengths, flux_at_earth, ebv=ebv, R_V=R_V)
    logging.info("UNRED extinction correction applied")
    return flux_unred

