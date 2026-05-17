import logging
import numpy as np
from domain.star import Star
from utils.constants import C_LIGHT_Angst, PARSEC_CM, R_V, MgII1w, C_LIGHT_km_s, MgII1w, vr_ISM, MgII2w, MgIw, FeIIw, PHOTON_ENERGY_CONVERSION_A
from configs.global_config import get_global_config, GlobalConfig
from flux.cute_line_core_emission import apply_line_core_emission
from flux.cute_extinction import extinction_amores
from flux.cute_ism_abs_all import cute_ism_abs_all
from flux.cute_unred import unred
from astropy.coordinates import SkyCoord
from astropy import units as u
from loaders.run_waltzer_context import RunContext
from loaders.load_model_temperature import load_model_for_temperature
from utils.debug_dumps import dump_1d_array, dump_3d_array
from utils.helpers import announce
from utils.flux_image_array import plot_flux_and_photons_windows, plot_model_input, plot_lce_comparison_four_panel, plot_two_line_comparison_four_panel


def convert_flux_to_photons(flux_unred, wavelengths):
    photon_flux = flux_unred * PHOTON_ENERGY_CONVERSION_A * wavelengths  # from ergs/s/cm2/A to photons/s/cm2/A
    return photon_flux


def calculate_flux_on_earth(star: Star, ctx: RunContext, wl_min_A: float, wl_max_A: float, announce_user: bool = False, background_star: bool = False):
    announce(f"Starting to calculate Flux on Earth for target star {star.name}", announce_user)
    cfg = get_global_config()
    dump_arrays = cfg.write_intermediate_arrays and not background_star
    dump_plots = cfg.produce_flux_convolution_plots and not background_star

    model_data = load_model_for_temperature(star.effective_temperature, wl_min_A, wl_max_A, announce_user=announce_user)

    if dump_arrays:
        dump_3d_array(model_data, ctx.output_dir, star.name, "FluxCalc_1_model_input", perChannel=True, zoom=True)
    if not background_star and dump_plots:
        plot_model_input(model_data, wl_min_A, wl_max_A, ctx.output_dir, star, filename_tag="FluxCalc_1_model_input", title_text="Model Input Spectrum")
        plot_flux_and_photons_windows(model_data[:, 0], model_data[:, 1], ctx.output_dir, star, "FluxCalc_1_model_input", "Model Input Spectrum (linear windows)", "Model flux [erg s⁻¹ cm⁻² Hz⁻¹]")

    flux_lambda_original = convert_stellar_model_to_flux(model_data, star.radius_sun_cm)
    if not background_star and dump_plots:
        plot_model_input(flux_lambda_original, wl_min_A, wl_max_A, ctx.output_dir, star, filename_tag="FluxCalc_2_convert_stellar_model_to_flux", title_text="Flux After Stellar Model Conversion")

    # keep undiluted flux
    flux_lambda_diluted = flux_lambda_original.copy()
    wavelengths = flux_lambda_original[:, 0]

    if dump_arrays:
        dump_3d_array(flux_lambda_original, ctx.output_dir, star.name, "FluxCalc_2_convertIntensityToLuminosity_snapshot", perChannel=True, zoom=True)

    if cfg.line_core_emission:
        # ACTUAL LCA EMISSION
        flux_lambda_diluted = apply_line_core_emission(flux_lambda_diluted, cfg.sigmaMg22, cfg.sigmaMg21, star.log_r, star.spectral_type, announce_user=announce_user)

        if dump_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_3_after_line_core_emission", perChannel=True, zoom=True)
        if not background_star and dump_plots:
            # plot_lce_comparison_four_panel(flux_lambda_original[:, 0], flux_lambda_original[:, 1], flux_lambda_diluted[:, 1], ctx.output_dir, star, 2794.5, 2796.0, 2802.0, 2803.3, filename_tag="FluxCalc_3_LCE_comparison_4panel", title_text="Mg II Line Core Emission")
            plot_lce_comparison_four_panel(flux_lambda_original[:,0], flux_lambda_original[:,1], flux_lambda_diluted[:,1], ctx.output_dir, star, 2794.5, 2796.5, 2801.70, 2803.70, "FluxCalc_3_LCE_comparison_4panel", "Mg II Line Core Emission", cfg.sigmaMg21, cfg.sigmaMg22)
            # MgII1w      = 2795.5280 #MgIIk wavelength 
            # MgII2w      = 2802.7050 #MgIIh wavelength


    else:
        logging.info("Line Core Emission not applied!")

    # Starting extinction
    ebv, _ = compute_ebv_av(star.right_ascension, star.declination, star.distance_pc)

    if cfg.interstellar_absorption:
        # ACTUAL ISM_ABS CALL
        flux_before_ism = flux_lambda_diluted.copy()
        flux_lambda_diluted = apply_ism_absorption(flux_lambda_diluted, ebv, cfg, announce_user=announce_user)

        if dump_arrays:
            dump_3d_array(flux_lambda_diluted, ctx.output_dir, star.name, "FluxCalc_4_after_ISM", perChannel=True, zoom=True)
        if not background_star and dump_plots:
            mgii1_ism_w = MgII1w + MgII1w * vr_ISM / C_LIGHT_km_s
            mgii2_ism_w = MgII2w + MgII2w * vr_ISM / C_LIGHT_km_s
            mgi_ism_w   = MgIw   + MgIw   * vr_ISM / C_LIGHT_km_s
            feii_ism_w  = FeIIw  + FeIIw  * vr_ISM / C_LIGHT_km_s
            # plot_model_input(flux_lambda_original, 2700, 2900, ctx.output_dir, star, filename_tag="FluxCalc_4_FluxPLot", title_text="ISM Transmission")
            plot_two_line_comparison_four_panel(flux_before_ism[:,0], flux_before_ism[:,1], flux_lambda_diluted[:,1], ctx.output_dir, star, 2794.5, 2796.0, mgii1_ism_w, "Mg II k", 2802.0, 2803.3, mgii2_ism_w, "Mg II h", filename_tag="FluxCalc_4_ISM_comparison_4panel_MGII", title_text="Mg II Interstellar Medium Absorption")
            plot_two_line_comparison_four_panel(flux_before_ism[:,0], flux_before_ism[:,1], flux_lambda_diluted[:,1], ctx.output_dir, star, 2794.5, 2796.0, mgii1_ism_w, "Mg II k", 2802.0, 2803.3, mgii2_ism_w, "Mg II h", filename_tag="FluxCalc_4_ISM_comparison_4panel_MGIFE", title_text="Mg II Interstellar Medium Absorption")
            plot_two_line_comparison_four_panel(flux_before_ism[:,0], flux_before_ism[:,1], flux_lambda_diluted[:,1], ctx.output_dir, star, 2598.5, 2600.3, feii_ism_w, "Fe II", 2851.5, 2852.8, mgi_ism_w, "Mg I", filename_tag="FluxCalc_4_ISM_MgI_FeII_4panel", title_text="Mg I and Fe II Interstellar Medium Absorption")

    else:
        logging.info("Interstellar Medium absorption not applied!")


    flux_di_before = flux_lambda_diluted[:, 1].copy()
    if dump_arrays:
        dump_1d_array(wavelengths, flux_di_before, ctx.output_dir, star.name, "FluxCalc_5_before_flux_at_earth", perChannel=True, zoom=True)

    # FINALLY FLUX ON EARTH
    flux_at_earth = compute_flux_at_earth(flux_lambda_diluted, star.distance_pc, announce_user=announce_user)

    if dump_arrays:
        dump_1d_array(wavelengths, flux_at_earth, ctx.output_dir, star.name, "FluxCalc_6_after_flux_at_earth", perChannel=True, zoom=True)

    # UNRED FLUX
    flux_unred = apply_unred(wavelengths, flux_at_earth, ebv, announce_user=announce_user)

    if dump_arrays:
        dump_1d_array(wavelengths, flux_unred, ctx.output_dir, star.name, "FluxCalc_7_after_unred", perChannel=True, zoom=True)
    if dump_plots:
        plot_flux_and_photons_windows(wavelengths, flux_unred, ctx.output_dir, star, "FluxCalc_1_Flux", "Flux", "Flux [erg s⁻¹ cm⁻² Å⁻¹]")

    photon_flux = convert_flux_to_photons(flux_unred, wavelengths)
    photon_flux = np.asarray(photon_flux, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)

    if dump_arrays:
        dump_1d_array(wavelengths, photon_flux, ctx.output_dir, star.name, "FluxCalc_8_photons_star", perChannel=True, zoom=True)
    if dump_plots:
        plot_flux_and_photons_windows(wavelengths, photon_flux, ctx.output_dir, star, "FluxCalc_photons", "Photon Flux", "Photon flux [photons s⁻¹ cm⁻² Å⁻¹]")

    return photon_flux, wavelengths


def convert_stellar_model_to_flux(model_data, r_star):
    '''
    Legacy model flux:
    frequency-based stellar model quantity, converted to per-wavelength
    and integrated over stellar surface and solid angle.
    Resulting quantity is stellar spectral luminosity (erg/s/A),
    later converted to flux at Earth by geometric dilution.
    '''
    flux_lambda = np.zeros(np.shape(model_data))
    wavelength = model_data[:, 0]
    wavelength_sq = wavelength * wavelength
    scale = C_LIGHT_Angst * 4 * np.pi * (r_star**2) * 4 * np.pi

    # we convert from frq to wavelength using lambda in Angstrom: F_lambda = F_nu * c / lambda^2
    # Unit before: erg/cm2/s/Hz, After:  ergs/cm2/s/A
    # Integrate over stellar surface area (4*pi*R^2) and over solid angle (4*pi)
    # multiply with surface area of star -> ergs/cm2/s/A to ergs/s/A
    # then multiply with 4*!pi for steradian conversion
    flux_lambda[:, 0] = wavelength
    flux_lambda[:, 1] = model_data[:, 1] * scale / wavelength_sq
    flux_lambda[:, 2] = model_data[:, 2] * scale / wavelength_sq

    logging.info("Converting intensity to luminosity for r_star=%.6e cm with %d wavelength points", r_star, model_data.shape[0])

    return flux_lambda

def apply_ism_absorption(data, ebv, cfg: GlobalConfig, announce_user: bool = False):
    announce("Starting ISM absorption", announce_user)
    if cfg.mg2_col is None:
        nh = 5.8e21 * ebv  # is the total hydrogen column density
        fractionMg2 = 0.825  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        # Safe runtime path: keep physically correct -inf for zero columns without triggering log10(0) warnings.
        mg2_col_linear = nh * fractionMg2 * 10.0**Mg_abn
        nmg2 = -np.inf if mg2_col_linear <= 0.0 else np.log10(mg2_col_linear)
        nmg2_source = "computed"
    else:
        nmg2 = float(cfg.mg2_col)
        nmg2_source = "cfg"

    if cfg.mg1_col is None:
        nh = 5.8e21 * ebv   # is the total hydrogen column density
        fractionMg1 = 0.00214  # (Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
        Mg_abn = -5.33  # (Frisch & Slavin 2003; this is the ISM abundance of Mg)
        mg1_col_linear = nh * fractionMg1 * 10.0**Mg_abn
        nmg1 = -np.inf if mg1_col_linear <= 0.0 else np.log10(mg1_col_linear)
        nmg1_source = "computed"
    else:
        nmg1 = float(cfg.mg1_col)
        nmg1_source = "cfg"

    if cfg.fe2_col is None:
        nh = 5.8e21 * ebv   # is the total hydrogen column density
        fractionFe2 = 0.967  # (Frisch & Slavin 2003; this is the fraction of Fe in the ISM that is singly ionised)
        Fe_abn = -5.73  # (Frisch & Slavin 2003; this is the ISM abundance of Fe)
        fe2_col_linear = nh * fractionFe2 * 10.0**Fe_abn
        nfe2 = -np.inf if fe2_col_linear <= 0.0 else np.log10(fe2_col_linear)
        nfe2_source = "computed"
    else:
        nfe2 = float(cfg.fe2_col)
        nfe2_source = "cfg"

    logging.info("ISM params: E(B-V)=%s nmg2=%s (%s) nmg1=%s (%s) nfe2=%s (%s)", ebv, nmg2, nmg2_source, nmg1, nmg1_source, nfe2, nfe2_source)

    flux_data = cute_ism_abs_all(data, nmg2, nmg1, nfe2)

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
    return flux_at_earth

def apply_unred(wavelengths, flux_at_earth, ebv, announce_user: bool = False):
    announce("Starting to apply UNRED extinction correction", announce_user)
    ebv = -1.0 * ebv
    flux_unred = unred(wavelengths, flux_at_earth, ebv=ebv, R_V=R_V)
    return flux_unred
