"""
Physical and simulator-wide constants.
Do not modify at runtime. Or ever, because tests will fail.
"""
import astropy.constants as ac
import astropy.units as u

# physics constants
C_LIGHT_km_s = ac.c.to("km/s").value
C_LIGHT_m_s = ac.c.to("m/s").value
C_LIGHT_cm_s = ac.c.to("cm/s").value 
C_LIGHT_Angst = ac.c.to("Angstrom/s").value # Angstrom/s (speed of light)

R_SUN_cm = ac.R_sun.to("cm").value
PARSEC_CM = ac.pc.to("cm").value
AU_cm = ac.au.to("cm").value  # cm (Astronomical Unit)
M_SUN_kg = ac.M_sun.to("kg").value
M_SUN_cgs = ac.M_sun.cgs.value  # g

boltzmann = ac.k_B.to("erg / K").value # erg/K = g cm**2/s**2/K (boltzmann const)
N_A   = ac.N_A.value # /mol (Avagadro constant)
sigma = ac.sigma_sb.to("erg / (cm2 s K4)").value # erg/cm**2/s/K**4 (stefan-boltzmann)

# Planck constant in erg*s
H_PLANCK = ac.h.to("erg s").value

# 1 / (h * c) with wavelength expected in Angstrom
# Derivation:
# photons = flux_erg * lambda_cm / (h * c)
# lambda_cm = lambda_Angstrom * 1e-8
# => factor = (1 / (h * c)) 
PHOTON_ENERGY_CONVERSION_A = (1.0 / (H_PLANCK * C_LIGHT_Angst))
# 1 steradian expressed in square arcseconds
# (used to convert surface brightness from per steradian to per arcsec² / pixel units)
ARCSEC2_PER_SR = (1 * u.sr).to(u.arcsec**2).value

#Parameters for MgII
MgII1w      = 2795.5280 #MgIIh wavelength 
MgII1_loggf = 0.100     #MgIIh loggf
MgII1_stark = -5.680    #MgIIh stark
MgII2w      = 2802.7050 #MgIIk wavelength
MgII2_loggf = -0.210    #MgIIk loggf
MgII2_stark =-5.680     #MgIIkStark damping constant
Mgaratio_loggf2to1=(10**MgII2_loggf)/(10**MgII1_loggf)

##Parameters for MgI
MgIw      = 2852.127 
MgI_loggf = 0.255
MgI_stark = -5.640

##Parameters for FeII
FeIIw      = 2599.39515
FeII_loggf = 0.378
FeII_stark = -6.53

# Extinction curve: R_V = A(V) / E(B-V), ratio of total to selective extinction.
R_V = 3.1

#ISM fixed parameters
ISM_b_Mg2=2.0        #b-parameter for the Ca2 ISM lines in km/s
vr_ISM=0.            #radial velocity of the ISM absorption lines in km/s


# Complete Wavelength ranges for plots or datadumps in A
# should be the complete WL range we actually want to see with Waltzer. 
# However, this is also just for data dumps, plots and checks
debug_wavelength_range_nuv = [2390, 3260]
debug_wavelength_range_vis = [4450, 8250]
debug_wavelength_range_ir = [8250, 18000]
# wavelengths for test_dumps and comparisons in A
DEBUG_WL_A_NUV = (2790.0, 2850.0)
DEBUG_WL_A_VIS = (5550.0, 5630.0)
DEBUG_WL_A_NIR  = (11000.0, 11100.0)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if not name.startswith("_") and not callable(value):
            print(f"{name:25s} = {value}")
