"""
Physical and simulator-wide constants.
Do not modify at runtime. Or ever, because tests will fail.
"""
import astropy.constants as ac

# physical constants
C_LIGHT_ROUNDED_m_s = 3e18                   # m / s
C_LIGHT_cm_s    = 2.99792458e10                # cm/s (speed of light)
C_LIGHT_km_s    = ac.c.to("km/s").value        # Speed of light, km/s
sigma = 5.67051e-5                   # erg/cm**2/s/K**4 (stefan-boltzmann)
k_B   = 1.380658e-16                 # erg/K = g cm**2/s**2/K (boltzmann const)
N_A   = 6.02214179e23                # /mol (Avagadro constant)
C_LIGHT_Angst    = 2.99792458e18                # Angstrom/s (speed of light)

# astronomy
R_SUN = 6.957e10                 # cm
PARSEC_CM = 3.086e18             # m
AU    = 1.49598e13                   # cm (Astronomical Unit)

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

#ISM fixed parameters
ISM_b_Mg2=2.0        #b-parameter for the Ca2 ISM lines in km/s
vr_ISM=0.            #radial velocity of the ISM absorption lines in km/s

#not yet used
H_PLANCK = 6.62607015e-34        # J s
K_BOLTZMANN = 1.380649e-23       # J / K
SIGMA_SB = 5.670374419e-8        # W / (m^2 K^4)
L_SUN = 3.828e26                 # W
M_SUN = 1.98847e30               # kg


# Complete Wavelength ranges for plots or datadumps in A
# should be the complete WL range we actually want to see with Waltzer. 
# However, this is also just for data dumps, plots and checks
debug_wavelength_range_nuv = [2290, 3360]
debug_wavelength_range_vis = [4200, 8310]
debug_wavelength_range_ir = [8200, 18000]
# wavelengths for test_dumps and comparisons in A
DEBUG_WL_A_NUV = (2790.0, 2850.0)
DEBUG_WL_A_VIS = (5550.0, 5630.0)
DEBUG_WL_A_IR  = (11000.0, 11100.0)
