# Patricio, 27.02.2026
# Will need to intall
#     git clone https://github.com/pcubillos/waltzer_etc
#     cd waltzer_etc
#     pip install -e .

import numpy as np
import matplotlib.pyplot as plt
import waltzer_etc as waltz
from waltzer_etc import sed
import pyratbay.spectrum as ps
import scipy.interpolate as si
plt.ion()


def nir_psf(npix):
    """
    Generate a 2D image of a Normalized PSF, i.e., np.sum(PSF) = 1.0

    Parameters
    ----------
    npix: Integer
        Set the 2D pixel array to size 2*npix + 1

    Returns
    -------
    psf: 2D float array
        WALTzER pixel array with PSF.
    """
    psf_file = (
        '/home/pcubillos/Dropbox/IWF/compendia/waltzer/inputs/'
        'psf_waltzer_nir_2025_11_27_edit.txt'
    )
    # Use only positive valus from center (PSF is simetric)
    rad, flux, _ = np.loadtxt(psf_file, unpack=True)
    mask = rad >= 0
    flux = flux[mask]
    rad = rad[mask]

    psf_r = si.interp1d(
        rad, flux,
        kind='slinear', bounds_error=False, fill_value='extrapolate',
    )

    x = np.arange(-npix, npix+1)
    y = np.arange(-npix, npix+1)
    X, Y = np.meshgrid(x, y)

    # TBD: PSF center is hard-coded at (0,0)
    x0 = y0 = 0.0
    r2 = np.sqrt((X - x0)**2 + (Y - y0)**2)

    psf = np.clip(psf_r(r2), 0.0, np.inf)
    psf /= np.sum(psf)
    return psf

    if False:
        import matplotlib.pyplot as plt
        # The PSF radial profile
        plt.figure(2)
        plt.clf()
        r = np.linspace(0, 5.5, 100)
        plt.plot(rad, flux, 'o', c='xkcd:blue', label='PSF data')
        plt.plot(r, psf_r(r), c='salmon', label='interpolation')
        plt.xlim(-0.05, 5.5)
        plt.ylim(0.25, 1.1)
        plt.legend(loc='lower left')
        plt.xlabel('distance from PSF center (pixels)')
        plt.ylabel('PSF')
        plt.savefig('waltzer_nir_psf_radial_profile.png', dpi=300)


def gauss2D(npix, sigma):
    x = np.arange(-npix, npix+1)
    y = np.arange(-npix, npix+1)
    X, Y = np.meshgrid(x, y)

    # Gaussian centered at (0, 0)
    # TBD: pass center coordinates as input arguments
    x0, y0 = 0.0, 0.0
    r2 = (X - x0)**2 + (Y - y0)**2
    psf = np.exp(-0.5 * r2 / sigma**2)
    psf /= np.sum(psf)

    return psf


def main():
    """
    Generate PSF
    calculate fluxes
    """
    # WALTzER detector
    nir = waltz.Detector('nir')

    # Spectral array at high resolution
    inst_resolution = nir.resolution
    wl_min = nir.hires_wl_min
    wl_max = nir.hires_wl_max
    resolution = nir.hires_resolution
    wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)

    # The source (Sirius)
    teff = 10_000
    logg = 4.5
    sed_type = 'llmodels'
    v_mag = -1.46

    sed_wl, flux = sed.load_sed(teff, logg, sed_type)
    sed_flux = np.interp(wl, sed_wl, flux)
    norm_flux = sed.normalize_vega(wl, sed_flux, v_mag)

    # Band-integrated fluxes in e- per second
    tso = nir.make_tso(wl, norm_flux)
    var_data = waltz.calc_variances(tso)
    band_wl, band_hw, source_flux, bkg_flux, dark, read = var_data

    # Aperture photometry
    radius = 0.5 * nir.aperture
    # For a Gaussian profile, assume that the standard deviation is a
    # third of the aperture radius
    sigma = radius/3.0
    # Map up to 5*sigma away from the center
    npix = int(5*sigma)

    theta = np.linspace(0, 2.0*np.pi, 100)
    xaper = radius * np.sin(theta)
    yaper = radius * np.cos(theta)

    psf = nir_psf(npix) * source_flux
    # alternative, gaussian psf
    #psf = gauss2D(npix, sigma=sigma) * source_flux

    fs = 12
    plt.figure(316)
    plt.clf()
    plt.subplots_adjust(0.135, 0.11, 0.98, 0.93)
    ax = plt.subplot(111)
    ax.plot(xaper, yaper, c='0.5', lw=1.5)
    # (left, right, bottom, top)
    extent = -npix-0.5, npix+0.5, -npix-0.5, npix+0.5
    cb = ax.imshow(
        psf, interpolation="nearest", origin="lower",
        extent=extent,
        cmap=plt.cm.magma_r,
        vmin=0,
        #aspect='auto',
    )
    ax.set_ylabel('Y axis (pixels)', fontsize=fs)
    ax.set_xlabel('X axis (pixels)', fontsize=fs)
    ax.set_title(f'{nir.band.upper()} brightest-pixel flux rate (Sirius)')
    ax.text(0.99, 0.95, f'Total flux = {np.sum(psf):_.0f} e/s', transform=ax.transAxes, ha='right', fontsize=fs-1)
    ax.text(0.99, 0.89, f'Max flux = {np.amax(psf):_.0f} e/s', transform=ax.transAxes, ha='right', fontsize=fs-1, weight='bold')
    ax.tick_params(which='both', direction='in', labelsize=fs-1)
    bar = plt.colorbar(cb)
    bar.ax.tick_params(which='both', direction='in', labelsize=fs-1)
    bar.set_label(f'PSF flux rate (e/s)', fontsize=fs)
    plt.savefig('waltzer_nir_psf_2D.png', dpi=300)
    print(f'Max flux {nir.band} = {np.amax(psf):_.0f} e/s')
    print(f'time_to_reach_50K = {50_000/np.amax(psf):.5f} s')



