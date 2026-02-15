# from utils.debug_dumps import dump_1d_array
# from instrument.detector import gaussbroad
# import numpy as np

# def testlegacy2(photon_flux_at_earth_A, wavelengths_total, nuv_cal, output_dir):

#     wave_new = wavelengths_total
#     photons_star_new = photon_flux_at_earth_A

#     mask = (wave_new >= 2000.0) & (wave_new < 4000.0)
#     wave_new = wave_new[mask]
#     photons_star_new = photons_star_new[mask]
#     dump_1d_array(wave_new, photons_star_new, output_dir, "HD 2685", "photons_star_new2",full=True, zoom=False)

#     # legacy uses HWHM = fwhm/2. For your NUV this equals pixel_scale (0.417684)
#     fwhm = nuv_cal.pixel_scale * 2
#     print("fwhm:", repr(fwhm))
#     dump_1d_array(np.array([0.0]), np.array([fwhm/2.0]), output_dir, "HD 2685", "new_hwhm_passed2", full=True, zoom=False)

#     smoothedflux = gaussbroad(wave_new, photons_star_new, fwhm/2.0)

#     dump_1d_array(wave_new, smoothedflux, output_dir, "HD 2685", "legacy_gauss_smoothedflux_check2",full=True, zoom=False)


#     ccd_flux = np.interp(nuv_cal.wavelength, wave_new, smoothedflux)
#     dump_1d_array(nuv_cal.wavelength, ccd_flux, output_dir, "HD 2685", "ccd_flux2", full=True, zoom=False)

    
#     aeff = np.interp(nuv_cal.wavelength, nuv_cal.wavelength, nuv_cal.effective_area)
#     dump_1d_array(nuv_cal.wavelength, aeff, output_dir, "HD 2685", "aeff2", full=True, zoom=False)

#     ccd_count1 = ccd_flux * aeff * fwhm
#     dump_1d_array(nuv_cal.wavelength, ccd_count1, output_dir, "HD 2685", "ccd_count12", full=True, zoom=False)


# def testlegacy3(photon_flux_at_earth_A, wavelengths_total, nuv_cal, output_dir):

#     # legacy uses HWHM = fwhm/2. For your NUV this equals pixel_scale (0.417684)
#     fwhm = nuv_cal.pixel_scale * 2
#     print("fwhm:", repr(fwhm))

#     smoothedflux = gaussbroad(wavelengths_total, photon_flux_at_earth_A, fwhm/2.0)

#     # dump_1d_array(wavelengths_total, smoothedflux, output_dir, "HD 2685", "legacy_gauss_smoothedflux_check3",full=True, zoom=False)


#     ccd_flux = np.interp(nuv_cal.wavelength, wavelengths_total, smoothedflux)
#     # dump_1d_array(nuv_cal.wavelength, ccd_flux, output_dir, "HD 2685", "ccd_flux3", full=True, zoom=False)

    
#     ccd_count1 = ccd_flux * nuv_cal.effective_area * fwhm
#     dump_1d_array(nuv_cal.wavelength, ccd_count1, output_dir, "HD 2685", "ccd_count12", full=True, zoom=False)
