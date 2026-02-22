from configs.channel_config import SpectroscopyChannel, PhotometryChannel

def test_spectroscopy_channel_init():
    ch = SpectroscopyChannel(
        channel_name="NUV",
        x_pixels=2048,
        y_pixels=512,
        resolution_factor=1.0,
        dark_noise=0.01,
        dark_current_sigma=0.001,
        read_noise=3.0,
        bias_offset=0.0,
        effective_area_file="ea.txt",
        ccd_gain=1.0,
        exposure_s=10.0,
        mode=1,
        spread_profile_file="",
        spread_half_height_pix=3,
        wavelength=[],
        effective_area=[],
        pixel_scale=1.0,
        spread_y_positions=None,
        spread_y_weights=None,
        spread_y_wavelengths=None,
        source_file="cfg",
        slit_position_x_arcsec=0.0,
        slit_position_y_arcsec=0.0,
        slope=0.0,
        intercept_pixels=0.0,
    )

    assert ch.channel_name == "NUV"
    assert ch.exposure_s == 10.0
    assert ch.x_pixels == 2048


def test_photometry_channel_init():
    ch = PhotometryChannel(
        channel_name="IR",
        x_pixels=100,
        y_pixels=100,
        resolution_factor=1.0,
        dark_noise=0.0,
        dark_current_sigma=0.0,
        read_noise=1.0,
        bias_offset=0.0,
        ccd_gain=1.0,
        exposure_s=5.0,
        source_file="cfg"
    )

    assert ch.channel_name == "IR"
    assert ch.exposure_s == 5.0