import logging
import numpy as np

from flux.cute_unred import unred


# Tests: unred
# Behavior: LMC branch flags select the corresponding coefficient set.
def test_unred_lmc_flag_selects_coefficients(caplog):
    wave = np.array([1500.0, 2000.0, 3000.0], dtype=np.float32)
    flux = np.ones_like(wave, dtype=np.float32)

    with caplog.at_level(logging.INFO):
        unred(wave, flux, ebv=0.1, R_V=3.1, LMC2=True, AVGLMC=False)
    lmc2_log = next(record.message for record in caplog.records if "unred_base_curve:" in record.message)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        unred(wave, flux, ebv=0.1, R_V=3.1, LMC2=False, AVGLMC=True)
    avglmc_log = next(record.message for record in caplog.records if "unred_base_curve:" in record.message)

    assert "LMC2=True" in lmc2_log
    assert "AVGLMC=False" in lmc2_log
    assert "c1=-2.16" in lmc2_log
    assert "c2=1.31" in lmc2_log

    assert "LMC2=False" in avglmc_log
    assert "AVGLMC=True" in avglmc_log
    assert "c1=-1.28" in avglmc_log
    assert "c2=1.11" in avglmc_log
