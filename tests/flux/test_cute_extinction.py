import numpy as np

from flux.cute_extinction import extinction_amores


# Tests: extinction_amores
# Behavior: returns finite, non-negative extinction that increases with distance.
def test_extinction_amores_monotonic_with_distance():
    glong = 150.0
    glat = 5.0

    ebv_near, av_near = extinction_amores(glong, glat, distance=0.1)
    ebv_far, av_far = extinction_amores(glong, glat, distance=1.0)

    assert np.isfinite(ebv_near)
    assert np.isfinite(av_near)
    assert np.isfinite(ebv_far)
    assert np.isfinite(av_far)

    assert ebv_near >= 0.0
    assert av_near >= 0.0
    assert ebv_far >= 0.0
    assert av_far >= 0.0

    assert ebv_far >= ebv_near
    assert av_far >= av_near
