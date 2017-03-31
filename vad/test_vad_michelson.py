""" Unit Tests for Py-ART's retrieve/vad_michelson.py module. """

from __future__ import print_function

import numpy as np
import pyart
from numpy.testing import assert_almost_equal, assert_raises

from pyart.retrieve import velocity_azimuth_display


def test_velocity_azimuth_display():
    test_radar = pyart.io.read(pyart.testing.NEXRAD_ARCHIVE_MSG1_FILE)
    corrected_velocity = None
    z_start = 0
    z_end = 1000
    z_count = 10
    norm_coh_power = None
    norm_coh_power_value = 0.5
    nyquist_velocity = None

    z_interval = ([0., 111.11111111, 222.22222222, 333.33333333,
                   444.44444444, 555.55555556, 666.66666667, 
                   777.77777778, 88.88888889, 1000.])
    u_mean = ([-2.29649931e-01, -1.14813107e+00, -6.54485761e-01,
               -9.62677803e-02, 3.18045369e-02, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               6.89043585e-04])
    v_mean = ([3.57859759e-02, 3.89782303e-01, -6.28990929e-01,
               -2.95026771e-01, -1.41892344e-01, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               -1.35308218e-04])

    vad = retrieve.velocity_azimuth_display(test_radar,
                                            corrected_velocity,
                                            z_start, z_end,
                                            z_count, norm_coh_power,
                                            norm_coh_power_value,
                                            nyquist_velocity)
    assert_almost_equal(vad['z_interval'], z_interval, 8)
    assert_almost_equal(vad['u_mean'], u_mean, 8)
    assert_almost_equal(vad['v_mean'], v_mean, 8)
