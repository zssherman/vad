""" Unit Tests for Py-ART's retrieve/vad_michelson.py module. """

from __future__ import print_function

from numpy.testing import assert_almost_equal

import pyart


def test_velocity_azimuth_display():
    test_radar = pyart.io.read(pyart.testing.NEXRAD_ARCHIVE_MSG1_FILE)
    velocity = 'velocity'
    height = ([50.0, 162.5, 275.0, 387.5, 500.0])
    valid_ray_min = 16
    gatefilter = None
    window = 2
    weight = 'equal'

    speed = ([1.921647204618432, 1.3827367541946036, 1.164585840293324,
              1.682006738330196, 2.1305502721695446])
    direction = ([356.78032862063486, 0.5040511489577615, 5.449921149639364,
                  352.1683403733644, 337.85762401398773])
    u_wind = ([0.10792796375637152, -0.012164265247251506,
               -0.11060735435152175, 0.22919529090527493, 0.803024469916098])
    v_wind = ([-1.9186139616028124, -1.382683247187013, -1.1593214362613435,
               -1.666318152819272, -1.9734224491876267])

    vad = pyart.retrieve.velocity_azimuth_display(test_radar, velocity,
                                                  height, valid_ray_min,
                                                  gatefilter, window, weight)

    assert_almost_equal(vad.height, height, 3)
    assert_almost_equal(vad.speed, speed, 3)
    assert_almost_equal(vad.direction, direction, 3)
    assert_almost_equal(vad.u_wind, u_wind, 3)
    assert_almost_equal(vad.v_wind, v_wind, 3)
