# This is VAD code adapted from VAD code created by Jonathan Helmus.
# VAD technique is based on Browning et al (1968).

"""
pyart.retrieve.velocity_azimuth_display
=======================================

Retrieval of VADs from a radar object.

.. autosummary::
    :toctreeL generated/
    :template: dev_template.rst

    velocity_azimuth_display
    _horizontal_winds
    _inverse_dist_squared
    _Average1D

"""

import numpy as np

from pyart.core import HorizontalWindProfile


def velocity_azimuth_display(
        radar, velocity=None, height=None,
        valid_ray_min=16, gatefilter=None, window=2,
        weight='equal'):
    """
    Velocity azimuth display.

    Note: If a specific sweep is desired, before using the
    velocity_azimuth_display function, use, for example:
    one_sweep_radar = radar.extract_sweeps([0])

    Parameters
    ----------
    radar : Radar
        Radar object used.
    velocity : string
        Velocity field to use for VAD calculation.
        If None, the default velocity field will be used.

    Other Parameters
    ----------------
    height : array
        Array of desired heights to be sampled for the vad
        calculation.
    valid_ray_min : int
        Amount of rays required to include that level in
        the VAD calculation.
    gatefilter : GateFilter
        A GateFilter indicating radar gates that should be excluded when
        from the import vad calculation.
    window : int
        Value to use for window when determing new values in the
        _Averag1D function.
    weight : string
        A string to indicate weighting method to use. 'equal' for
        equal weighting when interpolating or 'idw' for inverse
        distribution squared weighting for interpolating.
        Default is 'equal'.

    Returns
    -------
    height : array
        Heights in meters above sea level at which horizontal winds were
        sampled.
    speed : array
        Horizontal wind speed in meters per second at each height.
    direction : array
        Horizontal wind direction in degrees at each height.
    u_wind : array
        U-wind mean in meters per second.
    v_wind : array
        V-wind mean in meters per second.

    Reference
    ----------
    K. A. Browning and R. Wexler, 1968: The Determination
    of Kinematic Properties of a Wind Field Using Doppler
    Radar. J. Appl. Meteor., 7, 105–113

    """

    if velocity is None:
        velocity_used = 'velocity'
    else:
        velocity_used = velocity

    velocities = radar.fields[velocity_used]['data']
    #mask = velocities.mask
    #velocities[np.where(mask)] = np.nan
    if gatefilter is not None:
        velocities = np.ma.masked_where(
            gatefilter.gate_excluded, velocities)
    azimuths = radar.azimuth['data'][:]
    elevation = radar.fixed_angle['data'][0]

    u_wind, v_wind = _horizontal_winds(velocities, azimuths,
                                       elevation, valid_ray_min)
    bad = np.logical_or(np.isnan(u_wind), np.isnan(v_wind))
    good_u_wind = u_wind[~bad]
    good_v_wind = v_wind[~bad]

    radar_height = radar.gate_z['data'][0]
    good_height = radar_height[~bad]

    if height is None:
        z_want = np.linspace(0, 1000, 100)[:50]
    else:
        z_want = height

    try:
        print('max height', np.max(good_height), ' meters')
        print('min height', np.min(good_height), ' meters')
    except ValueError:
        raise ValueError('Not enough data in this radar sweep ' \
                         'for a vad calculation.')

    u_interp = _Average1D(good_height, good_u_wind,
                          z_want[1] - z_want[0] / window, weight)
    v_interp = _Average1D(good_height, good_v_wind,
                          z_want[1] - z_want[0] / window, weight)

    u_wanted = u_interp(z_want)
    v_wanted = v_interp(z_want)
    u_wanted = np.ma.masked_equal(u_wanted, 99999.)
    v_wanted = np.ma.masked_equal(v_wanted, 99999.)

    vad = HorizontalWindProfile.from_u_and_v(
        z_want, u_wanted, v_wanted)
    return vad

def _horizontal_winds(velocities, azimuths,
                      elevation, valid_ray_min):
    """ Calculates VAD for a scan and returns u_mean and
    v_mean. velocities is a 2D array, azimuths is a 1D
    array, elevation is a number.

    Note:
    We need to solve: Ax = b
    where:
    A = [sum_sin_squared_az, sum_sin_cos_az    ] = [a, b]
        [sum_sin_cos_az,     sum_cos_squared_az]   [c, d]
    b = [sum_sin_vel_dev] = [b_1]
        [sum_cos_vel_dev]   [b_2]
    The solution to this is:
    x = A-1 * b

    A-1 is:
     1    [ d,  -b ]
    --- * [ -c,  a ]
    |A|

    and the determinate, det is: det = a*d - b*c

    Therefore the elements of x are:

    x_1 = (d* b_1  + -b * b_2) / det = (d*b_1 - b*b_2) / det
    x_2 = (-c * b_1 +  a * b_2) / det = (a*b_2 - c*b_1) / det

    """
    velocities = velocities.filled(np.nan)
    shape = velocities.shape
    _, nbins = velocities.shape

    invalid = np.isnan(velocities)
    valid_rays_per_gate = np.sum(~np.isnan(velocities), axis=0)
    too_few_valid_rays = valid_rays_per_gate < valid_ray_min
    invalid[:, too_few_valid_rays] = True

    sin_az = np.sin(np.deg2rad(azimuths))
    cos_az = np.cos(np.deg2rad(azimuths))
    sin_az = np.repeat(sin_az, nbins).reshape(shape)
    cos_az = np.repeat(cos_az, nbins).reshape(shape)
    sin_az[invalid] = np.nan
    cos_az[invalid] = np.nan

    mean_velocity_per_gate = np.nanmean(velocities, axis=0).reshape(1, -1)
    velocity_deviation = velocities - mean_velocity_per_gate

    sum_cos_vel_dev = np.nansum(cos_az * velocity_deviation, axis=0)
    sum_sin_vel_dev = np.nansum(sin_az * velocity_deviation, axis=0)

    sum_sin_cos_az = np.nansum(sin_az * cos_az, axis=0)
    sum_sin_squared_az = np.nansum(sin_az**2, axis=0)
    sum_cos_squared_az = np.nansum(cos_az**2, axis=0)

    # The A matrix
    a = sum_sin_squared_az
    b = sum_sin_cos_az
    c = sum_sin_cos_az
    d = sum_cos_squared_az

    # The b vector
    b_1 = sum_sin_vel_dev
    b_2 = sum_cos_vel_dev

    # solve for the x vector
    determinant = a*d - b*c
    x_1 = (d*b_1 - b*b_2) / determinant
    x_2 = (a*b_2 - c*b_1) / determinant

    # calculate horizontal components of winds
    elevation_scale = 1 / np.cos(np.deg2rad(elevation))
    u_mean = x_1 * elevation_scale
    v_mean = x_2 * elevation_scale
    return u_mean, v_mean

def _inverse_dist_squared(dist):
    """ Obtaining distance weights by using distance weighting
    interpolation, using the inverse distance-squared relationship.
    """
    weights = 1 / (dist * dist)
    weights[np.isnan(weights)] = 99999.
    return weights


class _Average1D(object):
    """ Used to find the nearest gate height and horizontal wind
    value with respect to the user's desired height. """

    def __init__(self, x, y, window, weight,
                 fill_value=99999.):

        sort_idx = np.argsort(x)
        self.x_sorted = x[sort_idx]
        self.y_sorted = y[sort_idx]
        self.window = window
        self.fill_value = fill_value
        if weight == 'equal':
            self.weight_func = lambda x: None
        elif weight == 'idw':
            self.weight_func = _inverse_dist_squared
        elif callable(weight):
            self.weight_func = weight
        else:
            raise ValueError("Invalid weight argument:", weight)

    def __call__(self, x_new, window=None):

        if window is None:
            window = self.window

        y_new = np.zeros_like(x_new, dtype=self.y_sorted.dtype)
        for i, center in enumerate(x_new):

            bottom = center - window
            top = center + window
            start = np.searchsorted(self.x_sorted, bottom)
            stop = np.searchsorted(self.x_sorted, top)

            x_in_window = self.x_sorted[start:stop]
            y_in_window = self.y_sorted[start:stop]
            if len(x_in_window) == 0:
                y_new[i] = self.fill_value
            else:
                distances = x_in_window - center
                weights = self.weight_func(distances)
                y_new[i] = np.average(y_in_window, weights=weights)
        return y_new
