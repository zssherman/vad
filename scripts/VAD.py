"""
pyart.retrieve.velocity_azimuth_display
=======================================

Retrieval of VADs from a radar object.

Code adapted by Scott Collis,

"Below is some code from Susan Rennie (based off Michelson et al 2000)
from the Centre for Australian Weather and Climate Research (CAWCR)
for calculating VADs. I used her one tilt version and wrote my own
adaptation into Py-ART PyRadar object.. Note I convert to U and V before
averaging.. my concern is that if you have $\theta$ rapidly varying between
360$^\circ$ and 0$^\circ$ this will average to a nonsense number" (Collis).

.. autosummary::
    :toctreeL generated/
    :template: dev_template.rst

    Velocity Azimuth Display

"""


import numpy as np
import scipy

def velocity_azimuth_display(radar, corrected_velocity=None,
                             z_start=0, z_end=10500, z_count=101,
                             norm_coh_power=None,
                             norm_coh_power_value=0.5):
    """
    Velocity azimuth display.

    Parameters
    ----------
    radar : Radar
        Radar object used.
    corrected_velocity : string
        Velocity field to use for VAD calculation.
        If None, the field 'velocity' will be corrected using
        Py-ART's region based radar dealiasing algorithm,
        and field then used is the new 'corrected_velocity'.

    Other Parameters
    ----------------
    z_start : int
        Z location to begin VAD calculation in meters,
        default=0.
    z_end : int
        Z location to end VAD calculation in meters,
        default=10500.
    z_count : int
        Amount of data points used between z_start and z_end.
        Data points are evenly spread out using np.linspace,
        default=101.
    norm_coh_power : string
        Normalized coherent power field to use for VAD calculation.
        If None, a default normal_coherent_power field will be
        created for the user with a default array of 0.5's.
    norm_coh_power_value : float
        The value to use to create field 'norm_coh_power'
        if not present. Also used as mask to exclude_below,
        default=0.5.


    Returns
    -------
    u_mean : array
        U-wind mean in meters per second.
    v_mean : array
        V-wind mean in meters per second.

    References
    ----------
    Michelson, D. B., Andersson, T., Koistinen, J., Collier, C. G., Riedl, J.,
    Szturc, J., Gjertsen, U., Nielsen, A. and Overgaard, S. (2000) BALTEX Radar
    Data Centre Products and their Methodologies. In SMHI Reports. Meteorology
    and Climatology.Swedish Meteorological and Hydrological Institute, Norrkoping.

    """


    def interval_mean(data, current_z, wanted_z):
        """ Find the mean of data indexed by current_z
            at wanted_z on intervals wanted_z+/- delta
            wanted_z """
        delta = wanted_z[1] - wanted_z[0]
        pos_lower = [np.argsort((current_z - (wanted_z[i] - delta / 2.0))**2)[0]
                     for i in range(len(wanted_z))]
        pos_upper = [np.argsort((current_z - (wanted_z[i] + delta / 2.0))**2)[0]
                     for i in range(len(wanted_z))]
        new_values = np.array([data[pos_lower[i]:pos_upper[i]].mean()
                              for i in range(len(pos_upper))])
        return new_values


    def sd_to_uv(speed, direction):
        return (speed * np.sin(direction), speed * np.cos(direction))


    def VAD_algorithm(V, Az, El, Ra):
        ''' Calculates VAD for a scan, returns speed and angle
        outdic=VADf(V,Az,El,Ra)
        V is 2D array, Az, Ra are 1D arrays, El is number.
        All in degrees, m outdic contains speed, angle, variance. 
        '''
        nrays, nbins = V.shape
        nr2 = nrays / 2
        Vc = np.empty((nr2, nbins, 2))
        Vc[:, :, 0] = V[0:nr2, :]
        Vc[:, :, 1] = V[nr2:, :]
        sinaz = np.sin(np.radians(Az))
        cosaz = np.cos(np.radians(Az))
        sumv = np.ma.sum(Vc, 2)
        vals = np.isnan(sumv)
        vals2 = np.vstack((vals, vals))
        count = np.sum(np.isnan(sumv) == False, 0)
        aa = count < 8
        vals[:, aa] = 0
        vals2[:, aa] = 0
        count = np.float_(count)
        count[aa] = np.nan
        U_m = np.array([np.nansum(sumv, 0) / (2 * count)])
        count[aa] = 0

        CminusU_mcos = np.zeros((nrays, nbins))
        CminusU_msin = np.zeros((nrays, nbins))
        sincos = np.zeros((nrays, nbins))
        sin2 = np.zeros((nrays, nbins))
        cos2 = np.zeros((nrays, nbins))

        for i in range(nbins):
            CminusU_mcos[:, i] = cosaz * (V[:, i] - U_m[:, i])
            CminusU_msin[:, i] = sinaz * (V[:, i] - U_m[:, i])
            sincos[:, i] = sinaz * cosaz
            sin2[:, i] = sinaz**2
            cos2[:, i] = cosaz**2
        CminusU_mcos[vals2] = np.nan
        CminusU_msin[vals2] = np.nan
        sincos[vals2] = np.nan
        sin2[vals2] = np.nan
        cos2[vals2] = np.nan
        sumCminU_mcos = np.nansum(CminusU_mcos, 0)
        sumCminU_msin = np.nansum(CminusU_msin, 0)
        sumsincos = np.nansum(sincos, 0)
        sumsin2 = np.nansum(sin2, 0)
        sumcos2 = np.nansum(cos2, 0)
        a_value = (sumCminU_mcos - (sumsincos*sumCminU_msin / sumsin2))\
        / (sumcos2 - (sumsincos**2) / sumsin2)
        b_value = (sumCminU_msin - a_value*sumsincos) / sumsin2
        speed = np.sqrt(a_value**2 + b_value**2) / np.cos(np.radians(El))
        angle = np.arctan2(a_value, b_value)

        crv = np.empty((nrays, nbins))
        for i in range(nbins):
            crv[:, i] = np.sin(np.radians(Az) + angle[i])*speed[i]
        Vn = V.copy()
        Vn[vals2 == True] = np.nan
        var = np.nansum((crv - Vn)**2, 0) / (sum(np.isnan(Vn) == False) - 2)
        return {'speed' : speed, 'angle' : angle, 'variance' : var}


    def VAD_calculation(radar):
        speed = []
        angle = []
        height = []
        x = radar.gate_x['data']
        y = radar.gate_y['data']
        z = radar.gate_z['data']
        z_want = np.linspace(z_start, z_end, z_count)

        if norm_coh_power == None:
            # Copying a field and its shape to use as a
            # norm_coherent_power array with matching shape.
            radar.add_field_like('reflectivity', 'norm_coh_power',
                                 radar.fields['reflectivity']['data'].copy(),
                                 replace_existing=True)
            # Setting all values to norm_coh_power_value,
            # in the field norm_coh_power.
            (radar.fields['norm_coh_power']['data'])[
                radar.fields['norm_coh_power']['data'] !=
                norm_coh_power_value] = norm_coh_power_value
            norm_coh_power_used = 'norm_coh_power'
        else:
            norm_coh_power_used = norm_coh_power

        gatefilter = pyart.correct.GateFilter(radar)
        gatefilter.exclude_below(norm_coh_power_used, norm_coh_power_value)
        if 0 > norm_coh_power_value or norm_coh_power_value > 1:
            raise ValueError('Normalized coherent power is out of'
                             ' recommended range between 0 and 1.')
        if corrected_velocity == None:
            nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
            corr_vel = pyart.correct.dealias_region_based(
                radar, vel_field='velocity', keep_original=False, 
                gatefilter=gatefilter, nyquist_vel=nyq,
                centered=True)
            radar.add_field('corrected_velocity', corr_vel,
                              replace_existing=True)
            velocity_used = 'corrected_velocity'
        else:
            velocity_used = corrected_velocity

        if radar.fields[velocity_used]['units'] is not 'meters_per_second'\
        and radar.fields[velocity_used]['units'] is not 'm/s':
            raise ValueError('Field used for VAD Calculation needs to be'
                             ' a velocity field.')

        for i in range(len(radar.sweep_start_ray_index['data'])):
            i_s = radar.sweep_start_ray_index['data'][i]
            i_e = radar.sweep_end_ray_index['data'][i]

            if (i_e - i_s) % 2 == 0:
                print("even, all good")
            else:
                i_e = i_e - 1

            VR = radar.fields[velocity_used]['data'][i_s:i_e, :]
            #SQ = radar.fields[norm_coh_power_used]['data'][i_s:i_e, :]
            Az = radar.azimuth['data'][i_s:i_e]
            Ra = radar.range['data']
            El = radar.fixed_angle['data'][i]

            mask = VR.mask
            #crud = SQ < 0.5
            VR[np.where(mask)] = np.nan
            #VR[np.where(crud)] = np.nan
            one_level = VAD_algorithm(VR, Az, El, Ra / 1000.0)
            not_garbage = np.isfinite(one_level['speed'])

            print('max height', z[i_s, :][np.where(not_garbage)].max(),
                  ' meters')

            speed.append(one_level['speed'][np.where(not_garbage)])
            angle.append(one_level['angle'][np.where(not_garbage)])
            height.append(z[i_s, :][np.where(not_garbage)])

        speed_array = np.concatenate(speed)
        angle_array = np.concatenate(angle)
        height_array = np.concatenate(height)
        arg_order = height_array.argsort()
        speed_ordered = speed_array[arg_order]
        height_ordered = height_array[arg_order]
        angle_ordered = angle_array[arg_order]
        u_ordered, v_ordered = sd_to_uv(speed_ordered, angle_ordered)
        u_mean = interval_mean(u_ordered, height_ordered, z_want)
        v_mean = interval_mean(v_ordered, height_ordered, z_want)
        return z_want, u_mean, v_mean
    z_want, u_mean, v_mean = VAD_calculation(radar)
    return {'u_mean' : u_mean, 'v_mean' : v_mean, 'z_interval': z_want}
