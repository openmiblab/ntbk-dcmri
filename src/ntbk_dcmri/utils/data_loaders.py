import numpy as np
import pydmr

def load_spogis(datafile):

    data = pydmr.read(datafile, 'nest')

    subjects = list(data['rois'].keys())
    input_data = {}

    for subject in subjects:

        # Interpolate to the smallest time step
        time = data['rois'][subject]['study']['time_s']
        ts = np.min(np.diff(time))
        tint = np.arange(0, np.max(time), ts)

        s_ao_int = np.interp(tint, time, data['rois'][subject]['study']['aorta_aif_10%_mean'])
        s_lk_int = np.interp(tint, time, data['rois'][subject]['study']['kidney_left_mean'])
        s_rk_int = np.interp(tint, time, data['rois'][subject]['study']['kidney_right_mean'])

        input_data[subject] = {
            'dt': 0.1,
            'agent': data['pars'][subject]['study']['agent'],
            'dose': data['pars'][subject]['study']['dose'],
            'rate': data['pars'][subject]['study']['rate'],
            'FA': data['pars'][subject]['study']['FA'],
            'TR': data['pars'][subject]['study']['TR'],
            'field_strength': data['pars'][subject]['study']['field_strength'],
            'Nph': int(ts / data['pars'][subject]['study']['TR']),
            'Nk0': int(ts / data['pars'][subject]['study']['TR'] / 2),
            'S_ao': s_ao_int,
            'S_lk': s_lk_int,
            'S_rk': s_rk_int,
            'tS_ao': tint,
            'tS_lk': tint,
            'tS_rk': tint,
            'vol_lk': data['pars'][subject]['study']['vol_lk'],
            'vol_rk': data['pars'][subject]['study']['vol_rk'],
            'weight': data['pars'][subject]['study']['weight'],
            'nmGFR': data['pars'][subject]['study']['nmGFR'] / 60,
        }

    return input_data