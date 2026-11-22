import numpy as np
import pydmr

def load_spogis(datafile):

    # Some datasets are corrupted after a certain time point
    # Truncating these manually
    tmax = {
        '11': 9, # max time in minutes
        '14': 8,
        '19': 8.5,
    }

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

        if subject in tmax:
            idx = tint < tmax[subject] * 60
            tint = tint[idx]
            s_ao_int = s_ao_int[idx]
            s_lk_int = s_lk_int[idx]
            s_rk_int = s_rk_int[idx]

        input_data[subject] = {
            'dt': 0.1,
            'agent': data['pars'][subject]['study']['agent'],
            'dose': data['pars'][subject]['study']['dose'],
            'rate': data['pars'][subject]['study']['rate'],
            'FA': data['pars'][subject]['study']['FA'],
            'TR': data['pars'][subject]['study']['TR'],
            'TE': 0.001, #Estimate - awaiting data
            'field_strength': data['pars'][subject]['study']['field_strength'],
            'Nph': int(ts / data['pars'][subject]['study']['TR']),
            'Nk0': int(ts / data['pars'][subject]['study']['TR'] / 2),
            'S_ao': s_ao_int,
            'S_lk': s_lk_int,
            'S_rk': s_rk_int,
            'tS_ao': tint,
            'tS_lk': tint,
            'tS_rk': tint,
            'H': data['pars'][subject]['study']['H'],
            'vol_lk': data['pars'][subject]['study']['vol_lk'],
            'vol_rk': data['pars'][subject]['study']['vol_rk'],
            'weight': data['pars'][subject]['study']['weight'],
            'nmGFR': data['pars'][subject]['study']['nmGFR'] / 60,
        }

    return input_data