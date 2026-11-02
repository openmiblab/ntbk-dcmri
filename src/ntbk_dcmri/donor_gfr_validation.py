"""
=======================
SK-GFR in kidney donors
=======================

Data received from Jakob Spogis
"""

# %%
# Setup
# -----

# --- Import packages
import os
from pathlib import Path

# import matplotlib
# matplotlib.use("TkAgg")   # or "QtAgg" if you want to keep Qt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import dcmri as dc
import pydmr



from ntbk_dcmri.utils.data_loaders import load_spogis


def run(datafile, outputdir):
    fit_all_data(datafile, outputdir, parallel=True)
    concatenate_results(outputdir)
    gfr_scatter_plot(datafile, outputdir)



def fit_all_data(datafile, outputdir, parallel=True):
    all_data = load_spogis(datafile)
    if parallel:
        Parallel(n_jobs=-1)(delayed(fit_single_subj)(subj, data, outputdir) for subj, data in all_data.items())
    else:
        [fit_single_subj(subj, data, outputdir) for subj, data in all_data.items()]


def fit_single_subj(subj, data, outputdir):
    # dc.InverseAortaKidneys.print_configs()

    model = dc.InverseAortaKidneys(
        heartlung='chain',        
        organs='2cxm',          
        kidneys='2CF',         
        baseline='literature',      
        sequence='ZTE-3D-SPGR-SS',
        t2s_relaxation_ao=None,   
        t2s_relaxation_lk=None,   
        t2s_relaxation_rk=None,  
        estimate_bat='ttp',  
        tof_corr=False,
    )
    # model.print_inputs()

    data['pfree'] = {k: v for k, v in model.pfree().items() if k not in ['T_lka', 'T_rka']}
    #input_data['pfree']['T_hl'] = (0, 10)
    result = model(data, verbose=2)

    # Prepare output files
    pars_path = os.path.join(outputdir, 'Pars')
    plot_path = os.path.join(outputdir, 'Plots')
    pcov_path = os.path.join(outputdir, 'Cov')
    pvar_path = os.path.join(outputdir, 'Sdev')
    os.makedirs(pars_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(pcov_path, exist_ok=True)
    os.makedirs(pvar_path, exist_ok=True)

    plot_file = os.path.join(plot_path, 'subj_' + subj + '.png')
    plot_file_pcov = os.path.join(pcov_path, 'subj_' + subj + '.png')
    plot_file_pvar = os.path.join(pvar_path, 'subj_' + subj + '.png')
    pars_file = os.path.join(pars_path, 'subj_' + subj + '.dmr')

    # Create output
    model.save(result, pars_file, subj=subj, study='spogis kidney donors')
    model.plot(data, result, fname=plot_file, show=False)
    model.plot_pcov(result, plot_file_pcov, show=False, vmin=-0.01, vmax=0.01)
    model.plot_pvar(result, plot_file_pvar, show=False, xlim=[-3, 3])

    return model, result

    # model.print(result)

    # print('mrGFR (ml/min): ', result['pars']['GFR_lk'] * 60 + result['pars']['GFR_rk'] * 60)
    # print('nmGFR (ml/min): ', data['nmGFR'] * 60)


def concatenate_results(outputdir):
    all_outputdir = os.path.join(outputdir, 'all_results')

    all_results = os.path.join(all_outputdir, 'all_results')
    all_results_long = os.path.join(all_outputdir, 'all_results_long.csv')
    all_results_wide = os.path.join(all_outputdir, 'all_results_wide.csv')
    all_results_sdev_long = os.path.join(all_outputdir, 'sdev_all_results_long.csv')
    all_results_sdev_wide = os.path.join(all_outputdir, 'sdev_all_results_wide.csv')

    # Concatenate all numerical results
    subj_results = os.path.join(outputdir, 'Pars')
    pydmr.concat(subj_results, all_results)

    pydmr.pars_to_long(all_results, all_results_long)
    pydmr.pars_to_wide(all_results, all_results_wide)
    pydmr.sdev_to_long(all_results, all_results_sdev_long)
    pydmr.sdev_to_wide(all_results, all_results_sdev_wide)


def gfr_scatter_plot(datafile, outputdir):
    all_data = load_spogis(datafile)
    all_results_wide = os.path.join(outputdir, 'all_results', 'all_results_wide.csv')

    # Wide results: one row per subject, first column = subject id
    wide = pd.read_csv(all_results_wide, index_col=0, encoding='utf-8-sig')
    wide.index = wide.index.astype(str)  # match keys regardless of int/str

    # Build the two-column table, one row per subject
    rows = []
    for subj, params in all_data.items():
        key = str(subj)
        if key not in wide.index:
            continue  # subject missing from the results file
        rows.append({
            'subject': key,
            'nmGFR': 60 * params.get('nmGFR', np.nan),
            'GFR': 60 * wide.loc[key, 'GFR'],
        })

    table = pd.DataFrame(rows).set_index('subject')
    table = table.apply(pd.to_numeric, errors='coerce')
    table.to_csv(os.path.join(outputdir, 'gfr_vs_nmgfr.csv'))

    # Scatter plot (only subjects with both values)
    valid = table.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(valid['nmGFR'], valid['GFR'], color='C0')

    # Identity line over the full range
    ax.plot([0, 200], [0, 200], color='gray', linestyle='--', linewidth=1)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_xlabel('nm-GFR (mL/min)')
    ax.set_ylabel('mr-GFR (mL/min)')
    ax.set_title('mr-GFR vs nm-GFR', fontsize=14, pad=15)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'gfr_vs_nmgfr.png'), dpi=150)
    plt.close(fig)

    return table


if __name__ == '__main__':

    datafile = r'C:\Users\md1spsx\Documents\Data\dcmri\jakob_spogis\Patients_1_10_27.dmr'
    outputdir = r'C:\Users\md1spsx\Documents\Data\dcmri\jakob_spogis\results'
    
    run(datafile, outputdir)
