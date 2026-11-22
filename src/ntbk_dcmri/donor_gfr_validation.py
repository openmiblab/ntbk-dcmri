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
from scipy import stats

from joblib import Parallel, delayed
import dcmri as dc
import pydmr



from ntbk_dcmri.utils.data_loaders import load_spogis


def run(datafile, outputdir):
    fit_all_data(datafile, outputdir, parallel=True)
    concatenate_results(outputdir)
    gfr_scatter_plot(datafile, outputdir)
    gfr_bland_altman_plot(datafile, outputdir)


def fit_all_data(datafile, outputdir, parallel=True):
    all_data = load_spogis(datafile)
    if parallel:
        Parallel(n_jobs=-1)(delayed(fit_single_subj)(subj, data, outputdir) for subj, data in all_data.items())
    else:
        [fit_single_subj(subj, data, outputdir) for subj, data in all_data.items()]


def fit_single_subj(subj, data, outputdir):
    # dc.InverseAortaKidneys.print_configs()

    model = dc.InverseAortaKidneys(
        bolus_dispersion='None',
        heartlung='chain',        
        organs='2cxm',          
        kidneys='2CF',    
        baseline='literature',    
        sequence='3D-SPGR-SS', 
        # sequence='ZTE-3D-SPGR-SS',
        # t2s_relaxation_ao=None,   
        # t2s_relaxation_lk=None,   
        # t2s_relaxation_rk=None,  
        estimate_bat='custom',  
    )
    #model.print_inputs()

    # Initial values
    data |= {
        'TF': 0.1,
        'Tb': 10, 
        'Db': 0.01,

        'CO': 100, 
        'T_hl': 10, 
        'D_hl': 0.1,
        'T_b_or': 20, 
        'T_e_or': 120, 
        'E_or': 0.15, 
        'fCO_ki': 0.25, 
        'DRPF': 0.5, 
        'T_u_lk': 120, 
        'T_u_rk': 120, 
        'FF_lk': 0.1, 
        'FF_rk': 0.1, 
        'v_p_lk': 0.15, 
        'v_p_rk': 0.15,  
        # 'vol_ao': 100, # no effect as expected
    }

    data['pfree'] = {
        'Tb': (0, 30), 
        'Db': (0.0, 0.3),

        'BAT': (-30, 30), 
        'CO': (75, 125), 
        'T_hl': (0, 60), 
        'D_hl': (0.0, 1.0),
        'T_b_or': (0, 60), 
        'T_e_or': (0, 300), 
        'E_or': (0, 0.5), 
        'fCO_ki': (0.2, 0.3), 
        'DRPF': (0, 1), 
        'T_u_lk': (30, 1200), 
        'T_u_rk': (30, 1200), 
        'FF_lk': (0, 1.0), 
        'FF_rk': (0, 1.0), 
        'v_p_lk': (0, 0.2), 
        'v_p_rk': (0, 0.2),  
    }

    # data['pfree'] = {k: v for k, v in model.pfree().items() if k not in ['T_lka', 'T_rka']}
    #input_data['pfree']['T_hl'] = (0, 10)
    # data['B1corr_ao'] = 0.5 - reduces all values
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

    # Subject label next to each dot
    for subj, row in valid.iterrows():
        ax.annotate(
            str(subj), (row['nmGFR'], row['GFR']),
            xytext=(4, 4), textcoords='offset points',
            fontsize=8, clip_on=True,
        )

    # Identity line over the full range
    ax.plot([0, 200], [0, 200], color='gray', linestyle='--', linewidth=1)

    # Linear regression line (mr-GFR on nm-GFR)
    if len(valid) >= 3:
        fit = stats.linregress(valid['nmGFR'], valid['GFR'])
        x0, x1 = valid['nmGFR'].min(), valid['nmGFR'].max()
        xs = np.array([x0, x1])
        ax.plot(xs, fit.intercept + fit.slope * xs, color='C3', lw=1.5,
                label=(f'y = {fit.slope:.2f}x + {fit.intercept:.1f}\n'
                       f'R² = {fit.rvalue**2:.2f}, n = {len(valid)}'))
        ax.legend(loc='upper left', fontsize=9, frameon=False)

    #ax.set_xlim(0, 200)
    #ax.set_ylim(0, 200)
    ax.set_xlabel('nm-GFR (mL/min)')
    ax.set_ylabel('mr-GFR (mL/min)')
    ax.set_title('mr-GFR vs nm-GFR', fontsize=14, pad=15)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'gfr_vs_nmgfr.png'), dpi=150)
    plt.close(fig)

    return table



def gfr_bland_altman_plot(datafile, outputdir):
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

    # Bland-Altman quantities (only subjects with both values)
    valid = table.dropna()
    mean = (valid['GFR'] + valid['nmGFR']) / 2
    diff = valid['GFR'] - valid['nmGFR']  # mr-GFR minus nm-GFR

    bias = diff.mean()
    sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, color='C0', zorder=3)

    # Subject label next to each dot
    for subj in valid.index:
        ax.annotate(
            str(subj), (mean[subj], diff[subj]),
            xytext=(4, 4), textcoords='offset points',
            fontsize=8, clip_on=True,
        )

    # Zero line, bias and limits of agreement
    ax.axhline(0, color='lightgray', lw=1, zorder=1)
    ax.axhline(bias, color='C0', lw=1.2)
    for y in (lo, hi):
        ax.axhline(y, color='gray', lw=1, ls='--')
    for y, txt in [(hi, f'+1.96 SD: {hi:.1f}'),
                   (bias, f'bias: {bias:.1f}'),
                   (lo, f'-1.96 SD: {lo:.1f}')]:
        ax.text(0.99, y, txt, transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=9)

    ax.set_xlabel('Mean of mr-GFR and nm-GFR (mL/min)')
    ax.set_ylabel('mr-GFR - nm-GFR (mL/min)')
    ax.set_title('Bland-Altman: mr-GFR vs nm-GFR', fontsize=14, pad=15)

    # Keep the limit lines and their labels comfortably inside the axes
    pad = 0.15 * (hi - lo)
    ax.set_ylim(min(lo, diff.min()) - pad, max(hi, diff.max()) + pad)
    ax.set_xlim(0, None)

    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'gfr_bland_altman.png'), dpi=150)
    plt.close(fig)

    return table


if __name__ == '__main__':

    datafile = r'C:\Users\md1spsx\Documents\Data\dcmri\jakob_spogis\All_patients.dmr'
    outputdir = r'C:\Users\md1spsx\Documents\Data\dcmri\jakob_spogis\results'
    
    run(datafile, outputdir)
