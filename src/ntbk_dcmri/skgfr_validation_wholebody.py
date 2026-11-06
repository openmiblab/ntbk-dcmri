"""
========================================
Single-kidney glomerular filtration rate
========================================

This example illustrates the use of `~dcmri.Kidney` for measurement of 
single-kidney glomerular filtration rate (SK-GFR). 

The script uses data from a validation study comparing MRI-derived 
measurement of SK-GFR against reference measurements performed with 
radio-isotopes (Basak et al 2018). The study used 124 historical 
datasets collected in between the years 2000 and 2010 at 1 Tesla and 
3 Tesla MRI. 

The study was funded by 
`Kidney Research UK <https://www.kidneyresearchuk.org/>`_.

**Reference**

Basak S, Buckley DL, Chrysochou C, Banerji A, Vassallo D, Odudu A, Kalra PA, 
Sourbron SP. Analytical validation of single-kidney glomerular filtration 
rate and split renal function as measured with magnetic resonance renography. 
Magn Reson Imaging. 2019 Jun;59:53-60. doi: 10.1016/j.mri.2019.03.005. 
`[URL] <https://pubmed.ncbi.nlm.nih.gov/30849485/>`_.
"""

# %%
# Setup
# -----
import os

# --- Import packages
from joblib import delayed, Parallel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc


def run(outputdir):
    os.makedirs(outputdir, exist_ok=True)

    model = dc.InverseAortaKidneys(
        heartlung='chain',         # Smoother first pass data
        organs='2cxm',              # Long acquisition times reveal extravasation
        kidneys='2CF',              # Intracellular agent
        baseline='measured',        # Data are available on precontrast T1
        sequence='ZTE-3D-SPGR-SS',
        t2s_relaxation_ao=None,     # No T2* relaxation with TE=0
        t2s_relaxation_lk=None,     # No T2* relaxation with TE=0
        t2s_relaxation_rk=None,     # No T2* relaxation with TE=0
        estimate_bat='custom',      # Use the dedicated BAT estimator
    )

    results = fit_all_data(model, outputdir, parallel=True)

    # print(results.to_string())

    plot_skgfr(results, outputdir, 'skgfr_all.png')
    plot_skgfr(results, outputdir, 'skgfr_detail.png', vmax=80)

    print_error(results)



def fit_all_data(model, outputdir, parallel=True):
    records = []

    for visit in ['pre', 'post']:
        data = dc.load_kruk(visit=visit)

        if parallel:
            out = Parallel(n_jobs=-1)(
                delayed(fit_subject_data)(model, visit, subj, vals)
                for subj, vals in data.items()
            )
        else:
            out = [
                fit_subject_data(model, visit, subj, vals)
                for subj, vals in data.items()
            ]

        # Flatten the list of lists
        for r in out:
            records.extend(r)

    # Export long data
    csvfile = os.path.join(outputdir, 'results_long.csv')
    results = pd.DataFrame(records)
    results.to_csv(csvfile)

    # Export wide data
    csvfile = os.path.join(outputdir, 'results_wide.csv')
    wide_results = pd.pivot_table(results, values='value', columns='parameter', index=['subject', 'visit'])
    wide_results.to_csv(csvfile)

    return results



def fit_subject_data(model, visit, subj, vals):
    records = []
    if vals is None:
        return records

    print(f"Fitting visit {visit}, subject {subj}..")

    result = model(vals, verbose=0)

    info = {
        'subject': subj,
        'visit': visit,
        'field_strength': vals['field_strength'],
    }
    for parameter, value in result['pars'].items():
        records.append(info | {'parameter': parameter, 'value': value})

    for parameter in ['nmGFR_lk', 'nmGFR_rk']:
        records.append(info | {'parameter': parameter, 'value': vals[parameter]})

    return records


def extract_skgfr(results):
    # --- Separate SK-GFR values
    v1T = pd.pivot_table(results[results.field_strength==1], values='value', columns='parameter', index=['subject', 'visit'])
    v3T = pd.pivot_table(results[results.field_strength==3], values='value', columns='parameter', index=['subject', 'visit'])

    iso1T, iso3T = 60 * np.concatenate((v1T['nmGFR_lk'].values, v1T['nmGFR_rk'].values)), 60 * np.concatenate((v3T['nmGFR_lk'].values, v3T['nmGFR_rk'].values))
    mri1T, mri3T = 60 * np.concatenate((v1T['GFR_lk'].values, v1T['GFR_rk'].values)), 60 * np.concatenate((v3T['GFR_lk'].values, v3T['GFR_rk'].values))

    return {
        'iso1T': iso1T, 'iso3T': iso3T,
        'mri1T': mri1T, 'mri3T': mri3T,
    }


def plot_skgfr(results, outputdir, filename, vmax=None):
    pngfile = os.path.join(outputdir, filename)

    skgfr = extract_skgfr(results)
    # Plot MRI values and reference values

    # --- Plot full range
    if vmax is None:
        vmax = np.amax(np.concatenate((skgfr['iso1T'], skgfr['mri1T'], skgfr['iso3T'], skgfr['mri3T'])))

    plt.title('Single-kidney GFR (SK-GFR)')
    plt.plot(skgfr['iso1T'], skgfr['mri1T'], 'bo', linestyle='None', markersize=4, label='1T')
    plt.plot(skgfr['iso3T'], skgfr['mri3T'], 'ro', linestyle='None', markersize=4, label='3T')
    plt.plot(skgfr['iso3T'], skgfr['iso3T'], linestyle='-', color='black')
    plt.ylabel("MRI SK-GFR (mL/min)")
    plt.xlabel("Isotope SK-GFR (mL/min)")
    plt.xlim(0, vmax)
    plt.ylim(0, vmax)
    plt.legend()

    plt.savefig(pngfile, dpi=150)
    plt.close()



def print_error(results):

    v = pd.pivot_table(results, values='value', columns='parameter', index=['subject','visit'])

    iso = 60 * np.concatenate((v['nmGFR_lk'].values, v['nmGFR_rk'].values))
    mri = 60 * np.concatenate((v['GFR_lk'].values, v['GFR_rk'].values))

    diff = mri-iso
    bias = round(np.mean(diff), 0)
    bias_err = round(1.96 * np.std(diff) / np.sqrt(np.size(diff)), 0)
    err =  round(1.96 * np.std(diff), 0)

    def print_section(title):
        print(f"\n{title}\n{'-' * len(title)}")

    print_section('Single-kidney GFR')
    print(f"Bias of an MRI-based SK-GFR measurement: {bias:.2f} ± {bias_err:.2f} mL/min")  # paper: 0.56
    print(f"Error on a single bias-corrected SK-GFR measurement (95% CI): ± {err:.1f} mL/min")  # paper: [-28, 29]




if __name__ == '__main__':

    outputdir = r'C:\Users\md1spsx\Documents\Data\dcmri\skgfr_validation_wholebody'

    run(outputdir)
