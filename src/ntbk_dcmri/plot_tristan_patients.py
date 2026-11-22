"""
=====================================================================
Clinical - rifampicin effect in subjects with impaired liver function
=====================================================================

The data show in this example aimed to demonstrates the effect of rifampicin 
on liver function of patients with impaired function. The use 
case is provided by the liver work package of the 
`TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging 
biomarkers for drug safety assessment. 

The data were acquired in the aorta and liver in 3 patients with 
dynamic gadoxetate-enhanced MRI. The study participants take rifampicin 
as part of their routine clinical workup, with an aim to promote their liver 
function. For this study, they were taken off rifampicin 3 days before the 
first scan, and placed back on rifampicin 3 days before the second scan. The 
aim was to determine the effect if rifampicin in uptake and 
excretion function of the liver.

The data confirmed that patients had significantly reduced uptake and excretion 
function in the absence of rifampicin. Rifampicin adminstration promoted their 
excretory function but had no effect on their uptake function. 

Reference
--------- 

Manuscript in preparation..
"""

# %%
# Setup
# -----

# --- Import packages
import pandas as pd
import matplotlib.pyplot as plt
import dcmri as dc

# --- Source data
study = 'tristan_humans_patients_rifampicin'

# %%
# Data analysis
# -------------

# --- Configure the model
aol = dc.InverseAortaLiverSplitDrug(
    heartlung='pfcomp',         # Smoother first pass data
    organs='2cxm',              # Long acquisition times reveal extravasation
    liver='1I-IC',              # Intracellular agent
    non_stationary='U',         # Uptake rate changes during the course of the experiment
    baseline='measured',        # Data are available on precontrast T1
    sequence='ZTE-3D-SPGR-SS',  # Echo times are assumed to be zero
    t2s_relaxation_ao=None,     # No T2* relaxation with TE=0
    t2s_relaxation_li=None,     # No T2* relaxation with TE=0
    calibrate=True,             # Derive signal scaling factor from data
    trigger=True,               # Signals are retrospectively triggered
    bolus='dual',               # Two independent injections performed
    estimate_bat='custom',      # Use the dedicated BAT estimator
)

# --- Load all datasets
data = dc.load_tristan_humans(study)

# --- Fit all datasets and save results in a list
records = []
for subj, vals in data.items():
    result = aol(vals, verbose=2) 

    for parameter, value in result['pars'].items():
        visit = dc.parse_varname(parameter)['index']
        visit = {1: 'control', 2: 'drug', None: 'both'}[visit]
        records.append({
            'subject': subj,
            'visit': visit,
            'parameter': parameter,
            'value': value,
        })

# --- Return as a dataframe for analysis
results = pd.DataFrame(records)


# %%
# Plot individual results
# -----------------------
# Now lets visualise the main results from the study by plotting the drug 
# effect for all volunteers, and for both biomarkers: uptake rate ``khe`` 
# and excretion rate ``kbh``:

# Set up the figure
clr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
fs = 10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
fig.subplots_adjust(wspace=0.5)
ax1.set_title('Hepatocellular uptake rate', fontsize=fs, pad=10)
ax1.set_ylabel('khe (mL/min/100mL)', fontsize=fs)
ax1.set_ylim(0, 60)
ax1.tick_params(axis='x', labelsize=fs)
ax1.tick_params(axis='y', labelsize=fs)
ax2.set_title('Biliary excretion rate', fontsize=fs, pad=10)
ax2.set_ylabel('kbh (mL/min/100mL)', fontsize=fs)
ax2.set_ylim(0, 6)
ax2.tick_params(axis='x', labelsize=fs)
ax2.tick_params(axis='y', labelsize=fs)

# Pivot data for both visits to wide format for easy access:
v1 = results[results.visit=='control']
v2 = results[results.visit=='drug']
v1 = v1.pivot(index='subject', columns='parameter', values='value')
v2 = v2.pivot(index='subject', columns='parameter', values='value')

# Plot the rate constants in units of mL/min/100mL
for i, s in enumerate(v1.index):
    x = ['baseline']
    khe = [6000 * v1.at[s, 'k_1_e2h']]
    kbh = [6000 * v1.at[s, 'k_1_h2b']] 
    if s in v2.index:
        x += ['rifampicin']
        khe += [6000 * v2.at[s, 'k_2_e2h']]
        kbh += [6000 * v2.at[s, 'k_2_h2b']] 
    color = clr[i - 1]
    ax1.plot(x, khe, '-', label=s, marker='o', markersize=6, color=color)
    ax2.plot(x, kbh, '-', label=s, marker='o', markersize=6, color=color)
plt.show()

# sphinx_gallery_start_ignore
# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore
