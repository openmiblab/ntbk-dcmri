"""
=========================================================
Clinical - rifampicin induced inhibition (short protocol)
=========================================================

This example illustrates the use of `~dcmri.AortaLiver` for joint fitting of 
aorta and liver signals to a whole-body model. The use case is provided by the 
liver work package of the 
`TRISTAN project <https://www.imi-tristan.eu/liver>`_  which develops imaging 
biomarkers for drug safety assessment. The data and analysis was first 
presented at the ISMRM in 2024 (Min et al 2024, manuscript in press). 

The data were acquired in the aorta and liver of 10 healthy volunteers with 
dynamic gadoxetate-enhanced MRI, before and after administration of a drug 
(rifampicin) which is known to inhibit liver function. The assessments were 
done on two separate visits at least 2 weeks apart. 

The research question was to what extent rifampicin inhibits gadoxetate uptake 
rate from the extracellular space into the liver hepatocytes 
(khe, mL/min/100mL) and excretion rate from hepatocytes to bile 
(kbh, mL/100mL/min). 

2 of the volunteers only had the baseline assessment, the other 8 volunteers 
completed the full study. The results showed consistent and strong inhibition 
of khe (95%) and kbh (40%) by rifampicin. This implies that rifampicin poses 
a risk of drug-drug interactions (DDI), meaning it can cause another drug to 
circulate in the body for far longer than expected, potentially causing harm 
or raising a need for dose adjustment.

**Note**: this example is different to the 2 scan example of the same study 
in that this uses only the first scan to fit the model. 

Reference
--------- 

Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, Ebony Gunwhy, 
Gerry Kenna, Nicola Melillo, Geoff JM Parker, Gunnar Schuetz, Daniel Scotcher, 
John Waterton, Ian Rowe, and Steven Sourbron. *Measurement of liver function 
with dynamic gadoxetate-enhanced MRI: a validation study in healthy 
volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.
"""

# %%
# Setup
# -----

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import dcmri as dc

# %%
# Data analysis
# -------------

# --- Configure the model
aol = dc.InverseAortaLiver(
    heartlung='pfcomp',         # Smoother first pass data
    organs='2cxm',              # Long acquisition times reveal extravasation
    liver='1I-IC',              # Intracellular agent
    baseline='measured',        # Data are available on precontrast T1
    sequence='ZTE-3D-SPGR-SS',  # Echo times are assumed to be zero
    t2s_relaxation_ao=None,     # No T2* relaxation with TE=0
    t2s_relaxation_li=None,     # No T2* relaxation with TE=0
    calibrate=True,             # Derive signal scaling factor from data
    estimate_bat='ttp',      # Use the dedicated BAT estimator
)

# --- Load all datasets
study = 'tristan_humans_healthy_rifampicin'

data = {
    visit: dc.load_tristan_humans(study, visits=[visit], scans=[1])
    for visit in ['control', 'drug']
}

# --- Fit all datasets and save results in a list
records = []
for subj in data['control'].keys():
    for visit in ['control', 'drug']:
        result = aol(data[visit][subj], verbose=2) 

        for parameter, value in result['pars'].items():
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
    khe = [6000 * v1.at[s,'k_e2h']]
    kbh = [6000 * v1.at[s,'k_h2b']] 
    if s in v2.index:
        x += ['rifampicin']
        khe += [6000 * v2.at[s,'k_e2h']]
        kbh += [6000 * v2.at[s,'k_h2b']] 
    color = clr[i - 1]
    ax1.plot(x, khe, '-', label=s, marker='o', markersize=6, color=color)
    ax2.plot(x, kbh, '-', label=s, marker='o', markersize=6, color=color)

plt.show()

# sphinx_gallery_start_ignore
# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore
