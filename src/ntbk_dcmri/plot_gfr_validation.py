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

# --- Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcmri as dc


model = dc.InverseKidney(
    kinetics='2CFU',
    sequence='ZTE-3D-SPGR-SS', 
    t2s_relaxation=None, 
    aif='measured',
)

records = []
for visit in ['pre', 'post']:
    for kidney in ['LK', 'RK']:
        data = dc.load_kruk(visit=visit, kidney=kidney)
        for subj, vals in data.items():
            if vals is None:
                continue

            result = model(vals, verbose=2)

            info = {
                'subject': subj,
                'visit': visit,
                'kidney': kidney,
                'field_strength': vals['field_strength'],
            }
            for parameter, value in result['pars'].items():
                records.append(info | {
                    'parameter': parameter,
                    'value': value,
                })
            records.append(info | {
                'parameter': 'nmGFR',
                'value': vals['nmGFR'],
            })

# --- Return as a dataframe for analysis
results = pd.DataFrame(records)


# %%
# Plot MRI values and reference values

# --- Separate SK-GFR values
v1T = pd.pivot_table(results[results.field_strength==1], values='value', columns='parameter', index=['subject', 'kidney', 'visit'])
v3T = pd.pivot_table(results[results.field_strength==3], values='value', columns='parameter', index=['subject', 'kidney', 'visit'])

iso1T, iso3T = 60 * v1T['nmGFR'].values, 60 * v3T['nmGFR'].values
mri1T, mri3T = 60 * v1T['GFR'].values, 60 * v3T['GFR'].values

# --- Plot full range
vmax = np.amax(np.concatenate((iso1T, iso3T, mri1T, mri3T)))

plt.title('Single-kidney GFR (SK-GFR)')
plt.plot(iso1T, mri1T, 'bo', linestyle='None', markersize=4, label='1T')
plt.plot(iso3T, mri3T, 'ro', linestyle='None', markersize=4, label='3T')
plt.plot(iso3T, iso3T, linestyle='-', color='black')
plt.ylabel("MRI SK-GFR (mL/min)")
plt.xlabel("Isotope SK-GFR (mL/min)")
plt.xlim(0, vmax)
plt.ylim(0, vmax)
plt.legend()
plt.show()

# %%
# Plot a detail

vmax = 80

plt.title('Single-kidney GFR (SK-GFR)')
plt.plot(iso1T, mri1T, 'bo', linestyle='None', markersize=4, label='1T')
plt.plot(iso3T, mri3T, 'ro', linestyle='None', markersize=4, label='3T')
plt.plot(iso3T, iso3T, linestyle='-', color='black')
plt.ylabel("MRI SK-GFR (mL/min)")
plt.xlabel("Isotope SK-GFR (mL/min)")
plt.xlim(0, vmax)
plt.ylim(0, vmax)
plt.legend()
plt.show()


# %%
# Compute bias and accuracy

v = pd.pivot_table(results, values='value', columns='parameter', index=['subject','kidney','visit'])

iso = 60 * v['nmGFR'].values
mri = 60 * v['GFR'].values

diff = mri-iso
bias = round(np.mean(diff), 0)
bias_err = round(1.96 * np.std(diff) / np.sqrt(np.size(diff)), 0)
err =  round(1.96 * np.std(diff), 0)

print('-----------------')
print('Single-kidney GFR')
print('-----------------')
print(f"The bias in an MRI-based SK-GFR measurement is {bias} +/- {bias_err} ml/min") # paper 0.56
print(f"After bias correction, the error (95% CI) on a single SK-GFR measurement is +/- {err} mL/min") # paper [-28, 29]

# %%
# The results confirm the conclusion from the original study that 
# the precision of MR-derived SK-GFR with these historical data was 
# too low for clinical use. The exact numerical values are different 
# from those in the original study, showing the importance of 
# implementation detail.


# sphinx_gallery_start_ignore
# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = -1
# sphinx_gallery_end_ignore

