"""
.. _example_tristan_preclinical:

======================
Liver function in rats
======================

This example illustrates the use of `~dcmri.Liver` in a preclinical 
setting. The notebook replicates the findings from a series of 
multi-center experiments in rats performed by the 
`TRISTAN consortium <https://www.ihi.europa.eu/projects-results/project-factsheets/tristan>`_.

The studies aimed to validate the use 
of gadoxetate uptake and excretion rates in liver as a biomarker 
for drug-induced inhibition of liver function.

Data were collected in rats with and without administration of a drug 
to test if the inhibition is detectable. Scans were performed on 
preclinical MRI scanners at 3 different centers and 2 different field strengths. 
"""

# %%
# Setup
# -----

# %% 
# Import the required packages
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pydmr
import dcmri as dc


# %%
# The basic modelling approach is justified in a :ref:`separate 
# case study <example_gadoxetate_rat>` so we just replicate it here.
# We first define the helper function that builds a data dictionary 
# for a given study:

def get_data(study, subject, visit):

    # --- Load the data
    data = dc.load_tristan_rats(study, subject, visit)

    # --- Define time points for simulation
    dt = 0.5
    tmax = np.amax(data['tS_li']) + 120
    t = np.arange(0, tmax, dt)

    # --- Constants
    const = {
        'ci_li': dc.tristan_rat(t, BAT=data['BAT'], duration=data['duration']),
        'dt': dt,
        'H': 0.418, # Cremer et al, JCBFM 3, 254-256 (1983)
        'v_e_li': 0.23,
        'v_li': 1.0,
        'pfree': {'k_e2h':[0, 1], 'T_h': [0, 60 * 60]},
    }

    return data | const

# %%
# Next we build a helper function that computes the results for all 
# data in a given study:

def compute_liver_function(project):

    # --- Configure the model
    liver = dc.InverseLiver(kinetics='1I-IC-HF', calibrate=True)

    # --- Fit all datasets and save results in a list
    records = []
    for file in dc.fetch(project):
        data = pydmr.read(file, 'nest')
        for subj, visits in data['rois'].items():
            for visit in visits:
                study = Path(file).stem[:-4]
                inputs = get_data(study, subj, visit)
                result = liver(inputs) 
                visit_data = data['pars'][subj][visit]

                for parameter, value in result['pars'].items():
                    records.append({
                        'subject': subj,
                        'study': visit_data['study'],
                        'visit': visit_data['visit'],
                        'parameter': parameter,
                        'value': value,
                    })

    # --- Return as a dataframe for analysis
    return pd.DataFrame(records)

# %%
# The 6-compound study
# --------------------
# This section replicates key results from
# `Melillo et al (2023) <https://doi.org/10.3390/pharmaceutics15030896>`_.  
# The study determined the effect of 6 test drugs on liver function as 
# measured by gadoxetate uptake and excretion in healthy rats. 

# %% 
# We start by fitting all data in the study. The result is a data 
# table in long format:

results = compute_liver_function('tristan_rats_healthy_six_drugs')
print(results.to_string())


# %%
# We visualise the results by plotting the drug 
# effect for all subjects, and for both biomarkers: uptake rate ``k_e2h`` and 
# excretion rate ``k_h2b``:

studies = {5: 'Asunaprevir', 10: 'Bosentan', 8: 'Cyclosporine',
           7: 'Ketoconazole', 6: 'Pioglitazone', 12: 'Rifampicin'}

clr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
fs = 10
scale = 6000  # rate constants in mL/min/100mL
rows = [('k_e2h', 300), ('k_h2b', 30)]   # (parameter, y-axis maximum)

fig, ax = plt.subplots(len(rows), len(studies), figsize=(9, 8), sharey='row')
fig.subplots_adjust(wspace=0.2, hspace=0.1)

for col, (study, drug) in enumerate(studies.items()):
    ax[0, col].set_title(drug, fontsize=fs, pad=10)
    study_data = results[results.study == study]

    for row, (par, _) in enumerate(rows):
        # subjects x visits table for this parameter
        vals = (study_data[study_data.parameter == par]
                .pivot(index='subject', columns='visit', values='value'))
        for subj, v in vals.iterrows():
            ax[row, col].plot(v.index, scale * v, '-o', ms=6, label=subj,
                              color=clr[int(subj[-2:]) - 1])

# Axis formatting (y-axes are shared along each row)
for row, (par, ymax) in enumerate(rows):
    ax[row, 0].set_ylim(0, ymax)
    ax[row, 0].set_ylabel(f'{par} (mL/min/100cm3)', fontsize=fs)
    ax[row, 0].tick_params(axis='y', labelsize=fs)
for a in ax.flat:
    a.tick_params(labelbottom=False)

plt.show()

# %%
# We summarise the key findings from the study using a traffic-light 
# type visualisation:
# 
# - **red** means the inhibition is more than 20% (i.e. upper value of 
#   the 95% CI is less than -20%).
# - **orange** means the inhibition is less than 20% (i.e. upper value 
#   of the 95% CI is less than 0%).
# - **green** means no inhibition was detected with 95% confidence 
#   (i.e. 0% lies in the 95% CI).

params = ['k_e2h', 'k_h2b']

def effect_color(mean, ci):
    """Colour by the upper end of the 95% CI of the effect (in %)."""
    upper = mean + ci
    if upper < -20:
        return 'tab:red'
    if upper < 0:
        return 'tab:orange'
    return 'tab:green'

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
fig.subplots_adjust(left=0.3, right=0.7, wspace=0.25)
for ax, par in zip(axes, params):
    ax.set_title(f'{par} effect (%)', fontsize=fs, pad=10)
    ax.set_xlim(-100, 50)
    ax.grid(which='major', axis='x', linestyle='-')

# Loop over all studies
for study, drug in studies.items():
    study_data = results[results.study == study]

    # One pivot for both visits: columns are (visit, parameter)
    wide = study_data.pivot_table(index='subject', columns=['visit', 'parameter'],
                                  values='value')

    # Effect of the drug in % (subjects x parameters)
    effect = 100 * (wide[2] - wide[1]) / wide[1]

    # Mean effect and 95% CI on the mean
    mean = effect.mean()
    ci = 1.96 * effect.sem()

    for ax, par in zip(axes, params):
        ax.errorbar(mean[par], drug, xerr=ci[par], fmt='o',
                    color=effect_color(mean[par], ci[par]))

# Legend (proxy handles instead of dummy out-of-range points)
legend = [Line2D([], [], marker='o', ls='', color=c, label=label)
          for c, label in [('tab:red', 'inhibition > 20%'),
                           ('tab:orange', 'inhibition'),
                           ('tab:green', 'no inhibition')]]
axes[1].legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# %%
# These results confirm the key findings from the paper: 
# 
# - Two of the tested drugs (rifampicin and cyclosporine) showed 
#   strong inhibition of both uptake and excretion. 
# - One drug (ketoconazole) inhibits uptake but not excretion. 
# - Three drugs (pioglitazone, bosentan and asunaprevir) inhibit 
#   excretion but not uptake.

# %%
# Reproducibility study
# ---------------------
# This section replicates key results from 
# `Gunwhy et al (2024) <https://doi.org/10.1007/s10334-024-01192-5>`_. 
# The study aimed to identify the main sources of
# variability in DCE-MRI biomarkers of hepatocellular function in rats. This was
# done by comparing data measured at 3 different centres and 2 field strengths, at
# different days in the same subjects, and over the course of several months
# in the same centre. 

# %%
# Fit all data in the study:
results = compute_liver_function('tristan_rats_healthy_reproducibility')

# %%
# We visualise the results by plotting the 95% confidence intervals 
# for the two main biomarkers per substudy. The mean and 95% CI on the 
# mean are shown in color for reference. 

# Customise plot settings
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["axes.labelsize"] = 50
plt.rcParams["axes.titlesize"] = 50
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["axes.titleweight"] = 'bold'
plt.rcParams["font.weight"] = 'bold'
plt.rc('axes', linewidth=2)
plt.rc('xtick', labelsize=40)
plt.rc('ytick', labelsize=40)
plt.rcParams["lines.linewidth"] = 4
plt.rcParams['lines.markersize'] = 12

# Create list of biomarkers (parameters) of interest
params = ['k_e2h', 'k_h2b']

# Extract data of interest, i.e., visit 1 data for parameters of interest
data = results.query('parameter in @params and visit==1')
data['value'] *= 6000 # to units of mL/min/100cm3

# Get statistical summaries per parameter and study group
stat_summary = data.groupby(['parameter', 'study'])['value'].agg(['mean'])

# Calculate benchmark values per parameter by averaging all study group averages
benchmarks = stat_summary.groupby(['parameter'])['mean'].agg(['mean', 'sem'])

# Calculate the 95% confidence intervals for each parameter benchmark
benchmarks['CI95'] = benchmarks['sem'].mul(1.96)

# Panel order, left to right, with matching y-limits and labels
order = ['k_e2h', 'k_h2b']
ylims = [(0, 300), (0, 30)]
ylabels = ['$k_{e2h}$', '$k_{h2b}$']

g = sns.catplot(data=data,
                x='study',
                y='value',
                col='parameter',
                col_order=order,
                kind='point',
                capsize=0.2,
                sharey=False,
                linestyle='none',
                height=14,
                aspect=1.2,
                color='k',
                errorbar=('ci', 95))

g.set_titles("")

# Overlay y-labels, y-limits and benchmark lines on each panel
for ax, par, ylim, ylabel in zip(g.axes[0], order, ylims, ylabels):
    mean = benchmarks.loc[par, 'mean']
    ci = benchmarks.loc[par, 'CI95']
    ax.set(ylim=ylim)
    ax.set_ylabel(f"{ylabel} [mL/min/100cm3]")
    ax.axhline(mean, color='blue', ls=':')
    ax.axhline(mean - ci, color='red', ls='--')
    ax.axhline(mean + ci, color='red', ls='--')

plt.tight_layout()
plt.show()

# %%
# The results replicate the main finding from `Gunwhy et al (2024) <https://doi.org/10.1007/s10334-024-01192-5>`_., 
# that there are substantial study-dependent biases in absolute values.

# %%
# Multiple dosing study
# ---------------------
# This section replicates key results from Montelius et al (2021).
# The study aimed to identify the difference between acute and chronic dosing 
# effects of inhibitor drugs (Rifampicin, Cyclosporine, or Bosentan). 
#
# *Reference*
# 
# Mikael Montelius, Steven Sourbron, Nicola Melillo, Daniel Scotcher, 
# Aleksandra Galetin, Gunnar Schuetz, Claudia Green, Edvin Johansson, 
# John C. Waterton, and Paul Hockings. Acute and chronic rifampicin effect on 
# gadoxetate uptake in rats using gadoxetate DCE-MRI. Int Soc Mag Reson Med 
# 2021; 2674.

# %%
# Fit all data in the study:
results = compute_liver_function('tristan_rats_healthy_multiple_dosing')

# %%
# Now let's plot the biomarker values across visits for each of the 
# chronic dosing studies:

STUDY = {14: 'Rifampicin', 15: 'Cyclosporine', 11: 'Bosentan'}

plt.rcParams.update({
    'axes.titlesize': 25, 'axes.labelsize': 20,
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'font.weight': 'bold',
    'axes.linewidth': 1.5,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'lines.linewidth': 1.5, 'lines.markersize': 2,
})

# parameter -> y-axis limits, in row order (top to bottom)
rows = {'k_e2h': (0, 300), 'k_h2b': (0, 30)}

# Rate constants in units of mL/min/100cm3
data = (results[results.parameter.isin(list(rows))]
        .assign(value=lambda d: 6000 * d['value']))

g = sns.catplot(data=data,
                x='visit', y='value',
                hue='subject', palette='rocket',
                row='parameter', row_order=list(rows),
                col='study',
                kind='point',
                sharey='row',
                legend=False)

# One title per column, on the top row only
g.set_titles("")
for ax, study in zip(g.axes[0], g.col_names):
    ax.set_title(f"{STUDY[study]}", pad=15)

# One y-axis label and one set of limits per row
for ax_row, (par, ylim) in zip(g.axes, rows.items()):
    ax_row[0].set_ylabel(f"{par} [mL/min/100cm3]")
    ax_row[0].set_ylim(ylim)

plt.tight_layout()
plt.show()

# %%
# This replicates the findings from the original study that inhibition 
# happens in the acute stage, but that the effect is removed by biological 
# adaptation after chronic dosing.

# sphinx_gallery_start_ignore
# Choose the last image as a thumbnail for the gallery
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore
