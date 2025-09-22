#%%% header 
#! /usr/bin/env python3

import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

from rich import print
from rich.progress import track
from rich.traceback import install
from msp_tables import prepare_tables, log
from scipy.stats import linregress
from cmcrameri import cm as cmc

install(show_locals=True)
log.setLevel("ERROR")
bidsdir = '/mnt/bulk-vega/paulkuntke/mspaths'

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)  # or logging.ERROR, logging.CRITICAL
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def slope_per_patient(entry, colname):
    # convert dates to numeric (days since first date in group)
    x = (entry['encdate'] - entry['date'].min()).dt.days / 365.25  
    y = entry[colname].astype(float)
    if len(x) < 2 or y.nunique() == 1:
        return pd.Series({'slope_per_year': 0.0, 'intercept': y.iloc[0] if len(y)>0 else float('nan'), 'rvalue': 0.0})
    try:
        res = linregress(x, y)
    except ValueError:
        return pd.Series({'slope_per_year': 0.0, 'intercept': y.iloc[0] if len(y)>0 else float('nan'), 'rvalue': 0.0})
    return pd.Series({'slope_per_year': res.slope, 'intercept': res.intercept, 'rvalue': res.rvalue})



#%%% load Data
mri_df = pd.read_hdf('mri_data.hdf5')

with open("column_names.json", "r") as file:
    table_data = json.load(file)
prepared_tables = prepare_tables('888MS001 & 888MS002/Data Tables/', bidsdir, table_data)
prepared_tables_hc = prepare_tables('888MS005/Data Tables/', bidsdir, table_data)

neurological_df = pd.concat((prepared_tables["MSPT Neurological"],prepared_tables_hc["MSPT Neurological"] ))
patchar_df = pd.concat((prepared_tables['MSPT Patientcharacteristics'],prepared_tables_hc['MSPT Patientcharacteristics'] ))

#%%% Collect PST values

for clin_test in ["pst", "mdt_avg", "wst_avg", "cst_100", "cst_025"]:

    mri_baseline_df = mri_df.query('session_id == "ses-001"')[['mpi', 'date']]
    df = pd.merge(left=mri_baseline_df, right=neurological_df, on='mpi')[['mpi', 'date', 'encdate', clin_test]]
    df.drop_duplicates(inplace=True)
    df['encdate'] = pd.to_datetime(df.encdate, unit='s')


    df['distance'] = df['encdate'] - df['date']
    df['distance_days'] = df.distance.dt.days

    ax= sns.histplot(df, x="distance_days")
    ax.set_title(clin_test)

    plt.show()


    ax= sns.scatterplot(df, x='distance_days', y=clin_test, s=1)
    ax.set_title(clin_test)
    plt.show()

#%%% Collect PST values


for assessment in ["pst", "mdt_avg", "cst_100", "cst_025", "wst_avg"]:
    print(f"[bold]Showing data for {assessment}[/bold]")
    mri_baseline_df = mri_df.query('session_id == "ses-001"')[['mpi', 'date']]
    _df = pd.merge(left=mri_baseline_df, right=neurological_df, on='mpi')[['mpi', 'date', 'encdate', assessment]]
    _df.drop_duplicates(inplace=True)
    _df['encdate'] = pd.to_datetime(_df.encdate, unit='s')
    _df['distance'] = _df['encdate'] - _df['date']
    _df['distance_days'] = _df.distance.dt.days

    ax = sns.histplot(_df, x="distance_days")
    ax.set_title(assessment)
    plt.show()

    ax = sns.scatterplot(_df, x='distance_days', y=assessment, s=1)
    ax.set_title(assessment)
    plt.show()

    slopes = _df.groupby('mpi').apply(lambda x: slope_per_patient(x,assessment)).reset_index()
    slopes_df = pd.DataFrame(slopes)

    # ax = sns.barplot(data=slopes_df, x='mpi', y='slope_per_year')
    # ax.set_title(assessment)
    #plt.show()

    ax = sns.histplot(slopes_df, x='slope_per_year')
    ax.set_title(assessment)
    plt.show()

    slopes_df.to_csv(f'slope_{assessment}.csv')

mri_df["neuro_dist"] =  (pd.to_datetime(mri_df.neurological_date) - mri_df.date).dt.days
mri_df["neuro_dist_12m"] = (pd.to_datetime(mri_df.neurological_date_12m) - mri_df.date).dt.days
mri_df["neuro_dist_24m"] = (pd.to_datetime(mri_df.neurological_date_24m) - mri_df.date).dt.days


mri_df.rename(columns={
  'characteristics_date': 'characteristics_date_0m',
    'pdds_scr': 'pdds_scr_0m',
    'wst_avg': 'wst_avg_0m',
    'pst': 'pst_0m',
    'mdt_avg': 'mdt_avg_0m',
    'cst_100': 'cst_100_0m',
    'cst_025': 'cst_025_0m',

}, inplace=True) 


# Calculate deltas to get difference between baseline and follow-up-sessions
for score in  ['pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    for month in ["0m", "12m", "24m"]:
        mri_df[f'delta_{score}_{month}'] = mri_df[f'{score}_{month}'] - mri_df[f'{score}_0m']
    
long_df = pd.wide_to_long(mri_df,
                       stubnames=['characteristics_date', 'pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025'
                                  ],
                       i=['mpi',  'date'],
                       j='months',
                       sep='_',
                       suffix='\\d+m|0m|12m|24m').reset_index()


long_deltas_df = pd.wide_to_long(mri_df,
                       stubnames=['characteristics_date', 'delta_pdds_scr', 'delta_wst_avg', 'delta_pst', 'delta_mdt_avg', 'delta_cst_100', 'delta_cst_025'],
                       i=['mpi',  'date'],
                       j='months',
                       sep='_',
                       suffix='\\d+m|0m|12m|24m').reset_index()


unique_mpis = df['mpi'].drop_duplicates()
sample_mpis =  unique_mpis.sample(n=100, random_state=21)

sample_df = long_df[long_df['mpi'].isin(sample_mpis)].copy()   
sample_deltas_df = long_deltas_df[long_deltas_df['mpi'].isin(sample_mpis)].copy()        


 


# %%% 

sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_12m")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_24m")

# %%

# Working with pst
# Now generate HC-mean and use 2 STD to define CI Patients



hc_df = mri_df.query('group == "controls" and session_id == "ses-001"')

test_columns = ["pst_0m", "mdt_avg_0m", "wst_avg_0m", "cst_100_0m", "cst_025_0m"]

zscore_mean = {}
zscore_std = {}
for c in test_columns:
    zscore_mean[c] = np.mean(hc_df[c])
    zscore_std[c] = np.std(hc_df[c]) # mri_df.pst


# Cognitive-Impaired: 2STD > HC

for c in test_columns:
    mri_df[f"{c}_CI"] = mri_df[c] >= zscore_mean[c] +  1.5 * zscore_std[c]



# PDDS

mri_df["pdds_bin_wheelchair_0m"] = mri_df.pdds_scr_0m >= 7 # mark when wheelchair is needed
mri_df["pdds_bin_assistance_0m"] = mri_df.pdds_scr_0m >= 4 # mark from early cane as assistance
mri_df["pdds_bin_assistance_12m"] = mri_df.pdds_scr_12m >= 4 # mark from early cane as assistance
mri_df["pdds_bin_wheelchair_12m"] = mri_df.pdds_scr_12m >= 7 # mark when wheelchair is needed





# %% 
# # No

for c in test_columns:
    ax = sns.countplot(data=mri_df, x=f'{c}_CI', hue=f'{c}_CI')
    ax.set_title(f'Patients are Cognitive impaired on {c}')
    plt.show()





# %%%% Export

mri_df.query('session_id == "ses-001"')[[
    "mpi", 
    "pdds_scr_0m", 
    "pdds_scr_12m", 
    "pdds_scr_24m", 
    "pdds_bin_assistance_0m",
    "pdds_bin_wheelchair_0m", 
    "pdds_bin_assistance_12m", 
    "pdds_bin_wheelchair_12m"
    ]]

# %%%



     # keep all rows for them
 

# %%%

# sns.lineplot(data=mri_df, )

colortable = sns.color_palette("gray", n_colors=150)
colortable_hc = sns.dark_palette("#17C", n_colors=10)

for  score in ['pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    plt.figure(figsize=(8,6))
    sns.lineplot(data=sample_df, x="months", y=score, hue="mpi", marker="o", alpha=0.2, linewidth=1.5, palette=colortable, legend=False)
    sns.lineplot(data=long_df, x="months", y=score, hue="group",  marker="o", alpha=1, linewidth=1.5, legend=True)
    plt.xlabel("Time")
    plt.ylabel(f"{score}")
    plt.title(f"Progression of {score} over time")
    plt.tight_layout()
    plt.show()

# %%%


# TODO: Plot these with deltas from baseline.


         # keep all rows for them
# %%%
for  score in ['pdds_scr' , 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    print(f"Show delta for {score}")
    plt.figure(figsize=(8,6))
    sns.lineplot(data=sample_deltas_df, x="months", y=f"delta_{score}", hue="mpi", marker="o", alpha=0.2, linewidth=1.5, palette=colortable, legend=False)
    sns.lineplot(data=long_deltas_df, x="months", y=f"delta_{score}", hue="group",  marker="o", alpha=1, linewidth=1.5, legend=True)
    plt.xlabel("Time")
    plt.ylabel(f"delta {score}")
    plt.title(f"Progression of {score} over time")
    plt.tight_layout()
    plt.show()


# %%% Now add further timepoints
# Load Clinical Tests to get all timepoints

# neurological_df

sns.set_style("whitegrid")
neuro_df = pd.merge(
    left=mri_df.query('session_id == "ses-001"')[["mpi", "sex", "group", "birthyear", "mstype", "date", "characteristics_date_0m", "neurological_date", "mdt_avg_0m", "pst_0m", "wst_avg_0m", "cst_100_0m", "cst_025_0m", "pdds_scr_0m" ]],
    right=neurological_df,
    on='mpi',
)
neuro_df['neurological_date'] = pd.to_datetime(neuro_df.neurological_date)
neuro_df['date'] = pd.to_datetime(neuro_df.neurological_date)
neuro_df['encdate'] = pd.to_datetime(neuro_df.encdate, unit='s')

neuro_df["dist_to_baseline"] = neuro_df["encdate"] - neuro_df["date"]
neuro_df['dist_to_baseline_days'] = neuro_df["dist_to_baseline"].dt.days


for score in [ 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    neuro_df[f'delta_{score}'] = neuro_df[score] - neuro_df[f'{score}_0m']
    neuro_df[f'delta_{score}'] =  pd.to_numeric(neuro_df[f'delta_{score}'], errors='coerce').astype(float)

unique_mpis = neuro_df.query('group=="patients"')['mpi'].drop_duplicates()
sample_mpis =  unique_mpis.sample(n=100, random_state=42)
sample_df = neuro_df[neuro_df['mpi'].isin(sample_mpis)].copy()   

unique_mpis_hc = neuro_df.query('group=="controls"')['mpi'].drop_duplicates()
sample_mpis_hc =  unique_mpis_hc.sample(n=10, random_state=42)
sample_df_hc = neuro_df[neuro_df['mpi'].isin(sample_mpis_hc)].copy()   

cmap = cmc.berlin  # any cmcrameri colormap, e.g. 'vik', 'batlow', etc.
color1 = cmap(0.15)  # first color (float 0..1)
color2 = cmap(0.85)  # second color

# %%%
# Plot Test-Results of all datapoints
# compute text position: upper third of the axes in axis coordinates (y=~0.66)
text_x_axis = 0.3         # 80% from left
text_y_axis = 1.1        # ~upper third (0 bottom, 1 top)

# choose arrow/target point inside the axvspan (in data coords)
target_x = 0
# choose a y target slightly below top of data region near where you want arrow to point:

# add textbox with annotation
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.9)
arrow_props = dict(arrowstyle="->", color="black", linewidth=1.2)

for score in [ 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    plt.figure(figsize=(12,6)) 
    ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y=score, hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
    sns.lineplot(data=sample_df_hc, x="dist_to_baseline_days", y=score, hue="mpi", legend=False, palette=colortable_hc, markers=True, marker='o')
    sns.regplot(data=neuro_df.query('group=="controls"'), x="dist_to_baseline_days", y=score, scatter=False, color=color1, )
    sns.regplot(data=neuro_df.query('group=="patients"'), x="dist_to_baseline_days", y=score, scatter=False, color=color2)
    target_y = 0.9 * ax.get_ylim()[1]  # or pick a data y-value inside the span

    # create proxy legend handles for the regplot lines
    ctrl_line = mlines.Line2D([], [], color=color1, linestyle='-', label='controls')
    pat_line  = mlines.Line2D([], [], color=color2, linestyle='-', label='patients')

    # get existing handles (e.g., from hue) if you want to include them:
    handles, labels = ax.get_legend_handles_labels()

    # append the new handles and show legend
    handles.extend([ctrl_line, pat_line])
    ax.legend(handles=handles, labels=[*labels, 'controls', 'patients'], loc='best')

    ax.set_xlabel('distance to baseline MRI [days]')
    ax.set_ylabel(f'{score}')
    ax.axvspan(-90, 90, color='gray', alpha=0.3)


    ann = ax.annotate(
        f"Window of baseline\n {score} values",
        xy=(target_x, target_y),            # point being pointed to (data coords)
        xytext=(text_x_axis, text_y_axis),  # text position (can be axes coords)
        xycoords='data',                    # xy is in data coords
        textcoords='axes fraction',         # xytext is given in axes fraction
        ha='left',
        va='center',
        bbox=bbox_props,
        arrowprops=arrow_props,
        fontsize=10
    )

    # add drop shadow effect
    ann.get_bbox_patch().set_boxstyle("round,pad=0.4")
    ann.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace=(0,0,0), alpha=0.3),
        pe.Normal()
    ])



    plt.show()

# %%%
# Plot deltas from baseline

# compute text position: upper third of the axes in axis coordinates (y=~0.66)
text_x_axis = 0.2         # 80% from left
text_y_axis = 0.8        # ~upper third (0 bottom, 1 top)

# choose arrow/target point inside the axvspan (in data coords)
target_x = 0
# choose a y target slightly below top of data region near where you want arrow to point:

# add textbox with annotation
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.9)
arrow_props = dict(arrowstyle="->", color="black", linewidth=1.2)


for score in [ 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    plt.figure(figsize=(12,6)) 
    ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y=f'delta_{score}', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
    sns.lineplot(data=sample_df_hc, x="dist_to_baseline_days", y=f'delta_{score}', hue="mpi", legend=False, palette=colortable_hc, markers=True, marker='o')
    sns.regplot(data=neuro_df.query('group=="controls"'), x="dist_to_baseline_days", y=f'delta_{score}', scatter=False, color=color1, )
    sns.regplot(data=neuro_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'delta_{score}', scatter=False, color=color2)
    target_y = 0.7 * ax.get_ylim()[1]  # or pick a data y-value inside the span

    # create proxy legend handles for the regplot lines
    ctrl_line = mlines.Line2D([], [], color=color1, linestyle='-', label='controls')
    pat_line  = mlines.Line2D([], [], color=color2, linestyle='-', label='patients')

    # get existing handles (e.g., from hue) if you want to include them:
    handles, labels = ax.get_legend_handles_labels()

    # append the new handles and show legend
    handles.extend([ctrl_line, pat_line])
    ax.legend(handles=handles, labels=[*labels, 'controls', 'patients'], loc='best')

    ax.set_xlabel('distance to baseline MRI [days]')
    ax.set_ylabel(f'Δ of {score} to value at baseline')
    ax.axvspan(-90, 90, color='gray', alpha=0.3)


    ann = ax.annotate(
        f"Window of baseline\n {score} values",
        xy=(target_x, target_y),            # point being pointed to (data coords)
        xytext=(text_x_axis, text_y_axis),  # text position (can be axes coords)
        xycoords='data',                    # xy is in data coords
        textcoords='axes fraction',         # xytext is given in axes fraction
        ha='left',
        va='center',
        bbox=bbox_props,
        arrowprops=arrow_props,
        fontsize=10
    )

    # add drop shadow effect
    ann.get_bbox_patch().set_boxstyle("round,pad=0.4")
    ann.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace=(0,0,0), alpha=0.3),
        pe.Normal()
    ])



    plt.show()





# %%%

characteristics_df = pd.merge(
    left=mri_df.query('session_id == "ses-001"')[["mpi", "sex", "group", "birthyear", "mstype", "date", "characteristics_date_0m", "neurological_date", "mdt_avg_0m", "pst_0m", "wst_avg_0m", "cst_100_0m", "cst_025_0m", "pdds_scr_0m" ]],
    right=patchar_df,
    on='mpi',
)

characteristics_df['characteristics_date_0m'] = pd.to_datetime(characteristics_df['characteristics_date_0m'])
characteristics_df['date'] = pd.to_datetime(characteristics_df.date) 
characteristics_df['encdate'] = pd.to_datetime(neuro_df.encdate, unit='s')

characteristics_df["dist_to_baseline"] = characteristics_df["encdate"] - characteristics_df["date"]
characteristics_df['dist_to_baseline_days'] = characteristics_df["dist_to_baseline"].dt.days


characteristics_df.drop_duplicates(subset=["mpi", "encdate"], keep='last', inplace=True)




for score in [ 'pdds_scr']:
    characteristics_df[f'delta_{score}'] = characteristics_df[score] - characteristics_df[f'{score}_0m']
    characteristics_df[f'delta_{score}'] =  pd.to_numeric(characteristics_df[f'delta_{score}'], errors='coerce').astype(float)



unique_mpis = characteristics_df.query('group=="patients"')['mpi'].drop_duplicates()
sample_mpis =  unique_mpis.sample(n=100, random_state=42)
sample_df = characteristics_df[characteristics_df['mpi'].isin(sample_mpis)].copy()   

unique_mpis_hc = characteristics_df.query('group=="controls"')['mpi'].drop_duplicates()
sample_mpis_hc =  unique_mpis_hc.sample(n=10, random_state=42)
sample_df_hc = characteristics_df[characteristics_df['mpi'].isin(sample_mpis_hc)].copy()   


for score in [ 'pdds_scr']:
    plt.figure(figsize=(12,6)) 
    ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y=score, hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
    sns.lineplot(data=sample_df_hc, x="dist_to_baseline_days", y=score, hue="mpi", legend=False, palette=colortable_hc, markers=True, marker='o')
    sns.regplot(data=characteristics_df.query('group=="controls"'), x="dist_to_baseline_days", y=score, scatter=False, color=color1, )
    sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=score, scatter=False, color=color2)
    target_y = 0.9 * ax.get_ylim()[1]  # or pick a data y-value inside the span

    # create proxy legend handles for the regplot lines
    ctrl_line = mlines.Line2D([], [], color=color1, linestyle='-', label='controls')
    pat_line  = mlines.Line2D([], [], color=color2, linestyle='-', label='patients')

    # get existing handles (e.g., from hue) if you want to include them:
    handles, labels = ax.get_legend_handles_labels()

    # append the new handles and show legend
    handles.extend([ctrl_line, pat_line])
    ax.legend(handles=handles, labels=[*labels, 'controls', 'patients'], loc='best')

    ax.set_xlabel('distance to baseline MRI [days]')
    ax.set_ylabel(f'{score}')

    ax.axvspan(-90, 90, color='gray', alpha=0.3)


    ann = ax.annotate(
        f"Window of baseline\n {score} values",
        xy=(target_x, target_y),            # point being pointed to (data coords)
        xytext=(text_x_axis, text_y_axis),  # text position (can be axes coords)
        xycoords='data',                    # xy is in data coords
        textcoords='axes fraction',         # xytext is given in axes fraction
        ha='left',
        va='center',
        bbox=bbox_props,
        arrowprops=arrow_props,
        fontsize=10
    )

    # add drop shadow effect
    ann.get_bbox_patch().set_boxstyle("round,pad=0.4")
    ann.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace=(0,0,0), alpha=0.3),
        pe.Normal()
    ])



    plt.show()

# %%%


for score in [ 'pdds_scr']:
    plt.figure(figsize=(12,6)) 
    ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y=f'delta_{score}', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
    sns.lineplot(data=sample_df_hc, x="dist_to_baseline_days", y=f'delta_{score}', hue="mpi", legend=False, palette=colortable_hc, markers=True, marker='o')
    sns.regplot(data=characteristics_df.query('group=="controls"'), x="dist_to_baseline_days", y=f'delta_{score}', scatter=False, color=color1, )
    sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'delta_{score}', scatter=False, color=color2)
    target_y = 0.9 * ax.get_ylim()[1]  # or pick a data y-value inside the span

    # create proxy legend handles for the regplot lines
    ctrl_line = mlines.Line2D([], [], color=color1, linestyle='-', label='controls')
    pat_line  = mlines.Line2D([], [], color=color2, linestyle='-', label='patients')

    # get existing handles (e.g., from hue) if you want to include them:
    handles, labels = ax.get_legend_handles_labels()

    # append the new handles and show legend
    handles.extend([ctrl_line, pat_line])
    ax.legend(handles=handles, labels=[*labels, 'controls', 'patients'], loc='best')

    ax.set_xlabel('distance to baseline MRI [days]')
    ax.set_ylabel(f'Δ of {score} to value at baseline')
    ax.axvspan(-90, 90, color='gray', alpha=0.3)


    ann = ax.annotate(
        f"Window of baseline\n {score} values",
        xy=(target_x, target_y),            # point being pointed to (data coords)
        xytext=(text_x_axis, text_y_axis),  # text position (can be axes coords)
        xycoords='data',                    # xy is in data coords
        textcoords='axes fraction',         # xytext is given in axes fraction
        ha='left',
        va='center',
        bbox=bbox_props,
        arrowprops=arrow_props,
        fontsize=10
    )

    # add drop shadow effect
    ann.get_bbox_patch().set_boxstyle("round,pad=0.4")
    ann.get_bbox_patch().set_path_effects([
        pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace=(0,0,0), alpha=0.3),
        pe.Normal()
    ])



    plt.show()


# %%%%
# Find patient that progress into wheelchair
# pdds  >= 7 

wheelchair_patients = characteristics_df.query("pdds_scr >= 7").mpi.drop_duplicates().to_list()


sample_df = characteristics_df[characteristics_df['mpi'].isin(random.sample(wheelchair_patients, 100, ))].copy()   

plt.figure(figsize=(12,6)) 
ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y=f'{score}', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
sns.regplot(data=characteristics_df.query('mpi in @wheelchair_patients'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color='#010101', )
# sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color=color2)
# get existing handles (e.g., from hue) if you want to include them:

ax.set_xlabel('distance to baseline MRI [days]')
ax.set_ylabel(f'PDDS score')

ax.set_title(f' {len(wheelchair_patients)} Patients with wheelchair (pdds >=7 ) at any timepoint')



plt.show()


# %%%

# find patients that need no wheelchair at baseline mri but progress into wheelchair after the MRI

progressing_patients = characteristics_df.query("pdds_scr >= 7 and pdds_scr_0m < 7 and dist_to_baseline_days > 0").mpi.drop_duplicates().to_list()


sample_df = characteristics_df[characteristics_df['mpi'].isin(random.sample(progressing_patients, 100))].copy()   

plt.figure(figsize=(12,6)) 
ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y='pdds_scr', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
sns.regplot(data=characteristics_df.query('mpi in @progressing_patients'), x="dist_to_baseline_days", y='pdds_scr', scatter=False, color=color2, )
# sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color=color2)
# get existing handles (e.g., from hue) if you want to include them:

ax.set_xlabel('distance to baseline MRI [days]')
ax.set_ylabel(f'pdds')
ax.set_title(f' {len(progressing_patients)} Patients progress into wheelchair (pdds >=7 ) ')
plt.show()


plt.figure(figsize=(12,6)) 
ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y='delta_pdds_scr', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
sns.regplot(data=characteristics_df.query('mpi in @progressing_patients'), x="dist_to_baseline_days", y='delta_pdds_scr', scatter=False, color=color2, )
# sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color=color2)
# get existing handles (e.g., from hue) if you want to include them:

ax.set_xlabel('distance to baseline MRI [days]')
ax.set_ylabel(f'Δ of {score} to value at baseline')

ax.set_title(f' {len(progressing_patients)} Patients progress into wheelchair (pdds >=7 ) ')



plt.show()



# %%%

# find patients that progress after baseline mri by pdds of two

progressing_patients = characteristics_df.query("delta_pdds_scr >= 2 and dist_to_baseline_days > 0").mpi.drop_duplicates().to_list()
sample_df = characteristics_df[characteristics_df['mpi'].isin(random.sample(progressing_patients, 100))].copy()   

plt.figure(figsize=(12,6)) 
ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y='pdds_scr', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
sns.regplot(data=characteristics_df.query('mpi in @progressing_patients'), x="dist_to_baseline_days", y='pdds_scr', scatter=False, color=color2, )
# sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color=color2)
# get existing handles (e.g., from hue) if you want to include them:

ax.set_xlabel('distance to baseline MRI [days]')
ax.set_ylabel(f'pdds')
ax.set_title(f' {len(progressing_patients)} Patients show PDDS progress (dPDDS > 2 )')
plt.show()


plt.figure(figsize=(12,6)) 
ax = sns.lineplot(data=sample_df, x="dist_to_baseline_days", y='delta_pdds_scr', hue="mpi", legend=False, palette=colortable, markers=True, marker='o')
sns.regplot(data=characteristics_df.query('mpi in @progressing_patients'), x="dist_to_baseline_days", y='delta_pdds_scr', scatter=False, color=color2, )
# sns.regplot(data=characteristics_df.query('group=="patients"'), x="dist_to_baseline_days", y=f'{score}', scatter=False, color=color2)
# get existing handles (e.g., from hue) if you want to include them:

ax.set_xlabel('distance to baseline MRI [days]')
ax.set_ylabel(f'Δ of {score} to value at baseline')

ax.set_title(f' {len(progressing_patients)} Patients have  ΔPDDS > 2 after baseline MRI ')



plt.show()

# %%%

""" md
# Find Progressors 

Progressors are those with  e.g. 25% worsening p.a. 


"""


def extract_mpi(neuro_df):
    # Ensure dates are in datetime format
    neuro_df['neurological_date'] = pd.to_datetime(neuro_df['neurological_date'])
    neuro_df['encdate'] = pd.to_datetime(neuro_df['encdate'])
    
    # Sort by mpi and encdate
    neuro_df = neuro_df.sort_values(['mpi', 'encdate'])
    
    # Initialize lists to store results
    condition1_mpi = []
    condition2_mpi = []
    
    # Group by mpi and process each group
    for mpi, group in neuro_df.groupby('mpi'):
        # Get baseline mdt_avg and date
        baseline_date = group['neurological_date'].iloc[0]
        baseline_mdt = group.loc[group['encdate'] == baseline_date, 'mdt_avg'].iloc[0]
        
        # Filter data points after baseline
        post_baseline = group[group['encdate'] > baseline_date]
        
        # Condition 1: Any time point with >=25% worsening after baseline
        if not post_baseline.empty:
            post_baseline['change'] = post_baseline['mdt_avg'] / baseline_mdt
            if post_baseline['change'].max() >= 1.25:
                condition1_mpi.append(mpi)
        
        # Condition 2: Worsening rate of >=25% per year
        if len(post_baseline) >= 2:
            # Calculate time in years from baseline
            post_baseline['time_years'] = (post_baseline['encdate'] - baseline_date).dt.days / 365.25
            
            # Prepare data for regression
            X = post_baseline[['time_years']]
            y = post_baseline['mdt_avg']
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate slope (annual change rate)
            slope = model.coef_[0]
            
            # Check if the slope indicates >=25% worsening per year
            if slope >= 0.25 * baseline_mdt:
                condition2_mpi.append(mpi)
    
    return condition1_mpi, condition2_mpi



neuro_df = neuro_df.sort_values(['mpi', 'encdate'])


mpis_risen_by_25 = []
mpis_annual_change_25 = []

for mpi, group in track(neuro_df.groupby('mpi')):
    # print(group)
    
    baseline_date = group['neurological_date'].iloc[0]
    baseline_mdt = group['mdt_avg_0m'].iloc[0]

    post_baseline = group[group['encdate'] > baseline_date]


    if not post_baseline.empty:
        post_baseline['change'] = post_baseline['mdt_avg'] / baseline_mdt
        if post_baseline['change'].max() >= 1.25:
            mpis_risen_by_25.append(mpi)

    if not pd.isna(baseline_mdt):
        
         