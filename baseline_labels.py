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
from sklearn.linear_model import LinearRegression
from datetime import datetime

from rich import print
from rich.progress import track, Progress
# from rich.traceback import install
from msp_tables import prepare_tables, log
from scipy.stats import linregress
from cmcrameri import cm as cmc

# install(show_locals=False, )
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
for score in  ['pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025', ]:
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
neuro_df['date'] = pd.to_datetime(neuro_df.date)
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

for score in [ 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025', 'z_pst', 'z_mdt', 'z_wst', 'z_cst']:
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


neuro_df = neuro_df.sort_values(['mpi', 'encdate'])


mpis_risen_by_25 = []
mpis_annual_change_25 = []

for mpi, group in track(neuro_df.groupby('mpi')):
    # print(group)
    
    baseline_date = group['neurological_date'].iloc[0]
    baseline_mdt = group['pst_0m'].iloc[0]

    post_baseline = group[group['encdate'] > baseline_date]
    post_baseline.dropna(inplace=True)

    if not post_baseline.empty:
        post_baseline['change'] = post_baseline['pst'] / baseline_mdt
        if post_baseline['change'].max() >= 1.25:
            mpis_risen_by_25.append(mpi)

    if len(post_baseline) >= 2:
        post_baseline['time_years'] = (post_baseline['encdate'] - baseline_date).dt.days / 365.25

        # Prepare data for regression
        X = post_baseline[['time_years']]
        y = post_baseline['pst']
            
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate slope (annual change rate)
        slope = model.coef_[0]
            
        # Check if the slope indicates >=25% worsening per year
        if slope >= 0.25 * baseline_mdt:
            mpis_annual_change_25.append(mpi)     
# %%%


plt.figure(figsize=(12,6))
ax = sns.lineplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1])[['encdate','pst']].dropna(),
            x='encdate',
            y='pst',
            markers=True,
            marker='o'
            )
sns.scatterplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.to_list()[1])[['date','pst_0m']].dropna(),
            x='date',
            y='pst_0m',
            color='red'
            )
ax.axvline( mri_df.query('mpi == "' + neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1]).mpi.to_list()[0] + '" and session_id == "ses-001" ')['date'].to_list()[0], color="black")

plt.show()

plt.figure(figsize=(12,6))
ax = sns.lineplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1])[['encdate','wst_avg']].dropna(),
            x='encdate',
            y='wst_avg',
            markers=True,
            marker='o'
            )
sns.scatterplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.to_list()[1])[['date','wst_avg_0m']].dropna(),
            x='date',
            y='wst_avg_0m',
            color='red'
            )
ax.axvline( mri_df.query('mpi == "' + neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1]).mpi.to_list()[0] + '" and session_id == "ses-001" ')['date'].to_list()[0], color="black")

plt.show()


plt.figure(figsize=(12,6))
ax = sns.lineplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1])[['encdate','mdt_avg']].dropna(),
            x='encdate',
            y='mdt_avg',
            markers=True,
            marker='o'
            )
sns.scatterplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.to_list()[1])[['date','mdt_avg_0m']].dropna(),
            x='date',
            y='mdt_avg_0m',
            color='red'
            )
ax.axvline( mri_df.query('mpi == "' + neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1]).mpi.to_list()[0] + '" and session_id == "ses-001" ')['date'].to_list()[0], color="black")

plt.show()


plt.figure(figsize=(12,6))
ax = sns.lineplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1])[['encdate','cst_025']].dropna(),
            x='encdate',
            y='cst_025',
            markers=True,
            marker='o'
            )
sns.scatterplot(data=neuro_df.groupby('mpi').get_group(neuro_df.mpi.to_list()[1])[['date','cst_025_0m']].dropna(),
            x='date',
            y='cst_025_0m',
            color='red'
            )
ax.axvline( mri_df.query('mpi == "' + neuro_df.groupby('mpi').get_group(neuro_df.mpi.drop_duplicates().to_list()[1]).mpi.to_list()[0] + '" and session_id == "ses-001" ')['date'].to_list()[0], color="black", alpha=0.9)
plt.show()

# %%%%
# Draw Z-Scores over Time


def plot_zscores(mpi, date_col='encdate', scores=None, show_mean=True):
    """
    Plot Z-Scores development for given MPI
    """

    if scores is None:
        scores = ['z_cst', 'z_pst', 'z_wst', 'z_mdt']


    # Determine mean-zscore (are they adjusted? => it's not in the datesetdescription)
    zscore_means = pd.DataFrame(neuro_df.groupby(by=['mpi', date_col])[scores].mean().mean(axis=1))
    zscore_means.columns = ['z_score_mean']
    fig = plt.figure(figsize=(12,6))
    try:

        data = neuro_df.groupby('mpi').get_group(mpi)[[date_col] + scores ]
    except:
        return None

    for score in scores:

       sns.lineplot(data=data,
                    x=date_col,
                    y=score,
                    markers=True,
                    marker='o',
                    label=score,
                    alpha=0.3,
                    ci=None
                    )

    if show_mean:
        sns.lineplot(data=zscore_means.query('mpi == @mpi'),
                    x=date_col,
                    y='z_score_mean',
                    color='grey',
                    markers=True,
                    marker='o',
                    label='mean',
                    dashes=False,
                    linestyle='--',
                    linewidth=2
                    )
    legend = plt.legend()
    ax = fig.get_axes()[0]
    ax.set_ylabel('z-score')
    ax.set_xlabel('date')
    ax.set_title(mpi)
    if date_col == 'encdate':
        ax.axvline( mri_df.query('mpi == @mpi and session_id == "ses-001" ')['date'].to_list()[0], color="grey", alpha=0.5, linestyle='--')
    elif date_col == "dist_to_baseline_days":
        ax.axvline(0, alpha=0.5, color="grey", linestyle="--")
    ax.axhline(0, alpha=0.5, color="grey")

    return fig

# %%%%

pat_mpis = mri_df.query('group=="patients"').mpi.drop_duplicates().to_list()

for mpi in random.sample(pat_mpis, 10):

    fig = plot_zscores(mpi, 'dist_to_baseline_days')
    if fig is None:
        continue
    ax = fig.get_axes()[0]
    ax.set_ylabel('z-score')
    ax.set_xlabel('Days since MRI')
    plt.show()


# %%%
# 
# Find pogression
# (1) "worst" progression == any patient who _at any time point after baseline_ reached threshold T (even if it became better after)
# (2) "last" progression == any patient who reached threshold T at their _last_ time point 
# (3) averaged == these measurements are a bit zig-zaggy, so you can smooth them by using different kernels, e.g. the [1/3, 1/3, 1/3] kernel, 
#      or an [1/4, 1/2, 1/4] kernel and then do the worst progressor
# (4) you can make a linear fit for each patient and everyone who's trendline goes up is a progressor
        

# def find_progression:


# Find worst progression at any Timepoint after baseline
for score in ["pst", "mdt", "wst", "cst"]:
    progressors = list(neuro_df.query(f'encdate > date and z_{score} < -2').mpi.unique())
    mri_df[f'worst_progression_{score}_2z'] = mri_df['mpi'].isin(progressors)
for score in ["pst", "mdt", "wst", "cst"]:
    progressors = list(neuro_df.query(f'encdate > date and z_{score} < -1.5').mpi.unique())
    mri_df[f'worst_progression_{score}_15z'] = mri_df['mpi'].isin(progressors)


# any patient who reached threshold T at their _last_ time point
for score in ["pst", "mdt", "wst", "cst"]:
    progressors = list(neuro_df.query('encdate > date').sort_values(by=["mpi", "encdate"]).groupby('mpi').last().query(f'z_{score} < -2').reset_index().mpi.unique())
    mri_df[f'last_progression_{score}'] = mri_df['mpi'].isin(progressors) 
for score in ["pst", "mdt", "wst", "cst"]:
    progressors = list(neuro_df.query('encdate > date').sort_values(by=["mpi", "encdate"]).groupby('mpi').last().query(f'z_{score} < -1.5').reset_index().mpi.unique())
    mri_df[f'last_progression_{score}'] = mri_df['mpi'].isin(progressors)
# smoothed_pst = last_entries['pst'].rolling(window=3, min_periods=1).mean()


# you can make a linear fit for each patient and everyone who's trendline goes up is a progressor
def fit_poly(g, score):
    
    x = g['dist_to_baseline_days'].values
    y = g[score].values
    n = len(x)
    if n < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return pd.Series({f'{score}_n': n, f'{score}_slope': np.nan, f'{score}_intercept': np.nan, f'{score}_r_squared': np.nan})
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
    return pd.Series({f'{score}_n': n, f'{score}_slope': slope, f'{score}_intercept': intercept, f'{score}_r_squared': r2})

for score in ["pst", "wst_avg", "mdt_avg", "cst_100", "cst_025"]:
    lin_fit = neuro_df.groupby('mpi').apply(lambda x: fit_poly(x, score)).reset_index().drop_duplicates(subset='mpi')
    mri_df = mri_df.merge(right=lin_fit, on='mpi', how='left')




# %%%


# %%%


print("Controls")
print(mri_df.query('group == "controls"').mdt_avg_slope.mean())
print(mri_df.query('group == "controls"').mdt_avg_slope.std())

print("Patients")
print(mri_df.query('group == "patients"').mdt_avg_slope.mean())
print(mri_df.query('group == "patients"').mdt_avg_slope.std())


# Now we mark all those with a slope of < -0.01 as progressors
mri_df[f'pst_slope_progressor'] = False
mri_df.loc[mri_df.query(f'group=="patients" and pst_slope < -0.01').index, f'pst_slope_progressor'] = True
mri_df[f'wst_slope_progressor'] = False
mri_df.loc[mri_df.query(f'group=="patients" and wst_avg_slope > 0.0').index, f'wst_slope_progressor'] = True
mri_df[f'mdt_slope_progressor'] = False
mri_df.loc[mri_df.query(f'group=="patients" and mdt_avg_slope < 0.0').index, f'mdt_slope_progressor'] = True
mri_df[f'cst_025_slope_progressor'] = False
mri_df.loc[mri_df.query(f'group=="patients" and cst_025_slope < 0.0').index, f'cst_025_slope_progressor'] = True
mri_df[f'cst_100_slope_progressor'] = False
mri_df.loc[mri_df.query(f'group=="patients" and cst_100_slope < 0.0').index, f'cst_100_slope_progressor'] = True
# %%
neuro_df['pst_smoothed_3'] = neuro_df.query('encdate > date').sort_values(by=["mpi", "encdate"]).groupby('mpi')['pst'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
neuro_df['pst_smoothed_3'] = neuro_df['pst_smoothed_3'].fillna(neuro_df['pst'])


for mpi in random.sample(pat_mpis, 10):
    fig = plot_zscores(mpi, 'dist_to_baseline_days', ['pst', 'pst_smoothed_3'], show_mean=False)
    if fig is None:
        continue
    ax = fig.get_axes()[0]
    ax.set_ylabel('pst')
    ax.set_xlabel('Days since MRI')
    plt.show()

mpi = "500004010"
plot_zscores(mpi, 'dist_to_baseline_days', ['pst', 'pst_smoothed_3'], show_mean=False)


