#%%% header 
#! /usr/bin/env python3

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
from rich.traceback import install
from msp_tables import prepare_tables, log
from scipy.stats import linregress
from cmcrameri import cm

install(show_locals=True)
log.setLevel("ERROR")
bidsdir = '/mnt/bulk-vega/paulkuntke/mspaths'



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


#%%% show available data

mri_df["neuro_dist"] =  (pd.to_datetime(mri_df.neurological_date) - mri_df.date).dt.days
mri_df["neuro_dist_12m"] = (pd.to_datetime(mri_df.neurological_date_12m) - mri_df.date).dt.days
mri_df["neuro_dist_24m"] = (pd.to_datetime(mri_df.neurological_date_24m) - mri_df.date).dt.days

sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_12m")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_24m")

# %%

# Working with pst
# Now generate HC-mean and use 2 STD to define CI Patients



hc_df = mri_df.query('group == "controls" and session_id == "ses-001"')

test_columns = ["pst", "mdt_avg", "wst_avg", "cst_100", "cst_025"]

zscore_mean = {}
zscore_std = {}
for c in test_columns:
    zscore_mean[c] = np.mean(hc_df[c])
    zscore_std[c] = np.std(hc_df[c]) # mri_df.pst


# Cognitive-Impaired: 2STD > HC

for c in test_columns:
    mri_df[f"{c}_CI"] = mri_df[c] >= zscore_mean[c] +  1.5 * zscore_std[c]



# PDDS

mri_df["pdds_bin_wheelchair"] = mri_df.pdds_scr >= 7 # mark when wheelchair is needed
mri_df["pdds_bin_assistance"] = mri_df.pdds_scr >= 4 # mark from early cane as assistance
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
    "pdds_scr", 
    "pdds_scr_12m", 
    "pdds_scr_24m", 
    "pdds_bin_assistance",
    "pdds_bin_wheelchair", 
    "pdds_bin_assistance_12m", 
    "pdds_bin_wheelchair_12m"
    ]]

# %%%


m""" ri_df.rename(columns={
  'characteristics_date': 'characteristics_date_0m',
    'pdds_scr': 'pdds_scr_0m',
    'wst_avg': 'wst_avg_0m',
    'pst': 'pst_0m',
    'mdt_avg': 'mdt_avg_0m',
    'cst_100': 'cst_100_0m',
    'cst_025': 'cst_025_0m',

}, inplace=True) """

long_df = pd.wide_to_long(mri_df,
                       stubnames=['characteristics_date', 'pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025'
                                  ],
                       i=['mpi',  'date'],
                       j='months',
                       sep='_',
                       suffix='\\d+m|0m|12m|24m').reset_index()



unique_mpis = df['mpi'].drop_duplicates()
sample_mpis =  unique_mpis.sample(n=100, random_state=21)
sample_df = long_df[long_df['mpi'].isin(sample_mpis)].copy()                 # keep all rows for them
 

# %%%

# sns.lineplot(data=mri_df, )

colortable = sns.color_palette("gray", n_colors=150)

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


# %%%


# TODO: Plot these with deltas from baseline.


for score in  ['pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    for month in ["0m", "12m", "24m"]:
        mri_df[f'delta_{score}_{month}'] = mri_df[f'{score}_{month}'] - mri_df[f'{score}_0m']
    

long_df = pd.wide_to_long(mri_df,
                       stubnames=['characteristics_date', 'delta_pdds_scr', 'delta_wst_avg', 'delta_pst', 'delta_mdt_avg', 'delta_cst_100', 'delta_cst_025'],
                       i=['mpi',  'date'],
                       j='months',
                       sep='_',
                       suffix='\\d+m|0m|12m|24m').reset_index()

# %%%
for  score in ['pdds_scr', 'wst_avg', 'pst', 'mdt_avg', 'cst_100', 'cst_025']:
    print(f"Show delta for {score}")
    plt.figure(figsize=(8,6))
    sns.lineplot(data=sample_df, x="months", y=f"delta_{score}", hue="mpi", marker="o", alpha=0.2, linewidth=1.5, palette=colortable, legend=False)
    sns.lineplot(data=long_df, x="months", y=f"delta_{score}", hue="group",  marker="o", alpha=1, linewidth=1.5, legend=True)
    plt.xlabel("Time")
    plt.ylabel(f"{score}")
    plt.title(f"Progression of {score} over time")
    plt.tight_layout()
    plt.show()


# %%% show trends for 100 patients

