#%%% header 
#! /usr/bin/env python3

import pandas as pd
import seaborn as sns
from rich import print
from rich.progress import track
from msp_tables import prepare_tables, log


log.setLevel("NOTSET")


#%%% load Data
mri_df = pd.read_hdf('mri_data.hdf5')

with open("column_names.json", "r") as file:
    table_data = json.load(file)
prepared_tables = prepare_tables('888MS001 & 888MS002/Data Tables/', bidsdir, table_data)
prepared_tables_hc = prepare_tables('888MS005/Data Tables/', bidsdir, table_data)

#%%%


#%%% show available data

mri_df["neuro_dist"] =  (pd.to_datetime(mri_df.neurological_date) - mri_df.date).dt.days
mri_df["neuro_dist_12m"] = (pd.to_datetime(mri_df.neurological_date_12m) - mri_df.date).dt.days
mri_df["neuro_dist_24m"] = (pd.to_datetime(mri_df.neurological_date_24m) - mri_df.date).dt.days

sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_12m")
sns.histplot(mri_df.query('session_id =="ses-001"'), x="neuro_dist_24m")

