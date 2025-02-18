import re, os, json
import subprocess
import numpy  as np
import pandas as pd
from   ax.service.ax_client import AxClient, ObjectiveProperties

# Collect Ax Trials Dfs ----

def get_ax_table(fp):
    _, exp, phn, model, phase, _  = fp.split('/')
    metadata = pd.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase}, index=[0] )

    x = AxClient.load_from_json_file(fp)
    data = x.get_trials_data_frame()
    data['best_trial'] = False
    data.loc[(data.trial_index == x.get_best_trial()[0]), 'best_trial'] = True

    data = (metadata
            .assign(join_on_this = True)
            .merge(data.assign(join_on_this = True), on='join_on_this')
            .drop(columns='join_on_this')
            )
    return data
    

ax_tables = subprocess.check_output('find ../data_* -name [v,d]nn.json', shell=True).decode('utf-8').split('\n')[0:-1] # slice because it ends in newline
ax_tables_vnn = [e for e in ax_tables if e[-8:] == 'vnn.json' ]
ax_tables_dnn = [e for e in ax_tables if e[-8:] == 'dnn.json' ]

ax_tables_vnn = pd.concat([get_ax_table(fp=e) for e in ax_tables_vnn]).reset_index(drop=True)
ax_tables_dnn = pd.concat([get_ax_table(fp=e) for e in ax_tables_dnn]).reset_index(drop=True)

ax_tables_vnn.to_parquet('./ax_tables_vnn.parquet', index=False)
ax_tables_dnn.to_parquet('./ax_tables_dnn.parquet', index=False)
