import re, os, json
import subprocess
# import polars as pl
import pandas as pd
from   ax.service.ax_client import AxClient, ObjectiveProperties


# Collect metrics (training histories) ----
# def get_metrics(fp):
#     _, exp, phn, model, phase, _, version, _ = fp.split('/')
#     metadata = pl.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase, 'version':version} )

#     data = pl.read_csv(fp)
#     # pivot longer by stacking the train/val data.
#     data = pl.concat([
#         (data
#             .with_columns(split = pl.lit('train'), loss = pl.col('train_loss'))
#             .drop('train_loss', 'val_loss')
#             .drop_nulls()
#         ),
#         (data
#             .with_columns(split = pl.lit('val'), loss = pl.col('val_loss'))
#             .drop('train_loss', 'val_loss')
#             .drop_nulls()
#         )
#     ])
#     # join in the metadata to make everything easy to query
#     data = (metadata
#             .with_columns(join_on_this = pl.lit(True))
#             .join(data.with_columns(join_on_this = pl.lit(True)), on=['join_on_this'])
#             .drop('join_on_this')
#             )
#     return data

def get_metrics(fp):
    _, exp, phn, model, phase, _, version, _ = fp.split('/')
    metadata = pd.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase, 'version':version}, index=[0] )
    data = pd.read_csv(fp)

    data = pd.concat([
        (data
        .assign(split = 'train')
        .rename(columns={'train_loss':'loss'})
        .drop(columns='val_loss')
        .dropna()),
        (data
        .assign(split = 'val')
        .rename(columns={'val_loss':'loss'})
        .drop(columns='train_loss')
        .dropna())
    ])

    data = (metadata
            .assign(join_on_this = True)
            .merge(data.assign(join_on_this = True), on='join_on_this')
            .drop(columns='join_on_this')
            )
    return data


# metics is all of the training histories. This includes both the hyperparameter tuning and training
metrics = [
    e for e in 
    subprocess.check_output('find ../data_* -name metrics.csv', shell=True).decode('utf-8').split('\n')
    if e != '']

# metrics = pl.concat([get_metrics(fp = e) for e in metrics])
metrics = pd.concat([get_metrics(fp = e) for e in metrics]).reset_index(drop=True)
metrics.to_parquet('./metrics.parquet', index=False)

# Collect Ax Trials Dfs ----

# def get_ax_table(fp):
#     _, exp, phn, model, phase, _  = fp.split('/')
#     metadata = pl.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase} )

#     data = pl.from_pandas(AxClient.load_from_json_file(fp).get_trials_data_frame())

#     data = (metadata
#             .with_columns(join_on_this = pl.lit(True))
#             .join(data.with_columns(join_on_this = pl.lit(True)), on=['join_on_this'])
#             .drop('join_on_this')
#             )
#     return data

def get_ax_table(fp):
    _, exp, phn, model, phase, _  = fp.split('/')
    metadata = pd.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase}, index=[0] )

    data = AxClient.load_from_json_file(fp).get_trials_data_frame()

    data = (metadata
            .assign(join_on_this = True)
            .merge(data.assign(join_on_this = True), on='join_on_this')
            .drop(columns='join_on_this')
            )
    return data
    

ax_tables = subprocess.check_output('find ../data_* -name [v,d]nn.json', shell=True).decode('utf-8').split('\n')[0:-1] # slice because it ends in newline
ax_tables_vnn = [e for e in ax_tables if e[-8:] == 'vnn.json' ]
ax_tables_dnn = [e for e in ax_tables if e[-8:] == 'dnn.json' ]

# ax_tables_vnn = pl.concat([get_ax_table(fp=e) for e in ax_tables_vnn])
# ax_tables_dnn = pl.concat([get_ax_table(fp=e) for e in ax_tables_dnn])

ax_tables_vnn = pd.concat([get_ax_table(fp=e) for e in ax_tables_vnn]).reset_index(drop=True)
ax_tables_dnn = pd.concat([get_ax_table(fp=e) for e in ax_tables_dnn]).reset_index(drop=True)

ax_tables_vnn.to_parquet('./ax_tables_vnn.parquet', index=False)
ax_tables_dnn.to_parquet('./ax_tables_dnn.parquet', index=False)


# Placeholder
# it might become useful to aggregate and examine the run settings for each trained model. 
# If that becomes true, then `hparams.yaml` is what I need and it's in the subdir /models/*/version_\d/ 
# e.g. sparsevnn_study/data_gmx/phno_OilDry/vnn/models/vnn/version_0/hparams.yaml