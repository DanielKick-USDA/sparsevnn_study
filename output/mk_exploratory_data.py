import re, os, json
import subprocess
# import polars as pl
import numpy  as np
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

    # make sure there are the expected cols, even if they are nan
    if 'train_loss' not in list(data):
        data['train_loss'] = np.nan
    if 'val_loss' not in list(data):
        data['val_loss'] = np.nan

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


# find all the paths for the prediciton parquets

# use the file path to get information like
# hash
# model type 
# vnn or dnn
#   version number
#   do some cleanup to allow for merging later
    # fp = '../data_gmx/phno_OilDry/lin/models/blup/98c6637d04_bWGR_yhat.parquet'
    # fp = '../data_gmx/phno_OilDry/dnn/models/dnn/version_1/98c6637d04_yhat_trn.parquet'
    # fp = '../data_gmx/phno_OilDry/vnn/models/vnn/version_1/98c6637d04_yhat_val.parquet'


def get_yhat(fp):
    # figure out what kind of model we have. 
    # I'm using re instead of splitting the string because blups don't have version numbers
    # further I'm uising the parent dir so that we have exactly 3 characters (lin instead of blup/gwas/vnn/dnn)
    model = re.findall(r'.../models', fp)[0].split('/')[0]

    info = re.findall(r'[^/]+/[^/]+/.../models/.+', fp)[0]
    match model:
        case 'vnn' | 'dnn': 
            exp, phn, model, phase, _, version, file = info.split('/')
            uid_data, _, split = file.split('_')
            metadata = pd.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase, 'version':version, 'data_hash':uid_data}, index=[0] )
            data = pd.read_parquet(fp).drop(columns=['Phno_Idx']).drop_duplicates()  
            data = data.rename(columns= {e:e.lower() for e in list(data)})   
            yhat_col = [e for e in list(data) if e not in ['taxa', 'phno_idx']][0]
            data = data.rename(columns={yhat_col:'yhat'})
            
            # standarize naming so the merge works correctly
            data.loc[(data.split == 'Training'  ), 'split'] = 'train'
            data.loc[(data.split == 'Validation'), 'split'] = 'test'

        case 'lin':        
            exp, phn, model, phase, _,          file = info.split('/')
            uid_data, _, split = file.split('_')
            metadata = pd.DataFrame( {'exp':exp, 'phn':phn, 'model':model, 'phase':phase, 'version':'version_0', 'data_hash':uid_data}, index=[0] )
            data = pd.read_parquet(fp)
            data = data.rename(columns= {e:e.lower() for e in list(data)})
            obs_col = [e for e in list(data) if e not in ['taxa', 'phno_idx']][0]
            data = data.rename(columns={obs_col:'obs'})

    data = (metadata
            .assign(join_on_this = True)
            .merge(data.assign(join_on_this = True), on='join_on_this')
            .drop(columns='join_on_this')
            )

    return data


fps = (
    # blup predictions
    subprocess.check_output('find ../data_* -name *bWGR_yhat.parquet', shell=True).decode('utf-8').split('\n')[0:-1] + 
    # training set predictions
    subprocess.check_output('find ../data_* -name *yhat_trn\.parquet', shell=True).decode('utf-8').split('\n')[0:-1] +
    # validation set predictions
    subprocess.check_output('find ../data_* -name *yhat_val\.parquet', shell=True).decode('utf-8').split('\n')[0:-1]
    )

yhats = [get_yhat(fp = e) for e in fps]

yhat_vnn = pd.concat([e for e in yhats if e['model'][0] == 'vnn'])
yhat_dnn = pd.concat([e for e in yhats if e['model'][0] == 'dnn'])
yhat_lin = pd.concat([e for e in yhats if e['model'][0] == 'lin'])

yhat_vnn.to_parquet('./yhat_vnn.parquet', index=False)
yhat_dnn.to_parquet('./yhat_dnn.parquet', index=False)
yhat_lin.to_parquet('./yhat_lin.parquet', index=False)

# # how do we compare a hash?

# df = yhat_lin.loc[yhat_lin.data_hash == '98c6637d04' ].rename(columns={'yhat':'lin'})

# df = df.merge(
#     yhat_dnn.loc[:, ['exp', 'phn', 'split', 'taxa', 'data_hash', 'yhat']].rename(columns={'yhat':'dnn'})
# ).merge(
#     yhat_vnn.loc[:, ['exp', 'phn', 'split', 'taxa', 'data_hash', 'yhat']].rename(columns={'yhat':'vnn'})
# )

