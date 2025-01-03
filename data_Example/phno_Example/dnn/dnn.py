import os, argparse, re, json, hashlib, pickle # M_list is pickled

import numpy  as np
import pandas as pd
import scipy.stats # for spearmanr
## Model building ====
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import Dataset, DataLoader
import einops
import sparsevnn
import sparsevnn.util
import sparsevnn.qol
from   sparsevnn.core import \
    SparseLinearCustom,      \
    dist_scale_function,     \
    info_list_to_layer_list, \
    SparseVNN,               \
    MarkerDataset,           \
    VNNHelper,               \
    structured_layer_info,   \
    plDNN_general

## Logging with Pytorch Lightning ====
import lightning.pytorch as pl
from   lightning.pytorch.loggers import CSVLogger # used to save the history of each trial (used by ax)

## Adaptive Experimentation Platform ====
from ax.service.ax_client import AxClient, ObjectiveProperties

# write out tables
import pyarrow as pa
import pyarrow.parquet as pq
# and plots
import plotly.express as px

from tqdm import tqdm

torch.set_float32_matmul_precision('medium')


# Default values are stored in quality of life.
params_data = sparsevnn.qol.params_data()
params_run  = sparsevnn.qol.params_run()
# params      = sparsevnn.qol.params()
# params_list = sparsevnn.qol.params_list()
params = {
    'size_out': 0,
    'size_in': 0,
    'hidden_layers': 1,
    'width': 256,
    'drop': 0,
    'width_decay_rate': 0,
    'drop_decay_rate': 0,
    'width_decay_reverse': 1,
    'drop_decay_reverse': 1,
    }
params_list = [
    {
    'name': 'size_out',
    'type': 'fixed',
    'value': 1, #NOTE This will be overwritten below to match x.shape[1]
    'value_type': 'int',
    'log_scale': False
    },
    {
    'name': 'size_in',
    'type': 'fixed',
    'value': 1, #NOTE This will be overwritten below to match y.shape[1]
    'value_type': 'int',
    'log_scale': False
    },
    # not over written
    {
    'name': 'hidden_layers',
    'type': 'range',
    'bounds': [0, 8],
    'value_type': 'int',
    'log_scale': False
    },
    {
    'name': 'width',
    'type': 'range',
    'bounds': [1, 2048],
    'value_type': 'int',
    'log_scale': False
    },
    {
    'name': 'drop',
    'type': 'range',
    'bounds': [0, 99], # int will be transformed to float after decay is applied
    'value_type': 'int',
    'log_scale': False
    },
    {
    'name': 'width_decay_rate',
    'type': 'choice',
    'values': [0+(0.1*i) for i in range(10)]+[1.+(1*i) for i in range(11)],
    'value_type': 'float',
    'is_ordered': True,
    'sort_values':True
    },
    {
    'name': 'drop_decay_rate',
    'type': 'choice',
    'values': [0+(0.1*i) for i in range(10)]+[1.+(1*i) for i in range(11)],
    'value_type': 'float',
    'is_ordered': True,
    'sort_values':True
    },
    {
    'name': 'width_decay_reverse',
    'type': 'choice',
    'values': [0, 1],
    'value_type': 'int',
    'is_ordered': True,
    'sort_values':True
    },
    {
    'name': 'drop_decay_reverse',
    'type': 'choice',
    'values': [0, 1],
    'value_type': 'int',
    'is_ordered': True,
    'sort_values':True
    }
]

# values in `params_data` and `params_run` are the only ones I expect to be updated
params_data_keys = set(params_data.keys()).copy()
params_run_keys  = set(params_run.keys()).copy()

#FIXME 
# params_run_keys |= {'train_name'}  

# Update Default Parameters ---------------------------------------------------
# Default < JSON < Arguments

parser = argparse.ArgumentParser()
# Note type bool does not interpret the text following the flag as a bool, rather _passing_ the flag is a bool test
# See https://stackoverflow.com/questions/60999816/argparse-not-parsing-boolean-arguments
# and https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
def s2b(v):
    "Credit to user [Maxim](https://stackoverflow.com/users/805502/maxim) "
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# JSON files for experiment settings
parser.add_argument("--params_data", type=str, help="path to dict of `params_data` json")
parser.add_argument("--params_run",  type=str, help="path to dict of `params_run` json")
parser.add_argument("--params",      type=str, help="path to dict of `params` json")
parser.add_argument("--params_list", type=str, help="path to dict of `params_list` json")
# Small scale changes from default values
## Data / Graph Str (most common expected changes) ====

# params_data settings
parser.add_argument("--species",          type=str, help="KEGG species code of the current organism.")
parser.add_argument("--num_nucleotides",  type=int, help="length of the nucleotide dimension (usually 4 or 1).")
parser.add_argument("--graph_cache_path", type=str, help="path to local graph files' storage.")
parser.add_argument("--graph_source",     type=str, help="source of the graph. 'kegg' and 'local' are expected.")
parser.add_argument("--graph_cxn",        type=str, help="path to local graph connection table if being used.")
parser.add_argument("--gff_path",         type=str, help="path to gene annotation gff.")
parser.add_argument("--hmp_path",         type=str, help="path to hapmap file.")
parser.add_argument("--phno_path",        type=str, help="path to phenotype file.")
parser.add_argument("--cache_path",       type=str, help="path to *this* file's cache.")
parser.add_argument("--model_path",       type=str, help="path to a trained model to be loaded.")

parser.add_argument("--dataloader_shuffle_train",type=s2b, help="should this dataloader shuffle data?")
parser.add_argument("--dataloader_shuffle_valid",type=s2b, help="should this dataloader shuffle data?")


# params_run settings
parser.add_argument("--use_data_cache", type=s2b, help="If true, attempt to load/cache data using settings hash in save name.")
parser.add_argument("--patch",         type=s2b, help="If true, functions in `vnn_patch.py` will be loaded in.")
parser.add_argument("--batch_size",    type=int, help="batch size for dataloader")
parser.add_argument("--max_epoch",     type=int, help="max epoch for training")
parser.add_argument("--run_mode",      type=str, help="script run to `tune`, `train`, `predict`, or `eval`?")
parser.add_argument("--tune_trials",   type=int, help="if `tune` how many trials should be run?")
parser.add_argument("--tune_max",      type=int, help="if `tune` what is the maximum number of trials to be run?")
parser.add_argument("--tune_force",    type=s2b, help="if `tune` should trials be run even if there are `tune_max` already?")
parser.add_argument("--train_from_ax", type=s2b, help="if `train` should the best params from an Ax be used??")
parser.add_argument("--train_save",    type=s2b, help="if `train` should the model be saved?")
parser.add_argument("--train_name",    type=str, help="What should the model be saved as? If not provided will default to 'version_\d'")
parser.add_argument("--eval",          type=str, nargs='*', action='append', help="")
# --eval can be passed multiple times to create a list of 0 or more evaluations to be run. 
# per https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# at the time of writing, valid options are `saliency_inp`, `saliency_wab`, `rho_out`

args = parser.parse_args()

# If  --eval a b        -> [[a, b]]
#     --eval a --eval b -> [[a], [b]]
# so as_lat ensures that instead we have a flat list as `eval`
def as_lat(inp):
    lat = lambda x: True if list not in [type(e) for e in x] else False
    while lat(inp) != True:
        # turn everything that isn't a list into one, then use sum to concat the lists
        inp = sum([[e] if type(e) != list else e for e in inp], [])
    return inp

if args.eval != None:
    args.eval = as_lat(args.eval)

## JSON parameter files =======================================================
# in place update all k/v if an external file was passed.
# if args.params_data: params_data |= sparsevnn.qol.read_json(json_path=args.params_data)
# if args.params_run:  params_run  |= sparsevnn.qol.read_json(json_path=args.params_run)
# if args.params:      params      |= sparsevnn.qol.read_json(json_path=args.params)
# if args.params_list: params_list |= sparsevnn.qol.read_json(json_path=args.params_list)

def _get_json_if_exists(
        path, 
        file, # File name or regex (e.g. '\.gff$')
        is_regex = False
        ):
    res = {}
    files = os.listdir(path=path)
    if is_regex:
        file = sorted([e for e in files if re.match(file, e)])
        if file != []:
            file = file[0] 
            res = sparsevnn.qol.read_json(json_path=path+file)
    elif file in files:
            res = sparsevnn.qol.read_json(json_path=path+file)

    if res != {}: print(f'Loading and using {path+file}.')
    return res


if args.params_data:  
    params_data |= sparsevnn.qol.read_json(json_path=args.params_data)
else: 
    params_data |= _get_json_if_exists( # This messy expression will use a user path if provided or will pull params_data
        path = args.cache_path if args.cache_path else params_data['cache_path'],  
        file = 'params_data.json', 
        is_regex = False)
    
if args.params_run:  
    params_run |= sparsevnn.qol.read_json(json_path=args.params_run)
else: 
    params_run |= _get_json_if_exists( # This messy expression will use a user path if provided or will pull params_run
        path = args.cache_path if args.cache_path else params_data['cache_path'],  
        file = 'params_run.json', 
        is_regex = False)

if args.params:  
    params |= sparsevnn.qol.read_json(json_path=args.params)
else: 
    params |= _get_json_if_exists( # This messy expression will use a user path if provided or will pull params
        path = args.cache_path if args.cache_path else params_data['cache_path'],  
        file = 'params.json', 
        is_regex = False)


# the params_list for Ax is not so simple to merge. We have to pull the file, create an index based on the name key of 
# the dictionaries and then update each matching dictionary. It's also possible (but unexpected) that new parameters
# could be added here. To account for that any new parameters will be appended. 
new_params = []
if args.params_list:  
    new_params = sparsevnn.qol.read_json(json_path=args.params_list)
else: 
    new_params = _get_json_if_exists( # This messy expression will use a user path if provided or will pull params_list
        path = args.cache_path if args.cache_path else params_data['cache_path'],  
        file = 'params_list.json', 
        is_regex = False)
if new_params != []:
    # find what entries have the same names, update them
    params_list_lookup = {e['name']:i for i,e in enumerate(params_list)}
    for entry in new_params:
        if entry['name'] in params_list_lookup:
            # index of dict in list
            i = params_list_lookup[entry['name']]
            # update that entry in the list
            params_list[i] |= entry
        else:
            # otherwise append to end
            params_list.append(entry)
    del params_list_lookup
        

# It'll be most common to change the input files. The other paramters have so many values the would be inconvenient to provide as args.
# if args.graph_cache_path: params_data['graph_cache_path'] = args.graph_cache_path
# if args.gff_path:         params_data['gff_path']         = args.gff_path
# if args.hmp_path:         params_data['hmp_path']         = args.hmp_path
# if args.phno_path:        params_data['phno_path']        = args.phno_path
# if args.cache_path:       params_data['cache_path']       = args.cache_path
# if args.model_path:       params_data['model_path']       = args.model_path

## Individudal Arguments ======================================================
# Update each dict with the approprate args
for k in params_data_keys:
    if hasattr(args, k):
        _ = getattr(args, k)
        if _ != None:
            params_data[k] = _

for k in params_run_keys: 
    if hasattr(args, k):
        _ = getattr(args, k)
        if _ != None:
            params_run[k] = _


# if not specified, look in the cache path for a file ending in csv, gff, hmp.txt
possible_files = os.listdir(params_data['cache_path'])
if params_data['gff_path'] == None:
    _ = sorted([e for e in possible_files if ((re.match('.*\.gff$', e) != None ) | (re.match('.*\.gff3$', e) != None))])
    if _ != []:
        _ = _[0]
        params_data['gff_path'] = _
        print(f'No `gff_path` specified. Proceeding with {_}.')

if params_data['hmp_path'] == None:
    _ = sorted([e for e in possible_files if re.match('.*\.hmp\.txt$', e)])
    if _ != []:
        _ = _[0]
        params_data['hmp_path'] = _
        print(f'No `hmp_path` specified. Proceeding with {_}.')

if params_data['phno_path'] == None:
    _ = sorted([e for e in possible_files if re.match('.*\.csv$', e)])
    if _ != []:
        _ = 'phno.csv' if 'phno.csv' in _ else _[0] # use phno.csv if it exists, otherwise guess
        params_data['phno_path'] = _
        print(f'No `phno_path` specified. Proceeding with {_}.')

## Store Intermediate Data For Reuse ==========================================
# use_data_cache_load = True
use_data_cache = params_run['use_data_cache'] # option to use/ignore vnn_cache
params_data_subset = {e:params_data[e] for e in [
 'species',
 'num_nucleotides',
 'graph_cache_path',
 'gff_path',
 'hmp_path',
 'phno_path',
 'graph_source',
 'kegg_catalog',
 'graph_cxn',   
 'holdout_taxa_ignore',
 'holdout_type',
 'holdout_percent',
 'holdout_seed',
 'holdout_taxa'
]}

if not use_data_cache:
    use_data_cache_load = False

elif use_data_cache:
    sparsevnn.qol.ensure_dir_path_exists(dir_path = './vnn_cache/')

    hash_lookup = {}
    if os.path.exists('./vnn_cache/hash_lookup.json'):
        with open('./vnn_cache/hash_lookup.json', 'r') as f:
            hash_lookup = json.load(f)

    # to ensure the hash is the same we sort the keys then use encode to get the stirng as bytes
    _ = json.dumps(params_data_subset, sort_keys=True).encode()
    # using shake_256 instead sha256 because https://stackoverflow.com/questions/4567089/hash-function-that-produces-short-hashes
    params_data_subset_hash = hashlib.shake_256( _ ).hexdigest(5)

    if params_data_subset_hash not in hash_lookup.keys():
        use_data_cache_load = False
        # add new hash
        hash_lookup |= {str(params_data_subset_hash):params_data_subset}
        sparsevnn.qol.write_json(hash_lookup, './vnn_cache/hash_lookup.json')
    else:
        use_data_cache_load = True


# for reference write out these param dicts
_ = [
    sparsevnn.qol.write_json(a, f"{params_data['cache_path']}{b}.json") 
    for a,b in zip([params_data, params_run, params, params_list],
                   ['params_data', 'params_run', 'params', 'params_list'])
                    ]


## Make Global Variables ======================================================

cache_path = params_data['cache_path']
if cache_path == './': # in this case we can't get a meaningful value unless we use the pwd
    cache_path = os.getcwd()+'/'
lightning_log_dir = cache_path+"lightning"
exp_name = [e for e in cache_path.split('/') if e != ''][-1]
sparsevnn.qol.ensure_dir_path_exists(dir_path = cache_path)


def read_pq_or_pd(file_path:str):
    "Using file extention read a parquet -> csv -> table"
    match file_path.split('.')[-1]:
        case 'parquet': out = pq.read_table(file_path).to_pandas()
        case 'csv': out = pd.read_csv(file_path)
        case _: out = pd.read_table(file_path)
    return out


# Load Input Data -------------------------------------------------------------
if use_data_cache_load:
    phno           = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_phno.parquet')
    obs_geno_lookup= read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_obs_geno_lookup.parquet')
    cxn            = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_cxn.parquet')
    acgt_loci      = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_acgt_loci.parquet')
    gene_nodes_gff = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_gene_nodes_gff.parquet')

    # acgt = np.load(f'./vnn_cache/{params_data_subset_hash}_acgt.npz')
    # acgt = acgt['acgt']
    # no longer using a npz so that these data can be read into r as well (with accounting for ordering)
    acgt_shape = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_acgt_shape.parquet')
    acgt       = read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_acgt.parquet')
    acgt_shape = acgt_shape['shape'].tolist()
    acgt       = acgt['values'].to_numpy().reshape(acgt_shape)

    with open(f'./vnn_cache/{params_data_subset_hash}_inp_node_idx_dict.json', 'r') as f:
        inp_node_idx_dict = json.load(f)

    # calculate 
    y = np.array(phno.drop(columns='Taxa'))
    y_names = list(phno.drop(columns='Taxa'))

elif not use_data_cache_load:
    ## Load Marker Data ===========================================================
    hmp = read_pq_or_pd(file_path=params_data['hmp_path'])
    hmp = hmp.sort_values(['chrom', 'pos']).reset_index(drop=True)
    acgt= sparsevnn.util.hmp_table_to_matrix(hmp)

    # Currently acgt is in a "rotation" (order of dimenisons) that was convenient for its creation. 
    # We'll rotate it from snps, genotype, prob. -> genotype, prob., snps which is the order we need for the model.
    acgt = einops.rearrange(acgt, 's g p -> g p s')
    acgt_loci = hmp.loc[:, ['chrom', 'pos']]
    acgt_taxa = [e for e in list(hmp) 
        if e not in ['rs#', 'alleles', 'chrom', 'pos', 'strand', 'assembly#', 'center', 'protLSID', 'assayLSID', 'panelLSID', 'QCcode']]

    ## Load Response Data =========================================================
    # Taxa, y1, y2 ...
    phno = read_pq_or_pd(file_path=params_data['phno_path'])
    phno.Taxa = phno.Taxa.astype(str) # ensure this is a string

    # disallow specific taxa. Useful for keepig a test set fully separate.
    if params_data['holdout_taxa_ignore'] != []: 
        params_data['holdout_taxa_ignore'] = [str(e) for e in params_data['holdout_taxa_ignore'] ] # ensure this is a string
        mask = phno.Taxa.isin(params_data['holdout_taxa_ignore'])
        phno = phno.loc[~mask, ].reset_index(drop = True)

    # Filter Taxa based on availability
    phno_taxa   = list(set(phno.Taxa.tolist()))
    shared_taxa = sorted([e for e in phno_taxa if e in acgt_taxa])
    # instead of sorting phno and extracting Geno_Idx from the index, we'll create it here and then merge it in.
    shared_taxa_geno_idx = pd.DataFrame([(i,k) for i,k in enumerate(shared_taxa)]).rename(columns = {0:'Geno_Idx', 1:'Taxa'})
    
    phno = phno.loc[(phno.Taxa.isin(shared_taxa)), ].reset_index(drop = True)
    
    y = np.array(phno.drop(columns='Taxa'))
    y_names = list(phno.drop(columns='Taxa'))
    print(f'The output array is of shape {y.shape}.')


    ## Prepare Lookup Tables ======================================================
    unique_geno = phno.loc[:, ['Taxa']].drop_duplicates().reset_index().rename(columns={'index':'Is_Phno'}).merge(shared_taxa_geno_idx)

    obs_geno_lookup = phno.loc[:, ['Taxa']
                            ].reset_index(
                            ).rename(columns={'index':'Phno_Idx'}
                            ).merge(unique_geno, how='outer'
                            ).drop(columns = ['Taxa']
                            ).sort_values('Phno_Idx'
                            ).loc[:, ['Phno_Idx', 'Geno_Idx', 'Is_Phno']]

    # Build Graph Structure -------------------------------------------------------
    match params_data['graph_source']:
        case 'table':
            cxn = read_pq_or_pd(file_path=params_data['graph_cxn'])

        case 'kegg':
            # (Download if necessary)
            # catalog = sparsevnn.util._get_available_catalog(species = 'gmx') # How to see the options
            inp = sparsevnn.util._get_json(
                species = params_data['species'], 
                catalog_num = params_data['kegg_catalog'], 
                cache = True, 
                cache_dir = params_data['graph_cache_path'])
            inp = sparsevnn.util._peel(inp=inp)
            cxn = sparsevnn.util._connections_from_peeled_json(inp, max_iter = 1000, 
                                                            print_queue_len = False)
            cxn = pd.DataFrame(cxn, columns=['src', 'tgt'])


    # Match graph inputs to gene models (parsing GFF annotation file)
    kegg2ncbi = sparsevnn.util._get_kegg2ncbi(species = params_data['species'], 
                                            cache = True, 
                                            cache_dir = params_data['graph_cache_path'])
    ncbi2kegg = {kegg2ncbi[k]:k for k in kegg2ncbi}


    gff = sparsevnn.util._read_gene_annotation_table(filepath = params_data['gff_path'])
    gff = sparsevnn.util._gene_annotation_table_expand_attributes(gff)
    # project chromosome over rows
    gff = gff.loc[(gff.chromosome.notna()), ['seqid', 'chromosome']].merge( gff.drop(columns=['chromosome']) )
    # select columns
    gff = gff.loc[(gff.type == 'gene'), ['chromosome', 'start', 'end', 'ID', 'Dbxref']]
    # Drop any without a known chromosome.
    gff = gff.loc[(gff.chromosome != 'Unknown')]


    # Now we can use `kegg2ncbi` to match up the input nodes with those that have a ncbi-geneid
    gene_nodes_gff = sparsevnn.util.intersect_cxn_gff_nodes(
        gff=gff,
        cxn=cxn,
        kegg2ncbi=kegg2ncbi
        )

    cxn = sparsevnn.util.filter_connection_df(cxn = cxn, gene_nodes_gff = gene_nodes_gff)

    # Make sure that the highest node is `yhat` 
    if not 'yhat' in sparsevnn.util.order_connections(inp = cxn, node_names=None)[0]:
        print('`yhat` node is not the highest node')
        # Exists but is in the wrong place
        if 'yhat' in set(cxn.src.to_list()+cxn.tgt.to_list()):
            print('`yhat` exists. Removing _ALL_ links to it.')
            cxn = cxn.loc[~((cxn.src == 'yhat') |
                            (cxn.src == 'yhat')), ].reset_index(drop=True)

        print('Inserting `yhat`.')
        top_level_nodes = sparsevnn.util.order_connections(inp = cxn, node_names=None)[0]
        print('Adding `yhat` node.')
        cxn = pd.concat([
            pd.DataFrame([('yhat', e) for e in top_level_nodes], columns=['src', 'tgt']), 
            cxn]
            ).reset_index(drop=True)

    # acgt.shape 
    ## Update Genotype Data =======================================================

    # Because we use the position in acgt_taxa to map a shared_taxa entry to a position we only 
    # really need to check that all the shared entries exist in acgt_taxa
    assert [] == [e for e in shared_taxa if e not in acgt_taxa]

    acgt, taxa2idx = sparsevnn.util.acgt_filter_taxa(
        acgt=acgt,
        acgt_taxa=acgt_taxa,
        shared_taxa=shared_taxa)


    # Here is one way of linking snps to genes. Instead of finding snps within a given gene we look for the indices that are closest to or within.  
    # Now acgt_loci is updated to match acgt's reduced size
    acgt, inp_node_idx_dict, acgt_loci = sparsevnn.util.acgt_filter_snps(
        acgt = acgt, 
        acgt_loci = acgt_loci, 
        gene_nodes_gff = gene_nodes_gff, 
        include_adj = True)
    
    # Drop any empty nodes
    inp_node_idx_dict = {e:inp_node_idx_dict[e] for e in inp_node_idx_dict if inp_node_idx_dict[e] != []}
    # update cxn to only include leaf nodes with inputs
    leaves = cxn.loc[~(cxn.tgt.isin(cxn.src)), 'tgt'].tolist()
    # drop leaves without snps
    leaves = [e for e in leaves if e in inp_node_idx_dict]
    # trace tgt->src to make sure that removing empty leaved doesn't leave any broken branches
    branches = leaves.copy()

    for i in range(len(cxn.tgt)):
        new_branches = cxn.loc[(cxn.tgt.isin(branches)), 'src'].tolist()
        if set(branches) == set(branches+new_branches):
            # if no new branches were added then we've mapped the whole tree
            break
        else:
            branches = branches + new_branches
    branches = list(set(branches))
    # update and drop any 
    _ = cxn.shape[0]
    cxn = cxn.loc[(cxn.tgt.isin(branches)), ['src', 'tgt']].reset_index(drop = True)
    if _ != cxn.shape[0]:
        print(f'Excludinging {_ - cxn.shape[0]} connections without associated snps from `cxn`.')

    if use_data_cache: 
        ## Tables
        pq.write_table(pa.Table.from_pandas(phno),            f'./vnn_cache/{params_data_subset_hash}_phno.parquet')
        pq.write_table(pa.Table.from_pandas(obs_geno_lookup), f'./vnn_cache/{params_data_subset_hash}_obs_geno_lookup.parquet')
        pq.write_table(pa.Table.from_pandas(cxn),             f'./vnn_cache/{params_data_subset_hash}_cxn.parquet')
        pq.write_table(pa.Table.from_pandas(acgt_loci),       f'./vnn_cache/{params_data_subset_hash}_acgt_loci.parquet')
        pq.write_table(pa.Table.from_pandas(gene_nodes_gff),  f'./vnn_cache/{params_data_subset_hash}_gene_nodes_gff.parquet')
        # Because this might be modified directly we're going to write it out in non-parquet form too.
        cxn.to_csv(f'./vnn_cache/{params_data_subset_hash}_cxn.csv', index=False) 

        ## nd Array
        # np.savez_compressed(f'./vnn_cache/{params_data_subset_hash}_acgt.npz', acgt=acgt)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({'shape':list(acgt.shape)})),  f'./vnn_cache/{params_data_subset_hash}_acgt_shape.parquet')
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({'values':acgt.reshape(-1)})), f'./vnn_cache/{params_data_subset_hash}_acgt.parquet')


        ## Dict
        sparsevnn.qol.write_json(inp_node_idx_dict, f'./vnn_cache/{params_data_subset_hash}_inp_node_idx_dict.json') 


# Training Prep. --------------------------------------------------------------
## Model Prep. ================================================================
cxn_dict = sparsevnn.util.convert_connections(inp=cxn, to='dict', node_names=None)

myvnn = sparsevnn.util.mk_vnnhelper(
        edge_dict = cxn_dict,
        num_nucleotides = 4, # this could also be 1 for major/minor allele. 
        inp_tensor_lookup = inp_node_idx_dict,
        params = sparsevnn.qol.params() #NOTE Because we only need this for setup and the params variable contains different attributes we provide the default dict.
        )

# build dependancy dictionary
dd = sparsevnn.core.mk_NodeGroups(edge_dict=myvnn.edge_dict, dependancy_order=myvnn.dependancy_order)

M_list = [
    structured_layer_info(
    i = ii, 
    node_groups=dd, 
    node_props=myvnn.node_props, 
    edge_dict=myvnn.edge_dict, 
    as_sparse=True,
    inp_tensor_nucleotides= params_data['num_nucleotides'],
    # lambda to only provide the lookup for the 0th grouping (input level)
    inp_tensor_lookup = (lambda x: inp_node_idx_dict if x == 0 else None)(ii)
    )
    for ii in sorted(list(dd.keys()))]

print('\n'.join(
    ['Layer\tinp\tout\teye'
    ]+[f"{k}\t{len(dd[k]['inp'])}\t{len(dd[k]['out'])}\t{len(dd[k]['eye'])}" for k in dd]
    ))

## Dataloader Prep. ===========================================================
match params_data['holdout_type']:
    case 'percent':
        print(r'Setting Holdout as #% of Taxa')
        _ = obs_geno_lookup.Geno_Idx.drop_duplicates().tolist().copy()

        rng = np.random.default_rng(params_data['holdout_seed'])
        rng.shuffle(_)

        tst_cutoff = round(len(_) * params_data['holdout_percent'])
        tst_Geno_Idx = _[0:tst_cutoff]
        trn_Geno_Idx = _[tst_cutoff:]

    case 'taxa':
        print(r'Setting Holdout as specific Taxa')

        _ = phno['Taxa'].drop_duplicates().reset_index().rename(columns={'index':'Is_Phno'})

        mask = (_.Taxa.isin(params_data['holdout_taxa']))
        tst_Geno_Idx = set(_.loc[  mask,  ['Is_Phno']].merge(obs_geno_lookup).loc[:, 'Geno_Idx'])
        trn_Geno_Idx = set(_.loc[~(mask), ['Is_Phno']].merge(obs_geno_lookup).loc[:, 'Geno_Idx'])


    case 'parent':
        _ = phno['Taxa'].drop_duplicates().reset_index().rename(columns={'index':'Is_Phno'})

        # Standardize names to lower and set flag if either parent is in the holdout set
        drop_vals = [e.lower() for e in params_data['holdout_taxa']]

        mask = [True in [ee.lower() in drop_vals for ee in e.split('/')] for e in _.Taxa ]
        mask_inv = [e == False for e in mask]
        tst_Geno_Idx = set(_.loc[mask,    ['Is_Phno']].merge(obs_geno_lookup).loc[:, 'Geno_Idx'])
        trn_Geno_Idx = set(_.loc[mask_inv,['Is_Phno']].merge(obs_geno_lookup).loc[:, 'Geno_Idx'])


train_idx = obs_geno_lookup.loc[(obs_geno_lookup.Geno_Idx.isin(trn_Geno_Idx)), 'Phno_Idx'].tolist()
test_idx  = obs_geno_lookup.loc[(obs_geno_lookup.Geno_Idx.isin(tst_Geno_Idx)), 'Phno_Idx'].tolist()

if use_data_cache: 
    ## write out to aid modeling in R
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({'train_idx':train_idx})), f'./vnn_cache/{params_data_subset_hash}_train_idx.parquet')
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({'test_idx':test_idx  })), f'./vnn_cache/{params_data_subset_hash}_test_idx.parquet')

print(f'Train:\t{len(train_idx)}\nTest:\t{len(test_idx)}\nHolding out {round( 100*len(test_idx)/(len(train_idx)+len(test_idx)) , 3)}%')


acgt_tensor = torch.from_numpy(acgt
                  ).to(torch.float
                  ).swapaxes(1,2 # obs, nucleotide, length -> obs, length, nucleotide so that
                  ).reshape(acgt.shape[0], -1) # reshape will but the nucleotides right next to each other. This will make the gene lookup make sense.


y = torch.from_numpy(y
                  ).to(torch.float
                  )

y_c = y[train_idx].mean(axis=0)
y_s = y[train_idx].std(axis=0)

y = (y - y_c)/y_s


## send to gpu if available:
if torch.cuda.is_available():
    training_dataloader = DataLoader(
        MarkerDataset(
            lookup_obs = torch.from_numpy(np.array(train_idx)).to('cuda'),
            G = acgt_tensor.to('cuda'),
            y = y.to(torch.float32).to('cuda'),
            lookup_geno = torch.from_numpy(obs_geno_lookup.to_numpy()).to('cuda')
            ),
            batch_size = params_run['batch_size'],
            shuffle = params_data['dataloader_shuffle_train']
    )

    validation_dataloader = DataLoader(
        MarkerDataset(
            lookup_obs = torch.from_numpy(np.array(test_idx)).to('cuda'),
            G = acgt_tensor.to('cuda'),
            y = y.to(torch.float32).to('cuda'),
            lookup_geno = torch.from_numpy(obs_geno_lookup.to_numpy()).to('cuda')
            ),
            batch_size = params_run['batch_size'],
            shuffle = params_data['dataloader_shuffle_valid']  
    )
else:
    training_dataloader = DataLoader(
        MarkerDataset(
            lookup_obs = torch.from_numpy(np.array(train_idx)),
            G = acgt_tensor,
            y = y.to(torch.float32),
            lookup_geno = torch.from_numpy(obs_geno_lookup.to_numpy())
            ),
            batch_size = params_run['batch_size'],
            shuffle = params_data['dataloader_shuffle_train']
    )

    validation_dataloader = DataLoader(
        MarkerDataset(
            lookup_obs = torch.from_numpy(np.array(test_idx)),
            G = acgt_tensor,
            y = y.to(torch.float32),
            lookup_geno = torch.from_numpy(obs_geno_lookup.to_numpy())
            ),
            batch_size = params_run['batch_size'],
            shuffle = params_data['dataloader_shuffle_valid']  
    )


## Customizable Training Functions ============================================
def _layer_list_from_params(params):
    # 0th and -1th values will be set based on the in/out shape so we'll use placeholders here
    outs = ([0] + [params['width'] for i in range(params['hidden_layers'])] + [0])
    # scale 
    outs = [dist_scale_function(out=outs[i],  dist=i, decay_rate=params['width_decay_rate']) for i in range(len(outs))]
    if params['width_decay_reverse'] == 1:
        outs.reverse()
    outs[ 0] = params['size_in']
    outs[-1] = params['size_out']

    # since dropouts are between linear layers we'll have 1 fewer of them
    drops= ([params['drop'] for i in range(params['hidden_layers'])] + [0])
    drops= [dist_scale_function(out=drops[i], dist=i, decay_rate=params['drop_decay_rate']) for i in range(len(drops))]
    drops= drops[0:-1] # toss the placeholder level
    if params['drop_decay_reverse'] == 1:
        drops.reverse()
    drops += [None] # equal length but guard against dropout on last layer

    outs = [(i,j) for i,j in zip(outs, outs[1:])]
    layer_list = []
    for i in range(len(outs)):
        layer_list.append(nn.Linear(outs[i][0], outs[i][1]))
        if drops[i] != None:
            layer_list.append(nn.Dropout(p=(drops[i]/100)))
            layer_list.append(nn.ReLU())
    return layer_list

def train_one_model(
    params = params,
    params_data = params_data,
    params_run = params_run,
    edge_dict = cxn_dict,
    inp_tensor_lookup = inp_node_idx_dict,
    log_dir = lightning_log_dir # This is explicit istead of using the global scope so that hyps/training can log different dirs. 
    ):

    layer_list = _layer_list_from_params(params)

    model      = SparseVNN(layer_list = layer_list)
    VNN        = plDNN_general(model)
    optimizer = VNN.configure_optimizers()
    logger    = CSVLogger(log_dir, name=exp_name)
    logger.log_hyperparams(params={
        'params': params,
        'params_data': params_data,
        'params_run': params_run
    })
    trainer = pl.Trainer(max_epochs=params_run['max_epoch'], logger=logger)
    trainer.fit(model=VNN, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)
    return M_list, trainer


def vnn_from_state_dict(
    state_dict_path = None,
    params = params,
    params_data = params_data,
    edge_dict = cxn_dict,
    inp_tensor_lookup = inp_node_idx_dict,
    ):
    layer_list = _layer_list_from_params(params)
    model      = SparseVNN(layer_list = layer_list)
    # now we load the state dict into the model
    model.load_state_dict(torch.load(state_dict_path))
    return model



def evaluate(parameterization):
    "This is for Ax's use which is why it pulls variables from the global scope."
    _M_list, _ = train_one_model(
        params = parameterization,
        edge_dict = cxn_dict,
        inp_tensor_lookup = inp_node_idx_dict,
        log_dir = lightning_log_dir
    )
    # if we were optimizing number of training epochs this would be an effective loss to use.
    # trainer.callback_metrics['train_loss']
    # float(trainer.callback_metrics['train_loss'])
    # To potentially _overtrain_ models and still let the selction be based on their best possible performance,
    # I'll use the lowest average error in an epoch
    log_path = lightning_log_dir+'/'+exp_name
    fls = os.listdir(log_path)
    nums = [int(e.split('_')[-1]) for e in fls] 

    M = pd.read_csv(log_path+f"/version_{max(nums)}/metrics.csv")
    M = M.loc[:, ['epoch', 'train_loss']].dropna()

    M = M.groupby('epoch').agg(
        train_loss = ('train_loss', 'mean'),
        train_loss_sd = ('train_loss', 'std'),
        ).reset_index()

    train_metric = M.train_loss.min()
    print(train_metric)
    _ = {"train_loss": (train_metric, 0.0)}
    return _


# This is a sneaky feature I'm introducing to allow for customization of functions below.
# Overwrite exiting functions by customizing vnn_patch.py . 
if params_run['patch'] == True:
    if 'vnn_patch.py' in os.listdir():
        from vnn_patch import * 
        print(f'Patches applied from `vnn_patch.py`.')
    else:
        print(f'No patches applied. `vnn_patch.py` not found in {os.getcwd()}')


# Model Execution -------------------------------------------------------------
match params_run['run_mode']: # setup tune train predict eval
    case 'setup':
        print('`setup` complete. Ready to `tune` or `train`.')
    case 'tune':
        print('Running Hyperparameter Tuning')
        sparsevnn.qol.ensure_dir_path_exists(dir_path = lightning_log_dir)
        json_path = f"{lightning_log_dir}/{exp_name}.json"
        # overwrite params_list's output with the size with the right output size. Don't allow the user to enter the wrong value. 
        # This means we don't need to worry much about re-using these values. 
        i = [i for i in range(len(params_list)) if params_list[i]['name'] == 'size_out'][0]
        params_list[i]['value'] = y.shape[1]
        i = [i for i in range(len(params_list)) if params_list[i]['name'] == 'size_in'][0]
        params_list[i]['value'] = next(iter(training_dataloader))[1].shape[1]

        # Load ax json if it exists
        loaded_json = False
        if os.path.exists(json_path): 
            ax_client = (AxClient.load_from_json_file(filepath = json_path))
            loaded_json = True
        else:
            ax_client = AxClient()
            ax_client.create_experiment(
                name=exp_name,
                parameters=params_list,
                objectives={"train_loss": ObjectiveProperties(minimize=True)}
            )
        
        # Check if running hyps is forced or if we haven't yet reached tune_max
        run_trials_bool = True
        if params_run['tune_force'] == False:
            if loaded_json: 
                # check if we've reached the max number of hyperparamters combinations to test
                if params_run['tune_max'] <= (ax_client.generation_strategy.trials_as_df.index.max()+1):
                    run_trials_bool = False

        # Run trials 
        if run_trials_bool:
            for i in range(params_run['tune_trials']):
                parameterization, trial_index = ax_client.get_next_trial()
                # Local evaluation here can be replaced with deployment to external system.
                ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameterization)) # NOTE plDNN_general and other functions can be patched

            ax_client.save_to_json_file(filepath = json_path)

    case 'train':
        print('Running Training')
        json_path = f"{lightning_log_dir}/{exp_name}.json"
        params['default_out_nodes_out'] = y.shape[1]

        if params_run['train_from_ax']:
            if os.path.exists(json_path): 
                ax_client = (AxClient.load_from_json_file(filepath = json_path))
                params, _ = ax_client.get_best_parameters()
            else:
                print('Ax client not found! Using default.')

        model_log_dir = '/'.join(lightning_log_dir.split('/')[0:-1]+['models'])
        
        res_M_list, res = train_one_model(
            params = params,
            params_data = params_data,
            params_run = params_run,
            edge_dict = cxn_dict,
            inp_tensor_lookup = inp_node_idx_dict,
            log_dir= model_log_dir
            )
        

        # Turn all the tensors holding shape and index info into ints. This _massively_ reduces the space to save it.
        def _shrink_M_list(M_list): 
            # storing col and row info as tensors is convenient but has a massive effect on storage size
            # If all the size/start/stop values are coerced to ints the size for a test M_list goes from
            # 5.2G -> 7.5M. At that size we might as well save out al the attributes in it.
            for i in range(len(M_list)):
                e = M_list[i].col_info
                M_list[i].col_info = {k: {
                    'size':int(e[k]['size']),
                    'start':int(e[k]['start']),
                    'stop':int(e[k]['stop']),
                    } 
                    for k in e.keys()
                }
                
                e = M_list[i].row_info
                M_list[i].row_info = {k: {
                    'size':int(e[k]['size']),
                    'start':int(e[k]['start']),
                    'stop':int(e[k]['stop']),
                    } 
                    for k in e.keys()
                }
            return M_list
        
        res_M_list = _shrink_M_list(M_list=res_M_list)


        if params_run['train_save']:
            # save with the same numbering that pytorch lightning's record used
            log_path = model_log_dir+'/'+exp_name
            fls = os.listdir(log_path)
            nums = [int(e.split('_')[-1]) for e in fls]
            torch.save(res.model.mod.state_dict(), f'{log_path}/version_{max(nums)}/version_{max(nums)}.pt')
            sparsevnn.qol.write_json(params,       f'{log_path}/version_{max(nums)}/version_{max(nums)}.json')

            with open(f'{log_path}/version_{max(nums)}/version_{max(nums)}_M_list.pkl', 'wb') as f:
                pickle.dump(res_M_list, f, protocol=5)

            # Get the most recent history
            history = pd.read_csv(f'{log_path}/version_{max(nums)}/metrics.csv')
            # pivot longer
            history = pd.concat([
                history.drop(columns='val_loss').rename(columns={'train_loss':'loss'}).dropna().assign(split='trn'),
                history.drop(columns='train_loss').rename(columns={'val_loss':'loss'}).dropna().assign(split='val')]
            )

            plt = px.line(history, x = 'step', y = 'loss', color='split', hover_data=['epoch'])

            plt.write_html(f'{log_path}/version_{max(nums)}/metrics.html')
            plt.write_image(f'{log_path}/version_{max(nums)}/metrics.svg')


            if params_run.get('train_name', False):
                # allow for custom names instead of version_\d+
                # rename the paths.
                src_dir  = f'version_{max(nums)}'
                dst_dir = params_run['train_name']
                
                sparsevnn.qol.ensure_dir_path_exists(dir_path = f'{log_path}/{dst_dir}')

                current_names = os.listdir(f'{log_path}/{src_dir}')
                new_names     = [e.replace(src_dir, dst_dir) if re.match(r'^'+src_dir, e) else e 
                                 for e in current_names]                               
                import shutil
                # move everything and rename from version_\d -> name if provided
                for i,j in zip(current_names, new_names):
                    shutil.move(f'{log_path}/{src_dir}/{i}', f'{log_path}/{dst_dir}/{j}')

                # remove old dir
                shutil.rmtree(f'{log_path}/{src_dir}')

                                
    case 'predict':
        print('Running Prediction')

        # need to load params that are specifically associated with the saved model.
        # otherwise there's a chance the hyperparameters will have changed.
        model = vnn_from_state_dict(
            state_dict_path = params_data['model_path'],
            # Here we read in the associated json in case the model specification is different
            # from the current params file.
            params = sparsevnn.qol.read_json('.'.join(params_data['model_path'].split('.')[0:-1]+['json'])),
            # params = params,
            params_data = params_data,
            edge_dict = cxn_dict,
            inp_tensor_lookup = inp_node_idx_dict,
            ) 
        model = model.eval()

        save_dir = '/'.join(params_data['model_path'].split('/')[0:-1])

        pq.write_table(pa.Table.from_pandas(pd.DataFrame(y_c.numpy()[:,None].T, columns=y_names)), save_dir+f'/{params_data_subset_hash}_yvar_cs_center.parquet')
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(y_s.numpy()[:,None].T, columns=y_names)), save_dir+f'/{params_data_subset_hash}_yvar_cs_scale.parquet')

        def _collect_predictions(model, inp_dl):
            # check that model is on the same device as the data
            if next(iter(inp_dl))[0].get_device() != -1:
                model = model.to('cuda')
            # as a sanity check we'll save the true y's. That will allow for 
            yvar, yhat = [], []
            # only get the validation dataloader if the training dataloader is shuffled.
            for i, (y,x) in enumerate(inp_dl):
                yvar.append(       y.detach().cpu() )
                yhat.append(model(x).detach().cpu() )

            yvar = torch.concat(yvar)
            yhat = torch.concat(yhat)
            return yvar, yhat

        # mk df to apply taxa to predictions
        _ = phno.loc[:, ['Taxa']].reset_index(names='Is_Phno').merge(obs_geno_lookup).loc[:, ['Taxa', 'Phno_Idx']]

        if params_data['dataloader_shuffle_train'] == False:
            yvar, yhat = _collect_predictions(model = model, inp_dl  = training_dataloader)

            yvcs = pd.DataFrame(yvar.numpy(),            columns=y_names).assign(Split = 'Training', Phno_Idx = train_idx).merge(_)
            yhcs = pd.DataFrame(yhat.numpy(),            columns=y_names).assign(Split = 'Training', Phno_Idx = train_idx).merge(_)
            yvr = pd.DataFrame((yvar*y_s + y_c).numpy(), columns=y_names).assign(Split = 'Training', Phno_Idx = train_idx).merge(_)
            yhr = pd.DataFrame((yhat*y_s + y_c).numpy(), columns=y_names).assign(Split = 'Training', Phno_Idx = train_idx).merge(_)

            save_pq = lambda df, nom: pq.write_table(pa.Table.from_pandas(df), nom)
            save_pq(yvcs, save_dir+f'/{params_data_subset_hash}_yvar_cs_trn.parquet')
            save_pq(yhcs, save_dir+f'/{params_data_subset_hash}_yhat_cs_trn.parquet')
            save_pq(yhcs, save_dir+f'/{params_data_subset_hash}_yvar_trn.parquet')
            save_pq(yhr,  save_dir+f'/{params_data_subset_hash}_yhat_trn.parquet')

            for e in y_names:
                plt = px.scatter(pd.concat([yvcs.loc[:, [e, 'Taxa']].rename(columns={e:'Observed'}),
                                            yhcs.loc[:, [e]].rename(columns={e:'Predicted'})], axis = 1),
                                            x = 'Observed', y = 'Predicted', color = 'Taxa', hover_data=['Taxa'], title=e )
                
                plt.write_html(save_dir+f'/{params_data_subset_hash}_eval_cs_scatter_trn.html')
                plt.write_image(save_dir+f'/{params_data_subset_hash}_eval_cs_scatter_trn.svg')

                plt = px.scatter(pd.concat([yvr.loc[:, [e, 'Taxa']].rename(columns={e:'Observed'}),
                                            yhr.loc[:, [e]].rename(columns={e:'Predicted'})], axis = 1),
                                            x = 'Observed', y = 'Predicted', color = 'Taxa', hover_data=['Taxa'], title=e )
                
                plt.write_html(save_dir+f'/{params_data_subset_hash}_eval_scatter_trn.html')
                plt.write_image(save_dir+f'/{params_data_subset_hash}_eval_scatter_trn.svg')


        if params_data['dataloader_shuffle_valid'] == False:
            yvar, yhat = _collect_predictions(model = model, inp_dl  = validation_dataloader)

            yvcs = pd.DataFrame(yvar.numpy(),            columns=y_names).assign(Split = 'Validation', Phno_Idx = test_idx).merge(_)
            yhcs = pd.DataFrame(yhat.numpy(),            columns=y_names).assign(Split = 'Validation', Phno_Idx = test_idx).merge(_)
            yvr = pd.DataFrame((yvar*y_s + y_c).numpy(), columns=y_names).assign(Split = 'Validation', Phno_Idx = test_idx).merge(_)
            yhr = pd.DataFrame((yhat*y_s + y_c).numpy(), columns=y_names).assign(Split = 'Validation', Phno_Idx = test_idx).merge(_)

            save_pq = lambda df, nom: pq.write_table(pa.Table.from_pandas(df), nom)
            save_pq(yvcs, save_dir+f'/{params_data_subset_hash}_yvar_cs_val.parquet')
            save_pq(yhcs, save_dir+f'/{params_data_subset_hash}_yhat_cs_val.parquet')
            save_pq(yhcs, save_dir+f'/{params_data_subset_hash}_yvar_val.parquet')
            save_pq(yhr,  save_dir+f'/{params_data_subset_hash}_yhat_val.parquet')

            for e in y_names:
                plt = px.scatter(pd.concat([yvcs.loc[:, [e, 'Taxa']].rename(columns={e:'Observed'}),
                                            yhcs.loc[:, [e]].rename(columns={e:'Predicted'})], axis = 1),
                                            x = 'Observed', y = 'Predicted', color = 'Taxa', hover_data=['Taxa'], title=e )
                
                plt.write_html(save_dir+f'/{params_data_subset_hash}_eval_cs_scatter_val.html')
                plt.write_image(save_dir+f'/{params_data_subset_hash}_eval_cs_scatter_val.svg')

                plt = px.scatter(pd.concat([yvr.loc[:, [e, 'Taxa']].rename(columns={e:'Observed'}),
                                            yhr.loc[:, [e]].rename(columns={e:'Predicted'})], axis = 1),
                                            x = 'Observed', y = 'Predicted', color = 'Taxa', hover_data=['Taxa'], title=e )
                
                plt.write_html(save_dir+f'/{params_data_subset_hash}_eval_scatter_val.html')
                plt.write_image(save_dir+f'/{params_data_subset_hash}_eval_scatter_val.svg')            
            
    case 'eval':
        print('Running Evaluation')
        model = vnn_from_state_dict(
            state_dict_path = params_data['model_path'],
            # Here we read in the associated json in case the model specification is different
            # from the current params file.
            params = sparsevnn.qol.read_json('.'.join(params_data['model_path'].split('.')[0:-1]+['json'])),
            params_data = params_data,
            edge_dict = cxn_dict,
            inp_tensor_lookup = inp_node_idx_dict,
            ) 

        save_dir = '/'.join(params_data['model_path'].split('/')[0:-1])

        # load in cached M_list for model        
        with open(params_data['model_path'].replace('.pt', '_M_list.pkl'), 'rb') as f:
            M_list = pickle.load(f)


        if 'saliency_inp' in params_run['eval']:
            print('Calculating salience')
            model = model.eval()

            # iterate over dataloader and aggregate all of the saliences for the input data
            def _collect_salience_snpwise(model, inp_dl):
                # check that model is on the same device as the data
                if next(iter(inp_dl))[0].get_device() != -1:
                    model = model.to('cuda')

                # get saliency for all obs given model, dataloader
                def _get_saliency(y_i, x_i, model):
                    x_i.requires_grad_()
                    model.eval()

                    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
                    loss_fn = nn.MSELoss()

                    loss = loss_fn(model(x_i), y_i)
                    optimizer.zero_grad()
                    loss.backward()
                    out = x_i.grad
                    out = out.to('cpu').numpy()
                    # for output size 
                    # reshape to b,n,l
                    out = out.reshape(out.shape[0], -1, params_data['num_nucleotides']).swapaxes(1,2)
                    # reduce to max over nucleotide axis
                    out = out.max(axis = 1) # max over nucleotide axis
                    out.shape
                    return out

                out = []
                for i, (y,x) in enumerate(inp_dl):
                    out.append(_get_saliency(y_i = y, x_i = x, model = model))
                out = np.concatenate(out)

                #reverse workup for acgt_tensor to get (obs, nuc, len)
                return out


            ### SNP-wise Manhattan 
            def _plt_saliences(salience, save_dir,  plt_prefix):
                print('\n'.join(['Percentiles:']+[f'q {i} = {np.quantile(salience.salience, q = i)}' for i in [.95, .99, .999]]))
                
                # salience distribution
                plt_d = px.histogram(salience, x = 'salience')

                plt_d.add_vline(x=np.quantile(salience.salience, q = .95), line_dash="solid", line_color="#5d5d5d")
                plt_d.add_vline(x=np.quantile(salience.salience, q = .99), line_dash="dash",  line_color="#2a2a2a")
                plt_d.add_vline(x=np.quantile(salience.salience, q = .999),line_dash="dot",   line_color="#000000")

                plt_d.write_html(save_dir+f"{plt_prefix}_dist.html")
                plt_d.write_image(save_dir+f"{plt_prefix}_dist.svg")

                # salience manhattan
                plt_m = px.scatter(salience, x = 'index', y = 'salience', color = 'chrom', hover_data=['pos', 'cxn'])

                plt_m.add_hline(y=np.quantile(salience.salience, q = .95), line_dash="solid", line_color="#5d5d5d")
                plt_m.add_hline(y=np.quantile(salience.salience, q = .99), line_dash="dash",  line_color="#2a2a2a")
                plt_m.add_hline(y=np.quantile(salience.salience, q = .999),line_dash="dot",   line_color="#000000")

                plt_m.write_html(save_dir+f"{plt_prefix}_manhattan.html")
                plt_m.write_image(save_dir+f"{plt_prefix}_manhattan.svg")

                return plt_m, plt_d
            

            def _collapse_and_add_gff_annotations(acgt_loci, # = acgt_loci, 
                                     gene_nodes_gff, # = gene_nodes_gff, 
                                     e, # = validation_inp_sals
                                     ):
                salience = acgt_loci.copy()
                salience['salience'] = e.max(axis = 0) # max over observation axis
                salience = salience.reset_index()

                salience['chrom'] = salience['chrom'].astype(str)  
                salience['pos']   = salience['pos'].astype(int)  
                salience['cxn'] = ''
                for i in gene_nodes_gff.index:
                    chrom, start, end, cxn_val = gene_nodes_gff.loc[i, ['chromosome', 'start', 'end', 'cxn']]
                    salience.loc[(
                        (salience.chrom == str(chrom)) &
                        ((salience.pos  >= int(start)) & 
                         (salience.pos  <= int(end)))
                        ), 'cxn'] = cxn_val
                return salience
            
            
            # NOTE this could also be broken up into snpwise and genewise options
            ### Gene-wise Manhattan
            def _collapse_salience_genewise(M_list, acgt_loci, sals):
                # I can get the gene associations back like this. Not the same as a manhattan
                _ = pd.DataFrame([(
                        e, 
                        int(M_list[0].row_info[e]['start']),
                        int(M_list[0].row_info[e]['stop'])
                    ) for e in M_list[0].row_info.keys()], 
                    columns=['cxn', 'start', 'stop']
                    ).sort_values('start'
                    ).reset_index(drop = True)

                _['start'] = (_['start'] / params_data['num_nucleotides']).astype(int) # M_list is in reference to the flattened data.
                _['stop']  = (_['stop']  / params_data['num_nucleotides']).astype(int)

                _['chrom'] = ''
                _['pos'] = 0
                _['max_sal'] = 0.0

                for i in _.index:
                    start, stop = _.loc[i, ['start', 'stop']]
                    _.loc[i, ['salience']] = float( sals[:, start:stop].max().item() )
                    _.loc[i, ['chrom', 'pos']]  = acgt_loci.loc[round(np.mean([start, stop])), ['chrom','pos']]

                _['chrom'] = _['chrom'].astype(str) 
                _['pos'] = _['pos'].astype(str) 
                salience = _.reset_index()

                return salience


            # Training      
            if params_data['dataloader_shuffle_train'] == False:
                print('Calculating salience w.r.t. input (snp-wise) -- training')
                training_inp_sals   = _collect_salience_snpwise(model = model, inp_dl = training_dataloader)

                salience = _collapse_and_add_gff_annotations(
                    acgt_loci = acgt_loci, 
                    gene_nodes_gff = gene_nodes_gff, 
                    e = training_inp_sals)
                pq.write_table(pa.Table.from_pandas(salience), save_dir+f'/{params_data_subset_hash}_eval_salience_snpwise_trn.parquet')
                _ = _plt_saliences(salience = salience, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_trn_salience_snpwise')  

                print('Calculating salience w.r.t. input (gene-wise) -- training')
                training_inp_sals   = _collapse_salience_genewise(M_list = M_list, acgt_loci = acgt_loci, sals = training_inp_sals)
                _ = _plt_saliences(salience=training_inp_sals,   save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_trn_salience_genewise')
                pq.write_table(pa.Table.from_pandas(training_inp_sals), save_dir+f'/{params_data_subset_hash}_eval_salience_genewise_trn.parquet')

                del salience
                del training_inp_sals


            # Validation
            if params_data['dataloader_shuffle_valid'] == False:                  
                print('Calculating salience w.r.t. input (snp-wise) -- training')
                validation_inp_sals = _collect_salience_snpwise(model = model, inp_dl = validation_dataloader)            
                salience = _collapse_and_add_gff_annotations(
                    acgt_loci = acgt_loci, 
                    gene_nodes_gff = gene_nodes_gff, 
                    e = validation_inp_sals)
                pq.write_table(pa.Table.from_pandas(salience), save_dir+f'/{params_data_subset_hash}_eval_salience_snpwise_val.parquet')
                _ = _plt_saliences(salience = salience, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_val_salience_snpwise')  

                print('Calculating salience w.r.t. input (gene-wise) -- validation')
                validation_inp_sals = _collapse_salience_genewise(M_list = M_list, acgt_loci = acgt_loci, sals = validation_inp_sals)
                _ = _plt_saliences(salience=validation_inp_sals, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_val_salience_genewise')
                pq.write_table(pa.Table.from_pandas(validation_inp_sals), save_dir+f'/{params_data_subset_hash}_eval_salience_genewise_val.parquet')

                del salience
                del validation_inp_sals


        if 'saliency_wab' in params_run['eval']:
            print('Calculating salience w.r.t. weights and biases')
            # break this into two problems: 
            #   1. Collecting Gradients
            #   2. Organizing Gradients
            # 
            # Training a model with the same hyperparameters sometimes results in gradients that are max 0 and sometimes not. 
            # This seems to be a gradient attenuation problem. On one run I got max(abs(grads)) that look like so:
            # trn 0 -> 0 -> 0 -> 0 -> 0 -> 0 -> 0 -> 0.000 -> 0.001 -> 0.001
            # tst 0 -> 0 -> 0 -> 0 -> 0 -> 0 -> 0 -> 0.010 -> 0.031 -> 0.017
            # Observed with gmx data and params: 
            # "{'default_decay_rate': 0, 'default_drop_nodes_edge': 0.0, 'default_drop_nodes_inp': 0.0, 'default_drop_nodes_out': 0.0, 'default_out_nodes_edge': 2, 
            #   'default_out_nodes_inp': 1, 'default_out_nodes_out': 2, 'default_reps_nodes_edge': 2, 'default_reps_nodes_inp': 1, 'default_reps_nodes_out': 1}"
            def collect_gradients(model, inp_dl = training_dataloader):      
                if next(iter(inp_dl))[0].get_device() != -1:
                    model = model.to('cuda')         
                "Returns a tuple of weight grads, bias grads"
                # Setup list of lists [layer, ..., layer] with batch in layer
                gradient_weight_holder = [[] for i in model.layer_list]
                gradient_bias_holder = [[] for i in model.layer_list]
                gradient_obs = []

                for i, (y_i, x_i) in enumerate(inp_dl):
                    gradient_obs.append(len(y_i))
                    # set to train mode, setup optimizer
                    model = model.train()

                    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
                    loss_fn = nn.MSELoss()

                    loss = loss_fn(model(x_i), y_i)
                    optimizer.zero_grad()
                    loss.backward()

                    # now go through each of the layers and pull the gradient. 
                    # Convert to numpy and save
                    for level in range(len(model.layer_list)):
                        if type(model.layer_list[level]) == torch.nn.modules.linear.Linear:
                            gradient_weight_holder[level].append( model.layer_list[level].weight.grad.detach().cpu().numpy() ) # note: weight is for nn.Linear, weights for custom
                            gradient_bias_holder[level].append( model.layer_list[level].bias.grad.detach().cpu().numpy() )

                # drop empty idxs arising from dropout/relu
                gradient_weight_holder = [e for e in gradient_weight_holder if e !=[]]
                gradient_bias_holder   = [e for e in gradient_bias_holder if e !=[]]


                gradient_obs = np.array(gradient_obs)
                # convert to percent of training set
                gradient_obs = gradient_obs/gradient_obs.sum()
                gradient_obs = gradient_obs[:, None]

                def _scale_gradients(gradient_list, gradient_obs): # gradient list should be gradient_holder[-1]
                    _ = np.concatenate(gradient_list, 0).reshape(gradient_obs.shape[0], -1)
                    _ = _ * gradient_obs # Weight by the number of obs that went into the gradient
                    _ = _.sum(axis = 0)  # Get average gradient
                    return _

                # turn batches of gradients in gradient_holder into average (accumulated) gradient
                gradient_weight_holder = [_scale_gradients(gradient_list = e, gradient_obs = gradient_obs) for e in gradient_weight_holder]
                gradient_bias_holder = [_scale_gradients(gradient_list = e, gradient_obs = gradient_obs) for e in gradient_bias_holder]
                
                return gradient_weight_holder, gradient_bias_holder
            
            training_weight_grads   = collect_gradients(model = model, inp_dl = training_dataloader)
            validation_weight_grads = collect_gradients(model = model, inp_dl = validation_dataloader)

            df_weight = []
            df_bias = []

            # map layer list idx to gradient list (skips dropouts)
            grad2layer = [i for i in range(len(model.layer_list)) if type(model.layer_list[i]) == torch.nn.Linear]
            grad2layer = {i:grad2layer[i] for i in range(len(grad2layer))}

            for layer in range(len(training_weight_grads[0])):
                # node in info
                l_inp = model.layer_list[grad2layer[layer]].in_features 
                l_out = model.layer_list[grad2layer[layer]].out_features
                x1 = np.zeros((l_inp, l_out))
                x1 = x1.swapaxes(0,1) 
                x1[:, :] = np.arange(l_inp)
                x1 = x1.swapaxes(0,1) 
                x1 = x1.astype(int).reshape(-1)

                # node out info
                x2 = np.zeros((l_inp, l_out))
                x2[:, :] = np.arange(l_out)
                x2 = x2.astype(int).reshape(-1)


                df_weight.append(pd.DataFrame(
                    {
                    'layer':layer, 
                    'idx_inp':x1,
                    'idx_out':x2,
                    'trn_weight_grads': training_weight_grads[0][layer], # weights
                    'val_weight_grads': validation_weight_grads[0][layer], # weights
                    }
                    )
                    )

                df_bias.append(pd.DataFrame(
                    {
                    'layer':layer, 
                    'idx_out':np.arange(l_out),
                    'trn_bias_grads': training_weight_grads[1][layer], # bias
                    'val_bias_grads': validation_weight_grads[1][layer], # bias
                    }
                    )
                )
            
            df_weight = pd.concat(df_weight)
            df_bias   = pd.concat(df_bias)

            pq.write_table(pa.Table.from_pandas(df_weight), save_dir+f'/{params_data_subset_hash}_eval_gradients_nodewise_weights.parquet')
            pq.write_table(pa.Table.from_pandas(df_bias),   save_dir+f'/{params_data_subset_hash}_eval_gradients_nodewise_bias.parquet')
            del training_weight_grads
            del validation_weight_grads
            # del coordinates
            # del col_info
            del _
            del df_weight
            del df_bias

        if 'rho_out' in params_run['eval']:
            print('Calculating rho w.r.t. intermediate layer output')
            # NOTE: this can take a long time.

            def _collect_rho(model, M_list, inp_dl):
                if next(iter(inp_dl))[0].get_device() != -1:
                    model = model.to('cuda')

                # Convert M_list into a usable lookup
                out = []
                for i in tqdm(range(len(M_list)), ascii = True, desc = 'Building lookup table'):
                    # break down a `structured_layer_info` class into a df
                    slinfo = M_list[i]
                    _ = [(k, int(slinfo.row_info[k]['start']), int(slinfo.row_info[k]['stop'])) for k in slinfo.row_info.keys()]
                    _ = pd.DataFrame(_, columns=['node', 'start', 'stop'])
                    _['layer'] = i
                    out.append(_)

                out = pd.concat(out)

                out = out.reset_index(drop=True).reset_index().rename(columns={'index':'node_forward_idx'})

                # table with rows being (nodes x outputs per node) and cols being (obs) 
                output_tracker = pd.DataFrame(
                    {'node_forward_idx' : sum(
                        [[k for i in range(n)] 
                            for k,n in zip(
                                out['node_forward_idx'].tolist(),
                                (out['stop'] - out['start']).tolist())
                                ], 
                                [])}
                )

                # Improved verison
                _ = []
                for node in out.node_forward_idx:
                    mask = (out.node_forward_idx == node)

                    start = out.loc[mask, 'start'].values[0].astype(int)
                    stop  = out.loc[mask, 'stop' ].values[0].astype(int)

                    _.append(
                    pd.DataFrame({
                        'node_forward_idx': out.loc[mask, 'node_forward_idx' ].values[0].astype(int), 
                        'node':  out.loc[mask, 'node' ].values[0], 
                        'start': out.loc[mask, 'start' ].values[0].astype(int), 
                        'stop':  out.loc[mask, 'stop' ].values[0].astype(int), 
                        'layer': out.loc[mask, 'layer' ].values[0].astype(int),                            
                        'idx':   [i for i in range(start, stop)]
                        })
                    )
                _ = pd.concat(_, axis = 0).reset_index(drop = True)
                # make sure everything is sorted
                _ = _.sort_values(['layer', 'idx']).reset_index(drop = True)
                lookup = _.copy()


                inp_dl_G = inp_dl.dataset.G
                y_actual = [] # Track the actual y and the...
                cached_tensor_out = []
                cached_tensor_idx = []
                for i, (y,x) in tqdm(enumerate(inp_dl), ascii = True, desc = 'Conducting forward pass'):
                    # Find gene index in G (inp_dl_G) so we can store lookup values
                    # i_idx = [torch.where((x[j, :] == inp_dl_G).all(dim=1))[0] for j in range(len(x))]
                    # i_idx = torch.concatenate(i_idx).cpu().detach().numpy()
                    # There's a potential bug here. It's possible for some data to be identical after filtering (seen in gmx)
                    # In this case we return >1 value for where and break. The solution is to default to the minimum index OR 
                    # to have this be a list and apply the values to all matching entries in inp_dl_G
                    i_idx = [torch.where((x[j, :] == inp_dl_G).all(dim=1))[0] for j in range(len(x))]
                    i_idx = [e.cpu().detach().tolist() for e in i_idx]
                    # switching to using a list of lists (ideally each containing only one entry)
                    if False not in [jj in cached_tensor_idx for j in i_idx for jj in j]:
                        cached_tensor_idx = cached_tensor_idx+i_idx 
                        y_actual          = np.concatenate([ y_actual, y.swapaxes(0,1).cpu().detach().numpy()], axis=1)

                    else:
                        # This is the .forward() method adapted to store all the interediate tensors
                        with torch.no_grad():
                            # when using custom sparse linear layers the output is post linear->drop->transform
                            # That means that what we should return is the final output and all the relu outputs
                            tensor_out = [x]
                            for l in model.layer_list:
                                tensor_out.append(l(tensor_out[-1]))

                        tensor_out = [e.cpu().detach().numpy() for e in tensor_out]

                        layer_out_idxs = [
                                i for i in range(len(model.layer_list)) if (
                                    (type(model.layer_list[i]) == torch.nn.modules.activation.ReLU) or 
                                    (i+1 == len(model.layer_list)))]
                        
                        tensor_out = [tensor_out[i] for i in layer_out_idxs]

                        tmp = np.concatenate(tensor_out, axis = 1)

                        # _ = []
                        # for layer in sorted(list(set(lookup.layer))):
                        #     _.append( tensor_out[layer][:, lookup.loc[(lookup.layer == layer), 'idx'].tolist()] )
                        # tmp = np.concatenate(_, axis=1)#.swapaxes(0,1)

                        if type(y_actual) == list:
                            cached_tensor_idx = i_idx
                            y_actual = y.swapaxes(0,1).cpu().detach().numpy()
                            cached_tensor_out = np.zeros((len(inp_dl_G), tmp.shape[1]))

                        else:
                            # cached_tensor_idx = np.concatenate([cached_tensor_idx, i_idx], axis=0) # 1d 
                            cached_tensor_idx = cached_tensor_idx+i_idx 
                            y_actual          = np.concatenate([ y_actual, y.swapaxes(0,1).cpu().detach().numpy()], axis=1)
                            # This is doing way more operations than we need but it might be okay. 
                            # Ideally I would check if the index has already been cached. If so we don't need to calculate it or save it. 
                            # cached_tensor_out[i_idx, ] = tmp

                        # Add in the recovered values (even if they map to multiple inputs)
                        if 1 == max([len(e) for e in i_idx]):
                            # if i_idx's sub lists contain only one value then we can collapse the list and use it like an array. 
                            cached_tensor_out[sum(i_idx, []), ] = tmp
                        else:
                            # If not then we have to iterate through 
                            for j in range(len(tmp)):
                                for jj in i_idx[j]:
                                    cached_tensor_out[jj, ] = tmp[j]


                # Collapse to summary statistics
                rho_val = np.zeros((y_actual.shape[0], cached_tensor_out.shape[1]))
                rho_sig = np.zeros_like(rho_val)


                cached_tensor_idx = sum([[e[0]] for e in cached_tensor_idx], []) # if there are duplicate associations for a input arbitraily retain the 0th lookup.
                for y_i in tqdm(range(rho_val.shape[0]), ascii = True, desc = 'Calculating spearman\'s rho for all y vars'):
                    for n_i in tqdm(range(rho_val.shape[1]), leave = False, ascii = True, desc = '... and intermediates'):
                        if ((y_actual[y_i, :].std() == 0.0) | 
                            (cached_tensor_out[cached_tensor_idx, n_i].std() == 0.0)): # expand out tensor to account for duplicate inputs
                            # (node_out[n_i, :].std() == 0.0)):
                            val, sig = np.nan, np.nan
                        
                        else:
                            val, sig = scipy.stats.spearmanr( y_actual[y_i, :], cached_tensor_out[cached_tensor_idx, n_i] )
                        
                        rho_val[y_i, n_i] = val
                        rho_sig[y_i, n_i] = sig

                output_tracker = pd.concat([
                    output_tracker, 
                    pd.DataFrame(rho_val.swapaxes(0,1), columns=y_names),
                    pd.DataFrame(rho_sig.swapaxes(0,1), columns=[e+'_sig' for e in y_names])    
                    ], axis=1)

                # Add lookup info (node names)
                output_tracker = lookup.loc[:, ['node_forward_idx', 'node']].drop_duplicates().merge(output_tracker)
                return output_tracker

            trn_rho = _collect_rho(model = model, M_list = M_list, inp_dl = training_dataloader)
            pq.write_table(pa.Table.from_pandas(trn_rho), save_dir+f'/{params_data_subset_hash}_eval_rho_nodewise_trn.parquet')
            del trn_rho

            val_rho = _collect_rho(model = model, M_list = M_list, inp_dl = validation_dataloader)
            pq.write_table(pa.Table.from_pandas(val_rho), save_dir+f'/{params_data_subset_hash}_eval_rho_nodewise_val.parquet')
            del val_rho

    case _:
        print('Base case not implemented!!')
        
print('Done')