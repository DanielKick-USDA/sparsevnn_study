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


from   sparsevnn.clifn import \
    s2b,                      \
    as_lat,                   \
    _get_json_if_exists

# training
from   sparsevnn.clifn import \
    train_one_model,          \
    vnn_from_state_dict,      \
    evaluate,                 \
    _shrink_M_list

# prediciton and evaluation
from   sparsevnn.clifn import  \
    _collect_predictions,      \
    _collect_salience_snpwise, \
    _plt_saliences,            \
    _collapse_and_add_gff_annotations, \
    _collapse_salience_genewise, \
    collect_gradients,         \
    _collect_rho
 

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
params      = sparsevnn.qol.params()
params_list = sparsevnn.qol.params_list()
# values in `params_data` and `params_run` are the only ones I expect to be updated
params_data_keys = set(params_data.keys()).copy()
params_run_keys  = set(params_run.keys()).copy()



# Update Default Parameters ---------------------------------------------------
# Default < JSON < Arguments

parser = argparse.ArgumentParser()

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
parser.add_argument("--eval",          type=str, nargs='*', action='append', help="")
# --eval can be passed multiple times to create a list of 0 or more evaluations to be run. 
# per https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# at the time of writing, valid options are `saliency_inp`, `saliency_wab`, `rho_out`

args = parser.parse_args()

if args.eval != None:
    args.eval = as_lat(args.eval)

## JSON parameter files =======================================================
# in place update all k/v if an external file was passed.
# if args.params_data: params_data |= sparsevnn.qol.read_json(json_path=args.params_data)
# if args.params_run:  params_run  |= sparsevnn.qol.read_json(json_path=args.params_run)
# if args.params:      params      |= sparsevnn.qol.read_json(json_path=args.params)
# if args.params_list: params_list |= sparsevnn.qol.read_json(json_path=args.params_list)

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
# <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>


# FIXME -- eventually. This is a quick fix to use pickled data. 
if type(params_data['model_path']) is not type(None):
    if params_data['model_path'] != '':
        params_data['model_path'] = params_data['model_path'].replace('.pt', '_mod.pkl')


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


# Load Input Data -------------------------------------------------------------
if use_data_cache_load:
    phno           = sparsevnn.qol.read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_phno.parquet')
    obs_geno_lookup= sparsevnn.qol.read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_obs_geno_lookup.parquet')
    cxn            = sparsevnn.qol.read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_cxn.parquet')
    acgt_loci      = sparsevnn.qol.read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_acgt_loci.parquet')
    gene_nodes_gff = sparsevnn.qol.read_pq_or_pd(file_path=f'./vnn_cache/{params_data_subset_hash}_gene_nodes_gff.parquet')

    acgt = np.load(f'./vnn_cache/{params_data_subset_hash}_acgt.npz')
    acgt = acgt['acgt']

    with open(f'./vnn_cache/{params_data_subset_hash}_inp_node_idx_dict.json', 'r') as f:
        inp_node_idx_dict = json.load(f)

    # calculate 
    y = np.array(phno.drop(columns='Taxa'))
    y_names = list(phno.drop(columns='Taxa'))

elif not use_data_cache_load:
    ## Load Marker Data ===========================================================
    hmp = sparsevnn.qol.read_pq_or_pd(file_path=params_data['hmp_path'])
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
    phno = sparsevnn.qol.read_pq_or_pd(file_path=params_data['phno_path'])
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
            cxn = sparsevnn.qol.read_pq_or_pd(file_path=params_data['graph_cxn'])

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
        np.savez_compressed(f'./vnn_cache/{params_data_subset_hash}_acgt.npz', acgt=acgt)

        ## Dict
        sparsevnn.qol.write_json(inp_node_idx_dict, f'./vnn_cache/{params_data_subset_hash}_inp_node_idx_dict.json') 


# Training Prep. --------------------------------------------------------------
## Model Prep. ================================================================
cxn_dict = sparsevnn.util.convert_connections(inp=cxn, to='dict', node_names=None)


myvnn = sparsevnn.util.mk_vnnhelper(
        edge_dict = cxn_dict,
        num_nucleotides = 4, # this could also be 1 for major/minor allele. 
        inp_tensor_lookup = inp_node_idx_dict,
        params = params
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
        #print(f'in tuning') 
        sparsevnn.qol.ensure_dir_path_exists(dir_path = lightning_log_dir)
        json_path = f"{lightning_log_dir}/{exp_name}.json"
        # overwrite params_list's output with the size with the right output size. Don't allow the user to enter the wrong value. 
        # This means we don't need to worry much about re-using these values. 
        i = [i for i in range(len(params_list)) if params_list[i]['name'] == 'default_out_nodes_out'][0]
        params_list[i]['value'] = y.shape[1]

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
        #print(f'in tuning: loaded_json{loaded_json}') 
        # Check if running hyps is forced or if we haven't yet reached tune_max
        run_trials_bool = True
        if params_run['tune_force'] == False:
            if loaded_json: 
                # check if we've reached the max number of hyperparamters combinations to test
                if params_run['tune_max'] <= (ax_client.generation_strategy.trials_as_df.index.max()+1):
                    run_trials_bool = False

        # Run trials 
        #print(f'in tuning: run_trials_bool{run_trials_bool}') 
        if run_trials_bool:
            for i in range(params_run['tune_trials']):
                run_next_trial_bool = False
                #print(f'in tuning: run_next_trial_bool{run_next_trial_bool} i: {i}') 
                print(f"{ params_run['tune_max'] }") 
                if type(None) != type(ax_client):
                    #print(f"{ax_client.generation_strategy.trials_as_df.index.max()+1}")


                # At each step check if we have reached the maximum. This allows us to request extra trials in case initialization fails
                #if type(None) == type(ax_client.generation_strategy.trials_as_df):
                    run_next_trial_bool = True
                    #print('a')
                elif params_run['tune_max'] >= (ax_client.generation_strategy.trials_as_df.index.max()+1):
                    run_next_trial_bool = True
                    #print('b')
                #print(f'in tuning: run_next_trial_bool{run_next_trial_bool} i: {i}') 

                if run_next_trial_bool:
                    #print(f'in tuning: running')
                    parameterization, trial_index = ax_client.get_next_trial()
                    # Local evaluation here can be replaced with deployment to external system.


                    ax_client.complete_trial(
                        trial_index=trial_index, 
                        raw_data=evaluate(
                            parameterization = parameterization,
                            cxn_dict = cxn_dict,
                            inp_node_idx_dict = inp_node_idx_dict,
                            lightning_log_dir = lightning_log_dir,
                            exp_name = exp_name,
                            params_data = params_data,
                            params_run = params_run,
                            training_dataloader = training_dataloader,
                            validation_dataloader = validation_dataloader,
                            )) # NOTE plDNN_general and other functions can be patched

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
            log_dir= model_log_dir,
            exp_name = exp_name,
            training_dataloader = training_dataloader,
            validation_dataloader = validation_dataloader
            )

        # Turn all the tensors holding shape and index info into ints. This _massively_ reduces the space to save it.
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


            # NOTE
            # save the full pickled model. This is to get around an issue that seems to be introuded by torch.save
            with open(f'{log_path}/version_{max(nums)}/version_{max(nums)}_mod.pkl', 'wb') as f:
                pickle.dump(res.model.mod, f, protocol=5)


            # Get the most recent history
            history = pd.read_csv(f'{log_path}/version_{max(nums)}/metrics.csv')
            # pivot longer
            history = pd.concat([
                history.drop(columns='val_loss').rename(columns={'train_loss':'loss'}).dropna().assign(split='trn'),
                history.drop(columns='train_loss').rename(columns={'val_loss':'loss'}).dropna().assign(split='val')]
            )

            plt = px.line(history, x = 'step', y = 'loss', color='split')

            plt.write_html(f'{log_path}/version_{max(nums)}/metrics.html')
            plt.write_image(f'{log_path}/version_{max(nums)}/metrics.svg') 


    case 'predict':
        print('Running Prediction')

        if (params_data['model_path'].split('.')[-1] == 'pt'): # allow for legacy behavior
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
        else:
            # the path should be something like './models/vnn/version_1/version_1_mod.pkl'
            with open(params_data['model_path'], 'rb') as f:
                model = pickle.load(f)

        model = model.eval()

        save_dir = '/'.join(params_data['model_path'].split('/')[0:-1])

        pq.write_table(pa.Table.from_pandas(pd.DataFrame(y_c.numpy()[:,None].T, columns=y_names)), save_dir+f'/{params_data_subset_hash}_yvar_cs_center.parquet')
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(y_s.numpy()[:,None].T, columns=y_names)), save_dir+f'/{params_data_subset_hash}_yvar_cs_scale.parquet')



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
        save_dir = '/'.join(params_data['model_path'].split('/')[0:-1])

        print('Running Evaluation')
        if (params_data['model_path'].split('.')[-1] == 'pt'): # allow for legacy behavior
            model = vnn_from_state_dict(
                state_dict_path = params_data['model_path'],
                # Here we read in the associated json in case the model specification is different
                # from the current params file.
                params = sparsevnn.qol.read_json('.'.join(params_data['model_path'].split('.')[0:-1]+['json'])),
                params_data = params_data,
                edge_dict = cxn_dict,
                inp_tensor_lookup = inp_node_idx_dict,
                ) 
        else:
            # the path should be something like './models/vnn/version_1/version_1_mod.pkl'
            with open(params_data['model_path'], 'rb') as f:
                model = pickle.load(f)

        model = model.eval()


        if 'saliency_inp' in params_run['eval']:
            print('Calculating salience')
            model = model.eval()

            # Training      
            if params_data['dataloader_shuffle_train'] == False:
                print('Calculating salience w.r.t. input (snp-wise) -- training')
                training_inp_sals   = _collect_salience_snpwise(model = model, inp_dl = training_dataloader, params_data = params_data)

                salience = _collapse_and_add_gff_annotations(
                    acgt_loci = acgt_loci, 
                    gene_nodes_gff = gene_nodes_gff, 
                    e = training_inp_sals)
                pq.write_table(pa.Table.from_pandas(salience), save_dir+f'/{params_data_subset_hash}_eval_salience_snpwise_trn.parquet')
                _ = _plt_saliences(salience = salience, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_trn_salience_snpwise')  

                print('Calculating salience w.r.t. input (gene-wise) -- training')
                training_inp_sals   = _collapse_salience_genewise(M_list = M_list, acgt_loci = acgt_loci, sals = training_inp_sals, params_data = params_data)
                _ = _plt_saliences(salience=training_inp_sals,   save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_trn_salience_genewise')
                pq.write_table(pa.Table.from_pandas(training_inp_sals), save_dir+f'/{params_data_subset_hash}_eval_salience_genewise_trn.parquet')

                del salience
                del training_inp_sals


            # Validation
            if params_data['dataloader_shuffle_valid'] == False:                  
                print('Calculating salience w.r.t. input (snp-wise) -- training')
                validation_inp_sals = _collect_salience_snpwise(model = model, inp_dl = validation_dataloader, params_data = params_data)            
                salience = _collapse_and_add_gff_annotations(
                    acgt_loci = acgt_loci, 
                    gene_nodes_gff = gene_nodes_gff, 
                    e = validation_inp_sals)
                pq.write_table(pa.Table.from_pandas(salience), save_dir+f'/{params_data_subset_hash}_eval_salience_snpwise_val.parquet')
                _plt_saliences(salience = salience, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_val_salience_snpwise')  

                print('Calculating salience w.r.t. input (gene-wise) -- validation')
                validation_inp_sals = _collapse_salience_genewise(M_list = M_list, acgt_loci = acgt_loci, sals = validation_inp_sals, params_data = params_data)
                _ = _plt_saliences(salience=validation_inp_sals, save_dir = save_dir, plt_prefix = f'/{params_data_subset_hash}_eval_val_salience_genewise')
                pq.write_table(pa.Table.from_pandas(validation_inp_sals), save_dir+f'/{params_data_subset_hash}_eval_salience_genewise_val.parquet')

                del salience
                del validation_inp_sals


        if 'saliency_wab' in params_run['eval']:
            print('Calculating salience w.r.t. weights and biases')
            
            training_weight_grads   = collect_gradients(model = model, inp_dl = training_dataloader)
            validation_weight_grads = collect_gradients(model = model, inp_dl = validation_dataloader)

            coordinates = [e.connectivity for e in model.layer_list]
            # rearrange into records of (layer, key, start, stop)
            
            col_info = [[
                (layer, 
                k, 
                int(M_list[layer].col_info[k]['start']), 
                int(M_list[layer].col_info[k]['stop']) )
                for k in M_list[layer].col_info.keys() 
                ] for layer in 
                [layer for layer in range(len(M_list))]]

            # get rid of nested lists
            col_info = sum(col_info, [])

            _ = pd.DataFrame(col_info, columns=['layer', 'k', 'start', 'stop'])

            df_weight = []
            df_bias = []

            for i in tqdm(_.index, ascii = True, desc = 'Appending weights and biases'):
                # print(i)
                layer, k, start, stop = _.loc[i, ].values
                # the first two [0] are to get the first index of the (2, #) coordinate tensor
                matching_idx = torch.where(((coordinates[layer][0] >= start) & (coordinates[layer][0] <  stop)))
                df_weight.append(pd.DataFrame(
                    {
                    'layer':layer, 
                    'k':k, 
                    'start':start, 
                    'stop':stop,
                    'tensor_idx': matching_idx[0],
                    'trn_weight_grads': training_weight_grads[0][layer][matching_idx[0]], # weights
                    # 'trn_bias_grads': training_weight_grads[1][layer][matching_idx[0]], # bias
                    'val_weight_grads': validation_weight_grads[0][layer][matching_idx[0]], # weights
                    # 'val_bias_grads': validation_weight_grads[1][layer][matching_idx[0]] # bias
                    }
                    ))
                
                # NOTE That because weights are spread across 2 dims we have to use a lookup to turn the 1d representation into 2. 
                # Bias doesn't have that problem so we can directly index them.
                df_bias.append(pd.DataFrame(
                    {
                    'layer':layer, 
                    'k':k, 
                    'start':start, 
                    'stop':stop,
                    'tensor_idx': [j for j in range(start, stop)],
                    'trn_bias_grads': training_weight_grads[1][layer][start:stop], # bias
                    'val_bias_grads': validation_weight_grads[1][layer][start:stop] # bias
                    }
                    ))
            
            df_weight = pd.concat(df_weight)
            df_bias   = pd.concat(df_bias)

            pq.write_table(pa.Table.from_pandas(df_weight), save_dir+f'/{params_data_subset_hash}_eval_gradients_nodewise_weights.parquet')
            pq.write_table(pa.Table.from_pandas(df_bias),   save_dir+f'/{params_data_subset_hash}_eval_gradients_nodewise_bias.parquet')
            del training_weight_grads
            del validation_weight_grads
            del coordinates
            del col_info
            del _
            del df_weight
            del df_bias

        if 'rho_out' in params_run['eval']:
            print('Calculating rho w.r.t. intermediate layer output')
            # NOTE: this can take a long time.

            trn_rho = _collect_rho(model = model, M_list = M_list, inp_dl = training_dataloader, y_names = y_names)
            pq.write_table(pa.Table.from_pandas(trn_rho), save_dir+f'/{params_data_subset_hash}_eval_rho_nodewise_trn.parquet')
            del trn_rho

            val_rho = _collect_rho(model = model, M_list = M_list, inp_dl = validation_dataloader, y_names = y_names)
            pq.write_table(pa.Table.from_pandas(val_rho), save_dir+f'/{params_data_subset_hash}_eval_rho_nodewise_val.parquet')
            del val_rho

    case _:
        print('Base case not implemented!!')
        
print('Done')