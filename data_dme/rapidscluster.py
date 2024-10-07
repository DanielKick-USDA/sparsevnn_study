import json, argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

import cupy as cp
import cudf
import cuml


parser = argparse.ArgumentParser()
parser.add_argument("--hmp_path",  type=str, help="path to input hapmap file (as text or parquet)")
parser.add_argument("--max_snps",  type=int, help="Maximum SNPs to be retained across all chromosomes")
args = parser.parse_args()

if args.hmp_path:
    hmp_path = args.hmp_path
else: 
    hmp_path = './shared_data/5_Genotype_Data_All_Years.hmp.txt.parquet'
if args.max_snps:
    max_snps = args.max_snps
else:
    max_snps = 37500

# hmp_path = './shared_data/5_Genotype_Data_All_Years.hmp.txt'
# hmp_path = './shared_data/5_Genotype_Data_All_Years.hmp.txt.parquet'
# max_snps = 37500

if hmp_path.split('.')[-1] == 'parquet':
    M = pq.read_table(hmp_path).to_pandas()
else:
    M = pd.read_table(hmp_path)

M = M.sort_values(['chrom', 'pos']).reset_index(drop=True)

# number of chromosomes may vary. Vary the number of sites per chrom accoridingly.
# Where this may burn us is that it effectively up weights small chromosomes and penalizes densely sampled chromosomes
# Maybe the better thing to do is to to order them and select ever nth row. That's not perfect either. 
n_chrom = len(M.chrom.drop_duplicates().tolist()) 
snps_per_chrom = round(max_snps/ n_chrom)

pos_info = ['rs#', 'alleles', 'chrom', 'pos', 'strand', 'assembly#', 'center', 'protLSID', 'assayLSID', 'panelLSID', 'QCcode']
taxa_cols = [e for e in list(M) if e not in pos_info]


# drop any uniform rows
_ = M.loc[:, taxa_cols].nunique(axis = 1)
M = M.loc[(_ != 1), ].reset_index(drop=True)


# get x positions evenly spaced
def get_clusters_in_chrom(pos, k):
    ac = cuml.AgglomerativeClustering(n_clusters=k, metric='euclidean')
    ac.fit(cp.array(pos))
    return ac.labels_

# only run if the number of samples is above a threshold
# NOTE: there could be a bug here if the allowed number of snps is greater than the actual number
if M.shape[0] > max_snps:
    _ = [get_clusters_in_chrom(pos = M.loc[(M.chrom == chrom), 'pos'], k = snps_per_chrom ) for chrom in set(M.chrom)]
    _ = M.loc[:, ['chrom', 'pos']].assign(cluster = np.concatenate([e.tolist() for e in _]))
    _ = _.groupby(['chrom', 'cluster']).agg(pos = ('pos', 'min')).reset_index() # using min instead of median to ensure that even len arrays don't result in a float.  
    _.pos = _.pos.astype(int)
    M = _.merge(M, how = 'left')


Mn = M.loc[:, taxa_cols]
Mn = Mn.to_numpy()

acgt = cp.zeros((Mn.shape[0], Mn.shape[1], 4))
# The absolute most we can push this is ~ (5000, 150000, 4) -> ~23580MiB


def _mk_iupac():
    nuc2np = {
        'A': cp.array([1., 0, 0, 0]), 
        'C': cp.array([0., 1, 0, 0]), 
        'G': cp.array([0., 0, 1, 0]), 
        'T': cp.array([0., 0, 0, 1]), 
        'N': cp.array([.25, .25, .25, .25]), 
        '.': cp.array([0., 0, 0, 0]), 
    }

    iupac = {
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'S': ['G', 'C'],
        'W': ['A', 'T'],
        'K': ['G', 'T'],
        'M': ['A', 'C'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G']
        }

    # convert to array probs
    iupac = {e:cp.concatenate( [ nuc2np[ee][:, None] for ee in iupac[e]], axis=1 ).mean(axis=1) for e in iupac}

    # merge
    nuc2np |= iupac 
    iupac = nuc2np
    return iupac

iupac = _mk_iupac()

for k in iupac:
    acgt[(Mn == k)] = iupac[k]

acgt = acgt.swapaxes(0,1)
acgt = acgt.reshape((acgt.shape[0], -1))

# In the maize data I tested clustering with KMeans, AgglomerativeClustering, & HDBSCAN (DBSCAN is meaningfully slower than HDBSCAN)
# What I found was that 
# 
# KMeans is just fine. Decently large clusters with some skew but not bad. 
# at k=100 for 4928 the biggest are 162, 142, 116 so larger than expected but not terrible.
#
# Agglomerative clustering _really_ wants one big cluster. 
# at k=100 for 4928 the biggest are 4726, 20, 10.
#
# HDBSCAN is sort of intermediate. At alpha = 1.0 minimum samples doesn't matter overly much. 
# for 4928 each returns k=163 with the biggest being 1245, 387, 381 but rapidly falling off to <30

if False:
    from tqdm import tqdm
    def _test_clustering(cls):
        for i in tqdm([True]):
            cls.fit(acgt)
        labs = cls.labels_
        print(f'k= {labs.max()+1}')
        _ = cudf.DataFrame({'lab': labs}).assign(n=1).groupby('lab').count().reset_index().sort_values('n')
        _ = _.tail(20).n.to_arrow().to_pylist()
        _.reverse()
        print(_)
        # return _

    [_test_clustering(cls = cuml.cluster.KMeans(n_clusters = i) ) 
    for i in [10, 20, 50, 100]]

    [_test_clustering(cls = cuml.cluster.AgglomerativeClustering(n_clusters = i) ) 
    for i in [10, 20, 50, 100]]


    [f'{i}\t{j}\t{_test_clustering(cls = cuml.cluster.HDBSCAN(alpha = j, min_samples = i) )}' 
    for j in [1., 0.5, 0.00001] # inner loop
    for i in [1, 10, 20, 50, 100] ]


    for i in [1, 10, 20, 50, 100]:
        for j in [1., 0.5, 0.00001]:
            print(f'{i}\t{j}\t{_test_clustering(cls = cuml.cluster.HDBSCAN(alpha = j, min_samples = i) )}')



for params in (
    ('HDBSCAN', cuml.cluster.HDBSCAN(alpha = 1, min_samples = 5)),
    ('KMeans', cuml.cluster.KMeans(n_clusters = 20))):
    # same location as hmp
    save_dir = '/'.join(hmp_path.split('/')[0:-1])

    cls = params[1]

    cls.fit(acgt)
    labs = cls.labels_

    taxa_report = pd.DataFrame({'Taxa': taxa_cols, 'Cluster': labs.tolist()})

    _ = taxa_report.copy()
    _['n'] = 1
    _ = _.drop(columns=['Taxa']).groupby(['Cluster']).count().reset_index()
    _ = _.sort_values('n', ascending = False).reset_index(drop = True)
    _ = _.assign(pr = (_.n / (_.n.sum())) )
    _ = _.assign(cpr = _.pr.cumsum())

    _ = _.merge(taxa_report)

    _.to_csv(save_dir+f'/{params[0]}_taxa_report.csv')


    # Separate taxa into a 80% (largest clusters)
    t80  = _.loc[(_.cpr <= 0.8), 'Taxa'].tolist()
    t100 = _.loc[(_.cpr > 0.8), 'Taxa'].tolist()

    # get 5 cv folds
    cvf = {i:[] for i in range(5)}
    i = -1
    for e in t100:
        i = 0 if i == 4 else i+1
        cvf[i].append(e)

    # write out
    _ = json.dumps({
        'q80':t80, 
        'q100':t100, 
        'folds':cvf
        }, 
        # indent=4, 
        sort_keys=True)

    with open(save_dir+f'/{params[0]}_taxa_holdouts.json', 'w') as f:
        f.write(_) 