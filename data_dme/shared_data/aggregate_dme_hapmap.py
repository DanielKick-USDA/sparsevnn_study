# The goal of this file is to produce a tidy hapmap like parquet
# and a phenotype file, with some heterozygosity (hzg) both 
# containing only intersecting taxa:
# - SNPtable_AB_RILs_inferred_Release4Conv6_hzg.hmp.txt.parquet 
# - ADHReplicates_hzg.csv

# Because this process is slow, intermediate parquets are written
# in case this has to be run in several sessions. The inports here
# _shouldn't_ be version sensitve, but use sparsevnn.sif to be safe.

import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy  as np
from   tqdm import tqdm

phno = pd.read_table('./ADHReplicates.TruncRILs.110729.txt')
uniq_rils = sorted([str(e) for e in list(set(phno.RIL))])


def load_and_encode(file_path = './SNPtable_A_RILs_inferred_Release4Conv6_3L.hapmap.txt'):
    M = pd.read_table(file_path)
    M = M.drop(columns=['alleles'])
    M = M.drop_duplicates().reset_index(drop=True)

    left = M.loc[:, ['chrom', 'pos']]
    right = M.drop(columns=['chrom', 'pos'])
    right_cols = list(right)
    right = right.to_numpy()
    # these files seem to only have repeated nucleotides
    rnucs = {f"{['A', 'C', 'G', 'T', 'N'][i]}/{['A', 'C', 'G', 'T', 'N'][i]}": i for i in range(5)}

    right2 = np.zeros_like(right)
    for k in rnucs.keys():
        # note that the default value of get here doesn't do anything because we iterate over keys
        # I have included it because this makes what is happening more clear. 
        right2[right == k] = rnucs.get(k, -1)

    del right
    assert -1 not in right2
    M = pd.concat([left, pd.DataFrame(right2, columns = right_cols)], axis = 1)    
    return M


def _dedupe_table(x, uniq_rils):

    MA = load_and_encode(file_path = f'./SNPtable_A_RILs_inferred_Release4Conv6_{x}.hapmap.txt')
    MB = load_and_encode(file_path = f'./SNPtable_B_RILs_inferred_Release4Conv6_{x}.hapmap.txt')

    M = pd.merge(
        MA, 
        MB,
        on  = ['chrom', 'pos'], 
        how = 'inner'
    )

    return M


if not os.path.exists('./SNPtable_AB_RILs_inferred_Release4Conv6.hmp.txt.parquet'):

    res = [_dedupe_table(x = x, uniq_rils=uniq_rils) for x in tqdm(['2L', '2R', '3L', '3R', 'X'])]
    _ = pd.concat(res)

    table = pa.Table.from_pandas(_)

    pq.write_table(table, './SNPtable_AB_RILs_inferred_Release4Conv6.hmp.txt.parquet')


if not os.path.exists('./SNPtable_AB_RILs_inferred_Release4Conv6_hzg.hmp.txt.parquet'):
    _ = pq.read_table('./SNPtable_AB_RILs_inferred_Release4Conv6.hmp.txt.parquet').to_pandas()

    right = _.drop(columns=['chrom', 'pos']).to_numpy()
    # drop 0 variance rows
    _min = np.min(right, axis = 1)
    _max = np.max(right, axis = 1)
    mask = ((_max - _min) == 0)

    del right
    _ = _.loc[~mask, ]
    _ = _.drop_duplicates()
    table = _
    table = table.reset_index(drop=True)

    # back convert to nucleotides and save
    right = table.drop(columns=['chrom', 'pos']).to_numpy()

    # drop cols to save memory
    col_names = list(table.drop(columns=['chrom', 'pos']))
    table = table.loc[:, ['chrom', 'pos']]

    # a lot faster to create and fill a new array than convert the existing one
    out = np.array(['']).repeat(np.array(right.shape).prod()
                    ).reshape((right.shape[0], right.shape[1]))

    int2nuc = {i:f"{['A', 'C', 'G', 'T', 'N'][i]}" for i in range(5)}
    for k in int2nuc:
        out[right == k] = int2nuc[k]

    right = pd.DataFrame(out, columns=col_names)
    del out

    table = pd.concat([table, right], axis=1)
    pq.write_table(pa.Table.from_pandas(table), './SNPtable_AB_RILs_inferred_Release4Conv6_hzg.hmp.txt.parquet') 

    # constrain to only the phenotypes with genotypic data
    mask = phno.RIL.isin( [int(e) for e in uniq_rils if e not in list(table)] )
    phno = phno.loc[~mask, ['RIL', 'AdjADHSlope'] ].reset_index(drop = True)
    phno.to_csv('./ADHReplicates_hzg.csv')





























