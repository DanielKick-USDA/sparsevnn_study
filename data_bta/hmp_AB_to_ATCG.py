
import pandas as pd

fp = './shared_data/ASA_100K_Test.hmp.txt' # file path to hmp.txt
M = pd.read_table(fp)

meta_cols = ['rs#', 'alleles', 'chrom', 'pos', 'strand', 'assembly#', 'center',
             'protLSID', 'assayLSID', 'panelLSID', 'QCcode']
data_cols = [e for e in list(M) if e not in meta_cols]

enc = dict(zip(['A', 'X', 'N', '-'], 
               ['A', 'T', 'C', 'G']))

Mn  = M.loc[:, data_cols].to_numpy()
Mno = Mn.copy()
for e in list(enc.keys()):
    Mno[Mn == e] = enc[e]

M = pd.concat([
    M.loc[:, meta_cols], 
    pd.DataFrame(Mno, columns=data_cols)], 
    axis = 1)

# update the alleles column?
M.to_parquet(fp+'.parquet')
