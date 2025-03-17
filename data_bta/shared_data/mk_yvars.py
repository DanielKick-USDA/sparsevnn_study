import pandas as pd

M = pd.read_csv('ASA_Phenotypes.csv')
yvars = ['birth_wt', 'wean_wt', 'yrlng_wt']

for yvar in yvars:
    tmp = M.loc[:, ['animal']+[yvar]].rename(columns={'animal':'Taxa'})
    mask = tmp[yvar].notna()
    tmp.loc[mask, ].to_csv(f'./{yvar}.csv', index=False)