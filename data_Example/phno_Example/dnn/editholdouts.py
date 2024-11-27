import argparse, json, re

"""
Examples:
# Set holdout taxa ignore to the top 20% of smallest clusters
python editholdouts.py \
    --inp_fp ./shared_data/KMeans_taxa_holdouts.json \
    --out_fp ./params_data.json \
    --action add \
    --attribute holdout_taxa_ignore 


# Remove any taxa that were ignored
python editholdouts.py \
    --inp_fp ./shared_data/KMeans_taxa_holdouts.json \
    --out_fp ./params_data.json \
    --action clear \
    --attribute holdout_taxa_ignore 


# Set holdout_type to taxa and holdout_taxa to cv0
python editholdouts.py \
    --inp_fp ./shared_data/KMeans_taxa_holdouts.json \
    --out_fp ./params_data.json \
    --action add \
    --attribute holdout_taxa \
    --fold 0


# Remove taxa that were held out
python editholdouts.py \
    --inp_fp ./shared_data/KMeans_taxa_holdouts.json \
    --out_fp ./params_data.json \
    --action clear \
    --attribute holdout_taxa
"""


parser = argparse.ArgumentParser()
parser.add_argument("--inp_fp", type=str, help="File path of  `KMeans_taxa_holdouts.json`")
parser.add_argument("--out_fp", type=str, help="File path for `params_data.json` ")
parser.add_argument("--action", type=str, help="Action to be taken. Either `clear` or `add`.")
parser.add_argument("--attribute", type=str, help="Attribute of `params_data.json` to be modified either `holdout_taxa_ignore` or `holdout_taxa`.")
parser.add_argument("--fold", type=str, help="fold to be used expect number e.g. `0` or `fold\d+`")
args = parser.parse_args()

if args.inp_fp: inp_fp       = args.inp_fp  
if args.out_fp: out_fp       = args.out_fp  
if args.action: action       = args.action  
if args.attribute: attribute = args.attribute  
if args.attribute: fold      = args.fold  

def read_json(json_path):
    with open(json_path, 'r') as fp:
        dat = json.load(fp)
    return(dat)

def write_json(obj, json_path):
    with open(json_path, 'w') as f:
        f.write(json.dumps(obj, indent=4, sort_keys=True))

out = read_json(out_fp)
match action:
    case 'clear':
        match attribute:
            case 'holdout_taxa_ignore':
                print('Clearing `holdout_taxa_ignore`')
                out['holdout_taxa_ignore'] = []
                write_json(obj=out, json_path=out_fp)

            case 'holdout_taxa':
                print('Clearing `holdout_taxa`')
                out['holdout_taxa'] = []
                write_json(obj=out, json_path=out_fp)

            case _:
                print('Expected `attribute` to be `holdout_taxa_ignore` or `holdout_taxa`')
                print('No action taken.')

    case 'add':
        # only need inp if we're reading values from it
        inp = read_json(inp_fp)
        match attribute:
            case 'holdout_taxa_ignore':
                print(f'Setting `holdout_taxa_ignore` to the smallest 20% of clusters')
                # set the ignored taxa to the top 20% of clusters 
                # (sorted largest->smallest)
                out['holdout_taxa_ignore'] = inp['q100'] 
                write_json(obj=out, json_path=out_fp)

            case 'holdout_taxa':
                fold = re.findall(r'\d+', fold)[0]
                if fold not in inp['folds']:
                    print(f'No action taken.\nfold {fold} not in {inp["folds"].keys()}!')
                else:
                    print(f'Setting holdout to `taxa` and `holdout_taxa` to those in fold {fold}')
                    out['holdout_type'] = 'taxa'
                    out['holdout_taxa'] = inp['folds'][fold]
                    write_json(obj=out, json_path=out_fp)

            case _:
                print('Expected `attribute` to be `holdout_taxa_ignore` or `holdout_taxa`')
                print('No action taken.')   

    case _:
        print('Expected `action` to be`clear` or `add`')
        print('No action taken.')
