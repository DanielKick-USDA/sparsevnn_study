#!/usr/bin/env Rscript
# example usage is gapit_cli.R  --phno_file '../../shared_data/phno.csv' --geno_file '../../shared_data/example.hmp.txt' --hash '0f64b52c70' --hash_dir '../dnn/vnn_cache/' --out_dir './'
# note that only --phno_file and --geno_file are required arguments. If no hash is passed then the full hapmap will be used.
# otherwise the loci file written by dnn.py will be read and 

library(arrow)
library(vroom)
library(tidyverse)
library(GAPIT)

args = commandArgs(trailingOnly=TRUE)
# test if there is at least one named argument: if not, return an error
if (length(args)<2) {
	#FIXME
	stop("Expected arguments are `phno_file`:path `geno_file`:path with optional arguments `hash`:string `inp_dir`:string `out_dir`:string", call.=FALSE)
} else if(!('--hash' %in% args)){
	stop('--hash must be provided to supply the hash code of the data.')
}

print(args)
# confirm we have even arguments
stopifnot((length(args) %% 2) == 0)

args_names  <- args[seq(1, length(args), 2)]
args_values <- args[seq(2, length(args), 2)]
names(args_values) <- args_names
args <- args_values

# apply defaults if applicable
get_arg <- function(args, key, default){
    if(key %in% names(args)){
        # return value without name
	out <- as.character(args[key])
    } else {
        out <- default
    }
    return(out)
}

phno_file <- get_arg(args, key='--phno_file', '../../shared_data/phno.csv')
geno_file <- get_arg(args, key='--geno_file', '../../shared_data/notreal.hmp.txt')
hash    <- get_arg(args, key='--hash',    '000000000')
hash_dir <- get_arg(args, key='--hash_dir', '../dnn/vnn_cache/')
out_dir <- get_arg(args, key='--out_dir', './')


# load in hapmap. expect either text table or a parquet.
if(stringr::str_detect(geno_file, '.+\\.parquet$')){
    hmp <- arrow::read_parquet(geno_file)

} else {
    hmp  <- vroom::vroom(geno_file, delim = '\t')
}

hmp  <- as_tibble(hmp) |> mutate(chrom = as.character(chrom))
hmp_order <- names(hmp)

# only use hash if it is not default. Assume no hash of all 0s. 

if(hash != '000000000'){
    loci <- arrow::read_parquet(paste0(hash_dir, hash, '_acgt_loci.parquet'))
    # make sure the joining cols are characters
    loci <- as_tibble(loci) |> mutate(chrom = as.character(chrom))
    # filter down
    # This should practically be a left join, but inner will be slightly safer.
    hmp <- inner_join(loci, hmp, by=c('chrom', 'pos')) 
    # get ordering back to what it should be
    hmp <- hmp[, hmp_order]

    # to get this in the same format as if it were read in as a df of text
    # we need to set his to a matrix  (so the datatype is cast to text)
    # then rbind on the names 
    hmp_mn <- as.matrix(names(hmp))
    hmp_m  <- as.matrix(hmp)
    hmp    <- data.frame( rbind(t(hmp_mn), hmp_m))
} else {
    hmp    <- as.data.frame(hmp)
}

# have to set names to V1-V#
names(hmp) <- paste0('V', seq(1, dim(hmp)[2]))

ydf = read.csv(phno_file)
# There's a chance that there will be an index column. If there is drop it. 
ydf = ydf[, names(ydf)[names(ydf) != "X"]]

#FIXME! this part is failing on drosophila data.

myGAPIT <- GAPIT(
		 Y=ydf,
		 G=hmp,
		 PCA.total=3,
		 model="BLINK"
		 )

# I don't think GAPIT has an option to set an output directory. We'll move everything instead.
mv_files <- list.files()
mv_files <- mv_files[stringr::str_starts(mv_files, 'GAPIT.')]
for(mv_file in mv_files){
    cp_status <- file.copy(paste0('./', mv_file), paste0(out_dir, mv_file))
    if(cp_status){
        unlink(paste0('./', mv_file))        
    }
}




#  [1] "--phno_file"
#  [2] "../../shared_data/ADH_tiny_hzg.csv"
#  [3] "--geno_file"
#  [4] "../../shared_data/ADH_tiny_hzg.hmp.txt.parquet"
#  [5] "--hash"
#  [6] "6784d794cc"
#  [7] "--hash_dir"
#  [8] "../dnn/vnn_cache/"
#  [9] "--out_dir"
# [10] "./models/gwas/"



phno_file <- "../../shared_data/ADH_tiny_hzg.csv"
geno_file <- "../../shared_data/ADH_tiny_hzg.hmp.txt.parquet"
hash <- "6784d794cc"
hash_dir <- "../dnn/vnn_cache/"
out_dir <- "./models/gwas/"