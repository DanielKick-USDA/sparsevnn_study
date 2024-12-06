#!/usr/bin/env Rscript
# example usage is gapit_cli.R  --phno_file '../../shared_data/phno.csv' --geno_file '../../shared_data/example.hmp.txt' --hash '0f64b52c70' --hash_dir '../dnn/vnn_cache/' --out_dir './'
# note that only --phno_file and --geno_file are required arguments. If no hash is passed then the full hapmap will be used.
# otherwise the loci file written by dnn.py will be read and used.
# --max_snps <= 0 will be ignored. otherwise it will find max_snps/chromosomes snps spaced by distance in clustering.

library(arrow)
library(vroom)
library(tidyverse)
library(GAPIT)

args = commandArgs(trailingOnly=TRUE)
# test if there is at least one named argument: if not, return an error
if (length(args)<2) {
	#FIXME
	stop(paste0(
        "Expected arguments are ",
        "`phno_file`:path ", 
        "`geno_file`:path ", 
        "with optional arguments ", 
        "`max_snps`:int ", 
        "`hash`:string ", 
        "`hash_dir`:string ", 
        "`out_dir`:string"
    ), call.=FALSE)
} else if(!('--phno_file' %in% args)){
	stop('--phno_file must be provided.')
} else if(!('--geno_file' %in% args)){ 
	stop('--geno_file must be provided.')
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
max_snps    <- get_arg(args, key='--max_snps',    '-1')
hash    <- get_arg(args, key='--hash',    '000000000')
hash_dir <- get_arg(args, key='--hash_dir', '../dnn/vnn_cache/')
out_dir <- get_arg(args, key='--out_dir', './')

max_snps <- as.integer(max_snps)


ydf = read.csv(phno_file)
# There's a chance that there will be an index column. If there is drop it. 
ydf = ydf[, names(ydf)[names(ydf) != "X"]]
# make sure that Taxa is the first col
ydf = ydf[, c('Taxa', names(ydf)[names(ydf) != "Taxa"])]
ydf = ydf |> mutate(Taxa = as.character(Taxa))


# load in hapmap. expect either text table or a parquet.
if(stringr::str_detect(geno_file, '.+\\.parquet$')){
    hmp <- arrow::read_parquet(geno_file)

} else {
    hmp  <- vroom::vroom(geno_file, delim = '\t')
}

hmp  <- as_tibble(hmp) |> mutate(chrom = as.character(chrom))
hmp_order <- names(hmp)

shared_taxa <- hmp_order[hmp_order %in% unique(ydf$Taxa)]
# Filter ydf & select cols in hmp that are taxa shared between the two. 
ydf <- ydf |> filter(Taxa %in% shared_taxa)
hmp_order <- c("chrom", "pos", shared_taxa)


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
} else {
    hmp <- as.data.frame(hmp)
    hmp <- hmp[, hmp_order]
}

if(dim(hmp)[1] <= length(shared_taxa)){
    warning(paste(
        'shared_taxa >= snps!',
        paste('taxa:', length(shared_taxa)), 
        paste('snps:', dim(hmp)[1]))
        )
}


# constrain snps if need be
obs_snp <- dim(hmp)[1]
if((0 < max_snps) & (max_snps < obs_snp )){
    chrom_u <- unique(hmp$chrom)
    snp_per_chrom <- floor(max_snps/length(chrom_u))
    # This will use hierarcical clustring to find k clusters in the loci within one chromosome
    hclust_select_idx <- function(df, snp_per_chrom){
        if(class(df) == "data.frame"){
            # tibbles and df have slightly different semantics. This will ensure we're looking at a vector.
            df <- df$pos
        }

        if(length(df) >= 8192){
            # if length is over 65536L (2**16) then we will need to split this into chunks
            # This is a limit in R's hclust
            # calculating the distance matrix is a pain for large arrays so I'm using a smaller chunk size than needed. 
            num_bins <- ceiling(length(df) / (2**13))
            df_bins <- cut(df, num_bins)
            df_lst <- map(unique(df_bins), function(i){ df[(df_bins == i) ] })
        } else {
            df_lst <- list(df)
        }

        # set df_lst as a list so that we can work with a number of snps above or below the limit for hclust
        snp_per_bin <- floor(snp_per_chrom/ length(df_lst))

        idx_snps <- function(x, snp_per_bin){
            mask <- cutree(hclust(dist(x), method = 'average'), k = snp_per_bin)
            mask_idx <- unlist( map(seq(1, snp_per_bin), function(i){ min(which(mask == i)) }) )
            out <- rep(F, times = length(mask))
            out[mask_idx] <- TRUE
            return(out)
        }

        df_idxs <- map(df_lst, function(x){ idx_snps(x =x, snp_per_bin=snp_per_bin) })

        df_idxs <- do.call(c, df_idxs)
        return(df_idxs)
    }

    hmp[['downsample_bool']] <- TRUE
    pb = txtProgressBar(min = 0, max = length(chrom_u), initial = 0) 
    print('Downsampling Chromosomes:')
    for(chrom_i in chrom_u){
        setTxtProgressBar(pb, which(chrom_i == chrom_u) ) 
        mask = hclust_select_idx(
            df = hmp[(hmp$chrom == chrom_i), 'pos'], 
            snp_per_chrom = snp_per_chrom
            )
        hmp[(hmp$chrom == chrom_i), 'downsample_bool'] <- mask
    }
    hmp <- hmp |> 
        filter(downsample_bool) |> 
        select(-downsample_bool) |> 
        as.data.frame() 
}

# check if the critical cols exist and create them if not. 
if((!('chrom' %in% names(hmp))) |
   (!('pos'   %in% names(hmp))) ){
    # this shouldn't trigger. hmp[, hmp_order] should fail first. 
    stop('Missing either `chrom` or `pos` column in hapmap.')
} 
# Add in the non-fatal missing cols. 
if(!('rs#' %in% names(hmp))){
    hmp[['rs#']] <- paste0(hmp$chrom, '-', hmp$pos)
}

# taxa
hmp_t <- as.matrix(hmp[, shared_taxa])

if(!('alleles' %in% names(hmp))){
    hmp[['alleles']] <- apply(hmp_t, 1, function(x) paste(unique(x), collapse='/') )
}

# to get this in the same format as if it were read in as a df of text
# we need to set his to a matrix  (so the datatype is cast to text)

# metadata
hmp_mn<- as.matrix(names(hmp[, c('rs#', 'alleles', 'chrom', 'pos')]))
hmp_m <- as.matrix(      hmp[, c('rs#', 'alleles', 'chrom', 'pos')])
hmp_m <- data.frame( rbind(t(hmp_mn), hmp_m))

# taxa revisited
hmp_tn <- as.matrix(names(hmp[, shared_taxa]))
hmp_t <- data.frame( rbind(t(hmp_tn), hmp_t))

hmp <- cbind(hmp_m, hmp_t)

# have to set names to V1-V#
names(hmp) <- paste0('V', seq(1, dim(hmp)[2]))

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