#!/usr/bin/env Rscript

# Sample use to run on at most 500 snp*nucl inputs and write results to ./vnn_cache/
# Rscript bWGR_cli.R 'ca4b9c6991' './vnn_cache/' 500

library(jsonlite)
library(arrow)
library(bWGR)

args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("Expected unamed arguments `hash`:string `output_dir`:string `max_inp`:int\n If `max_inp` <= 0 filtering will be skipped ", call.=FALSE)
} else if (length(args)==1) {
  args[2] = './vnn_cache'
} else if (length(args)==2) {
  args[3] = 0
}
# hash <- 'ca4b9c6991'
# output_dir <- './vnn_cache'
# max_inp <- 500 # allow 0 or negative values to skip
print('Running with `hash`, `output_dir`\t`max_inp`')
args[3] <- as.integer(args[3]) # allow 0 or negative values to skip
print(args)
hash       <- args[1]
output_dir <- args[2]
max_inp    <- as.integer(args[3]) # allow 0 or negative values to skip


# Data Prep ----
hash_lookup <- jsonlite::read_json('./vnn_cache/hash_lookup.json')

phno <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_phno.parquet'))
oglu <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_obs_geno_lookup.parquet'))

# holdout info
test_idx  <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_test_idx.parquet')) 
train_idx <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_train_idx.parquet'))

# fix off by one on indices
oglu      <- as.matrix(oglu) +1
test_idx  <- as.matrix(test_idx) +1
train_idx <- as.matrix(train_idx) +1

# should anything be excised from oglu?
if( length(hash_lookup[[hash]][['holdout_taxa_ignore']]) > 0){
  # There should NOT be any entries that need to be removed. This is a sanity check.
  holdout_taxa_ignore <- unlist(hash_lookup[[hash]][['holdout_taxa_ignore']])
  stopifnot(sum(phno$Taxa %in% holdout_taxa_ignore) == 0)
} 

acgt_loci  <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_acgt_loci.parquet'))

# bWGR prep ----
acgt_shape <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_acgt_shape.parquet'))
acgt       <- arrow::read_parquet(paste0('./vnn_cache/', hash,'_acgt.parquet'))
acgt_shape <- acgt_shape$shape
x <- array(acgt$values, dim = rev(acgt_shape))
x <- aperm(x, rev(seq_along(acgt_shape)))
# reshape to obs, (nuc*snp) should have s1 s1 s1 s1, s2, s2, s2, s2 ...
x <- array(x, c(acgt_shape[1], prod(acgt_shape[2:length(acgt_shape)]) ))


# expand out loci ref? not necessary right now.
if (max_inp > 0){
  print(paste0('Filtering to at most ', as.character(max_inp), ' inputs (snp*nucl).'))
  sdx <- apply(x, 2, sd)
  mask <- sdx >= quantile(sdx, (1 - (max_inp/length(sdx))))
  
  # How many values need to be added to the mask due to ties?
  # If there are ties, we may be _under_ the maximum input number
  add_mask <- sum(mask) - max_inp
  if(add_mask > 0){
    tmp <- sdx[mask]
    tmp_cutoff <- sort(tmp)[add_mask]
    mask[mask == TRUE][tmp <= tmp_cutoff] <- FALSE
  }
  x <- x[, mask]
  print(paste0('Filtered to ', as.character(sum(mask)), ' inputs (snp*nucl).'))
}

y <- phno[oglu[,1], 2] #NOTE this assumes only one yhat. Should repeat for all non Taxa cols?
gen  <- x[oglu[,2],  ] # expand to appropriate size
y[test_idx] <- NA #Overwrite y testing set

# bWGR fit and save out ----
fm <- wgr(y, gen, iv=FALSE, pi=0) # don't have to transpose with this version

res <- phno[oglu[,1], ]
res['yhat'] <- fm$hat
res['split'] <- ''
res[train_idx[,1], 'split'] <- 'train'
res[ test_idx[,1], 'split'] <- 'test'



arrow::write_parquet(
  as_arrow_table(data.frame(mu = fm$mu)),
  paste(c(output_dir, '/', hash, '_bWGR_mu.parquet'), collapse = ''))

arrow::write_parquet(
  as_arrow_table(data.frame(b = fm$b)),
  paste(c(output_dir, '/', hash, '_bWGR_b.parquet'), collapse = ''))

arrow::write_parquet(
  as_arrow_table(data.frame(Vb = fm$Vb)),
  paste(c(output_dir, '/', hash, '_bWGR_Vb.parquet'), collapse = ''))

arrow::write_parquet(
  as_arrow_table(data.frame(d = fm$d)),
  paste(c(output_dir, '/', hash, '_bWGR_d.parquet'), collapse = ''))

arrow::write_parquet(
  as_arrow_table(data.frame(Ve = fm$Ve)),
  paste(c(output_dir, '/', hash, '_bWGR_Ve.parquet'), collapse = ''))

arrow::write_parquet(
  as_arrow_table(data.frame(cxx = fm$cxx)),
  paste(c(output_dir, '/', hash, '_bWGR_cxx.parquet'), collapse = ''))
