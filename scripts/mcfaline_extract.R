# Extract the raw counts + cell metadata of the McFaline-Figueroa et al. 2024 sci-Plex-GxE kinome screen
# (GEO GSM7056149, sciPlexGxE_2_preprocessed_cds.list.RDS.gz: a list of three monocle3 cell_data_sets, one per line,
# gzip-compressed twice) into the flat layout of scripts/viperturb_extract.R, one directory per line.
# monocle3 is not installed; S4 slots are plain attributes, so attr() reaches them. R only moves bytes.
#
#   Rscript scripts/mcfaline_extract.R <in.RDS.gz> <out_dir>      -> <out_dir>/<line>/{i,p,x}.bin, genes.txt,
#                                                                    ensembl.txt, cells.txt, meta.csv, info.txt
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
x <- readRDS(gzcon(gzfile(args[1])))
for (line in names(x)) {
  o <- x[[line]]; d <- file.path(args[2], line)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  cnt <- attr(get("data", envir = attr(attr(o, "assays"), ".xData")), "listData")$counts   # ShallowSimpleListAssays
  if (!inherits(cnt, "dgCMatrix")) stop(line, ": counts are ", class(cnt)[1], ", expected dgCMatrix")
  if (any(cnt@x != round(cnt@x)) || any(cnt@x < 0)) stop(line, ": counts are not raw integers")
  md <- as.data.frame(attr(attr(o, "colData"), "listData"), stringsAsFactors = FALSE)
  rr <- attr(o, "rowRanges")
  sym <- attr(attr(rr, "elementMetadata"), "listData")$gene_short_name
  ens <- attr(rr, "partitioning")@NAMES
  if (length(sym) != nrow(cnt) || length(ens) != nrow(cnt)) stop(line, ": row annotation length differs from counts")
  write.csv(md, file.path(d, "meta.csv"), row.names = FALSE)
  writeLines(as.character(sym), file.path(d, "genes.txt"))
  writeLines(sub("\\..*$", "", as.character(ens)), file.path(d, "ensembl.txt"))
  writeLines(as.character(md$cell), file.path(d, "cells.txt"))
  wb <- function(v, f) { con <- file(f, "wb"); n <- length(v); s <- 1
    while (s <= n) { e <- min(n, s + 2e8 - 1); writeBin(as.integer(v[s:e]), con, size = 4L); s <- e + 1 }; close(con) }
  wb(cnt@i, file.path(d, "i.bin")); wb(cnt@p, file.path(d, "p.bin")); wb(cnt@x, file.path(d, "x.bin"))
  writeLines(c(paste("n_genes", nrow(cnt)), paste("n_cells", ncol(cnt)), paste("nnz", length(cnt@x)),
               paste("source", basename(args[1])), paste("line", line)), file.path(d, "info.txt"))
  cat(line, ":", nrow(cnt), "genes x", ncol(cnt), "cells, nnz", length(cnt@x), "\n"); flush.console()
}
