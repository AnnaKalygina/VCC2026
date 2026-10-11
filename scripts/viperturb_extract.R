# Extract the RAW RNA counts + full cell metadata of one VIPerturb-seq Seurat object (Bradu et al. 2026,
# Zenodo 18460279, genome_wide_bin{A,B,C}.RDS) into flat files that src/ingest_viperturb.py reads.
#
# Why this shape: the only public form of these cells is Seurat .rds; SeuratObject/Seurat are NOT installed on
# the cluster R, but S4 slots are plain attributes, so attr() reaches them without the class definitions
# (same trick as scripts/mixscale_ntc.R). All statistics are computed in Python, R only moves bytes.
#
#   Rscript scripts/viperturb_extract.R <in.rds> <out_dir>
#
# Output in <out_dir>:
#   i.bin  int32  row (gene) index, 0-based        -- dgCMatrix@i, genes x cells, column-compressed
#   p.bin  int32  column pointers (n_cells + 1)    -- so (p, i, x) is a CSR of cells x genes
#   x.bin  int32  UMI counts (verified integral before the cast)
#   genes.txt / cells.txt   one name per line, in matrix order
#   meta.csv                the whole meta.data (first column `cell` = rowname), for Python to pick the labels
#   info.txt                dims / nnz / assay description

suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; outdir <- args[2]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

t0 <- Sys.time()
el <- function() sprintf("[%.1f min]", as.numeric(difftime(Sys.time(), t0, units = "mins")))
cat("reading", rds, "\n"); flush.console()
obj <- readRDS(rds)
cat("class:", class(obj), " slots:", paste(names(attributes(obj)), collapse = ", "), el(), "\n")
md <- attr(obj, "meta.data")

assays <- attr(obj, "assays")
active <- attr(obj, "active.assay")
cat("assays:", paste(names(assays), collapse = ", "), " active:", active, "\n")
for (nm in names(assays)) {
  a <- assays[[nm]]
  lay <- attr(a, "layers")
  if (!is.null(lay)) {
    cat(sprintf("  %s: %s, layers %s\n", nm, class(a)[1],
                paste(sprintf("%s[%s]", names(lay), sapply(lay, function(m) paste(dim(m), collapse = "x"))), collapse = ", ")))
  } else {
    cn <- attr(a, "counts")
    cat(sprintf("  %s: %s, counts %s\n", nm, class(a)[1], paste(dim(cn), collapse = "x")))
  }
}
an <- if ("RNA" %in% names(assays)) "RNA" else active
cat("using assay", an, "\n")
rna <- assays[[an]]

# LogMap (Seurat v5 features/cells maps) is an S4 logical matrix; strip the class to index it without SeuratObject
as_lm <- function(x) { y <- unclass(x); attr(y, "package") <- NULL; y }

lay <- attr(rna, "layers")
if (!is.null(lay)) {                                   # Seurat v5 Assay5
  keys <- grep("^counts", names(lay), value = TRUE)
  if (length(keys) == 0) stop("no counts layer among: ", paste(names(lay), collapse = ", "))
  fm <- as_lm(attr(rna, "features")); cm <- as_lm(attr(rna, "cells"))
  allg <- rownames(fm); allc <- rownames(cm)
  parts <- list(); cells <- character(0); genes <- NULL
  for (k in keys) {
    m <- lay[[k]]
    g <- allg[fm[, k]]; cc <- allc[cm[, k]]
    if (length(g) != nrow(m) || length(cc) != ncol(m)) stop("layer ", k, ": LogMap sizes disagree with the matrix")
    if (is.null(genes)) genes <- g
    if (!identical(g, genes)) stop("counts layers carry different gene sets -- not handled")
    parts[[k]] <- m; cells <- c(cells, cc)
    cat(sprintf("  layer %s: %d genes x %d cells, class %s\n", k, nrow(m), ncol(m), class(m)[1]))
  }
  if (length(parts) > 1) cat("WARNING", length(parts), "counts layers -> cbind in layer order\n")
  cnt <- if (length(parts) == 1) parts[[1]] else do.call(cbind, parts)
} else {                                               # Seurat v3/v4 Assay
  cnt <- attr(rna, "counts")
  genes <- rownames(cnt); cells <- colnames(cnt)
}
suppressWarnings(rm(obj, assays, rna, lay, a, cn, m, parts)); invisible(gc())
if (!is(cnt, "dgCMatrix")) { cat("WARNING counts are", class(cnt)[1], "-> coercing to dgCMatrix\n"); cnt <- as(cnt, "CsparseMatrix") }
cat(sprintf("counts: %d genes x %d cells, nnz %.0f %s\n", nrow(cnt), ncol(cnt), length(cnt@x), el()))
if (length(cnt@x) >= 2^31) stop("nnz >= 2^31 does not fit int32 pointers")

# --- meta.data --------------------------------------------------------------------------------
cat("meta.data:", nrow(md), "x", ncol(md), " cols:", paste(colnames(md), collapse = ", "), "\n")
if (nrow(md) != ncol(cnt)) cat("WARNING meta.data rows", nrow(md), "!= count columns", ncol(cnt), "\n")
if (!all(rownames(md) %in% cells)) cat("WARNING", sum(!rownames(md) %in% cells), "meta.data rows have no count column\n")
for (k in colnames(md)) {
  v <- md[[k]]; u <- length(unique(v))
  cat(sprintf("  %-28s %-10s %8d unique; head: %s\n", k, class(v)[1], u, paste(head(unique(as.character(v)), 6), collapse = " | ")))
  if (u <= 40) print(head(sort(table(v, useNA = "ifany"), decreasing = TRUE), 40))
  else if (!is.numeric(v)) print(head(sort(table(v, useNA = "ifany"), decreasing = TRUE), 8))
}
write.csv(cbind(data.frame(cell = rownames(md), stringsAsFactors = FALSE), md),
          file.path(outdir, "meta.csv"), row.names = FALSE)   # base R: the pixi environment has no data.table

# --- counts: chunked writeBin (a single call is capped at 2^31-1 bytes) ------------------------
wr <- function(v, path, as_int) {
  con <- file(path, "wb"); n <- length(v); step <- 1e8
  for (lo in seq(1, max(n, 1), by = step)) {
    hi <- min(n, lo + step - 1); blk <- v[lo:hi]
    if (as_int && !is.integer(blk)) {
      if (any(blk != round(blk))) stop(path, ": non-integer values -- this is not a raw count matrix")
      blk <- as.integer(blk)
    }
    writeBin(blk, con, size = 4L)
  }
  close(con)
}
wr(cnt@i, file.path(outdir, "i.bin"), TRUE)
wr(cnt@p, file.path(outdir, "p.bin"), TRUE)
wr(cnt@x, file.path(outdir, "x.bin"), TRUE)
writeLines(genes, file.path(outdir, "genes.txt"))
writeLines(cells, file.path(outdir, "cells.txt"))
writeLines(c(paste("n_genes", nrow(cnt)), paste("n_cells", ncol(cnt)), paste("nnz", length(cnt@x)),
             paste("assay", an), paste("source", basename(rds)),
             paste("max_count", max(cnt@x)), paste("median_library", median(Matrix::colSums(cnt)))),
           file.path(outdir, "info.txt"))
cat("wrote", outdir, el(), "\n")
cat(readLines(file.path(outdir, "info.txt")), sep = "\n")
cat("done\n")
