# Dump colData / rowData of one GSE337988 SingleCellExperiment skeleton (`*_se.rds`) to csv.gz, so Python can
# read the counts straight from `*_assays.h5` with h5py.
#
# Why attr() everywhere: the object is a SingleCellExperiment, and SingleCellExperiment is NOT installed on the
# cluster R (a first version died in S4 dispatch on dim()/colData()). S4 slots are plain R attributes, so reading
# them with attr() needs no class definition and no method dispatch. S4Vectors (for DFrame) IS installed.
#
#   Rscript scripts/gse337988_coldata.R <in_se.rds> <out_prefix>

suppressPackageStartupMessages({ library(S4Vectors) })
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; out <- args[2]
rows_only <- length(args) >= 3 && args[3] == "rows"     # colData already dumped -> redo rowData only

obj <- readRDS(rds)
cat("class:", class(obj), "\n")
cat("slots:", paste(names(attributes(obj)), collapse = ", "), "\n")

# DFrame -> data.frame of atomic columns; S4 / list columns are reported (not silently dropped) and skipped
flat <- function(df, what) {
  cols <- attr(df, "listData")
  keep <- list()
  for (k in names(cols)) {
    v <- cols[[k]]
    if (is.factor(v)) v <- as.character(v)
    if (is.atomic(v) && !isS4(v)) { keep[[k]] <- v; next }
    if (isS4(v)) {
      cat(sprintf("WARNING %s column %s is S4 %s (slots: %s): trying as.vector\n", what, k, class(v)[1],
                  paste(names(attributes(v)), collapse = ",")))
      vv <- tryCatch(as.vector(v), error = function(e) { cat("  as.vector failed:", conditionMessage(e), "\n"); NULL })
      if (!is.null(vv) && is.atomic(vv)) { keep[[k]] <- vv; next }
    }
    if (is.list(v)) {
      cat(sprintf("WARNING %s column %s is a list: ';'-joining\n", what, k))
      keep[[k]] <- vapply(v, function(x) paste(as.character(x), collapse = ";"), ""); next
    }
    cat(sprintf("WARNING %s column %s (class %s) SKIPPED\n", what, k, class(v)[1]))
  }
  as.data.frame(keep, stringsAsFactors = FALSE, check.names = FALSE)
}

if (!rows_only) {
cd <- attr(obj, "colData")
rn <- attr(cd, "rownames")
cat("colData nrows:", attr(cd, "nrows"), " rownames:", length(rn), "\n")
cdf <- flat(cd, "colData")
if (!is.null(rn)) cdf <- cbind(barcode = rn, cdf)
for (k in colnames(cdf)) {
  v <- cdf[[k]]
  u <- length(unique(v))
  cat(sprintf("  %-28s %-10s %8d unique; head: %s\n", k, class(v)[1], u, paste(head(unique(v), 6), collapse = " | ")))
  if (u <= 60) print(head(sort(table(v, useNA = "ifany"), decreasing = TRUE), 60))
}
write.csv(cdf, gzfile(paste0(out, "_coldata.csv.gz")), row.names = FALSE)
}

# In a RangedSummarizedExperiment (SCE) rowData = mcols(rowRanges), and the row names are the names of
# rowRanges' partitioning; the SE-level `NAMES` slot is an R NULL symbol and `elementMetadata` is empty.
rr <- attr(obj, "rowRanges")
cat("rowRanges class:", class(rr), " slots:", paste(names(attributes(rr)), collapse = ", "), "\n")
rd <- attr(rr, "elementMetadata")
rnames <- attr(obj, "NAMES")
if (!is.character(rnames)) {
  part <- attr(rr, "partitioning")
  rnames <- if (!is.null(part)) attr(part, "NAMES") else attr(rr, "NAMES")
  cat("row names taken from rowRanges:", class(rnames), length(rnames), "\n")
}
if (!is.character(rnames)) { cat("WARNING no character row names found\n"); rnames <- NULL }
cat("rowData nrows:", attr(rd, "nrows"), " cols:", paste(names(attr(rd, "listData")), collapse = ", "),
    " NAMES:", length(rnames), "\n")
rdf <- flat(rd, "rowData")
if (!is.null(rnames)) rdf <- cbind(feature = rnames, rdf)
print(head(rdf, 5))
write.csv(rdf, gzfile(paste0(out, "_rowdata.csv.gz")), row.names = FALSE)

md <- attr(obj, "metadata")
cat("metadata names:", paste(names(md), collapse = ", "), "\n")
for (k in names(md)) cat("  metadata", k, "class", class(md[[k]])[1], "length", length(md[[k]]), "\n")
cat("done\n")
