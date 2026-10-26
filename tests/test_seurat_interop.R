# Manual R regression test, with no full Seurat or converter dependency:
# Rscript tests/test_seurat_interop.R /path/to/rustscenic /tmp/seurat-check
args <- commandArgs(trailingOnly = TRUE)
source(file.path(args[[1]], "examples", "seurat_export.R"))
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
out <- tempfile("seurat-check-", tmpdir = args[[2]])
dir.create(out)

expect_error <- function(expr, pattern) {
  message <- tryCatch({force(expr); NULL}, error = function(e) conditionMessage(e))
  stopifnot(!is.null(message), grepl(pattern, message))
}

object <- tiny_seurat_example()
stopifnot(inherits(object[["RNA"]], "Assay5"))
data <- SeuratObject::LayerData(object[["RNA"]], layer = "data")
stopifnot(inherits(data, "sparseMatrix"), all(data@x >= 0))
scaled <- SeuratObject::LayerData(object[["RNA"]], layer = "scale.data")
stopifnot(nrow(scaled) == 32L, nrow(data) == 64L, any(scaled < 0))

export_rna_data(object, file.path(out, "single"))
actual <- Matrix::readMM(file.path(out, "single", "expression.mtx"))
stopifnot(identical(dim(actual), c(64L, 8L)))
stopifnot(max(abs(actual - data)) < 1e-14)
stopifnot(identical(readLines(file.path(out, "single", "genes.txt")), rownames(data)))
stopifnot(identical(readLines(file.path(out, "single", "cells.txt")), colnames(data)))
expect_error(export_rna_data(object, file.path(out, "single")), "existing exchange files")

missing <- SeuratObject::CreateSeuratObject(
  counts = SeuratObject::CreateAssay5Object(
    counts = SeuratObject::LayerData(object[["RNA"]], layer = "counts")))
expect_error(export_rna_data(missing, file.path(out, "missing")), "Expected one")
stopifnot(!dir.exists(file.path(out, "missing")))

split <- object
split[["RNA"]] <- split(split[["RNA"]], f = rep(c("A", "B"), each = 4))
expect_error(export_rna_data(split, file.path(out, "split")), "Expected one")
stopifnot(!dir.exists(file.path(out, "split")))
split[["RNA"]] <- SeuratObject::JoinLayers(split[["RNA"]], layers = "data", new = "data")
export_rna_data(split, file.path(out, "joined"))
joined <- Matrix::readMM(file.path(out, "joined", "expression.mtx"))
stopifnot(max(abs(joined - data)) < 1e-14)
stopifnot(identical(readLines(file.path(out, "joined", "cells.txt")), colnames(data)))

invalid <- object
bad_data <- data
bad_data@x[[1]] <- -1
SeuratObject::LayerData(invalid[["RNA"]], layer = "data") <- bad_data
expect_error(export_rna_data(invalid, file.path(out, "negative")), "finite, nonnegative")

activity <- data.frame("R1(+)" = seq_len(ncol(object)), check.names = FALSE,
                       row.names = colnames(object))
csv <- file.path(out, "activity.csv")
utils::write.csv(activity[rev(rownames(activity)), , drop = FALSE], csv)
with_activity <- add_regulon_activity(object, csv)
stopifnot(identical(with_activity[[]][["R1(+)"]], as.numeric(activity[["R1(+)"]])))
stopifnot(identical(colnames(with_activity), colnames(object)))
expect_error(add_regulon_activity(with_activity, csv), "already exist")
utils::write.csv(activity[-1, , drop = FALSE], csv)
expect_error(add_regulon_activity(object, csv), "exactly the object's cells")

# Numeric-looking IDs must not lose leading zeros during CSV import.
numeric_ids <- object
colnames(numeric_ids) <- sprintf("%03d", seq_len(ncol(object)))
rownames(activity) <- colnames(numeric_ids)
utils::write.csv(activity, csv)
with_activity <- add_regulon_activity(numeric_ids, csv)
stopifnot(identical(colnames(with_activity), sprintf("%03d", seq_len(ncol(object)))))
stopifnot(identical(with_activity[[]][["R1(+)"]], as.numeric(activity[["R1(+)"]])))
cat("PASS: real Assay5 data selection, scale exclusion, split/missing rejection, explicit join, ",
    "sparse values/names, and barcode-indexed AddMetaData return\n")
