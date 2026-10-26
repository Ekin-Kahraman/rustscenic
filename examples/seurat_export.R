# Explicit sparse RNA exchange. Requires SeuratObject >= 5.0 and Matrix.
# Source this file for export_rna_data() and add_regulon_activity(), or run:
# Rscript examples/seurat_export.R --example export_dir
# Rscript examples/seurat_export.R input.rds export_dir
# Rscript examples/seurat_export.R --add-activity input.rds activity.csv result.rds

check_exchange_ids <- function(ids, label) {
  if (is.null(ids) || !length(ids) || anyNA(ids) || any(!nzchar(ids)) ||
      anyDuplicated(ids) || any(grepl("[\r\n]", ids))) {
    stop(label, " must be nonempty, unique names without line breaks")
  }
}

export_rna_data <- function(object, directory, assay = "RNA") {
  if (!requireNamespace("SeuratObject", quietly = TRUE) ||
      !requireNamespace("Matrix", quietly = TRUE)) {
    stop("Install SeuratObject >= 5.0 and Matrix in your R environment")
  }
  if (utils::packageVersion("SeuratObject") < "5.0.0") {
    stop("This example requires SeuratObject >= 5.0")
  }
  rna <- object[[assay]]
  layers <- SeuratObject::Layers(rna, search = NA)
  data_layers <- grep("^data($|\\.)", layers, value = TRUE)
  if (!identical(data_layers, "data")) {
    stop("Expected one log-normalized RNA 'data' layer; found: ",
         paste(data_layers, collapse = ", "),
         ". Normalize explicitly if missing; deliberately JoinLayers if split.")
  }
  expression <- SeuratObject::LayerData(rna, layer = "data")
  if (!is.matrix(expression) && !inherits(expression, "sparseMatrix")) {
    stop("This example expects an in-memory matrix or Matrix sparseMatrix")
  }
  check_exchange_ids(rownames(expression), "Gene IDs")
  check_exchange_ids(colnames(expression), "Cell IDs")
  values <- if (inherits(expression, "sparseMatrix")) expression@x else c(expression)
  if (!is.numeric(values) || any(!is.finite(values)) || any(values < 0)) {
    stop("Expected finite, nonnegative log-normalized RNA data")
  }
  expression <- Matrix::Matrix(expression, sparse = TRUE)
  paths <- file.path(directory, c("expression.mtx", "genes.txt", "cells.txt"))
  if (any(file.exists(paths))) stop("Use an export directory without existing exchange files")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(expression, paths[[1]])
  writeLines(rownames(expression), paths[[2]])
  writeLines(colnames(expression), paths[[3]])
  invisible(directory)
}

add_regulon_activity <- function(object, csv_path) {
  activity <- utils::read.csv(csv_path, row.names = 1, check.names = FALSE,
                             colClasses = "character")
  check_exchange_ids(rownames(activity), "Activity cell IDs")
  check_exchange_ids(colnames(activity), "Regulon names")
  if (!setequal(rownames(activity), colnames(object))) {
    stop("Activity must contain exactly the object's cells, matched by barcode")
  }
  for (name in colnames(activity)) {
    values <- suppressWarnings(as.numeric(activity[[name]]))
    if (any(!is.finite(values))) stop("Activity values must be finite numbers")
    activity[[name]] <- values
  }
  if (any(colnames(activity) %in% colnames(object[[]]))) {
    stop("Activity column names already exist in object metadata")
  }
  activity <- activity[colnames(object), , drop = FALSE]
  SeuratObject::AddMetaData(object, metadata = activity)
}

tiny_seurat_example <- function() {
  genes <- c("SPI1", "CD14", "LYZ", "CSF1R", "PAX5", "CD79A", "MS4A1", paste0("G", 8:64))
  cells <- paste0("CELL", 1:8, "-1")
  counts <- outer(1:64, 1:8, function(g, c) (g * 3 + c * 5 + g * c) %% 17)
  counts[counts < 6] <- 0
  dimnames(counts) <- list(genes, cells)
  counts <- Matrix::Matrix(counts, sparse = TRUE)
  # Explicit synthetic normalization, without loading the full Seurat package.
  data <- log1p(counts %*% Matrix::Diagonal(x = 10000 / Matrix::colSums(counts)))
  dimnames(data) <- dimnames(counts)
  rna <- SeuratObject::CreateAssay5Object(counts = counts, data = data)
  scaled <- as.matrix(data[1:32, , drop = FALSE])
  scaled <- sweep(scaled, 1, rowMeans(scaled), "-")
  SeuratObject::LayerData(rna, layer = "scale.data") <- scaled
  SeuratObject::CreateSeuratObject(counts = rna, assay = "RNA")
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 2L && args[[1]] == "--example") {
    object <- tiny_seurat_example()
    export_rna_data(object, args[[2]])
    saveRDS(object, file.path(args[[2]], "seurat.rds"))
    writeLines(c('{',
      '  "SPI1_regulon": ["SPI1", "CD14", "LYZ", "CSF1R"],',
      '  "PAX5_regulon": ["PAX5", "CD79A", "MS4A1"]',
      '}'), file.path(args[[2]], "regulons.json"))
  } else if (length(args) == 4L && args[[1]] == "--add-activity") {
    object <- add_regulon_activity(readRDS(args[[2]]), args[[3]])
    saveRDS(object, args[[4]])
  } else if (length(args) == 2L) {
    export_rna_data(readRDS(args[[1]]), args[[2]])
  } else {
    stop("Usage: seurat_export.R input.rds export_dir | --example export_dir | ",
         "--add-activity input.rds activity.csv result.rds")
  }
}
