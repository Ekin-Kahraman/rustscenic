# Independent implementation check against R's standard multiple-testing routine.
suppressPackageStartupMessages(library(data.table))
setDTthreads(1L)
dir.create('verification', showWarnings=FALSE)
for (dataset in c('k562', 'rpe1')) {
  d <- fread(file.path('results', dataset, 'discovery.csv'))
  valid <- d$pass_qc & is.finite(d$p_value)
  q <- rep(NA_real_, nrow(d))
  q[valid] <- p.adjust(d$p_value[valid], method='BH')
  con <- file(file.path('verification', paste0(dataset, '-r-bh.bin')), 'wb')
  writeBin(q, con, size=8L, endian='little')
  close(con)
  stopifnot(all((q[valid] < .05) == d$significant[valid]))
  cat(dataset, ': R BH and SCEPTRE significance agree for', sum(valid), 'tests\n')
}
