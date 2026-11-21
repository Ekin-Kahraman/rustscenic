args<-commandArgs(trailingOnly=TRUE); dataset<-args[[1]];phase<-args[[2]]
root<-normalizePath('.');.libPaths(c(file.path(root,'r-library'),.libPaths()))
suppressPackageStartupMessages({library(sceptre);library(Matrix);library(data.table);library(jsonlite)})
data.table::setDTthreads(1L)
input<-file.path(root,'inputs',dataset);out<-file.path(root,'results',dataset);dir.create(out,recursive=TRUE,showWarnings=FALSE)
meta<-fromJSON(file.path(input,'metadata.json'));logdir<-file.path(out,'logs');dir.create(logdir,showWarnings=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R-session.txt'))
readints<-function(name,n){con<-gzfile(file.path(input,name),'rb');on.exit(close(con));readBin(con,integer(),n=n,size=4,endian='little')}
if(phase=='calibration'){
  genes<-readLines(file.path(input,'genes.txt'));cells<-fread(file.path(input,'cells.tsv.gz'),data.table=FALSE)
  response<-new('dgCMatrix',Dim=as.integer(c(meta$genes,meta$cells)),i=readints('counts.i.bin.gz',meta$nnz),p=readints('counts.p.bin.gz',meta$cells+1),x=as.numeric(readints('counts.x.bin.gz',meta$nnz)),Dimnames=list(genes,cells$barcode))
  stopifnot(ncol(response)==nrow(cells),all(colSums(response)<=cells$full_ncounts))
  guide_names<-sort(unique(cells$construct));guide_ids<-sprintf('construct_%04d',seq_along(guide_names));gidx<-match(cells$construct,guide_names)
  guide_targets<-vapply(guide_names,function(g){v<-unique(cells$target[cells$construct==g]);stopifnot(length(v)==1);if(v=='control') 'non-targeting' else v},character(1))
  assignments<-sparseMatrix(i=gidx,j=seq_len(nrow(cells)),x=1,dims=c(length(guide_names),nrow(cells)),dimnames=list(guide_ids,cells$barcode))
  mapping<-data.frame(grna_id=guide_ids,grna_target=guide_targets,original_construct=guide_names);fwrite(mapping,file.path(out,'construct-map.csv'))
  covariates<-data.frame(full_ncounts=cells$full_ncounts,batch=factor(cells$batch),row.names=cells$barcode)
  s<-import_data(response_matrix=response,grna_matrix=assignments,grna_target_data_frame=mapping[,c('grna_id','grna_target')],moi='low',extra_covariates=covariates,response_names=genes)
  rm(response,assignments);gc()
  targets<-sort(setdiff(unique(cells$target),'control'));pairs<-expand.grid(grna_target=targets,response_id=genes,stringsAsFactors=FALSE);pairs<-pairs[pairs$grna_target!=pairs$response_id,]
  positive<-data.frame(grna_target=intersect(targets,genes),response_id=intersect(targets,genes))
  s<-set_analysis_parameters(s,discovery_pairs=pairs,positive_control_pairs=positive,side='both',grna_integration_strategy='union',formula_object=~log(full_ncounts)+batch,resampling_approximation='skew_normal',control_group='nt_cells',resampling_mechanism='permutations',multiple_testing_method='BH',multiple_testing_alpha=.05)
  s<-assign_grnas(s,method='maximum',min_grna_n_umis_threshold=1L,print_progress=FALSE)
  assigned<-as(get_grna_assignments(s),'CsparseMatrix')
  stopifnot(all(colSums(assigned)==1L),ncol(assigned)==nrow(cells),identical(rownames(assigned)[assigned@i+1L],guide_ids[gidx]));rm(assigned)
  s<-run_qc(s,n_nonzero_trt_thresh=7L,n_nonzero_cntrl_thresh=7L,response_n_umis_range=c(0,1),response_n_nonzero_range=c(0,1),p_mito_threshold=1)
  fwrite(s@discovery_pairs_with_info,file.path(out,'pair-qc.csv'));writeLines(capture.output(print(s)),file.path(out,'qc-summary.txt'))
  saveRDS(s,file.path(out,'object.rds'),compress=FALSE)
  specifications<-data.frame(run=c('median-a','median-b','single','upper'),size=c(meta$calibration_group_sizes[2],meta$calibration_group_sizes[2],meta$calibration_group_sizes[1],meta$calibration_group_sizes[3]),n=c(10000,10000,3000,3000),seed=2026111201:2026111204)
  rows<-list()
  for(k in seq_len(nrow(specifications))){
    spec<-specifications[k,];set.seed(spec$seed);start<-proc.time()[[3]]
    cat('CALIBRATION',dataset,spec$run,'group size',spec$size,'\n');flush.console()
    s<-run_calibration_check(s,n_calibration_pairs=spec$n,calibration_group_size=spec$size,parallel=TRUE,n_processors=2,log_dir=logdir,print_progress=TRUE)
    result<-get_result(s,analysis='run_calibration_check');fwrite(result,file.path(out,paste0('calibration-',spec$run,'.csv')))
    keep<-result$pass_qc & is.finite(result$p_value);v<-result$p_value[keep];lfc<-result$log_2_fold_change[keep];lfc<-lfc[is.finite(lfc)]
    row<-data.frame(dataset=dataset,run=spec$run,seed=spec$seed,construct_group_size=spec$size,n=length(v),p05=mean(v<=.05),p01=mean(v<=.01),p001=mean(v<=.001),p50=mean(v<=.5),bh05=sum(p.adjust(v,method='BH')<=.05),median_lfc=median(lfc),mean_lfc=mean(lfc),seconds=proc.time()[[3]]-start)
    row$pass<-row$p05<=.065 && row$p01<=.015 && row$p001<=.0025 && row$bh05<=10 && abs(row$median_lfc)<=.1 && row$n>=1000
    rows[[k]]<-row;fwrite(do.call(rbind,rows),file.path(out,'calibration-summary.csv'));print(row);flush.console()
  }
  summaries<-do.call(rbind,rows);passed<-all(summaries$pass)
  write_json(list(status=if(passed)'passed' else 'failed',gate='Frozen operational control-calibration gate; not proof of universal FDR',discovery_inspected=FALSE),file.path(out,'CALIBRATION-STATUS.json'),auto_unbox=TRUE,pretty=TRUE)
  saveRDS(s,file.path(out,'object.rds'),compress=FALSE)
  if(!passed)stop('Calibration failed. Discovery analysis is blocked pending documented diagnosis.')
  cat('ALL CALIBRATION CHECKS PASSED',dataset,'\n')
}else if(phase=='discovery'){
  gate<-fromJSON(file.path(out,'CALIBRATION-STATUS.json'));stopifnot(gate$status=='passed')
  s<-readRDS(file.path(out,'object.rds'));set.seed(2026111210)
  s<-run_power_check(s,parallel=TRUE,n_processors=2,log_dir=logdir);fwrite(get_result(s,'run_power_check'),file.path(out,'positive-controls.csv'))
  set.seed(2026111211);s<-run_discovery_analysis(s,parallel=TRUE,n_processors=2,log_dir=logdir)
  d<-get_result(s,'run_discovery_analysis');fwrite(d,file.path(out,'discovery.csv'))
  write_json(list(status='complete',dataset=dataset,pairs=nrow(d),qc_pass=sum(d$pass_qc),finite_p=sum(is.finite(d$p_value)),significant=sum(d$significant,na.rm=TRUE),scope='Post-hoc corrected response analysis; no model refits'),file.path(out,'DISCOVERY-STATUS.json'),auto_unbox=TRUE,pretty=TRUE)
  cat('DISCOVERY COMPLETE',dataset,'\n')
}else stop('Unknown phase')
