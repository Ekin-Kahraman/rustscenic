args<-commandArgs(trailingOnly=TRUE);dataset<-args[[1]];root<-normalizePath('.');.libPaths(c(file.path(root,'r-library'),.libPaths()))
suppressPackageStartupMessages({library(sceptre);library(data.table);library(jsonlite)});setDTthreads(1L)
out<-file.path(root,'results',dataset);meta<-fromJSON(file.path(root,'inputs',dataset,'metadata.json'));s<-readRDS(file.path(out,'object.rds'))
write_json(list(status='pending_distinct_pair_validation',discovery_inspected=FALSE),file.path(out,'CALIBRATION-STATUS.json'),auto_unbox=TRUE,pretty=TRUE)
s<-run_calibration_check(s,n_calibration_pairs=20000,calibration_group_size=meta$calibration_group_sizes[2],parallel=TRUE,n_processors=2,log_dir=file.path(out,'logs'))
d<-get_result(s,'run_calibration_check');fwrite(d,file.path(out,'calibration-distinct-pool.csv'))
a<-fread(file.path(out,'calibration-median-a.csv'),data.table=FALSE);key<-function(x)paste(x$grna_target,x$response_id,sep='::');stopifnot(!anyDuplicated(key(d)))
new<-which(!key(d)%in%key(a));stopifnot(length(new)>=10000);set.seed(2026111202);x<-d[sample(new,10000),];fwrite(x,file.path(out,'calibration-validation.csv'))
keep<-x$pass_qc&is.finite(x$p_value);p<-x$p_value[keep];lfc<-x$log_2_fold_change[keep];lfc<-lfc[is.finite(lfc)]
v<-data.frame(dataset=dataset,run='distinct-validation',seed=2026111202,construct_group_size=meta$calibration_group_sizes[2],n=length(p),p05=mean(p<=.05),p01=mean(p<=.01),p001=mean(p<=.001),p50=mean(p<=.5),bh05=sum(p.adjust(p,'BH')<=.05),median_lfc=median(lfc),mean_lfc=mean(lfc),distinct_pool_available=length(new))
v$pass<-v$p05<=.065&&v$p01<=.015&&v$p001<=.0025&&v$bh05<=10&&abs(v$median_lfc)<=.1&&v$n>=1000;fwrite(v,file.path(out,'validation-summary.csv'));print(v)
initial<-fread(file.path(out,'calibration-summary.csv'));pass<-all(initial$pass)&&v$pass
write_json(list(status=if(pass)'passed' else 'failed',discovery_inspected=FALSE,distinct_validation_pairs=10000,distinct_validation_overlap_with_median_a=0,median_b_is_duplicate=TRUE,unique_counted_tests=26000,scope='Empirical calibration diagnostic, not independent biological replication or proof of universal FDR'),file.path(out,'CALIBRATION-STATUS.json'),auto_unbox=TRUE,pretty=TRUE)
if(!pass)stop('Distinct-pair validation failed')
