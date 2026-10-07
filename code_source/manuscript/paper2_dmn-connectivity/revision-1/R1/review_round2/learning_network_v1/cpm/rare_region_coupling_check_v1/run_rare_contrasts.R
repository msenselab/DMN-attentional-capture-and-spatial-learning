#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(lme4);library(lmerTest);library(jsonlite)})
P <- normalizePath(getwd());R <- normalizePath(file.path(P,'../../..'));SOURCE <- file.path(R,'location_standardized_capture_v1/trial_model')
sha <- function(p) unname(strsplit(system2('sha256sum',shQuote(p),stdout=TRUE),' +')[[1]][1])
inputs <- file.path(SOURCE,c('final_selected_model.rds','final_selected_beta.csv','final_selected_vcov_RT100.csv','verification_independent.json','exact_model_frame.csv'))
inputs <- c(inputs,file.path(P,'ANALYSIS_SPEC.md'),file.path(P,'run_rare_contrasts.R'))
old <- list.files(file.path(P,'..'),full.names=TRUE,recursive=TRUE,all.files=TRUE);old<-old[!dir.exists(old)&!startsWith(normalizePath(old),paste0(P,'/'))]
hashes<-setNames(vapply(inputs,sha,''),inputs);protected<-setNames(vapply(old,sha,''),normalizePath(old))
write_json(list(inputs=as.list(hashes),protected_old_cpm=as.list(protected)),file.path(P,'input_manifest.json'),auto_unbox=TRUE,pretty=TRUE)
m <- readRDS(file.path(SOURCE,'final_selected_model.rds'));stopifnot(inherits(m,'lmerModLmerTest'),nrow(model.frame(m))==10995)
beta<-fixef(m);V<-as.matrix(vcov(m));oldbeta<-read.csv(file.path(SOURCE,'final_selected_beta.csv'));stopifnot(max(abs(beta-oldbeta$beta_RT100))<1e-10)
write.csv(data.frame(term=names(beta),beta_RT100=as.numeric(beta)),file.path(P,'model_beta.csv'),row.names=FALSE)
write.csv(data.frame(term=rownames(V),V,check.names=FALSE),file.path(P,'model_vcov.csv'),row.names=FALSE)
write.csv(m@vcov_varpar,file.path(P,'variance_parameter_covariance.csv'),row.names=FALSE)
jacs<-do.call(rbind,lapply(seq_along(m@Jac_list),function(k){J<-m@Jac_list[[k]];z<-expand.grid(row=seq_len(nrow(J)),col=seq_len(ncol(J)));data.frame(parameter=k,row=z$row,col=z$col,value=as.vector(J))}))
write.csv(jacs,file.path(P,'vcov_beta_derivative_jacobians.csv'),row.names=FALSE)
rows<-list();vectors<-list();k<-0
for(scope in c('Pooled','SS','DS','Moderation')){
  for(type in c('BR','UR','BR_minus_UR')){
    v<-setNames(rep(0,length(beta)),names(beta));base<-if(type=='BR')c(BR=1) else if(type=='UR')c(UR=1) else c(BR=1,UR=-1)
    for(cond in names(base)){
      if(scope!='Moderation')v[paste0('condition',cond,':X')]<-base[cond]
      if(scope %in% c('Pooled','DS','Moderation'))v[paste0('condition',cond,':X:D')]<-base[cond]*(if(scope=='Pooled')17/33 else 1)
    }
    z<-as.data.frame(contest1D(m,v,confint=TRUE,ddf='Satterthwaite'));lo<-names(z)[grepl('lower',names(z),ignore.case=TRUE)][1];hi<-names(z)[grepl('upper',names(z),ignore.case=TRUE)][1]
    k<-k+1;id<-paste(scope,type,sep='_')
    rows[[k]]<-data.frame(contrast_id=id,scope=scope,contrast=type,estimate_ms_per_globalFCSD=z$Estimate*100,SE_ms_per_globalFCSD=z[['Std. Error']]*100,df=z$df,t=z[['t value']],ci_low_ms=z[[lo]]*100,ci_high_ms=z[[hi]]*100,p_raw=z[['Pr(>|t|)']],family=if(scope=='Moderation')'Holm-3 group moderation DS-minus-SS' else 'Holm-9 rare-region simple/direct slopes')
    vectors[[k]]<-data.frame(contrast_id=id,coefficient=names(v),weight=as.numeric(v))
  }
}
out<-do.call(rbind,rows);out$p_Holm<-ave(out$p_raw,out$family,FUN=function(x)p.adjust(x,'holm'));write.csv(out,file.path(P,'rare_region_contrasts.csv'),row.names=FALSE);write.csv(do.call(rbind,vectors),file.path(P,'contrast_vectors.csv'),row.names=FALSE)
stopifnot(identical(unname(vapply(names(hashes),sha,'')),unname(hashes)),identical(unname(vapply(names(protected),sha,'')),unname(protected)))
write_json(list(status='PASS_EXECUTED_NO_REFIT',rows=10995,participants=33,SS=16,DS=17,new_contrasts=12,original_model_and_sources_unchanged=TRUE,old_cpm_files_unchanged=length(protected)),file.path(P,'execution.json'),auto_unbox=TRUE,pretty=TRUE)
print(out,row.names=FALSE)
