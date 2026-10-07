#!/usr/bin/env Rscript
# Independent saved-object verification using design-grid contrast weights.
suppressPackageStartupMessages({library(lme4);library(lmerTest);library(jsonlite);library(digest)})
a<-commandArgs(trailingOnly=TRUE);stopifnot(length(a)==1L);OUT<-normalizePath(a[[1]],mustWork=TRUE)
sha<-function(p)digest(file=p,algo='sha256',serialize=FALSE)
manifest<-fromJSON(file.path(OUT,'source_manifest_before.json'));paths<-unlist(manifest$paths);hashes<-unlist(manifest$sha256);stopifnot(identical(unname(vapply(paths,sha,'')),unname(hashes)))
d<-readRDS(paths['frame_rds']);old<-readRDS(paths['original_fit']);fit<-readRDS(file.path(OUT,'sensitivity_selected_model.rds'));mf<-model.frame(fit)
expected<-RT100 ~ L*X*D + L*X*run2 + (1+L|subject)
stopifnot(identical(deparse(formula(fit)),deparse(expected)),isREML(fit),nobs(fit)==6721L,nrow(d)==6721L,length(unique(d$subject))==33L,identical(names(getME(fit,'cnms')),names(getME(old,'cnms'))),identical(getME(fit,'cnms'),getME(old,'cnms')))
for(k in c('RT100','L','X','D','run2'))stopifnot(max(abs(as.numeric(mf[[k]])-as.numeric(d[[k]])))<1e-12)
stopifnot(identical(as.character(mf$subject),as.character(d$subject)),all(d$correct==1),all(d$RT_ms>=200 & d$RT_ms<=3000),!anyDuplicated(d[c('subject','run','trial')]))
u<-unique(d[c('subject','group','X','dmn_conn_primary')]);stopifnot(nrow(u)==33L,sum(u$group=='SS')==16L,sum(u$group=='DS')==17L);x<-u$dmn_conn_primary-ave(u$dmn_conn_primary,u$group,FUN=mean);stopifnot(max(abs(u$X-x/sd(x)))<1e-12)
weights<-table(u$group)/nrow(u)
# The 16-grid is an algebraic design fixture, not synthetic empirical data.
grid<-expand.grid(L=0:1,X=0:1,D=0:1,run2=0:1)
make_vectors<-function(f){MM<-model.matrix(delete.response(terms(nobars(formula(f)))),grid);MM<-MM[,names(fixef(f)),drop=FALSE];slope<-function(g,r){i<-which(grid$D==g & grid$run2==r);z<-grid[i,];coef<-ifelse(z$L==z$X,1,-1);as.numeric(crossprod(coef,MM[i,,drop=FALSE]))};v<-list();ss0<-slope(0,0);ss1<-slope(0,1);ds0<-slope(1,0);ds1<-slope(1,1);v$run2_minus_run1_common<-ss1-ss0;stopifnot(max(abs((ds1-ds0)-v$run2_minus_run1_common))<1e-12);v$Pooled_run1<-as.numeric(weights['SS'])*ss0+as.numeric(weights['DS'])*ds0;v$SS_run1<-ss0;v$DS_run1<-ds0;v$Pooled_run2<-as.numeric(weights['SS'])*ss1+as.numeric(weights['DS'])*ds1;v$SS_run2<-ss1;v$DS_run2<-ds1;v$Pooled_equal_run_average<-(v$Pooled_run1+v$Pooled_run2)/2;v$SS_equal_run_average<-(ss0+ss1)/2;v$DS_equal_run_average<-(ds0+ds1)/2;v$DS_minus_SS_equal_run_average<-v$DS_equal_run_average-v$SS_equal_run_average;v}
vectors<-make_vectors(fit);term<-names(fixef(fit));targ<-setNames(vectors$run2_minus_run1_common,term);stopifnot(sum(abs(targ)>1e-12)==1L,targ['L:X:run2']==1)
recompute<-function(f,v){z<-as.data.frame(contest1D(f,v,confint=TRUE,ddf='Satterthwaite'));c(estimate_ms=z$Estimate*100,SE_ms=z[['Std. Error']]*100,df=z$df,t=z[['t value']],ci_low_ms=z$lower*100,ci_high_ms=z$upper*100,p_raw=z[['Pr(>|t|)']])}
recomputed<-t(sapply(vectors,function(v)recompute(fit,v)));saved<-read.csv(file.path(OUT,'neural_learning_contrasts.csv'));stopifnot(nrow(saved)==11L,identical(saved$id,rownames(recomputed)))
maxerrors<-sapply(colnames(recomputed),function(k)max(abs(saved[[k]]-recomputed[,k])));stopifnot(all(maxerrors<1e-8))
for(family in unique(saved$family)){i<-which(saved$family==family);p<-if(length(i)==1L)saved$p_raw[i]else p.adjust(saved$p_raw[i],'holm');stopifnot(max(abs(p-saved$p_adjusted[i]))<1e-12)}
sm<-summary(fit)$coefficients;fx<-read.csv(file.path(OUT,'fixed_effects.csv'));stopifnot(identical(fx$term,rownames(sm)),max(abs(fx$estimate_RT100-sm[,'Estimate']))<1e-10,max(abs(fx$SE_RT100-sm[,'Std. Error']))<1e-10,max(abs(fx$p_raw-sm[,'Pr(>|t|)']))<1e-12)
comparison<-read.csv(file.path(OUT,'original_vs_run_averaged_sensitivity.csv'));ov<-make_vectors(old);oldgrid<-t(sapply(ov[comparison$id],function(v)recompute(old,v)));for(k in colnames(oldgrid))stopifnot(max(abs(comparison[[paste0('original_',k)]]-oldgrid[,k]))<1e-8)
stopifnot(max(abs(comparison$original_p_adjusted-p.adjust(oldgrid[,'p_raw'],'holm')))<1e-12,max(abs(comparison$estimate_change_ms-(comparison$estimate_ms-comparison$original_estimate_ms)))<1e-10)
original_export<-read.csv(paths['original_contrasts']);stopifnot(max(abs(original_export$estimate_ms_per_global_corrected_DMN_SD-oldgrid[,'estimate_ms']))<1e-8)
records<-read.csv(file.path(OUT,'optimizer_summary.csv'));stopifnot(nrow(records)==5L)
for(i in seq_len(nrow(records))){f<-readRDS(file.path(OUT,paste0('fit_',records$fit[i],'.rds')));stopifnot(isREML(f),nobs(f)==6721L,identical(deparse(formula(f)),deparse(expected)),!isSingular(f,tol=1e-4),!isSingular(f,tol=1e-5),isTRUE(all.equal(as.numeric(logLik(f)),records$logLik[i],tolerance=1e-10)),qr(getME(f,'X'))$rank==12L,all(is.finite(getME(f,'theta'))),length(f@optinfo$conv$lme4$messages)==0L);stopifnot(min(eigen(f@optinfo$derivs$Hessian,symmetric=TRUE,only.values=TRUE)$values)>0)}
stopifnot(identical(unname(vapply(paths,sha,'')),unname(hashes)))
result<-list(status='PASS',independent_saved_object_readback=TRUE,exact_source_and_frame_preserved=TRUE,new_formula_and_original_random_structure_verified=TRUE,N=33L,trials=6721L,all_11_design_grid_contrasts_verified=TRUE,all_pointwise_intervals_and_Satterthwaite_tests_verified=TRUE,new_Holm6_and_Holm4_families_verified=TRUE,original_comparison_and_exports_verified=TRUE,five_saved_optimizer_fits_verified=TRUE,max_numeric_errors=as.list(maxerrors),focal=list(estimate_ms=saved$estimate_ms[1],ci_low_ms=saved$ci_low_ms[1],ci_high_ms=saved$ci_high_ms[1],p_raw=saved$p_raw[1]),manuscript_changed=FALSE)
write_json(result,file.path(OUT,'verification_independent.json'),pretty=TRUE,auto_unbox=TRUE);cat('PASS independent verifier: original hashes/frame, 11 design-grid contrasts, raw/Holm p values, five fitted objects verified.\n');print(saved[1,])
outputs<-list.files(OUT,full.names=TRUE);outputs<-outputs[!grepl('output_hashes.json$',outputs)];write_json(as.list(setNames(vapply(outputs,sha,''),basename(outputs))),file.path(OUT,'output_hashes.json'),pretty=TRUE,auto_unbox=TRUE)
