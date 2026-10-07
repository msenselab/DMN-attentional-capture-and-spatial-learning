#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(lme4);library(lmerTest);library(jsonlite);library(digest)})
options(stringsAsFactors=FALSE,width=180)
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L)
OUT<-normalizePath(args[[1]],mustWork=TRUE);BASE<-file.path(dirname(OUT),'trial_model')
ROOT<-Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
sha<-function(p)digest(file=p,algo='sha256',serialize=FALSE)
inputs<-c(original_fit=file.path(BASE,'final_selected_model.rds'),frame_rds=file.path(BASE,'exact_model_frame.rds'),frame_csv=file.path(BASE,'exact_model_frame.csv'),original_contrasts=file.path(BASE,'final_selected_neural_location_contrasts_Holm4.csv'),original_verification=file.path(BASE,'verification.json'),neural=file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv'),trials=file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv'),new_spec=file.path(OUT,'ANALYSIS_SPEC.md'),new_code=file.path(OUT,'fit_run_moderation.R'))
stopifnot(all(file.exists(inputs)))
hashes<-vapply(inputs,sha,'');write_json(list(paths=as.list(inputs),sha256=as.list(hashes)),file.path(OUT,'source_manifest_before.json'),pretty=TRUE,auto_unbox=TRUE)
lock<-fromJSON(inputs['original_verification']);stopifnot(hashes['frame_csv']==lock$exact_model_frame_sha256,hashes['neural']==lock$source_hashes$neural,hashes['trials']==lock$source_hashes$trials)
old<-readRDS(inputs['original_fit']);d<-readRDS(inputs['frame_rds']);mf<-model.frame(old)
oldform<-RT100 ~ L*X*D + run2 + L:run2 + (1+L|subject)
newform<-RT100 ~ L*X*D + L*X*run2 + (1+L|subject)
stopifnot(identical(deparse(formula(old)),deparse(oldform)),isREML(old),nobs(old)==6721L,nrow(d)==6721L,length(unique(d$subject))==33L,all(d$correct==1),all(d$RT_ms>=200 & d$RT_ms<=3000),all(d$RT100==d$RT_ms/100),all(d$category=='biased'),all(d$L==as.integer(d$DisRegion=='Rare')),all(d$D==as.integer(d$group=='DS')),all(d$run2==as.integer(d$run==2)),!anyDuplicated(d[c('subject','run','trial')]))
for(nm in c('RT100','L','X','D','run2'))stopifnot(isTRUE(all.equal(as.numeric(mf[[nm]]),as.numeric(d[[nm]]),tolerance=1e-12)))
stopifnot(identical(as.character(mf$subject),as.character(d$subject)))
u<-unique(d[c('subject','group','D','dmn_conn_primary','X')]);stopifnot(nrow(u)==33L,sum(u$group=='SS')==16L,sum(u$group=='DS')==17L)
xc<-u$dmn_conn_primary-ave(u$dmn_conn_primary,u$group,FUN=mean);stopifnot(max(abs(u$X-xc/sd(xc)))<1e-12)
ss<-sum(u$group=='SS');ds<-sum(u$group=='DS');N<-nrow(u);wds<-ds/N
write.csv(aggregate(list(n=rep(1,nrow(d))),d[c('group','run','DisRegion')],sum),file.path(OUT,'cell_support.csv'),row.names=FALSE)
write.csv(aggregate(list(n=rep(1,nrow(d))),d[c('subject','group','run','DisRegion')],sum),file.path(OUT,'participant_cell_support.csv'),row.names=FALSE)

# Registry is written before any new fits. New families do not alter old ones.
registry<-list()
add<-function(id,family,weights){registry[[length(registry)+1L]]<<-list(id=id,family=family,weights=weights)}
add('run2_minus_run1_common','targeted_exploratory_single_test',c('L:X:run2'=1))
for(r in 0:1)for(g in c('Pooled','SS','DS')){
 w<-c('L:X'=1,'L:X:D'=switch(g,Pooled=wds,SS=0,DS=1),'L:X:run2'=r)
 add(paste0(g,'_run',r+1),'new_run_specific_Holm6',w)
}
for(g in c('Pooled','SS','DS'))add(paste0(g,'_equal_run_average'),'new_run_averaged_Holm4',c('L:X'=1,'L:X:D'=switch(g,Pooled=wds,SS=0,DS=1),'L:X:run2'=0.5))
add('DS_minus_SS_equal_run_average','new_run_averaged_Holm4',c('L:X:D'=1))
write_json(lapply(registry,function(x){x$weights<-as.list(x$weights);x}),file.path(OUT,'contrast_registry.json'),pretty=TRUE,auto_unbox=TRUE)

capture_fit<-function(expr){warn<-character();msg<-character();f<-withCallingHandlers(expr,warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart('muffleWarning')},message=function(m){msg<<-c(msg,conditionMessage(m));invokeRestart('muffleMessage')});list(fit=f,warnings=unique(warn),messages=unique(msg))}
ctrlB<-lmerControl(optimizer='bobyqa',calc.derivs=TRUE,optCtrl=list(maxfun=200000))
ctrlN<-lmerControl(optimizer='nloptwrap',calc.derivs=TRUE,optCtrl=list(maxeval=200000))
fits<-list()
store<-function(name,obj){saveRDS(obj$fit,file.path(OUT,paste0('fit_',name,'.rds')));write_json(list(warnings=obj$warnings,messages=obj$messages),file.path(OUT,paste0(name,'_messages.json')),pretty=TRUE,auto_unbox=TRUE);cat('Completed',name,'logLik',as.numeric(logLik(obj$fit)),'singular',isSingular(obj$fit),'\n');flush.console();fits[[name]]<<-obj}
store('canonical_bobyqa',capture_fit(lmer(newform,data=d,REML=TRUE,control=ctrlB)))
store('alternative_nloptwrap',capture_fit(lmer(newform,data=d,REML=TRUE,control=ctrlN)))
store('bobyqa_restart',capture_fit(lmer(newform,data=d,REML=TRUE,control=ctrlB,start=list(theta=getME(fits$alternative_nloptwrap$fit,'theta')))))
store('alternative_nloptwrap_repeat',capture_fit(lmer(newform,data=d,REML=TRUE,control=ctrlN)))
store('bobyqa_restart_repeat',capture_fit(lmer(newform,data=d,REML=TRUE,control=ctrlB,start=list(theta=getME(fits$alternative_nloptwrap_repeat$fit,'theta')))))
diag<-function(name,obj){f<-obj$fit;der<-f@optinfo$derivs;ev<-eigen(der$Hessian,symmetric=TRUE,only.values=TRUE)$values;scaled<-as.numeric(backsolve(chol(der$Hessian),der$gradient));rp<-rePCA(f)$subject;list(summary=data.frame(fit=name,formula=paste(deparse(formula(f)),collapse=' '),REML=isREML(f),nobs=nobs(f),fixed_columns=ncol(getME(f,'X')),fixed_rank=qr(getME(f,'X'))$rank,optimizer=f@optinfo$optimizer,code=paste(f@optinfo$conv$opt,collapse=';'),messages=paste(unique(c(obj$warnings,obj$messages,f@optinfo$conv$lme4$messages)),collapse=';'),logLik=as.numeric(logLik(f)),singular_tol1e4=isSingular(f,tol=1e-4),singular_tol1e5=isSingular(f,tol=1e-5),min_hessian_eigen=min(ev),max_scaled_gradient=max(abs(scaled))),gradient=data.frame(fit=name,index=seq_along(scaled),raw=der$gradient,scaled=scaled),hessian=data.frame(fit=name,index=seq_along(ev),eigenvalue=ev),covariance=transform(as.data.frame(VarCorr(f)),fit=name),pca=data.frame(fit=name,index=seq_along(rp$sdev),sdev=rp$sdev),theta=data.frame(fit=name,parameter=names(getME(f,'theta')),theta=as.numeric(getME(f,'theta'))))}
diags<-mapply(diag,names(fits),fits,SIMPLIFY=FALSE)
for(kind in c('summary','gradient','hessian','covariance','pca','theta'))write.csv(do.call(rbind,lapply(diags,`[[`,kind)),file.path(OUT,paste0('optimizer_',kind,'.csv')),row.names=FALSE)
selected<-fits$canonical_bobyqa$fit;saveRDS(selected,file.path(OUT,'sensitivity_selected_model.rds'))
ll<-vapply(fits,function(o)as.numeric(logLik(o$fit)),0);fixedM<-sapply(fits,function(o)fixef(o$fit));thetaM<-sapply(fits,function(o)getME(o$fit,'theta'))
agreement<-list(logLik_maxdiff=max(abs(ll-ll[1])),fixed_RT100_maxdiff=max(abs(fixedM-fixedM[,1])),theta_maxdiff=max(abs(thetaM-thetaM[,1])))
write.csv(data.frame(term=rownames(fixedM),fixedM,check.names=FALSE),file.path(OUT,'optimizer_fixed_effect_comparison.csv'),row.names=FALSE)
write.csv(data.frame(parameter=rownames(thetaM),thetaM,check.names=FALSE),file.path(OUT,'optimizer_theta_comparison.csv'),row.names=FALSE)
diagnostic<-do.call(rbind,lapply(diags,`[[`,'summary'))
reproducible<-agreement$logLik_maxdiff<1e-6 && agreement$fixed_RT100_maxdiff<1e-6 && agreement$theta_maxdiff<1e-5
stopifnot(reproducible,all(!diagnostic$singular_tol1e4),all(diagnostic$code=='0'),all(diagnostic$messages==''),all(diagnostic$fixed_columns==12),all(diagnostic$fixed_rank==12),all(diagnostic$min_hessian_eigen>0),all(diagnostic$max_scaled_gradient<0.002))
sm<-summary(selected)$coefficients;fx<-data.frame(term=rownames(sm),estimate_RT100=sm[,'Estimate'],SE_RT100=sm[,'Std. Error'],df=sm[,'df'],t=sm[,'t value'],p_raw=sm[,'Pr(>|t|)']);fx$estimate_ms<-fx$estimate_RT100*100;fx$SE_ms<-fx$SE_RT100*100;fx$ci_low_ms<-(fx$estimate_RT100-qt(.975,fx$df)*fx$SE_RT100)*100;fx$ci_high_ms<-(fx$estimate_RT100+qt(.975,fx$df)*fx$SE_RT100)*100
write.csv(fx,file.path(OUT,'fixed_effects.csv'),row.names=FALSE)
canon<-function(x)paste(sort(strsplit(x,':',fixed=TRUE)[[1]]),collapse=':')
contrast<-function(f,w){v<-setNames(rep(0,length(fixef(f))),names(fixef(f)));for(term in names(w)){j<-which(vapply(names(v),canon,'')==canon(term));stopifnot(length(j)==1L);v[j]<-w[[term]]};z<-as.data.frame(contest1D(f,v,confint=TRUE,ddf='Satterthwaite'));data.frame(estimate_ms=z$Estimate*100,SE_ms=z[['Std. Error']]*100,df=z$df,t=z[['t value']],ci_low_ms=z$lower*100,ci_high_ms=z$upper*100,p_raw=z[['Pr(>|t|)']])}
result<-do.call(rbind,lapply(registry,function(x)cbind(data.frame(id=x$id,family=x$family),contrast(selected,x$weights))))
result$p_adjusted<-result$p_raw;result$adjustment<-'unadjusted targeted exploratory test'
for(fam in c('new_run_specific_Holm6','new_run_averaged_Holm4')){j<-which(result$family==fam);result$p_adjusted[j]<-p.adjust(result$p_raw[j],'holm');result$adjustment[j]<-fam}
result$unit<-'ms per global participant SD of group-centered corrected Default91 connectivity';result$N<-N;result$trials<-nrow(d)
write.csv(result,file.path(OUT,'neural_learning_contrasts.csv'),row.names=FALSE)
comparison<-result[result$family=='new_run_averaged_Holm4',];oldcons<-list(c('L:X'=1,'L:X:D'=wds),c('L:X'=1),c('L:X'=1,'L:X:D'=1),c('L:X:D'=1));oldres<-do.call(rbind,lapply(oldcons,function(w)contrast(old,w)))
oldres$p_adjusted<-p.adjust(oldres$p_raw,'holm');names(oldres)<-paste0('original_',names(oldres));comparison<-cbind(comparison,oldres);comparison$estimate_change_ms<-comparison$estimate_ms-comparison$original_estimate_ms
write.csv(comparison,file.path(OUT,'original_vs_run_averaged_sensitivity.csv'),row.names=FALSE)
stopifnot(identical(hashes,vapply(inputs,sha,'')))
write_json(list(status='PASS_EMPIRICAL_COMPUTATION_PENDING_INDEPENDENT_READBACK',formula=paste(deparse(newform),collapse=' '),N=N,SS=ss,DS=ds,trials=nrow(d),original_inputs_unchanged=TRUE,exact_frame_original_fit_verified=TRUE,corrected_predictor_participant_scaling_verified=TRUE,optimizer_reproducible=reproducible,agreement=agreement,all_five_fits_nonsingular=TRUE,target='common run2-minus-run1 change in DMN modulation of rare-minus-frequent RT',random_structure='original correlated participant intercept/location slope retained',manuscript_changed=FALSE,limitations='Exploratory post hoc sensitivity; common temporal change across groups, not a group-specific four-way interaction; conditional on retained random-effects structure'),file.path(OUT,'computation_verification.json'),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(OUT,'R_sessionInfo.txt'));print(result);print(comparison)
