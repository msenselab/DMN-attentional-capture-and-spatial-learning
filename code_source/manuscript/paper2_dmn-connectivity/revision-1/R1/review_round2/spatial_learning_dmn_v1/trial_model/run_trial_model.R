#!/usr/bin/env Rscript
# Locked scanner-only rare-minus-frequent DMN trial model; outputs only this directory.
suppressPackageStartupMessages({library(lme4); library(lmerTest); library(jsonlite)})
options(stringsAsFactors=FALSE, width=160)
ROOT <- Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
OUT <- paste0(Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE)), "/manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/spatial_learning_dmn_v1/trial_model")
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)
NEURAL <- file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv')
TRIALS <- file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv')
SPEC <- file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/spatial_learning_dmn_v1/ANALYSIS_SPEC.md')
PREP <- file.path(ROOT,'manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/prepare_data.py')
DOCS <- file.path(ROOT,'docs/README.md')
sha <- function(p) unname(system2('sha256sum', p, stdout=TRUE) |> strsplit(' +') |> lapply(`[[`,1) |> unlist())
stopifnot(file.exists(NEURAL),file.exists(TRIALS),file.exists(SPEC),file.exists(PREP),file.exists(DOCS))
expected <- c(neural='cc3cddf8377ff9aefba18e8781d2512151e335f280d232c5b23d791266e6feef', trials='fac39301cb3b828f925937e080213bc5ed39fcaa03cbe0e6c558cf2ea18b34c6')
hashes <- c(neural=sha(NEURAL),trials=sha(TRIALS),analysis_spec=sha(SPEC),preparation_code=sha(PREP),task_documentation=sha(DOCS))
stopifnot(identical(unname(hashes['neural']),unname(expected['neural'])), identical(unname(hashes['trials']),unname(expected['trials'])))

# Immutable-source audit and locked recoding.
s <- read.csv(NEURAL,check.names=FALSE); d0 <- read.csv(TRIALS,check.names=FALSE)
stopifnot(nrow(s)==33L, length(unique(s$subject))==33L, all(c('subject','group','dmn_conn_primary') %in% names(s)),
          all(is.finite(s$dmn_conn_primary)), identical(sort(unique(s$group)),c('DS','SS')), sum(s$group=='SS')==16L, sum(s$group=='DS')==17L,
          nrow(d0)==10995L, length(unique(d0$subject))==33L, all(d0$correct==1L), all(is.finite(d0$RT_ms)), all(d0$RT_ms>=200 & d0$RT_ms<=3000))
expected_biased <- (d0$group=='SS' & d0$Dimension=='Same') | (d0$group=='DS' & d0$Dimension=='Different')
recorded_location_valid <- d0$DisRegion %in% c('Rare','Frequent')
source_event_expected <- paste0('sub-',d0$subject,'_task-suppression_run-',d0$run,'_events.tsv')
audit <- data.frame(
  audit=c('raw_trial_rows','raw_subjects','correct_and_finite_200_3000ms','SS_orientation_is_biased','DS_color_is_biased',
          'recorded_category_matches_mapping','biased_trials_recorded_Rare_or_Frequent','all_source_events_task_suppression_and_run_matched',
          'neural_rows','neural_subjects','neural_SS','neural_DS','source_hash_neural_matches_lock','source_hash_trials_matches_lock'),
  observed=c(nrow(d0),length(unique(d0$subject)),sum(d0$correct==1 & is.finite(d0$RT_ms) & d0$RT_ms>=200 & d0$RT_ms<=3000),
             sum(d0$group=='SS' & d0$Dimension=='Same'),sum(d0$group=='DS' & d0$Dimension=='Different'),
             sum((d0$category=='biased')==expected_biased),sum(expected_biased & recorded_location_valid),
             sum(grepl('task-suppression',d0$source_events,fixed=TRUE) & endsWith(d0$source_events,source_event_expected)), 
             nrow(s),length(unique(s$subject)),sum(s$group=='SS'),sum(s$group=='DS'),as.integer(hashes['neural']==expected['neural']),as.integer(hashes['trials']==expected['trials'])),
  expected=c(10995,33,10995,sum(d0$group=='SS' & d0$Dimension=='Same'),sum(d0$group=='DS' & d0$Dimension=='Different'),nrow(d0),sum(expected_biased),nrow(d0),33,33,16,17,1,1),
  pass=NA
)
audit$pass <- audit$observed==audit$expected
write.csv(audit,file.path(OUT,'source_and_coding_audit.csv'),row.names=FALSE)
if(!all(audit$pass)) stop('Source/coding audit failed; no model fitted.')

# X is recomputed solely from corrected primary Default91 FC, group-centered then scaled by global participant SD.
s$D <- as.integer(s$group=='DS')
s$group_mean_correctedFC <- ave(s$dmn_conn_primary,s$group,FUN=mean)
s$correctedFC_group_centered <- s$dmn_conn_primary-s$group_mean_correctedFC
global_correctedFC_group_centered_SD <- sd(s$correctedFC_group_centered)
s$X_corrected_globalSD <- s$correctedFC_group_centered/global_correctedFC_group_centered_SD
stopifnot(isTRUE(all.equal(mean(s$correctedFC_group_centered),0,tolerance=1e-14)),
          isTRUE(all.equal(sd(s$X_corrected_globalSD),1,tolerance=1e-14)))
write.csv(s[,c('subject','group','D','dmn_conn_primary','group_mean_correctedFC','correctedFC_group_centered','X_corrected_globalSD')],file.path(OUT,'neural_coding_corrected_default91.csv'),row.names=FALSE)

# The exact analytic frame: biased dimension and recorded locations only; no imputation/equalization/exclusions.
d <- d0[expected_biased & recorded_location_valid, , drop=FALSE]
d$L <- as.integer(d$DisRegion=='Rare')
d$run2 <- as.integer(d$run==2L)
d <- merge(d, s[,c('subject','group','D','dmn_conn_primary','group_mean_correctedFC','correctedFC_group_centered','X_corrected_globalSD')],
           by=c('subject','group','D'),all.x=TRUE,sort=FALSE,validate='many-to-one')
d <- d[order(d$subject,d$run,d$trial),,drop=FALSE]
d$subject <- factor(d$subject)
d$X <- d$X_corrected_globalSD
stopifnot(nrow(d)==sum(expected_biased), all(d$category=='biased'), all(d$L %in% 0:1), all(d$run2 %in% 0:1),
          all(is.finite(as.matrix(d[,c('RT100','L','X','D','run2')]))), length(unique(d$subject))==33L,
          all(table(d$subject)>0), all(table(d$subject,d$L)>0), all(table(d$subject,d$run2)>0),
          !anyDuplicated(d[,c('subject','run','trial')]), all(d$RT100==d$RT_ms/100))
write.csv(d,file.path(OUT,'exact_model_frame.csv'),row.names=FALSE)
saveRDS(d,file.path(OUT,'exact_model_frame.rds'))
frame_hash <- sha(file.path(OUT,'exact_model_frame.csv'))
frame_counts <- aggregate(list(n=rep(1,nrow(d))),d[c('group','run','DisRegion')],sum)
frame_subject_counts <- aggregate(list(n=rep(1,nrow(d))),d[c('subject','group','run','DisRegion','L')],sum)
write.csv(frame_counts,file.path(OUT,'exact_frame_counts.csv'),row.names=FALSE)
write.csv(frame_subject_counts,file.path(OUT,'exact_frame_subject_counts.csv'),row.names=FALSE)

form <- RT100 ~ L*X*D + run2 + L:run2 + (1+L|subject)
form_sensitivity <- RT100 ~ L*X*D + run2 + L:run2 + (1+L||subject)
fit_capture <- function(expr) {
  warns <- character(); msgs <- character()
  fit <- withCallingHandlers(expr,
    warning=function(w){warns <<- c(warns,conditionMessage(w)); invokeRestart('muffleWarning')},
    message=function(m){msgs <<- c(msgs,conditionMessage(m)); invokeRestart('muffleMessage')})
  list(fit=fit,warnings=unique(warns),messages=unique(msgs))
}
ctrl_bobyqa <- lmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000))
ctrl_nlopt <- lmerControl(optimizer='nloptwrap',optCtrl=list(maxeval=200000))
canonical <- fit_capture(lmer(form,data=d,REML=TRUE,control=ctrl_bobyqa))
alt <- fit_capture(lmer(form,data=d,REML=TRUE,control=ctrl_nlopt))
restart_control <- lmerControl(optimizer='bobyqa',optCtrl=list(maxfun=200000))
restart <- fit_capture(lmer(form,data=d,REML=TRUE,control=restart_control,start=list(theta=getME(alt$fit,'theta'))))
# Independent repetition of the alternative-optimizer-plus-restart sequence required for a clean non-singular solution.
alt_repeat <- fit_capture(lmer(form,data=d,REML=TRUE,control=ctrl_nlopt))
restart_repeat <- fit_capture(lmer(form,data=d,REML=TRUE,control=restart_control,start=list(theta=getME(alt_repeat$fit,'theta'))))
fits <- list(canonical_bobyqa=canonical, alternative_nloptwrap=alt, bobyqa_restart_from_nloptwrap=restart,
             alternative_nloptwrap_repeat=alt_repeat, bobyqa_restart_from_nloptwrap_repeat=restart_repeat)

# Numerical diagnostics preserve raw/scaled gradient, Hessian spectrum, PCA, all covariance parameters and convergence records.
diag_fit <- function(name,obj) {
  f <- obj$fit; oi <- f@optinfo; der <- oi$derivs; h <- der$Hessian; g <- der$gradient
  eig <- if(is.null(h)) rep(NA_real_,0) else eigen(h,symmetric=TRUE,only.values=TRUE)$values
  scaled <- if(is.null(h)||is.null(g)) rep(NA_real_,length(g)) else tryCatch(as.numeric(backsolve(chol(h),g)),error=function(e)rep(NA_real_,length(g)))
  vc <- as.data.frame(VarCorr(f)); rp <- rePCA(f)$subject
  list(
    summary=data.frame(fit=name,REML=TRUE,formula=paste(deparse(form),collapse=' '),optimizer=oi$optimizer,
      optimizer_code=if(is.null(oi$conv$opt)) NA else oi$conv$opt, optimizer_message=paste(c(obj$warnings,obj$messages,oi$conv$lme4$messages),collapse=' | '),
      logLik=as.numeric(logLik(f)),AIC=AIC(f),BIC=BIC(f),sigma_RT100=sigma(f),sigma_ms=sigma(f)*100,
      isSingular_tol1e4=isSingular(f,tol=1e-4),isSingular_tol1e5=isSingular(f,tol=1e-5),
      Hessian_min_eigen=if(length(eig))min(eig) else NA,Hessian_max_eigen=if(length(eig))max(eig) else NA,
      Hessian_nonpositive_eigen=if(length(eig))sum(eig<=0) else NA,stringsAsFactors=FALSE),
    gradient=data.frame(fit=name,index=seq_along(g),raw_gradient=as.numeric(g),scaled_gradient=scaled,abs_raw=abs(g),abs_scaled=abs(scaled)),
    hessian=if(is.null(h)) data.frame(fit=name,index=integer(),eigenvalue=numeric()) else data.frame(fit=name,index=seq_along(eig),eigenvalue=eig),
    theta=data.frame(fit=name,parameter=names(getME(f,'theta')),theta=as.numeric(getME(f,'theta'))),
    varcorr=transform(vc,fit=name),
    pca=data.frame(fit=name,component=seq_along(rp$sdev),sd=rp$sdev,eigenvalue=rp$sdev^2),
    warnings=data.frame(fit=rep(name,length(c(obj$warnings,obj$messages))),kind=c(rep('warning',length(obj$warnings)),rep('message',length(obj$messages))),message=c(obj$warnings,obj$messages),stringsAsFactors=FALSE)
  )
}
diags <- mapply(diag_fit,names(fits),fits,SIMPLIFY=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'summary')),file.path(OUT,'optimizer_diagnostics_summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'gradient')),file.path(OUT,'optimizer_gradients.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'hessian')),file.path(OUT,'optimizer_hessian_eigenvalues.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'theta')),file.path(OUT,'optimizer_theta.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'varcorr')),file.path(OUT,'optimizer_random_covariance.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'pca')),file.path(OUT,'optimizer_rePCA.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(diags,`[[`,'warnings')),file.path(OUT,'optimizer_warnings_messages.csv'),row.names=FALSE)

# Reproducibility comparison is fit-level and never changes formula, rows, REML, or random structure.
comp <- data.frame(term=names(fixef(canonical$fit)),canonical_bobyqa=fixef(canonical$fit),check.names=FALSE)
for(nm in setdiff(names(fits),'canonical_bobyqa')) comp[[nm]] <- fixef(fits[[nm]]$fit)
for(nm in setdiff(names(fits),'canonical_bobyqa')) comp[[paste0('absdiff_',nm,'_vs_canonical')]] <- abs(comp[[nm]]-comp$canonical_bobyqa)
write.csv(comp,file.path(OUT,'optimizer_fixed_effect_comparison_RT100.csv'),row.names=FALSE)
theta_comp <- Reduce(function(x,y)merge(x,y,by='parameter',all=TRUE),lapply(names(fits),function(n)data.frame(parameter=names(getME(fits[[n]]$fit,'theta')),setNames(data.frame(as.numeric(getME(fits[[n]]$fit,'theta'))),n))))
write.csv(theta_comp,file.path(OUT,'optimizer_theta_comparison.csv'),row.names=FALSE)
lls <- sapply(fits,function(x)as.numeric(logLik(x$fit)))
all_singular <- all(sapply(fits,function(x)isSingular(x$fit,tol=1e-4)))
no_messages_nonsingular <- sapply(fits,function(x)!isSingular(x$fit,tol=1e-4) && length(unique(c(x$warnings,x$messages,x$fit@optinfo$conv$lme4$messages)))==0)
reproducible <- max(abs(lls-lls[1]))<1e-6 && max(as.matrix(comp[,grepl('^absdiff_',names(comp)),drop=FALSE]))<1e-6 && max(abs(theta_comp[,-1,drop=FALSE]-theta_comp[,2]))<1e-5

# Prespecified changed-covariance sensitivity only if every same-formula correlated fit is singular.
sensitivity <- NULL
if(all_singular) {
  sensitivity <- fit_capture(lmer(form_sensitivity,data=d,REML=TRUE,control=ctrl_bobyqa))
  saveRDS(sensitivity$fit,file.path(OUT,'sensitivity_independent_random_effects.rds'))
  ss <- summary(sensitivity$fit)$coefficients
  sdf <- data.frame(term=rownames(ss),estimate_RT100=ss[,'Estimate'],SE_RT100=ss[,'Std. Error'],df=ss[,'df'],t=ss[,'t value'],p_raw=ss[,'Pr(>|t|)'])
  sdf$estimate_ms=sdf$estimate_RT100*100; sdf$SE_ms=sdf$SE_RT100*100
  write.csv(sdf,file.path(OUT,'sensitivity_independent_random_effects_fixed_effects.csv'),row.names=FALSE)
  write.csv(as.data.frame(VarCorr(sensitivity$fit)),file.path(OUT,'sensitivity_independent_random_effects_covariance.csv'),row.names=FALSE)
}

# The locked correlated canonical fit remains final-selected; diagnostic/sensitivity fits are not adopted by significance.
selected <- canonical$fit
for(nm in names(fits)) saveRDS(fits[[nm]]$fit,file.path(OUT,paste0('fit_',nm,'.rds')))
saveRDS(canonical$fit,file.path(OUT,'fit_canonical_bobyqa_correlated.rds'))
saveRDS(alt$fit,file.path(OUT,'fit_alternative_nloptwrap_correlated.rds'))
saveRDS(restart$fit,file.path(OUT,'fit_bobyqa_restart_from_nloptwrap_correlated.rds'))
saveRDS(selected,file.path(OUT,'final_selected_model.rds'))
cf <- summary(selected)$coefficients
fixed <- data.frame(term=rownames(cf),estimate_RT100=cf[,'Estimate'],SE_RT100=cf[,'Std. Error'],df=cf[,'df'],t=cf[,'t value'],p_raw=cf[,'Pr(>|t|)'])
fixed$estimate_ms <- fixed$estimate_RT100*100; fixed$SE_ms <- fixed$SE_RT100*100
fixed$ci_low_ms <- fixed$estimate_ms-qt(.975,fixed$df)*fixed$SE_ms; fixed$ci_high_ms <- fixed$estimate_ms+qt(.975,fixed$df)*fixed$SE_ms
write.csv(fixed,file.path(OUT,'final_selected_fixed_effects.csv'),row.names=FALSE)
rv <- as.data.frame(VarCorr(selected)); rv$vcov_ms2 <- rv$vcov*10000; rv$sdcor_ms <- ifelse(is.na(rv$var2),rv$sdcor*100,rv$sdcor)
write.csv(rv,file.path(OUT,'final_selected_random_covariance.csv'),row.names=FALSE)
re <- ranef(selected,condVar=TRUE)$subject
re_out <- data.frame(subject=rownames(re),re,check.names=FALSE); names(re_out)[2:3] <- c('random_intercept_RT100','random_L_slope_RT100'); re_out$random_intercept_ms <- re_out$random_intercept_RT100*100; re_out$random_L_slope_ms <- re_out$random_L_slope_RT100*100
write.csv(re_out,file.path(OUT,'final_selected_subject_random_effects.csv'),row.names=FALSE)

canon_term <- function(x) paste(sort(strsplit(x,':',fixed=TRUE)[[1]]),collapse=':')
make_contrast <- function(label,w) {
  v <- setNames(rep(0,length(fixef(selected))),names(fixef(selected)))
  for(term in names(w)) { idx <- which(vapply(names(v),canon_term,'')==canon_term(term)); stopifnot(length(idx)==1L); v[idx] <- w[[term]] }
  z <- as.data.frame(contest1D(selected,v,confint=TRUE,ddf='Satterthwaite'))
  nms <- names(z); lo <- nms[grepl('lower',nms,ignore.case=TRUE)][1]; hi <- nms[grepl('upper',nms,ignore.case=TRUE)][1]
  data.frame(question=label,weight_spec=paste(names(w),w,sep='*',collapse=' + '),estimate_RT100=z$Estimate,SE_RT100=z[['Std. Error']],df=z$df,t=z[['t value']],ci_low_RT100=z[[lo]],ci_high_RT100=z[[hi]],p_raw=z[['Pr(>|t|)']])
}
contrasts <- rbind(
 make_contrast('Pooled participant-weighted DMN modulation of Rare-minus-Frequent location effect',list('L:X'=1,'L:X:D'=17/33)),
 make_contrast('SS DMN modulation of Rare-minus-Frequent location effect',list('L:X'=1)),
 make_contrast('DS DMN modulation of Rare-minus-Frequent location effect',list('L:X'=1,'L:X:D'=1)),
 make_contrast('DS-minus-SS difference in DMN modulation of Rare-minus-Frequent location effect',list('L:X:D'=1))
)
for(k in c('estimate','SE','ci_low','ci_high')) contrasts[[paste0(k,'_ms_per_global_corrected_DMN_SD')]] <- contrasts[[paste0(k,'_RT100')]]*100
contrasts$p_Holm4 <- p.adjust(contrasts$p_raw,method='holm'); contrasts$Holm_family <- 'four prespecified new neural-location slope contrasts'; contrasts$unit <- 'ms per global SD of group-centered corrected dmn_conn_primary'; contrasts$positive_sign_interpretation <- 'positive: higher corrected DMN predicts a larger Rare-minus-Frequent RT contrast (greater rare-location slowing relative to frequent)'
write.csv(contrasts,file.path(OUT,'final_selected_neural_location_contrasts_Holm4.csv'),row.names=FALSE)

# Full residual numerical summaries, without using trials as participant-level information.
r <- residuals(selected)*100
resdiag <- do.call(rbind,lapply(split(seq_along(r),interaction(d$group,d$DisRegion,drop=TRUE)),function(i){x<-r[i]; data.frame(cell=as.character(interaction(d$group,d$DisRegion,drop=TRUE)[i[1]]),n=length(x),mean_ms=mean(x),sd_ms=sd(x),min_ms=min(x),max_ms=max(x),skewness=mean((x-mean(x))^3)/mean((x-mean(x))^2)^1.5,kurtosis=mean((x-mean(x))^4)/mean((x-mean(x))^2)^2)}))
write.csv(resdiag,file.path(OUT,'final_selected_residual_diagnostics.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(OUT,'R_sessionInfo.txt'))

verification <- list(
 status='PASS', scope='scanner-only N=33, biased-dimension trials; RT100 ~ L*X*D + run2 + L:run2 + (1+L|subject), REML',
 source_hashes=as.list(hashes), expected_source_hashes=as.list(expected), exact_model_frame_sha256=frame_hash,
 checks=list(raw_rows_10995=nrow(d0)==10995, neural_N33=nrow(s)==33, SS16_DS17=sum(s$group=='SS')==16 && sum(s$group=='DS')==17,
  source_and_coding_audit_pass=all(audit$pass), model_rows=nrow(d), model_subjects=length(unique(d$subject)), all_biased=d$category |> unique() |> identical('biased'),
  L_is_Rare_1_Frequent_0=identical(sort(unique(d$L)),c(0L,1L)), exact_key_unique=!anyDuplicated(d[,c('subject','run','trial')]),
  corrected_primary_only=all(is.finite(d$dmn_conn_primary)) && all(is.finite(d$X_corrected_globalSD)), X_global_participant_SD=global_correctedFC_group_centered_SD,
  first_run_trials_retained=sum(as.character(d$first_run)=='True'), formula=paste(deparse(form),collapse=' '), REML=TRUE,
  same_formula_optimizer_fits=names(fits), all_same_formula_singular=all_singular,
  optimizer_numerically_reproducible=reproducible, selected_model='canonical_bobyqa_correlated; diagnostics do not replace locked model',
  sensitivity_ran=all_singular, sensitivity_changed_covariance=if(all_singular)'(1+L||subject), numeric L; labelled sensitivity only' else 'not run by policy'),
 artifacts=list(final_selected_model='final_selected_model.rds',exact_frame='exact_model_frame.csv',fixed_effects='final_selected_fixed_effects.csv',random_covariance='final_selected_random_covariance.csv',random_effects='final_selected_subject_random_effects.csv',four_contrasts='final_selected_neural_location_contrasts_Holm4.csv')
)
write_json(verification,file.path(OUT,'verification.json'),pretty=TRUE,auto_unbox=TRUE)
cat(sprintf('PASS: %d model rows, %d participants; all_same_formula_singular=%s; reproducible=%s\n',nrow(d),length(unique(d$subject)),all_singular,reproducible))
print(contrasts)
