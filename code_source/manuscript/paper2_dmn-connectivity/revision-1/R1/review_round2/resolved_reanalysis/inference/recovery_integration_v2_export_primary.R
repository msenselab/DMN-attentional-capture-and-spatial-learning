#!/usr/bin/env Rscript
# Re-export one already fitted, verified primary_joint model; never optimize or refit it.
suppressPackageStartupMessages({library(lme4); library(lmerTest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5L) stop('usage: Rscript recovery_integration_v2_export_primary.R SELECTED_RDS COMPARATOR_RDS ORIGINAL_RDS TRIALS_CSV OUTPUT_MIXED_MODELS_DIR')
selected_path <- args[[1]]; comparator_path <- args[[2]]; original_path <- args[[3]]; trials_path <- args[[4]]; out <- args[[5]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
for (p in c(selected_path, comparator_path, original_path, trials_path)) if (!file.exists(p)) stop('Missing required input: ', p)
fit <- readRDS(selected_path); comparator <- readRDS(comparator_path); original <- readRDS(original_path)
for (object in list(fit, comparator, original)) if (!inherits(object, 'merMod')) stop('Input RDS is not a merMod object.')
d <- read.csv(trials_path, check.names=FALSE)
if (nrow(d) != 10995L || length(unique(d$subject)) != 33L) stop('Expected exact 10,995-row, N=33 handoff.')
d$subject <- factor(d$subject)
formula_primary <- RT100 ~ B*X*D + U*X*D + (1+B+U|subject)
formula_text <- paste(deparse(formula_primary), collapse=' ')
if (!isTRUE(all.equal(formula(fit, fixed.only=FALSE), formula_primary))) stop('Selected recovery RDS formula mismatch.')
if (!isTRUE(all.equal(formula(comparator, fixed.only=FALSE), formula_primary))) stop('Comparator RDS formula mismatch.')
if (!isTRUE(isREML(fit)) || !isTRUE(isREML(comparator))) stop('Selected/comparator must be REML fits.')
if (nobs(fit) != 10995L || nobs(comparator) != 10995L) stop('Saved RDS nobs mismatch.')
for (saved in list(fit, original)) {
  mf <- model.frame(saved)
  for (nm in c('RT100','B','U','D','X')) if (!identical(as.numeric(mf[[nm]]), as.numeric(d[[nm]]))) stop('Saved model frame differs from trial handoff: ', nm)
  if (!identical(as.character(mf$subject), as.character(d$subject))) stop('Saved model frame subject order differs from trial handoff.')
}
if (!identical(as.numeric(fixef(fit)), as.numeric(fixef(readRDS(selected_path))))) stop('RDS reload instability.')
# The recovery RDS is an lme4 merMod. Build only lmerTest's deviance-function
# handoff from the exact saved frame, then attach inference machinery to the
# saved optimum. No optimizer is invoked and no coefficients/theta are changed.
ddf_devfun <- lme4::lmer(formula_primary, data=d, REML=TRUE,
  control=lmerControl(optimizer='bobyqa', calc.derivs=TRUE, optCtrl=list(maxfun=200000)), devFunOnly=TRUE)
fit_lt <- lmerTest:::as_lmerModLT(fit, ddf_devfun)
canonical <- function(x) paste(sort(strsplit(x, ':', fixed=TRUE)[[1]]), collapse=':')
contrast_row <- function(fit, label, weights, model) {
  L <- setNames(rep(0, length(fixef(fit))), names(fixef(fit)))
  for (term in names(weights)) {
    hit <- which(vapply(names(L), canonical, '') == canonical(term))
    if (length(hit) != 1L) stop('Contrast term missing/nonunique: ', term)
    L[hit] <- weights[[term]]
  }
  z <- as.data.frame(contest1D(fit, L, confint=TRUE, ddf='Satterthwaite'))
  data.frame(model=model, contrast=label, N=33L, trials=nrow(d), estimate_ms=z$Estimate*100,
             se_ms=z[['Std. Error']]*100, df=z$df, t=z[['t value']], p_raw=z[['Pr(>|t|)']],
             ci_low_ms=z$lower*100, ci_high_ms=z$upper*100)
}
make_contrasts <- function(fit, model) {
  C <- rbind(
    contrast_row(fit,'Pooled participant-weighted biased DMN slope',list('B:X'=1,'B:X:D'=17/33),model),
    contrast_row(fit,'Pooled participant-weighted unbiased DMN slope',list('U:X'=1,'U:X:D'=17/33),model),
    contrast_row(fit,'SS biased DMN slope',list('B:X'=1),model),
    contrast_row(fit,'DS biased DMN slope',list('B:X'=1,'B:X:D'=1),model),
    contrast_row(fit,'SS unbiased DMN slope',list('U:X'=1),model),
    contrast_row(fit,'DS unbiased DMN slope',list('U:X'=1,'U:X:D'=1),model),
    contrast_row(fit,'Group moderation biased DS-SS',list('B:X:D'=1),model),
    contrast_row(fit,'Group moderation unbiased DS-SS',list('U:X:D'=1),model),
    contrast_row(fit,'Pooled biased-minus-unbiased DMN slope',list('B:X'=1,'B:X:D'=17/33,'U:X'=-1,'U:X:D'=-17/33),model)
  )
  C$p_Holm9 <- p.adjust(C$p_raw, method='holm')
  C$Holm_family <- 'nine exploratory contrasts from this joint LMM'
  C
}
Xmat <- getME(fit,'X')
if (ncol(Xmat) != 12L || qr(Xmat)$rank != 12L) stop('Selected RDS must retain rank-12 fixed design.')
fx <- as.data.frame(coef(summary(fit_lt))); fx$term <- rownames(fx); rownames(fx) <- NULL
names(fx)[names(fx)=='Estimate'] <- 'estimate_RT100'; names(fx)[names(fx)=='Std. Error'] <- 'se_RT100'; names(fx)[names(fx)=='Pr(>|t|)'] <- 'p_raw'; names(fx)[names(fx)=='t value'] <- 't'
fx$model <- 'primary_joint'; fx$estimate_ms <- fx$estimate_RT100*100; fx$se_ms <- fx$se_RT100*100
fx$ci_low_ms <- fx$estimate_ms - qt(.975,fx$df)*fx$se_ms; fx$ci_high_ms <- fx$estimate_ms + qt(.975,fx$df)*fx$se_ms
write.csv(fx,file.path(out,'primary_joint_fixed_effects_complete.csv'),row.names=FALSE)
write.csv(data.frame(model='primary_joint',column_index=seq_len(ncol(Xmat)),column=colnames(Xmat)),file.path(out,'primary_joint_model_matrix_columns.csv'),row.names=FALSE)
C <- make_contrasts(fit_lt,'primary_joint'); write.csv(C,file.path(out,'primary_joint_simple_slopes_and_contrasts.csv'),row.names=FALSE)
omni <- as.data.frame(anova(fit_lt,type=3,ddf='Satterthwaite')); omni$term <- rownames(omni); rownames(omni) <- NULL; omni$model <- 'primary_joint'; write.csv(omni,file.path(out,'primary_joint_omnibus_type3.csv'),row.names=FALSE)
vc <- as.matrix(VarCorr(fit)$subject); write.csv(vc*10000,file.path(out,'primary_joint_random_covariance_ms2.csv'))
msg <- unique(fit@optinfo$conv$lme4$messages); grad <- fit@optinfo$derivs$gradient
bad_pattern <- 'failed to converge|unable to evaluate|degenerate'
selected_ok <- isTRUE(fit@optinfo$conv$opt == 0) && !isSingular(fit,tol=1e-4) && !any(grepl(bad_pattern,msg,ignore.case=TRUE))
comp_msg <- unique(comparator@optinfo$conv$lme4$messages)
comparator_ok <- isTRUE(comparator@optinfo$conv$opt == 0) && !isSingular(comparator,tol=1e-4) && !any(grepl(bad_pattern,comp_msg,ignore.case=TRUE))
if (!selected_ok || !comparator_ok) stop('Recovered saved models do not satisfy retained numerical gates.')
diag <- data.frame(model='primary_joint',N=33L,trials=nrow(d),REML=TRUE,formula=formula_text,fixed_columns=ncol(Xmat),fixed_rank=qr(Xmat)$rank,expected_fixed_columns=12L,singular=isSingular(fit,tol=1e-4),optimizer_code=fit@optinfo$conv$opt,messages=paste(msg,collapse=' | '),max_gradient=if(length(grad)) max(abs(grad)) else NA_real_,residual_SD_ms=sigma(fit)*100,random_min_eigenvalue_RT100=min(eigen(vc,symmetric=TRUE,only.values=TRUE)$values),random_max_eigenvalue_RT100=max(eigen(vc,symmetric=TRUE,only.values=TRUE)$values),logLik=as.numeric(logLik(fit)),bobyqa_converged=selected_ok,nloptwrap_converged=comparator_ok,nloptwrap_messages=paste(comp_msg,collapse=' | '),diagnostic_status='PASSED',diagnostic_reasons='')
write.csv(diag,file.path(out,'primary_joint_diagnostics.csv'),row.names=FALSE)
write.csv(data.frame(model='primary_joint',term=names(fixef(fit)),bobyqa_RT100=fixef(fit),nloptwrap_RT100=fixef(comparator),abs_difference_RT100=abs(fixef(fit)-fixef(comparator))),file.path(out,'primary_joint_optimizer_agreement.csv'),row.names=FALSE)
capture.output(summary(fit_lt),file=file.path(out,'primary_joint_summary.txt'))
recovery <- data.frame(model='primary_joint',selected_solution='bobyqa_restart_from_nloptwrap',selected_rds=normalizePath(selected_path),comparator_rds=normalizePath(comparator_path),original_rds=normalizePath(original_path),nobs=nobs(fit),fixed_columns=ncol(Xmat),fixed_rank=qr(Xmat)$rank,REML=isREML(fit),optimizer=fit@optinfo$optimizer,optimizer_code=fit@optinfo$conv$opt,singular_tol_1e4=isSingular(fit,tol=1e-4),hessian_min_eigenvalue=min(eigen((fit@optinfo$derivs$Hessian+t(fit@optinfo$derivs$Hessian))/2,symmetric=TRUE,only.values=TRUE)$values),raw_gradient_max=max(abs(grad)),selected_logLik=as.numeric(logLik(fit)),comparator_logLik=as.numeric(logLik(comparator)),likelihood_gain_vs_comparator=as.numeric(logLik(fit)-logLik(comparator)),original_logLik=as.numeric(logLik(original)),likelihood_gain_vs_original=as.numeric(logLik(fit)-logLik(original)),stringsAsFactors=FALSE)
write.csv(recovery,file.path(out,'primary_joint_optimizer_recovery_diagnostics.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'recovery_export_R_sessionInfo.txt'))
cat('PASS: primary_joint re-exported from saved recovery RDS without fitting.\n')
