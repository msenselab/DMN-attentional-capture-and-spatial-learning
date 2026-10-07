#!/usr/bin/env Rscript
# One sequential heavy job: primary and history-adjusted trial-level LMMs.
suppressPackageStartupMessages({library(lme4); library(lmerTest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('usage: Rscript fit_new_predictor_mixed_models.R TRIALS_CSV OUTPUT_DIR')
trials_path <- args[[1]]; out <- args[[2]]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
d <- read.csv(trials_path, check.names=FALSE)
required <- c('subject','group','run','RT100','B','U','D','X','log_order','prior_biased','prior_unbiased')
if (!all(required %in% names(d))) stop('Missing required trial columns: ', paste(setdiff(required,names(d)), collapse=', '))
if (nrow(d) != 10995L || length(unique(d$subject)) != 33L) stop('History handoff must preserve 10,995 trials and 33 participants.')
if (!all(sort(as.integer(tapply(d$subject, d$group, function(x) length(unique(x))))) == c(16,17))) stop('Expected SS=16 and DS=17.')
if (any(!is.finite(as.matrix(d[,c('RT100','B','U','D','X','log_order','prior_biased','prior_unbiased')])))) stop('Nonfinite model values.')
d$subject <- factor(d$subject)
d$run2 <- as.integer(d$run == 2)
for (nm in c('log_order','prior_biased','prior_unbiased')) {
  zname <- paste0('z_',nm)
  d[[zname]] <- as.numeric(ave(d[[nm]], d$subject, FUN=function(x) {
    s <- sd(x)
    if (!is.finite(s) || s == 0) stop('Zero/invalid within-subject SD for ', nm)
    (x - mean(x))/s
  }))
}
# Exact authorized structures: 12 fixed columns for the primary joint model,
# 144 for the history model (12 core + 33 subjects x 4 nuisance slopes).
formulas <- list(
  primary_joint = RT100 ~ B*X*D + U*X*D + (1+B+U|subject),
  primary_joint_history = RT100 ~ B*X*D + U*X*D + subject:run2 + subject:z_log_order + subject:z_prior_biased + subject:z_prior_unbiased + (1+B+U|subject)
)
expected_columns <- c(primary_joint=12L, primary_joint_history=144L)
control <- lmerControl(optimizer='bobyqa', calc.derivs=TRUE, optCtrl=list(maxfun=200000))
canonical <- function(x) paste(sort(strsplit(x, ':', fixed=TRUE)[[1]]), collapse=':')
contrast_row <- function(fit, label, weights, model) {
  L <- setNames(rep(0, length(fixef(fit))), names(fixef(fit)))
  for (term in names(weights)) {
    hit <- which(vapply(names(L), canonical, '') == canonical(term))
    if (length(hit) != 1L) stop('Contrast term missing/nonunique: ', term, ' in ', model)
    L[hit] <- weights[[term]]
  }
  z <- as.data.frame(contest1D(fit, L, confint=TRUE, ddf='Satterthwaite'))
  data.frame(model=model, contrast=label, N=33L, trials=nrow(d),
             estimate_ms=z$Estimate*100, se_ms=z[['Std. Error']]*100,
             df=z$df, t=z[['t value']], p_raw=z[['Pr(>|t|)']],
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
fit_one <- function(model, formula) {
  warnings <- character()
  fit <- withCallingHandlers(
    lmer(formula, data=d, REML=TRUE, control=control),
    warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart('muffleWarning') },
    message=function(m) { warnings <<- c(warnings, conditionMessage(m)); invokeRestart('muffleMessage') }
  )
  Xmat <- getME(fit,'X')
  if (ncol(Xmat) != expected_columns[[model]] || qr(Xmat)$rank != expected_columns[[model]]) {
    stop(model, ' design mismatch: expected full-rank ', expected_columns[[model]],
         ', got ', ncol(Xmat), ' columns and rank ', qr(Xmat)$rank)
  }
  saveRDS(fit, file.path(out,paste0(model,'.rds')))
  fx <- as.data.frame(coef(summary(fit))); fx$term <- rownames(fx); rownames(fx) <- NULL
  names(fx)[names(fx)=='Estimate'] <- 'estimate_RT100'
  names(fx)[names(fx)=='Std. Error'] <- 'se_RT100'
  names(fx)[names(fx)=='Pr(>|t|)'] <- 'p_raw'
  names(fx)[names(fx)=='t value'] <- 't'
  fx$model <- model; fx$estimate_ms <- fx$estimate_RT100*100; fx$se_ms <- fx$se_RT100*100
  fx$ci_low_ms <- fx$estimate_ms - qt(.975,fx$df)*fx$se_ms
  fx$ci_high_ms <- fx$estimate_ms + qt(.975,fx$df)*fx$se_ms
  write.csv(fx,file.path(out,paste0(model,'_fixed_effects_complete.csv')),row.names=FALSE)
  write.csv(data.frame(model=model,column_index=seq_len(ncol(Xmat)),column=colnames(Xmat)),file.path(out,paste0(model,'_model_matrix_columns.csv')),row.names=FALSE)
  C <- make_contrasts(fit,model); write.csv(C,file.path(out,paste0(model,'_simple_slopes_and_contrasts.csv')),row.names=FALSE)
  omni <- as.data.frame(anova(fit, type=3, ddf='Satterthwaite')); omni$term <- rownames(omni); rownames(omni) <- NULL; omni$model <- model
  write.csv(omni,file.path(out,paste0(model,'_omnibus_type3.csv')),row.names=FALSE)
  vc <- as.matrix(VarCorr(fit)$subject); write.csv(vc*10000,file.path(out,paste0(model,'_random_covariance_ms2.csv')))
  eig <- eigen(vc,symmetric=TRUE,only.values=TRUE)$values
  msg <- unique(c(warnings,fit@optinfo$conv$lme4$messages))
  grad <- fit@optinfo$derivs$gradient
  diagnostics <- data.frame(model=model,N=33L,trials=nrow(d),REML=TRUE,
    formula=paste(deparse(formula),collapse=' '),fixed_columns=ncol(Xmat),fixed_rank=qr(Xmat)$rank,
    expected_fixed_columns=expected_columns[[model]],singular=isSingular(fit,tol=1e-4),
    optimizer_code=fit@optinfo$conv$opt,messages=paste(msg,collapse=' | '),
    max_gradient=if(length(grad)) max(abs(grad)) else NA_real_,residual_SD_ms=sigma(fit)*100,
    random_min_eigenvalue_RT100=min(eig),random_max_eigenvalue_RT100=max(eig),logLik=as.numeric(logLik(fit)))
  alt <- lmer(formula,data=d,REML=TRUE,control=lmerControl(optimizer='nloptwrap',calc.derivs=TRUE,optCtrl=list(maxeval=200000)))
  alt_msg <- unique(alt@optinfo$conv$lme4$messages)
  bad_pattern <- 'failed to converge|unable to evaluate|degenerate'
  bobyqa_ok <- isTRUE(fit@optinfo$conv$opt == 0) && !any(grepl(bad_pattern,msg,ignore.case=TRUE))
  nloptwrap_ok <- isTRUE(alt@optinfo$conv$opt == 0) && !any(grepl(bad_pattern,alt_msg,ignore.case=TRUE))
  diagnostic_reasons <- c(if (!bobyqa_ok) 'bobyqa convergence/message flag', if (!nloptwrap_ok) 'nloptwrap convergence/message flag', if (isSingular(fit,tol=1e-4)) 'singular random-effects fit')
  diagnostics$bobyqa_converged <- bobyqa_ok
  diagnostics$nloptwrap_converged <- nloptwrap_ok
  diagnostics$nloptwrap_messages <- paste(alt_msg,collapse=' | ')
  diagnostics$diagnostic_status <- if (length(diagnostic_reasons)) 'DIAGNOSTIC_FLAGGED_NOT_READY' else 'PASSED'
  diagnostics$diagnostic_reasons <- paste(diagnostic_reasons,collapse=' | ')
  write.csv(diagnostics,file.path(out,paste0(model,'_diagnostics.csv')),row.names=FALSE)
  write.csv(data.frame(model=model,term=names(fixef(fit)),bobyqa_RT100=fixef(fit),nloptwrap_RT100=fixef(alt),abs_difference_RT100=abs(fixef(fit)-fixef(alt))),file.path(out,paste0(model,'_optimizer_agreement.csv')),row.names=FALSE)
  capture.output(summary(fit),file=file.path(out,paste0(model,'_summary.txt')))
  list(diagnostics=diagnostics, contrasts=C)
}
# Deliberately sequential: no duplicate concurrent heavy mixed-model runs.
results <- mapply(fit_one, names(formulas), formulas, SIMPLIFY=FALSE)
write.csv(do.call(rbind,lapply(results,`[[`,'diagnostics')),file.path(out,'mixed_model_diagnostics_all.csv'),row.names=FALSE)
write.csv(do.call(rbind,lapply(results,`[[`,'contrasts')),file.path(out,'mixed_model_simple_slopes_all.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
cat('Sequential mixed-model outputs written: primary_joint (12 columns), primary_joint_history (144 columns). Inspect diagnostics before treating results as ready.\n')
