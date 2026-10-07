#!/usr/bin/env Rscript
# Standalone RDS-to-table validation. It only reads saved RDS/table outputs.
suppressPackageStartupMessages({library(lme4); library(lmerTest)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop('usage: Rscript recovery_integration_v2_validate_rds_tables.R PRIMARY_RDS MIXED_MODELS_DIR AUDIT_JSON')
rds_path <- args[[1]]; out <- args[[2]]; audit <- args[[3]]
fit <- readRDS(rds_path)
if (!inherits(fit,'merMod')) stop('Primary RDS not a merMod.')
mf <- model.frame(fit)
ddf_devfun <- lme4::lmer(formula(fit), data=mf, REML=isREML(fit),
  control=lmerControl(optimizer='bobyqa', calc.derivs=TRUE, optCtrl=list(maxfun=200000)), devFunOnly=TRUE)
fit_lt <- lmerTest:::as_lmerModLT(fit, ddf_devfun)
num_equal <- function(a,b,tol=2e-10) length(a)==length(b) && all(is.finite(a)) && all(is.finite(b)) && all(abs(a-b)<=tol)
canon <- function(x) paste(sort(strsplit(x,':',fixed=TRUE)[[1]]),collapse=':')
failures <- character(); check <- function(name, ok) {if(!isTRUE(ok)) failures <<- c(failures,name)}
fixed <- read.csv(file.path(out,'primary_joint_fixed_effects_complete.csv'),check.names=FALSE)
expected <- as.data.frame(coef(summary(fit_lt))); expected$term <- rownames(expected); rownames(expected)<-NULL
names(expected)[names(expected)=='Estimate'] <- 'estimate_RT100'; names(expected)[names(expected)=='Std. Error'] <- 'se_RT100'; names(expected)[names(expected)=='Pr(>|t|)'] <- 'p_raw'; names(expected)[names(expected)=='t value'] <- 't'
check('fixed_terms', identical(as.character(fixed$term),as.character(expected$term)))
for(nm in c('estimate_RT100','se_RT100','df','t','p_raw')) check(paste0('fixed_',nm), num_equal(fixed[[nm]],expected[[nm]]))
check('fixed_ms_estimates',num_equal(fixed$estimate_ms,fixed$estimate_RT100*100));check('fixed_ms_se',num_equal(fixed$se_ms,fixed$se_RT100*100));check('fixed_ci_low',num_equal(fixed$ci_low_ms,(fixed$estimate_RT100-qt(.975,fixed$df)*fixed$se_RT100)*100));check('fixed_ci_high',num_equal(fixed$ci_high_ms,(fixed$estimate_RT100+qt(.975,fixed$df)*fixed$se_RT100)*100))
contrast_row <- function(label,weights) {L<-setNames(rep(0,length(fixef(fit_lt))),names(fixef(fit_lt)));for(term in names(weights)){hit<-which(vapply(names(L),canon,'')==canon(term));if(length(hit)!=1L)stop('contrast term failure');L[hit]<-weights[[term]]};z<-as.data.frame(contest1D(fit_lt,L,confint=TRUE,ddf='Satterthwaite'));data.frame(contrast=label,estimate_ms=z$Estimate*100,se_ms=z[['Std. Error']]*100,df=z$df,t=z[['t value']],p_raw=z[['Pr(>|t|)']],ci_low_ms=z$lower*100,ci_high_ms=z$upper*100)}
wantc<-rbind(contrast_row('Pooled participant-weighted biased DMN slope',list('B:X'=1,'B:X:D'=17/33)),contrast_row('Pooled participant-weighted unbiased DMN slope',list('U:X'=1,'U:X:D'=17/33)),contrast_row('SS biased DMN slope',list('B:X'=1)),contrast_row('DS biased DMN slope',list('B:X'=1,'B:X:D'=1)),contrast_row('SS unbiased DMN slope',list('U:X'=1)),contrast_row('DS unbiased DMN slope',list('U:X'=1,'U:X:D'=1)),contrast_row('Group moderation biased DS-SS',list('B:X:D'=1)),contrast_row('Group moderation unbiased DS-SS',list('U:X:D'=1)),contrast_row('Pooled biased-minus-unbiased DMN slope',list('B:X'=1,'B:X:D'=17/33,'U:X'=-1,'U:X:D'=-17/33)))
gotc<-read.csv(file.path(out,'primary_joint_simple_slopes_and_contrasts.csv'),check.names=FALSE)
check('contrast_labels',identical(as.character(gotc$contrast),as.character(wantc$contrast)))
for(nm in c('estimate_ms','se_ms','df','t','p_raw','ci_low_ms','ci_high_ms'))check(paste0('contrast_',nm),num_equal(gotc[[nm]],wantc[[nm]]))
check('contrast_holm9',num_equal(gotc$p_Holm9,p.adjust(gotc$p_raw,method='holm')))
want_o<-as.data.frame(anova(fit_lt,type=3,ddf='Satterthwaite'));want_o$term<-rownames(want_o);rownames(want_o)<-NULL
got_o<-read.csv(file.path(out,'primary_joint_omnibus_type3.csv'),check.names=FALSE)
check('type3_terms',identical(as.character(got_o$term),as.character(want_o$term)))
for(nm in c('NumDF','DenDF','F value','Pr(>F)'))check(paste0('type3_',nm),num_equal(got_o[[nm]],want_o[[nm]]))
cols<-read.csv(file.path(out,'primary_joint_model_matrix_columns.csv'),check.names=FALSE);check('matrix_columns',nrow(cols)==ncol(getME(fit,'X')) && identical(as.character(cols$column),colnames(getME(fit,'X'))))
status<-if(length(failures))'FAIL' else 'PASS'; esc<-function(x)gsub('"','\\\\"',x,fixed=TRUE); body<-paste0('{\n  "validator": "recovery_integration_v2_validate_rds_tables.R",\n  "status": "',status,'",\n  "primary_rds": "',esc(normalizePath(rds_path)),'",\n  "checks": {"fixed_effect_estimates_se_df_p": ',if(any(grepl('^fixed_',failures)))'false' else 'true',', "contrasts": ',if(any(grepl('^contrast_',failures)))'false' else 'true',', "typeIII": ',if(any(grepl('^type3_',failures)))'false' else 'true',', "matrix_columns": ',if('matrix_columns'%in%failures)'false' else 'true','},\n  "failures": [',paste(sprintf('"%s"',esc(failures)),collapse=', '),']\n}\n');writeLines(body,audit);cat(status,'\n',sep='');quit(status=if(status=='PASS')0 else 1)
