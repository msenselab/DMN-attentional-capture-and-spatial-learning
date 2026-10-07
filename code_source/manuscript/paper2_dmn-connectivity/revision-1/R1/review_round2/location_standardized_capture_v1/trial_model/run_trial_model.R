#!/usr/bin/env Rscript
# Exploratory five-condition, location-standardized capture LMM. Outputs only this directory.
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(jsonlite) })
options(stringsAsFactors = FALSE, width = 180)

ROOT <- Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
OUT <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/trial_model")
TRIALS <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv")
NEURAL <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv")
SPEC <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/ANALYSIS_SPEC.md")
args_all <- commandArgs(trailingOnly = FALSE)
SCRIPT <- normalizePath(sub("^--file=", "", args_all[grep("^--file=", args_all)]))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

sha <- function(path) unname(strsplit(system2("sha256sum", path, stdout = TRUE), " +")[[1]][1])
expected_hashes <- c(
  trials = "fac39301cb3b828f925937e080213bc5ed39fcaa03cbe0e6c558cf2ea18b34c6",
  neural = "cc3cddf8377ff9aefba18e8781d2512151e335f280d232c5b23d791266e6feef"
)
stopifnot(file.exists(TRIALS), file.exists(NEURAL), file.exists(SPEC), file.exists(SCRIPT))
hashes <- c(trials = sha(TRIALS), neural = sha(NEURAL), analysis_spec = sha(SPEC), run_script = sha(SCRIPT))
stopifnot(identical(hashes[names(expected_hashes)], expected_hashes))

# Locked source checks, trial coding, and X reconstruction.
t0 <- read.csv(TRIALS, check.names = FALSE)
s <- read.csv(NEURAL, check.names = FALSE)
stopifnot(
  nrow(t0) == 10995L, length(unique(t0$subject)) == 33L,
  all(t0$correct == 1L), all(is.finite(t0$RT_ms)), all(t0$RT_ms >= 200 & t0$RT_ms <= 3000),
  nrow(s) == 33L, length(unique(s$subject)) == 33L,
  all(c("subject", "group", "dmn_conn_primary") %in% names(s)),
  all(is.finite(s$dmn_conn_primary)), identical(sort(unique(s$group)), c("DS", "SS")),
  sum(s$group == "SS") == 16L, sum(s$group == "DS") == 17L
)
s$D_recomputed <- as.integer(s$group == "DS")
s$group_mean_dmn_conn_primary <- ave(s$dmn_conn_primary, s$group, FUN = mean)
s$dmn_conn_primary_group_centered <- s$dmn_conn_primary - s$group_mean_dmn_conn_primary
x_sd <- sd(s$dmn_conn_primary_group_centered)
s$X_recomputed <- s$dmn_conn_primary_group_centered / x_sd
stopifnot(isTRUE(all.equal(mean(s$dmn_conn_primary_group_centered), 0, tolerance = 1e-14)), isTRUE(all.equal(sd(s$X_recomputed), 1, tolerance = 1e-14)))
write.csv(s[, c("subject", "group", "D_recomputed", "dmn_conn_primary", "group_mean_dmn_conn_primary", "dmn_conn_primary_group_centered", "X_recomputed")], file.path(OUT, "neural_X_reconstruction.csv"), row.names = FALSE)

expected_biased <- (t0$group == "SS" & t0$Dimension == "Same") | (t0$group == "DS" & t0$Dimension == "Different")
expected_unbiased <- (t0$group == "SS" & t0$Dimension == "Different") | (t0$group == "DS" & t0$Dimension == "Same")
condition_chr <- ifelse(t0$Dimension == "Absent", "Absent",
  ifelse(expected_biased & t0$DisRegion == "Frequent", "BF",
  ifelse(expected_biased & t0$DisRegion == "Rare", "BR",
  ifelse(expected_unbiased & t0$DisRegion == "Frequent", "UF",
  ifelse(expected_unbiased & t0$DisRegion == "Rare", "UR", NA_character_)))))
stopifnot(!anyNA(condition_chr), all(condition_chr %in% c("Absent", "BF", "BR", "UF", "UR")))

# Exact analytic frame preserves all eligible trials and original keys, with no first-trial exclusion.
d <- t0
d$condition <- factor(condition_chr, levels = c("Absent", "BF", "BR", "UF", "UR"))
d$BF <- as.integer(d$condition == "BF")
d$BR <- as.integer(d$condition == "BR")
d$UF <- as.integer(d$condition == "UF")
d$UR <- as.integer(d$condition == "UR")
d$D <- as.integer(d$group == "DS")
d$run2 <- as.integer(d$run == 2L)
idx <- match(d$subject, s$subject)
stopifnot(!anyNA(idx), identical(as.character(d$group), as.character(s$group[idx])), identical(d$D, s$D_recomputed[idx]))
d$dmn_conn_primary <- s$dmn_conn_primary[idx]
d$group_mean_dmn_conn_primary <- s$group_mean_dmn_conn_primary[idx]
d$dmn_conn_primary_group_centered <- s$dmn_conn_primary_group_centered[idx]
d$X <- s$X_recomputed[idx]
d$RT100 <- d$RT_ms / 100
d$subject <- factor(d$subject)
stopifnot(
  nrow(d) == 10995L, length(unique(d$subject)) == 33L,
  !anyDuplicated(d[, c("subject", "run", "trial")]),
  all(is.finite(as.matrix(d[, c("RT100", "X", "D", "run2", "BF", "BR", "UF", "UR")]))),
  all(d$RT100 == d$RT_ms / 100), all(d$run2 %in% 0:1),
  all(rowSums(d[, c("BF", "BR", "UF", "UR")]) <= 1L)
)
expected_counts <- data.frame(
  group = rep(c("SS", "DS"), each = 5),
  condition = rep(c("Absent", "BF", "BR", "UF", "UR"), 2),
  expected_n = c(953L, 2632L, 559L, 596L, 596L, 1013L, 2897L, 633L, 566L, 550L),
  stringsAsFactors = FALSE
)
observed_counts <- as.data.frame(table(group = d$group, condition = d$condition), stringsAsFactors = FALSE)
names(observed_counts)[3] <- "observed_n"
count_audit <- merge(expected_counts, observed_counts, by = c("group", "condition"), sort = FALSE)
count_audit$pass <- count_audit$expected_n == count_audit$observed_n
observed_cell_counts <- aggregate(list(n = rep(1L, nrow(d))), d[, c("subject", "group", "run", "condition")], sum)
subject_groups <- unique(data.frame(subject = as.character(d$subject), group = as.character(d$group), stringsAsFactors = FALSE))
cell_counts <- merge(subject_groups, expand.grid(run = sort(unique(d$run)), condition = levels(d$condition), stringsAsFactors = FALSE), by = NULL)
cell_counts <- merge(cell_counts, observed_cell_counts, by = c("subject", "group", "run", "condition"), all.x = TRUE, sort = FALSE)
cell_counts$n[is.na(cell_counts$n)] <- 0L
cell_counts$all_required_cell_nonempty <- cell_counts$n > 0L
stopifnot(all(count_audit$pass), all(cell_counts$all_required_cell_nonempty))
write.csv(count_audit, file.path(OUT, "condition_count_inventory.csv"), row.names = FALSE)
write.csv(cell_counts, file.path(OUT, "participant_run_condition_inventory.csv"), row.names = FALSE)
write.csv(d, file.path(OUT, "exact_model_frame.csv"), row.names = FALSE)
saveRDS(d, file.path(OUT, "exact_model_frame.rds"))
frame_hash <- sha(file.path(OUT, "exact_model_frame.csv"))

source_audit <- data.frame(
  check = c("locked_trials_hash", "locked_neural_hash", "all_eligible_trials_retained", "N_subjects", "SS_n", "DS_n", "correct_RT_200_3000_inclusive", "no_additional_first_trial_exclusion", "five_condition_inventory", "all_participant_run_condition_cells_nonempty", "unique_original_subject_run_trial_keys", "primary_DMN_only", "X_global_participant_SD"),
  observed = c(hashes["trials"], hashes["neural"], nrow(d), length(unique(d$subject)), sum(s$group == "SS"), sum(s$group == "DS"), sum(t0$correct == 1L & t0$RT_ms >= 200 & t0$RT_ms <= 3000), sum(as.character(d$first_run) == "True"), sum(count_audit$pass), sum(cell_counts$all_required_cell_nonempty), !anyDuplicated(d[, c("subject", "run", "trial")]), all(is.finite(d$dmn_conn_primary)), format(x_sd, digits = 16)),
  expected = c(expected_hashes["trials"], expected_hashes["neural"], "10995", "33", "16", "17", "10995", as.character(sum(as.character(t0$first_run) == "True")), "10", "330", "TRUE", "TRUE", format(x_sd, digits = 16)),
  stringsAsFactors = FALSE
)
source_audit$pass <- source_audit$observed == source_audit$expected
write.csv(source_audit, file.path(OUT, "source_and_coding_audit.csv"), row.names = FALSE)
stopifnot(all(source_audit$pass))

fixed_form <- "RT100 ~ condition*X*D + condition:run2"
full_form <- as.formula(paste(fixed_form, "+ (1 + BF + BR + UF + UR | subject)"))
independent_form <- as.formula(paste(fixed_form, "+ (1 + BF + BR + UF + UR || subject)"))
ctrl_bobyqa <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
ctrl_nlopt <- lmerControl(optimizer = "nloptwrap", optCtrl = list(maxeval = 200000))

fit_capture <- function(formula, control, start = NULL) {
  warns <- character(); msgs <- character()
  fit <- withCallingHandlers(
    if (is.null(start)) lmer(formula, data = d, REML = TRUE, control = control) else lmer(formula, data = d, REML = TRUE, control = control, start = start),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") }
  )
  list(fit = fit, warnings = unique(warns), messages = unique(msgs))
}
run_optimizer_triplet <- function(formula, prefix) {
  b <- fit_capture(formula, ctrl_bobyqa)
  n <- fit_capture(formula, ctrl_nlopt)
  r <- fit_capture(formula, ctrl_bobyqa, start = list(theta = getME(n$fit, "theta")))
  out <- list()
  out[[paste0(prefix, "_bobyqa")]] <- b
  out[[paste0(prefix, "_nloptwrap")]] <- n
  out[[paste0(prefix, "_bobyqa_restart_from_nloptwrap")]] <- r
  out
}

# The three mandatory same-formula correlated fits are always preserved.
full_fits <- run_optimizer_triplet(full_form, "full_correlated")
for (nm in names(full_fits)) saveRDS(full_fits[[nm]]$fit, file.path(OUT, paste0("fit_", nm, ".rds")))

fit_messages <- function(obj) unique(c(obj$warnings, obj$messages, obj$fit@optinfo$conv$lme4$messages))
clean_nonsingular <- function(obj) !isSingular(obj$fit, tol = 1e-4) && length(fit_messages(obj)) == 0L
# Mandatory independent repeat only if a reproducible clean full solution exists; equivalence is assessed below.
if (any(vapply(full_fits, clean_nonsingular, logical(1)))) {
  full_repeat <- run_optimizer_triplet(full_form, "full_correlated_repeat")
  # Preserve only the independent alternative and its restart as predeclared; repeated bobyqa documents the attempt too.
  full_fits <- c(full_fits, full_repeat)
  for (nm in names(full_repeat)) saveRDS(full_repeat[[nm]]$fit, file.path(OUT, paste0("fit_", nm, ".rds")))
}

safe_eigen <- function(x) tryCatch(eigen(x, symmetric = TRUE, only.values = TRUE)$values, error = function(e) numeric())
diag_one <- function(name, obj, structure_label) {
  f <- obj$fit; oi <- f@optinfo; der <- oi$derivs; derivative_source <- "optimizer"
  if (is.null(der) || is.null(der$gradient) || is.null(der$Hessian)) {
    derivative_source <- "lme4_deriv12_numerical_readback"
    der <- tryCatch({
      devfun <- lmer(formula(f), data = model.frame(f), REML = TRUE, devFunOnly = TRUE)
      lme4:::deriv12(devfun, getME(f, "theta"))
    }, error = function(e) NULL)
  }
  g <- if (is.null(der) || is.null(der$gradient)) numeric() else as.numeric(der$gradient); h <- if (is.null(der)) NULL else der$Hessian
  eig <- if (is.null(h)) numeric() else safe_eigen(h)
  scaled <- if (!length(g) || is.null(h)) rep(NA_real_, length(g)) else tryCatch(as.numeric(backsolve(chol(h), g)), error = function(e) rep(NA_real_, length(g)))
  rp <- rePCA(f)
  pca <- do.call(rbind, lapply(names(rp), function(grp) data.frame(fit = name, group = grp, component = seq_along(rp[[grp]]$sdev), sd = rp[[grp]]$sdev, eigenvalue = rp[[grp]]$sdev^2, stringsAsFactors = FALSE)))
  vc <- as.data.frame(VarCorr(f)); vc$fit <- name; vc$structure <- structure_label; vc$sdcor_ms <- vc$sdcor * 100
  theta <- getME(f, "theta")
  cn <- getME(f, "cnms")
  list(
    summary = data.frame(fit = name, structure = structure_label, formula = paste(deparse(formula(f)), collapse = " "), optimizer = oi$optimizer, derivative_source = derivative_source, optimizer_code = if (is.null(oi$conv$opt)) NA_integer_ else oi$conv$opt, optimizer_messages = paste(fit_messages(obj), collapse = " | "), logLik = as.numeric(logLik(f)), AIC = AIC(f), BIC = BIC(f), sigma_RT100 = sigma(f), sigma_ms = sigma(f) * 100, singular_tol1e4 = isSingular(f, tol = 1e-4), singular_tol1e5 = isSingular(f, tol = 1e-5), hessian_min_eigen = if (length(eig)) min(eig) else NA_real_, hessian_max_eigen = if (length(eig)) max(eig) else NA_real_, hessian_nonpositive_eigen = if (length(eig)) sum(eig <= 0) else NA_integer_, stringsAsFactors = FALSE),
    gradient = if (length(g)) data.frame(fit = name, index = seq_along(g), raw_gradient = g, scaled_gradient = scaled, abs_raw_gradient = abs(g), abs_scaled_gradient = abs(scaled), stringsAsFactors = FALSE) else data.frame(fit = character(), index = integer(), raw_gradient = numeric(), scaled_gradient = numeric(), abs_raw_gradient = numeric(), abs_scaled_gradient = numeric()),
    hessian = if (length(eig)) data.frame(fit = name, index = seq_along(eig), eigenvalue = eig, stringsAsFactors = FALSE) else data.frame(fit = character(), index = integer(), eigenvalue = numeric()),
    theta = data.frame(fit = name, structure = structure_label, parameter = names(theta), theta = as.numeric(theta), stringsAsFactors = FALSE),
    varcorr = vc,
    pca = pca,
    messages = data.frame(fit = rep(name, length(c(obj$warnings, obj$messages))), kind = c(rep("warning", length(obj$warnings)), rep("message", length(obj$messages))), message = c(obj$warnings, obj$messages), stringsAsFactors = FALSE),
    structure = data.frame(fit = name, structure = structure_label, cnms_json = toJSON(cn, auto_unbox = TRUE), theta_n = length(theta), theta_names = paste(names(theta), collapse = " | "), independent_numeric_dummy_slopes = identical(structure_label, "independent") && length(theta) == length(unlist(cn)), stringsAsFactors = FALSE)
  )
}
export_diagnostics <- function(fit_list, structure_label, append = FALSE) {
  di <- Map(function(n, o) diag_one(n, o, structure_label), names(fit_list), fit_list)
  items <- c("summary", "gradient", "hessian", "theta", "varcorr", "pca", "messages", "structure")
  for (item in items) {
    x <- do.call(rbind, lapply(di, `[[`, item))
    p <- file.path(OUT, paste0("optimizer_", if (item == "summary") "diagnostics_summary" else item, ".csv"))
    if (append && file.exists(p)) x <- rbind(read.csv(p, check.names = FALSE), x)
    write.csv(x, p, row.names = FALSE)
  }
  invisible(di)
}
full_diags <- export_diagnostics(full_fits, "full_correlated")

# Same-formula equality against canonical bobyqa using the predeclared tolerances.
compare_fit_set <- function(fit_list, canonical_name) {
  can <- fit_list[[canonical_name]]$fit; beta0 <- fixef(can); theta0 <- getME(can, "theta"); ll0 <- as.numeric(logLik(can))
  rows <- lapply(names(fit_list), function(nm) {
    f <- fit_list[[nm]]$fit; th <- getME(f, "theta")
    data.frame(fit = nm, abs_logLik_difference = abs(as.numeric(logLik(f)) - ll0), max_abs_fixed_difference_RT100 = max(abs(fixef(f) - beta0)), max_abs_theta_difference = if (identical(names(th), names(theta0))) max(abs(th - theta0)) else NA_real_, logLik_within_1e_5 = abs(as.numeric(logLik(f)) - ll0) <= 1e-5, fixed_within_1e_4_RT100 = max(abs(fixef(f) - beta0)) <= 1e-4, theta_within_1e_3 = identical(names(th), names(theta0)) && max(abs(th - theta0)) <= 1e-3, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}
full_equality <- compare_fit_set(full_fits, "full_correlated_bobyqa")
write.csv(full_equality, file.path(OUT, "full_correlated_optimizer_equality.csv"), row.names = FALSE)
all_full_boundary <- all(vapply(full_fits, function(z) isSingular(z$fit, tol = 1e-4), logical(1)))
full_boundary_profile <- do.call(rbind, lapply(names(full_fits), function(nm) {
  p <- rePCA(full_fits[[nm]]$fit)$subject$sdev
  data.frame(fit = nm, n_nearzero_pca_tol1e4 = sum(p <= 1e-4), min_rePCA_sd = min(p), stringsAsFactors = FALSE)
}))
write.csv(full_boundary_profile, file.path(OUT, "full_correlated_boundary_profile.csv"), row.names = FALSE)

# Predeclared covariance sensitivity follows only an all-boundary correlated result.
independent_fits <- list(); reduced_fits <- list(); removed_terms <- character(); selected_name <- "full_correlated_bobyqa"; selected_structure <- "full_correlated"; selection_reason <- "Canonical full correlated bobyqa fit retained; no covariance sensitivity triggered."
if (all_full_boundary) {
  independent_fits <- run_optimizer_triplet(independent_form, "independent")
  for (nm in names(independent_fits)) saveRDS(independent_fits[[nm]]$fit, file.path(OUT, paste0("fit_", nm, ".rds")))
  export_diagnostics(independent_fits, "independent", append = TRUE)
  indep_equality <- compare_fit_set(independent_fits, "independent_bobyqa")
  write.csv(indep_equality, file.path(OUT, "independent_optimizer_equality.csv"), row.names = FALSE)
  # In || syntax the theta entries are response-scale independent scalar SD parameters.
  independent_theta <- getME(independent_fits[["independent_bobyqa"]]$fit, "theta")
  independent_cnms <- getME(independent_fits[["independent_bobyqa"]]$fit, "cnms")
  scalar_terms <- setdiff(unlist(independent_cnms), "(Intercept)")
  scalar_theta <- independent_theta[grep("^subject\\.", names(independent_theta))]
  # Names retain the dummy in this lme4 version; align by order after checking scalar count.
  stopifnot(length(scalar_theta) == length(scalar_terms), length(independent_theta) == length(unlist(independent_cnms)))
  scalar_audit <- data.frame(term = scalar_terms, theta_SD_over_residual = as.numeric(scalar_theta), fitted_SD_RT100 = as.numeric(scalar_theta) * sigma(independent_fits[["independent_bobyqa"]]$fit), fitted_SD_ms = as.numeric(scalar_theta) * sigma(independent_fits[["independent_bobyqa"]]$fit) * 100, boundary_le_1e_4 = as.numeric(scalar_theta) <= 1e-4, stringsAsFactors = FALSE)
  write.csv(scalar_audit, file.path(OUT, "independent_scalar_variance_audit.csv"), row.names = FALSE)
  removed_terms <- scalar_audit$term[scalar_audit$boundary_le_1e_4]
  if (length(removed_terms)) {
    remain <- c("1", setdiff(scalar_terms, removed_terms))
    rhs <- paste(remain, collapse = " + ")
    reduced_form <- as.formula(paste(fixed_form, "+ (", rhs, " || subject)"))
    reduced_fits <- run_optimizer_triplet(reduced_form, "reduced_independent")
    for (nm in names(reduced_fits)) saveRDS(reduced_fits[[nm]]$fit, file.path(OUT, paste0("fit_", nm, ".rds")))
    export_diagnostics(reduced_fits, "reduced_independent", append = TRUE)
    red_equality <- compare_fit_set(reduced_fits, "reduced_independent_bobyqa")
    write.csv(red_equality, file.path(OUT, "reduced_independent_optimizer_equality.csv"), row.names = FALSE)
    selected_name <- "reduced_independent_bobyqa"; selected_structure <- "reduced_independent"
    selection_reason <- paste0("All full correlated fits were boundary fits. Predeclared independent covariance sensitivity had scalar boundary terms removed deterministically: ", paste(removed_terms, collapse = ", "), ". Fixed formula and exact frame unchanged; selection was not p-value based.")
  } else {
    selected_name <- "independent_bobyqa"; selected_structure <- "independent"
    selection_reason <- "All full correlated fits were boundary fits. Predeclared independent covariance sensitivity retained all non-boundary scalar random terms; fixed formula and exact frame unchanged; selection was not p-value based."
  }
}
all_fits <- c(full_fits, independent_fits, reduced_fits)
selected <- all_fits[[selected_name]]$fit
saveRDS(selected, file.path(OUT, "final_selected_model.rds"))

# Full fixed/random output for every saved fitted model and selected beta/vcov export.
export_model_tables <- function(name, obj, structure) {
  f <- obj$fit; co <- summary(f)$coefficients
  fixed <- data.frame(fit = name, structure = structure, term = rownames(co), estimate_RT100 = co[, "Estimate"], SE_RT100 = co[, "Std. Error"], df = co[, "df"], t = co[, "t value"], p_raw = co[, "Pr(>|t|)"], stringsAsFactors = FALSE)
  fixed$estimate_ms <- fixed$estimate_RT100 * 100; fixed$SE_ms <- fixed$SE_RT100 * 100
  fixed$ci_low_ms <- fixed$estimate_ms - qt(.975, fixed$df) * fixed$SE_ms; fixed$ci_high_ms <- fixed$estimate_ms + qt(.975, fixed$df) * fixed$SE_ms
  vc <- as.data.frame(VarCorr(f)); vc$fit <- name; vc$structure <- structure; vc$sdcor_ms <- vc$sdcor * 100
  list(fixed = fixed, random = vc)
}
structure_for <- function(nm) if (grepl("^full_correlated", nm)) "full_correlated" else if (grepl("^reduced_independent", nm)) "reduced_independent" else "independent"
mt <- lapply(names(all_fits), function(nm) export_model_tables(nm, all_fits[[nm]], structure_for(nm))); names(mt) <- names(all_fits)
write.csv(do.call(rbind, lapply(mt, `[[`, "fixed")), file.path(OUT, "all_models_fixed_effects.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(mt, `[[`, "random")), file.path(OUT, "all_models_random_covariance.csv"), row.names = FALSE)
sel_fixed <- mt[[selected_name]]$fixed
sel_random <- mt[[selected_name]]$random
write.csv(sel_fixed, file.path(OUT, "final_selected_fixed_effects.csv"), row.names = FALSE)
write.csv(sel_random, file.path(OUT, "final_selected_random_covariance.csv"), row.names = FALSE)
re <- ranef(selected, condVar = TRUE)$subject
re_out <- data.frame(subject = rownames(re), re, check.names = FALSE)
names(re_out)[-1] <- paste0("conditional_mode_", names(re_out)[-1], "_RT100")
for (nm in names(re_out)[-1]) re_out[[sub("_RT100$", "_ms", nm)]] <- re_out[[nm]] * 100
write.csv(re_out, file.path(OUT, "final_selected_subject_random_effects.csv"), row.names = FALSE)
beta <- fixef(selected); V <- vcov(selected)
write.csv(data.frame(term = names(beta), beta_RT100 = as.numeric(beta), beta_ms = as.numeric(beta) * 100), file.path(OUT, "final_selected_beta.csv"), row.names = FALSE)
Vout <- data.frame(term = rownames(V), as.matrix(V), check.names = FALSE)
write.csv(Vout, file.path(OUT, "final_selected_vcov_RT100.csv"), row.names = FALSE)

canon <- function(x) paste(sort(strsplit(x, ":", fixed = TRUE)[[1]]), collapse = ":")
make_vector <- function(weights, fit) {
  v <- setNames(rep(0, length(fixef(fit))), names(fixef(fit)))
  for (term in names(weights)) {
    ix <- which(vapply(names(v), canon, "") == canon(term))
    stopifnot(length(ix) == 1L)
    v[ix] <- weights[[term]]
  }
  v
}
weights_for <- function(base, group) {
  # base is a named mix of BF/BR/UF/UR; capture terms are condition-specific X slopes versus absent.
  w <- setNames(as.numeric(base), paste0("condition", names(base), ":X"))
  if (identical(group, "DS")) w <- c(w, setNames(as.numeric(base), paste0("condition", names(base), ":X:D")))
  if (identical(group, "Pooled")) w <- c(w, setNames(as.numeric(base) * 17/33, paste0("condition", names(base), ":X:D")))
  w
}
Bmix <- c(BF = .5, BR = .5); Umix <- c(UF = .5, UR = .5); Dmix <- c(BF = .5, BR = .5, UF = -.5, UR = -.5)
contrast_spec <- rbind(
  data.frame(panel = "Holm9_simple", contrast_id = "Pooled_B50", estimand = "Pooled B50: 0.5(BF−Absent X slope + BR−Absent X slope); SS16/33 + DS17/33", group = "Pooled", mix = "B", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "Pooled_U50", estimand = "Pooled U50: 0.5(UF−Absent X slope + UR−Absent X slope); SS16/33 + DS17/33", group = "Pooled", mix = "U", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "Pooled_B50_minus_U50", estimand = "Pooled B50−U50; SS16/33 + DS17/33", group = "Pooled", mix = "D", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "SS_B50", estimand = "SS B50: 0.5(BF−Absent X slope + BR−Absent X slope)", group = "SS", mix = "B", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "SS_U50", estimand = "SS U50: 0.5(UF−Absent X slope + UR−Absent X slope)", group = "SS", mix = "U", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "SS_B50_minus_U50", estimand = "SS B50−U50", group = "SS", mix = "D", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "DS_B50", estimand = "DS B50: 0.5(BF−Absent X slope + BR−Absent X slope)", group = "DS", mix = "B", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "DS_U50", estimand = "DS U50: 0.5(UF−Absent X slope + UR−Absent X slope)", group = "DS", mix = "U", stringsAsFactors = FALSE),
  data.frame(panel = "Holm9_simple", contrast_id = "DS_B50_minus_U50", estimand = "DS B50−U50", group = "DS", mix = "D", stringsAsFactors = FALSE),
  data.frame(panel = "Holm3_moderation", contrast_id = "DS_minus_SS_B50", estimand = "Group moderation: DS−SS in B50", group = "Moderation", mix = "B", stringsAsFactors = FALSE),
  data.frame(panel = "Holm3_moderation", contrast_id = "DS_minus_SS_U50", estimand = "Group moderation: DS−SS in U50", group = "Moderation", mix = "U", stringsAsFactors = FALSE),
  data.frame(panel = "Holm3_moderation", contrast_id = "DS_minus_SS_B50_minus_U50", estimand = "Group moderation: DS−SS in B50−U50", group = "Moderation", mix = "D", stringsAsFactors = FALSE)
)
make_weights <- function(row) {
  mix <- switch(row$mix, B = Bmix, U = Umix, D = Dmix)
  if (row$group == "Moderation") return(setNames(as.numeric(mix), paste0("condition", names(mix), ":X:D")))
  weights_for(mix, row$group)
}
eval_contrasts <- function(fit, tag) {
  out <- vector("list", nrow(contrast_spec)); vecs <- vector("list", nrow(contrast_spec))
  for (i in seq_len(nrow(contrast_spec))) {
    w <- make_weights(contrast_spec[i, ])
    v <- make_vector(w, fit)
    z <- as.data.frame(contest1D(fit, v, confint = TRUE, ddf = "Satterthwaite"))
    lo <- names(z)[grepl("lower", names(z), ignore.case = TRUE)][1]; hi <- names(z)[grepl("upper", names(z), ignore.case = TRUE)][1]
    out[[i]] <- cbind(contrast_spec[i, ], data.frame(model = tag, weight_spec = paste(names(w), format(unname(w), trim = TRUE), sep = "*", collapse = " + "), estimate_RT100 = z$Estimate, SE_RT100 = z[["Std. Error"]], df = z$df, t = z[["t value"]], ci_low_RT100 = z[[lo]], ci_high_RT100 = z[[hi]], p_raw = z[["Pr(>|t|)"]], stringsAsFactors = FALSE))
    vecs[[i]] <- data.frame(model = tag, panel = contrast_spec$panel[i], contrast_id = contrast_spec$contrast_id[i], coefficient = names(v), weight_RT100 = as.numeric(v), stringsAsFactors = FALSE)
  }
  ans <- do.call(rbind, out)
  for (prefix in c("estimate", "SE", "ci_low", "ci_high")) ans[[paste0(prefix, "_ms_per_globalFCSD")]] <- ans[[paste0(prefix, "_RT100")]] * 100
  ans$unit <- "ms per global participant-level SD of group-centered dmn_conn_primary"
  ans$positive_sign_interpretation <- "positive: higher DMN connectivity predicts a larger capture slope (present-minus-absent RT slope)"
  ans$p_Holm <- ave(ans$p_raw, ans$panel, FUN = function(x) p.adjust(x, method = "holm"))
  ans$Holm_family <- ifelse(ans$panel == "Holm9_simple", "locked exploratory equal-region simple-contrast Holm-9", "locked exploratory equal-region group-moderation Holm-3")
  list(results = ans, vectors = do.call(rbind, vecs))
}
selected_con <- eval_contrasts(selected, selected_name)
full_con <- eval_contrasts(full_fits[["full_correlated_bobyqa"]]$fit, "full_correlated_bobyqa")
write.csv(selected_con$results[selected_con$results$panel == "Holm9_simple", ], file.path(OUT, "contrasts_Holm9_simple.csv"), row.names = FALSE)
write.csv(selected_con$results[selected_con$results$panel == "Holm3_moderation", ], file.path(OUT, "contrasts_Holm3_moderation.csv"), row.names = FALSE)
write.csv(selected_con$results, file.path(OUT, "contrasts_all_12.csv"), row.names = FALSE)
write.csv(selected_con$vectors, file.path(OUT, "contrast_vectors_selected.csv"), row.names = FALSE)
write.csv(full_con$vectors, file.path(OUT, "contrast_vectors_full_correlated.csv"), row.names = FALSE)
comparison <- merge(full_con$results[, c("contrast_id", "estimate_ms_per_globalFCSD", "SE_ms_per_globalFCSD", "df", "t", "p_raw")], selected_con$results[, c("contrast_id", "estimate_ms_per_globalFCSD", "SE_ms_per_globalFCSD", "df", "t", "p_raw")], by = "contrast_id", suffixes = c("_full_correlated", "_selected"), sort = FALSE)
comparison$estimate_difference_selected_minus_full_ms <- comparison$estimate_ms_per_globalFCSD_selected - comparison$estimate_ms_per_globalFCSD_full_correlated
write.csv(comparison, file.path(OUT, "full_vs_selected_contrast_comparison.csv"), row.names = FALSE)

res <- residuals(selected) * 100
resdiag <- do.call(rbind, lapply(split(seq_along(res), interaction(d$group, d$condition, drop = TRUE)), function(ii) { x <- res[ii]; data.frame(cell = as.character(interaction(d$group, d$condition, drop = TRUE)[ii[1]]), n = length(x), mean_ms = mean(x), sd_ms = sd(x), min_ms = min(x), max_ms = max(x), stringsAsFactors = FALSE) }))
write.csv(resdiag, file.path(OUT, "final_selected_residual_diagnostics.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(OUT, "R_sessionInfo.txt"))

artifacts <- list.files(OUT, full.names = TRUE, recursive = FALSE)
artifact_hashes <- setNames(vapply(artifacts, sha, character(1)), normalizePath(artifacts))
manifest <- list(
  status = "PASS",
  scope = "Exploratory five-condition location-standardized capture LMM; all 10,995 eligible correct RT 200–3000 ms trials; no first-trial exclusion.",
  paths = list(root = normalizePath(ROOT), output = normalizePath(OUT), trials = normalizePath(TRIALS), neural = normalizePath(NEURAL), analysis_spec = normalizePath(SPEC), run_script = SCRIPT),
  source_hashes = as.list(hashes), expected_source_hashes = as.list(expected_hashes),
  exact_frame = list(path = normalizePath(file.path(OUT, "exact_model_frame.csv")), sha256 = frame_hash, rows = nrow(d), participants = length(unique(d$subject)), SS = sum(s$group == "SS"), DS = sum(s$group == "DS"), X_global_participant_SD = x_sd),
  model = list(fixed_formula = fixed_form, full_formula = paste(deparse(full_form), collapse = " "), selected_formula = paste(deparse(formula(selected)), collapse = " "), REML = TRUE, selected_model = selected_name, selected_structure = selected_structure, selection_reason = selection_reason, all_full_correlated_boundary = all_full_boundary, reduced_terms_removed = as.list(removed_terms)),
  checks = list(source_and_coding_audit_pass = all(source_audit$pass), condition_inventory_pass = all(count_audit$pass), all_participant_run_conditions_present = all(cell_counts$all_required_cell_nonempty), all_full_correlated_fits_boundary = all_full_boundary, full_correlated_optimizer_equality = normalizePath(file.path(OUT, "full_correlated_optimizer_equality.csv"))),
  artifact_sha256 = as.list(artifact_hashes)
)
write_json(manifest, file.path(OUT, "run_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("PASS: %d rows; %d participants; selected=%s; all_full_correlated_boundary=%s\n", nrow(d), length(unique(d$subject)), selected_name, all_full_boundary))
print(selected_con$results[, c("panel", "contrast_id", "estimate_ms_per_globalFCSD", "SE_ms_per_globalFCSD", "df", "t", "p_raw", "p_Holm")])
