#!/usr/bin/env Rscript
# Independent read-back verifier: does not source or import run_trial_model.R.
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(jsonlite) })
options(stringsAsFactors = FALSE)
ROOT <- Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
OUT <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/trial_model")
TRIALS <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv")
NEURAL <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv")
sha <- function(path) unname(strsplit(system2("sha256sum", path, stdout = TRUE), " +")[[1]][1])
expected_hashes <- c(trials = "fac39301cb3b828f925937e080213bc5ed39fcaa03cbe0e6c558cf2ea18b34c6", neural = "cc3cddf8377ff9aefba18e8781d2512151e335f280d232c5b23d791266e6feef")
need <- c("exact_model_frame.csv", "final_selected_model.rds", "final_selected_fixed_effects.csv", "final_selected_random_covariance.csv", "final_selected_beta.csv", "final_selected_vcov_RT100.csv", "contrasts_all_12.csv", "contrasts_Holm9_simple.csv", "contrasts_Holm3_moderation.csv", "contrast_vectors_selected.csv", "source_and_coding_audit.csv", "optimizer_diagnostics_summary.csv", "run_manifest.json")
stopifnot(all(file.exists(file.path(OUT, need))))
actual_hashes <- c(trials = sha(TRIALS), neural = sha(NEURAL))
stopifnot(identical(actual_hashes, expected_hashes))
manifest <- fromJSON(file.path(OUT, "run_manifest.json"), simplifyVector = FALSE)

# Independently reconstruct the locked all-trial frame and compare it to the saved analytic frame.
t0 <- read.csv(TRIALS, check.names = FALSE)
s <- read.csv(NEURAL, check.names = FALSE)
stopifnot(nrow(t0) == 10995L, nrow(s) == 33L, all(t0$correct == 1L), all(t0$RT_ms >= 200 & t0$RT_ms <= 3000))
biased <- (t0$group == "SS" & t0$Dimension == "Same") | (t0$group == "DS" & t0$Dimension == "Different")
unbiased <- (t0$group == "SS" & t0$Dimension == "Different") | (t0$group == "DS" & t0$Dimension == "Same")
cond <- ifelse(t0$Dimension == "Absent", "Absent", ifelse(biased & t0$DisRegion == "Frequent", "BF", ifelse(biased & t0$DisRegion == "Rare", "BR", ifelse(unbiased & t0$DisRegion == "Frequent", "UF", ifelse(unbiased & t0$DisRegion == "Rare", "UR", NA_character_)))))
stopifnot(!anyNA(cond))
s$group_centered <- s$dmn_conn_primary - ave(s$dmn_conn_primary, s$group, FUN = mean)
s$X_rebuilt <- s$group_centered / sd(s$group_centered)
i <- match(t0$subject, s$subject)
rebuilt <- data.frame(subject = t0$subject, group = t0$group, run = t0$run, trial = t0$trial, condition = cond, RT100 = t0$RT_ms / 100, D = as.integer(t0$group == "DS"), run2 = as.integer(t0$run == 2L), BF = as.integer(cond == "BF"), BR = as.integer(cond == "BR"), UF = as.integer(cond == "UF"), UR = as.integer(cond == "UR"), X = s$X_rebuilt[i], dmn_conn_primary = s$dmn_conn_primary[i], stringsAsFactors = FALSE)
saved <- read.csv(file.path(OUT, "exact_model_frame.csv"), check.names = FALSE)
keys <- c("subject", "group", "run", "trial", "condition")
numeric_vars <- c("RT100", "D", "run2", "BF", "BR", "UF", "UR", "X", "dmn_conn_primary")
frame_keys_ok <- nrow(saved) == 10995L && identical(saved[, keys], rebuilt[, keys]) && !anyDuplicated(saved[, c("subject", "run", "trial")])
frame_numeric_ok <- all(vapply(numeric_vars, function(v) isTRUE(all.equal(saved[[v]], rebuilt[[v]], tolerance = 1e-12)), logical(1)))
count_tab <- as.data.frame(table(saved$group, factor(saved$condition, levels = c("Absent", "BF", "BR", "UF", "UR"))))
count_ok <- identical(as.integer(count_tab$Freq), c(1013L, 953L, 2897L, 2632L, 633L, 559L, 566L, 596L, 550L, 596L))
stopifnot(frame_keys_ok, frame_numeric_ok, count_ok)

fit <- readRDS(file.path(OUT, "final_selected_model.rds"))
fixed_formula <- RT100 ~ condition * X * D + condition:run2
fixed_formula_ok <- identical(colnames(getME(fit, "X")), colnames(model.matrix(fixed_formula, data = saved))) && max(abs(getME(fit, "X") - model.matrix(fixed_formula, data = saved))) < 1e-12
mf <- model.frame(fit)
model_frame_ok <- nrow(mf) == nrow(saved) && isTRUE(all.equal(mf$RT100, saved$RT100, tolerance = 1e-12)) && identical(as.character(mf$condition), saved$condition) && isTRUE(all.equal(mf$X, saved$X, tolerance = 1e-12)) && isTRUE(all.equal(as.numeric(mf$D), as.numeric(saved$D), tolerance = 0)) && isTRUE(all.equal(as.numeric(mf$run2), as.numeric(saved$run2), tolerance = 0)) && identical(as.character(mf$subject), saved$subject)
stopifnot(isREML(fit), fixed_formula_ok, model_frame_ok)

# Fixed coefficients and beta/vcov are checked directly against the serialized selected model.
sm <- summary(fit)$coefficients
fixed <- read.csv(file.path(OUT, "final_selected_fixed_effects.csv"), check.names = FALSE)
fixed_ok <- identical(fixed$term, rownames(sm)) && max(abs(fixed$estimate_RT100 - sm[, "Estimate"])) < 1e-10 && max(abs(fixed$SE_RT100 - sm[, "Std. Error"])) < 1e-10 && max(abs(fixed$df - sm[, "df"])) < 1e-10
beta <- read.csv(file.path(OUT, "final_selected_beta.csv"), check.names = FALSE)
V <- as.matrix(vcov(fit)); Vsave <- as.matrix(read.csv(file.path(OUT, "final_selected_vcov_RT100.csv"), check.names = FALSE)[, -1])
beta_vcov_ok <- identical(beta$term, names(fixef(fit))) && max(abs(beta$beta_RT100 - fixef(fit))) < 1e-12 && all(dim(Vsave) == dim(V)) && max(abs(Vsave - V)) < 1e-12
stopifnot(fixed_ok, beta_vcov_ok)

# Independently rebuild all twelve locked equal-region vectors and contrast tests.
canon <- function(x) paste(sort(strsplit(x, ":", fixed = TRUE)[[1]]), collapse = ":")
make_vector <- function(weights) {
  v <- setNames(rep(0, length(fixef(fit))), names(fixef(fit)))
  for (term in names(weights)) { j <- which(vapply(names(v), canon, "") == canon(term)); stopifnot(length(j) == 1L); v[j] <- weights[[term]] }
  v
}
mk <- function(mix, group) {
  w <- setNames(as.numeric(mix), paste0("condition", names(mix), ":X"))
  if (group == "DS") w <- c(w, setNames(as.numeric(mix), paste0("condition", names(mix), ":X:D")))
  if (group == "Pooled") w <- c(w, setNames(as.numeric(mix) * 17/33, paste0("condition", names(mix), ":X:D")))
  if (group == "Moderation") w <- setNames(as.numeric(mix), paste0("condition", names(mix), ":X:D"))
  w
}
B <- c(BF = .5, BR = .5); U <- c(UF = .5, UR = .5); DIFF <- c(BF = .5, BR = .5, UF = -.5, UR = -.5)
spec <- data.frame(panel = c(rep("Holm9_simple", 9), rep("Holm3_moderation", 3)), contrast_id = c("Pooled_B50", "Pooled_U50", "Pooled_B50_minus_U50", "SS_B50", "SS_U50", "SS_B50_minus_U50", "DS_B50", "DS_U50", "DS_B50_minus_U50", "DS_minus_SS_B50", "DS_minus_SS_U50", "DS_minus_SS_B50_minus_U50"), group = c("Pooled", "Pooled", "Pooled", "SS", "SS", "SS", "DS", "DS", "DS", "Moderation", "Moderation", "Moderation"), mix = c("B", "U", "D", "B", "U", "D", "B", "U", "D", "B", "U", "D"), stringsAsFactors = FALSE)
rows <- vector("list", nrow(spec)); vectors <- vector("list", nrow(spec))
for (k in seq_len(nrow(spec))) {
  mix <- switch(spec$mix[k], B = B, U = U, D = DIFF); v <- make_vector(mk(mix, spec$group[k])); z <- as.data.frame(contest1D(fit, v, confint = TRUE, ddf = "Satterthwaite")); lo <- names(z)[grepl("lower", names(z), ignore.case = TRUE)][1]; hi <- names(z)[grepl("upper", names(z), ignore.case = TRUE)][1]
  rows[[k]] <- data.frame(contrast_id = spec$contrast_id[k], panel = spec$panel[k], estimate_RT100 = z$Estimate, SE_RT100 = z[["Std. Error"]], df = z$df, t = z[["t value"]], ci_low_RT100 = z[[lo]], ci_high_RT100 = z[[hi]], p_raw = z[["Pr(>|t|)"]])
  vectors[[k]] <- data.frame(contrast_id = spec$contrast_id[k], coefficient = names(v), weight_RT100 = as.numeric(v))
}
recomputed <- do.call(rbind, rows); recomputed$p_Holm <- ave(recomputed$p_raw, recomputed$panel, FUN = function(x) p.adjust(x, "holm"))
saved_con <- read.csv(file.path(OUT, "contrasts_all_12.csv"), check.names = FALSE)
saved_vec <- read.csv(file.path(OUT, "contrast_vectors_selected.csv"), check.names = FALSE)
contrast_columns <- c("estimate_RT100", "SE_RT100", "df", "t", "ci_low_RT100", "ci_high_RT100", "p_raw", "p_Holm")
contrast_ok <- nrow(saved_con) == 12L && identical(saved_con$contrast_id, recomputed$contrast_id) && all(vapply(contrast_columns, function(v) max(abs(saved_con[[v]] - recomputed[[v]])) < 1e-10, logical(1))) && nrow(read.csv(file.path(OUT, "contrasts_Holm9_simple.csv"))) == 9L && nrow(read.csv(file.path(OUT, "contrasts_Holm3_moderation.csv"))) == 3L
recomputed_vec <- do.call(rbind, vectors)
vector_ok <- nrow(saved_vec) == nrow(recomputed_vec) && identical(saved_vec$contrast_id, recomputed_vec$contrast_id) && identical(saved_vec$coefficient, recomputed_vec$coefficient) && max(abs(saved_vec$weight_RT100 - recomputed_vec$weight_RT100)) < 1e-15
stopifnot(contrast_ok, vector_ok)

optimizer <- read.csv(file.path(OUT, "optimizer_diagnostics_summary.csv"), check.names = FALSE)
audit <- read.csv(file.path(OUT, "source_and_coding_audit.csv"), check.names = FALSE)
optimizer_ok <- nrow(optimizer) >= 3L && all(optimizer$REML %in% TRUE) && any(optimizer$fit == "full_correlated_bobyqa") && all(c("singular_tol1e4", "singular_tol1e5", "hessian_min_eigen", "derivative_source") %in% names(optimizer))
manifest_frame_hash_ok <- identical(sha(file.path(OUT, "exact_model_frame.csv")), manifest$exact_frame$sha256)
stopifnot(all(audit$pass), optimizer_ok, manifest_frame_hash_ok)

result <- list(status = "PASS", independent_readback = TRUE, paths = list(output = normalizePath(OUT), selected_model = normalizePath(file.path(OUT, "final_selected_model.rds")), contrasts = normalizePath(file.path(OUT, "contrasts_all_12.csv"))), source_hashes = as.list(actual_hashes), checks = list(exact_rebuilt_frame_matches_saved = frame_keys_ok && frame_numeric_ok, fixed_formula_matches = fixed_formula_ok, REML = isREML(fit), fit_model_frame_matches_saved = model_frame_ok, fixed_effects_reproduced = fixed_ok, beta_vcov_reproduced = beta_vcov_ok, all_12_contrasts_and_Holm_panels_reproduced = contrast_ok, explicit_vectors_reproduced = vector_ok, source_coding_audit_pass = all(audit$pass), optimizer_diagnostics_present = optimizer_ok, frame_hash_matches_manifest = manifest_frame_hash_ok), recomputed = list(rows = nrow(saved), participants = length(unique(saved$subject)), selected_formula = paste(deparse(formula(fit)), collapse = " "), contrast_estimates_ms = as.list(saved_con$estimate_ms_per_globalFCSD), contrast_p_Holm = as.list(saved_con$p_Holm)))
write_json(result, file.path(OUT, "verification_independent.json"), pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("PASS independent verifier: %d trials, %d participants, 12 contrasts (9+3 panels) reproduced.\n", nrow(saved), length(unique(saved$subject))))
