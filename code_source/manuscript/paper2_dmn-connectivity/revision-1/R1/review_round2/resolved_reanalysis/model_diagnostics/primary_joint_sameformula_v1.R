#!/usr/bin/env Rscript
# Diagnostic-only rerun of the locked primary_joint LMM. Does not alter canonical outputs.
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE)
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
default_trials <- file.path(root, "inference", "results_common399_n33", "joint_history_trials_new_primary.csv")
default_out <- file.path(dirname(script_path), "primary_joint_sameformula_v1")
if (!(length(args) %in% c(0L, 2L))) stop("usage: Rscript primary_joint_sameformula_v1.R [TRIALS_CSV OUTPUT_DIR]")
trials_path <- if (length(args) == 2L) args[[1]] else default_trials
out <- if (length(args) == 2L) args[[2]] else default_out
canonical_rds <- file.path(root, "inference", "results_common399_n33", "mixed_models", "primary_joint.rds")

if (!file.exists(trials_path)) stop("Missing trial file: ", trials_path)
if (!file.exists(canonical_rds)) stop("Missing locked canonical RDS: ", canonical_rds)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (length(list.files(out, all.files = TRUE, no.. = TRUE))) stop("Refusing to overwrite nonempty output directory: ", out)

formula_primary <- RT100 ~ B * X * D + U * X * D + (1 + B + U | subject)
formula_text <- paste(deparse(formula_primary), collapse = " ")
d <- read.csv(trials_path, check.names = FALSE)
required <- c("subject", "group", "run", "RT100", "B", "U", "D", "X")
if (!all(required %in% names(d))) stop("Missing required trial columns: ", paste(setdiff(required, names(d)), collapse = ", "))
if (nrow(d) != 10995L || length(unique(d$subject)) != 33L) stop("Expected exact primary trial universe: 10,995 rows and 33 subjects.")
if (!all(sort(as.integer(tapply(d$subject, d$group, function(x) length(unique(x))))) == c(16L, 17L))) stop("Expected SS=16 and DS=17.")
if (any(!is.finite(as.matrix(d[, c("RT100", "B", "U", "D", "X")]))) ) stop("Nonfinite primary-model values.")
d$subject <- factor(d$subject)

saved <- readRDS(canonical_rds)
if (!inherits(saved, "merMod")) stop("Canonical object is not an lme4 fitted model.")
if (nobs(saved) != nrow(d)) stop("Saved model and diagnostic input have different row counts.")
X_saved <- getME(saved, "X")
if (ncol(X_saved) != 12L || qr(X_saved)$rank != 12L) stop("Locked canonical fixed-effect design is not full rank 12.")
# The locked fit carries the data used for fitting. Check the five numeric model columns
# and subject identities exactly before refitting from the separately saved CSV.
mf_saved <- model.frame(saved)
for (nm in c("RT100", "B", "U", "D", "X")) {
  if (!isTRUE(all.equal(as.numeric(mf_saved[[nm]]), as.numeric(d[[nm]]), tolerance = 0))) stop("Saved fit differs from diagnostic CSV for ", nm)
}
if (!identical(as.character(mf_saved$subject), as.character(d$subject))) stop("Saved fit differs from diagnostic CSV for subject order/identity.")

capture_fit <- function(expr) {
  warnings <- character(); messages <- character()
  fit <- withCallingHandlers(
    expr,
    warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { messages <<- c(messages, conditionMessage(m)); invokeRestart("muffleMessage") }
  )
  attr(fit, "captured_warnings") <- unique(warnings)
  attr(fit, "captured_messages") <- unique(messages)
  fit
}
msg_text <- function(fit) {
  unique(c(attr(fit, "captured_warnings"), attr(fit, "captured_messages"), fit@optinfo$conv$lme4$messages))
}
num_or_na <- function(x) if (length(x)) as.numeric(x) else NA_real_
fit_diagnostics <- function(fit, solution) {
  theta <- getME(fit, "theta")
  grad <- fit@optinfo$derivs$gradient
  H <- fit@optinfo$derivs$Hessian
  he <- if (is.matrix(H) && nrow(H)) eigen((H + t(H))/2, symmetric = TRUE, only.values = TRUE)$values else NA_real_
  scaled <- NA_real_
  if (is.matrix(H) && nrow(H) && all(is.finite(H))) {
    ch <- try(chol((H + t(H))/2), silent = TRUE)
    if (!inherits(ch, "try-error")) scaled <- max(abs(backsolve(ch, grad, transpose = TRUE)))
  }
  pc <- rePCA(fit)$subject
  data.frame(
    solution = solution,
    optimizer = fit@optinfo$optimizer,
    optimizer_code = num_or_na(fit@optinfo$conv$opt),
    formula = formula_text,
    REML = TRUE,
    nobs = nobs(fit),
    subjects = length(unique(getME(fit, "flist")$subject)),
    fixed_columns = ncol(getME(fit, "X")),
    fixed_rank = qr(getME(fit, "X"))$rank,
    logLik = as.numeric(logLik(fit)),
    REML_criterion = REMLcrit(fit),
    isSingular_tol_1e4 = isSingular(fit, tol = 1e-4),
    isSingular_tol_1e5 = isSingular(fit, tol = 1e-5),
    theta_length = length(theta),
    max_abs_gradient = if (length(grad)) max(abs(grad)) else NA_real_,
    max_abs_scaled_gradient = scaled,
    hessian_min_eigenvalue = min(he),
    hessian_max_eigenvalue = max(he),
    hessian_negative_eigenvalues = sum(he < 0),
    rePCA_min_SD = min(pc$sdev),
    rePCA_max_SD = max(pc$sdev),
    messages = paste(msg_text(fit), collapse = " | "),
    stringsAsFactors = FALSE
  )
}
write_solution_detail <- function(fit, solution) {
  theta <- getME(fit, "theta")
  grad <- fit@optinfo$derivs$gradient
  H <- fit@optinfo$derivs$Hessian
  pc <- rePCA(fit)$subject
  vc <- as.matrix(VarCorr(fit)$subject)
  cn <- colnames(vc)
  write.csv(data.frame(solution = solution, theta_name = names(theta), theta = as.numeric(theta)), file.path(out, paste0(solution, "_theta.csv")), row.names = FALSE)
  write.csv(data.frame(solution = solution, theta_index = seq_along(grad), gradient = as.numeric(grad)), file.path(out, paste0(solution, "_gradient.csv")), row.names = FALSE)
  write.csv(data.frame(solution = solution, hessian_row = rep(seq_len(nrow(H)), each = ncol(H)), hessian_col = rep(seq_len(ncol(H)), times = nrow(H)), hessian_value = as.numeric(H)), file.path(out, paste0(solution, "_hessian.csv")), row.names = FALSE)
  he <- eigen((H + t(H))/2, symmetric = TRUE, only.values = TRUE)$values
  write.csv(data.frame(solution = solution, hessian_eigen_rank = seq_along(he), hessian_eigenvalue = he), file.path(out, paste0(solution, "_hessian_eigenvalues.csv")), row.names = FALSE)
  write.csv(data.frame(solution = solution, random_PC = seq_along(pc$sdev), SD_RT100 = pc$sdev, variance_RT100 = pc$sdev^2), file.path(out, paste0(solution, "_rePCA_sdev.csv")), row.names = FALSE)
  rot_long <- expand.grid(random_effect = cn, random_PC = seq_len(ncol(pc$rotation)), stringsAsFactors = FALSE)
  rot_long$loading <- as.numeric(pc$rotation)
  rot_long$solution <- solution
  write.csv(rot_long[, c("solution", "random_PC", "random_effect", "loading")], file.path(out, paste0(solution, "_rePCA_rotation.csv")), row.names = FALSE)
  vc_long <- expand.grid(random_effect_row = cn, random_effect_col = cn, stringsAsFactors = FALSE)
  vc_long$covariance_RT100 <- as.numeric(vc)
  vc_long$solution <- solution
  write.csv(vc_long[, c("solution", "random_effect_row", "random_effect_col", "covariance_RT100")], file.path(out, paste0(solution, "_random_covariance.csv")), row.names = FALSE)
  sdtab <- as.data.frame(VarCorr(fit))
  sdtab$solution <- solution
  write.csv(sdtab, file.path(out, paste0(solution, "_VarCorr.csv")), row.names = FALSE)
}

# Sequential by design: a fresh nloptwrap fit, bobyqa initialized at its theta,
# then an independent repeat of the same nloptwrap-plus-restart sequence.
nlopt_control <- lmerControl(optimizer = "nloptwrap", calc.derivs = TRUE, optCtrl = list(maxeval = 200000))
bobyqa_restart_control <- lmerControl(optimizer = "bobyqa", calc.derivs = TRUE, optCtrl = list(maxfun = 200000))
nlopt <- capture_fit(lmer(formula_primary, data = d, REML = TRUE, control = nlopt_control))
bobyqa_restart <- capture_fit(lmer(formula_primary, data = d, REML = TRUE, control = bobyqa_restart_control, start = getME(nlopt, "theta")))
nlopt_repeat <- capture_fit(lmer(formula_primary, data = d, REML = TRUE, control = nlopt_control))
bobyqa_restart_repeat <- capture_fit(lmer(formula_primary, data = d, REML = TRUE, control = bobyqa_restart_control, start = getME(nlopt_repeat, "theta")))

saveRDS(nlopt, file.path(out, "primary_joint_nloptwrap.rds"))
saveRDS(bobyqa_restart, file.path(out, "primary_joint_bobyqa_restart_from_nloptwrap.rds"))
saveRDS(nlopt_repeat, file.path(out, "primary_joint_nloptwrap_repeat.rds"))
saveRDS(bobyqa_restart_repeat, file.path(out, "primary_joint_bobyqa_restart_from_nloptwrap_repeat.rds"))
solutions <- list(saved_bobyqa_locked = saved, fresh_nloptwrap = nlopt, bobyqa_restart_from_nloptwrap = bobyqa_restart, fresh_nloptwrap_repeat = nlopt_repeat, bobyqa_restart_from_nloptwrap_repeat = bobyqa_restart_repeat)
diag <- do.call(rbind, Map(fit_diagnostics, solutions, names(solutions)))
write.csv(diag, file.path(out, "primary_joint_sameformula_optimizer_diagnostics.csv"), row.names = FALSE)
for (nm in names(solutions)) write_solution_detail(solutions[[nm]], nm)

fixed <- do.call(rbind, lapply(names(solutions), function(nm) data.frame(solution = nm, term = names(fixef(solutions[[nm]])), estimate_RT100 = as.numeric(fixef(solutions[[nm]])))))
wide_fixed <- reshape(fixed, idvar = "term", timevar = "solution", direction = "wide")
names(wide_fixed) <- sub("estimate_RT100\\.", "", names(wide_fixed))
solution_columns <- setdiff(names(wide_fixed), "term")
reference <- wide_fixed$saved_bobyqa_locked
for (nm in setdiff(solution_columns, "saved_bobyqa_locked")) wide_fixed[[paste0("abs_difference_from_saved_", nm)]] <- abs(wide_fixed[[nm]] - reference)
wide_fixed$max_abs_difference_across_solutions <- apply(as.matrix(wide_fixed[, solution_columns, drop = FALSE]), 1, function(x) max(x) - min(x))
write.csv(wide_fixed, file.path(out, "primary_joint_sameformula_fixed_effect_comparison.csv"), row.names = FALSE)

theta_table <- do.call(rbind, lapply(names(solutions), function(nm) data.frame(solution = nm, theta_name = names(getME(solutions[[nm]], "theta")), theta = as.numeric(getME(solutions[[nm]], "theta")))))
wide_theta <- reshape(theta_table, idvar = "theta_name", timevar = "solution", direction = "wide")
names(wide_theta) <- sub("theta\\.", "", names(wide_theta))
theta_solution_columns <- setdiff(names(wide_theta), "theta_name")
for (nm in setdiff(theta_solution_columns, "saved_bobyqa_locked")) wide_theta[[paste0("abs_difference_from_saved_", nm)]] <- abs(wide_theta[[nm]] - wide_theta$saved_bobyqa_locked)
write.csv(wide_theta, file.path(out, "primary_joint_sameformula_theta_comparison.csv"), row.names = FALSE)

# A non-singular, no-message solution requires an independent repeat before it can be reported as a reproducible diagnostic solution.
non_singular_good <- diag$optimizer_code == 0 & !diag$isSingular_tol_1e4 & !grepl("failed to converge|unable to evaluate|degenerate|negative eigenvalue", diag$messages, ignore.case = TRUE)
names(non_singular_good) <- diag$solution
agreement_row <- function(a, b) {
  fa <- solutions[[a]]; fb <- solutions[[b]]
  ll_diff <- abs(as.numeric(logLik(fa)) - as.numeric(logLik(fb)))
  fixed_diff <- max(abs(fixef(fa) - fixef(fb)))
  theta_diff <- max(abs(getME(fa, "theta") - getME(fb, "theta")))
  data.frame(
    first_solution = a, second_solution = b,
    abs_logLik_difference = ll_diff,
    max_abs_fixed_effect_difference_RT100 = fixed_diff,
    max_abs_theta_difference = theta_diff,
    first_non_singular_good = non_singular_good[[a]],
    second_non_singular_good = non_singular_good[[b]],
    logLik_tolerance = 1e-6,
    fixed_effect_tolerance_RT100 = 1e-6,
    theta_tolerance = 1e-5,
    agreement_pass = non_singular_good[[a]] && non_singular_good[[b]] && ll_diff <= 1e-6 && fixed_diff <= 1e-6 && theta_diff <= 1e-5,
    stringsAsFactors = FALSE
  )
}
repeat_agreement <- rbind(
  agreement_row("fresh_nloptwrap", "fresh_nloptwrap_repeat"),
  agreement_row("bobyqa_restart_from_nloptwrap", "bobyqa_restart_from_nloptwrap_repeat")
)
write.csv(repeat_agreement, file.path(out, "primary_joint_sameformula_repeat_agreement.csv"), row.names = FALSE)
new_solution_names <- setdiff(names(solutions), "saved_bobyqa_locked")
if (any(non_singular_good[new_solution_names]) && !all(repeat_agreement$agreement_pass)) stop("Non-singular same-formula solution found but independent repeat agreement did not pass; do not interpret or adopt it.")
if (any(non_singular_good[new_solution_names]) && !all(non_singular_good[new_solution_names])) stop("Mixed good/flagged same-formula solutions; do not interpret or adopt any replacement.")

sha256 <- function(path) {
  z <- suppressWarnings(system2("sha256sum", path, stdout = TRUE, stderr = TRUE))
  status <- attr(z, "status")
  if (!length(z) || (!is.null(status) && status != 0)) return(NA_character_)
  strsplit(z[[1]], "\\s+")[[1]][1]
}
prov <- data.frame(
  item = c("script", "diagnostic_input_trials", "locked_canonical_primary_joint_rds"),
  path = c(normalizePath(script_path), normalizePath(trials_path), normalizePath(canonical_rds)),
  sha256 = c(sha256(script_path), sha256(trials_path), sha256(canonical_rds)),
  stringsAsFactors = FALSE
)
write.csv(prov, file.path(out, "provenance.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "R_sessionInfo.txt"))

interpretation <- if (all(non_singular_good[new_solution_names])) {
  c(
    "A reproducible, non-singular same-formula optimizer solution was found. The fresh nloptwrap fit, its bobyqa restart, and the independent repeat pair all have optimizer code 0, no retained convergence message, positive Hessian eigenvalue minima, and `isSingular(..., tol=1e-4) == FALSE`.",
    sprintf("The independent repeat-agreement table passed both pairs: maximum repeat logLik difference %.9g, maximum fixed-effect difference %.9g RT100, and maximum theta difference %.9g.", max(repeat_agreement$abs_logLik_difference), max(repeat_agreement$max_abs_fixed_effect_difference_RT100), max(repeat_agreement$max_abs_theta_difference)),
    "The locked canonical bobyqa object remains a distinct singular/nonconverged boundary solution with a negative Hessian eigenvalue. This diagnostic does not replace, revise, or adopt a model for inference; any adoption requires the parent analysis owners to decide how to handle the locked canonical-output status."
  )
} else {
  c(
    "No non-singular converged same-formula solution was found. The rePCA output identifies a zero/near-zero random-effects principal component: the full 3-dimensional subject intercept/B/U covariance is on a boundary and its complete covariance parameterization is not estimable from these data under this specification.",
    "Options not run and not adopted: (1) a pre-specified lower-dimensional random-effects covariance model, (2) an uncorrelated random-effects sensitivity, or (3) a regularizing Bayesian hierarchical model with independently specified priors. Each changes the random-effects restriction or inferential framework and requires separate labelling. Retaining the locked fit as flagged and reporting the limitation is also an option."
  )
}
report <- c(
  "# Primary joint LMM: same-formula optimizer diagnostic v1",
  "",
  "## Scope",
  "",
  "Diagnostic-only fit of the locked primary formula with the same 10,995 trials, 33 subjects, fixed effects, REML criterion, and `(1+B+U|subject)` random-effects structure. No canonical output was modified and no alternative was adopted.",
  "",
  "Formula:",
  "",
  "```r",
  formula_text,
  "```",
  "",
  "## Sequential fits executed",
  "",
  "1. Inspected locked canonical `primary_joint.rds` (bobyqa).",
  "2. Fresh same-formula REML fit with `nloptwrap` (`maxeval=200000`).",
  "3. Same-formula REML `bobyqa` restart initialized at the fresh nloptwrap theta (`maxfun=200000`).",
  "4. Independently repeated steps 2--3 for agreement verification.",
  "",
  "See `primary_joint_sameformula_optimizer_diagnostics.csv` for exact log likelihoods, raw/scaled gradients, convergence messages, singularity flags, Hessian eigenvalue bounds/counts, and rePCA extrema. Full gradients and Hessians are saved one file per solution; `primary_joint_sameformula_repeat_agreement.csv` records the prespecified numerical repeat checks.",
  "",
  "## Interpretation",
  "",
  interpretation,
  "",
  "No scientific outcome-driven simplification was performed.",
  ""
)
writeLines(report, file.path(out, "primary_joint_sameformula_diagnostic_report.md"))
cat("Diagnostic-only same-formula fits complete. Output: ", normalizePath(out), "\n", sep = "")
