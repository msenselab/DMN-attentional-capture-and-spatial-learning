#!/usr/bin/env Rscript
# Derived contrasts only: reads the frozen selected model; never refits or writes upstream.
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(jsonlite) })
options(stringsAsFactors = FALSE, width = 180)

ROOT <- Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
OUT <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/same_dimension_hierarchical_v1")
UP <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/trial_model")
MODEL <- file.path(UP, "final_selected_model.rds")
BETA <- file.path(UP, "final_selected_beta.csv")
VCOV <- file.path(UP, "final_selected_vcov_RT100.csv")
PRIOR_VERIFY <- file.path(UP, "verification_independent.json")
MANIFEST <- file.path(UP, "run_manifest.json")
SPEC <- file.path(OUT, "ANALYSIS_SPEC.md")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
sha <- function(path) unname(strsplit(system2("sha256sum", path, stdout = TRUE), " +")[[1]][1])

# Preflight is read-only and requires the frozen upstream independent verifier.
stopifnot(all(file.exists(c(MODEL, BETA, VCOV, PRIOR_VERIFY, MANIFEST, SPEC))))
prior <- fromJSON(PRIOR_VERIFY)
man <- fromJSON(MANIFEST)
stopifnot(identical(prior$status, "PASS"), isTRUE(prior$independent_readback), all(unlist(prior$checks)),
          identical(man$status, "PASS"), isTRUE(man$model$REML),
          identical(man$model$selected_model, "full_correlated_bobyqa"))
expected_model_sha <- "afacc506fda7801db929df4c06f6dd033a507521907bbe7be2843394f86b983d"
stopifnot(identical(sha(MODEL), expected_model_sha))

fit <- readRDS(MODEL)
beta <- fixef(fit); V <- vcov(fit)
saved_beta <- read.csv(BETA, check.names = FALSE)
saved_V <- as.matrix(read.csv(VCOV, check.names = FALSE)[, -1])
stopifnot(identical(names(beta), saved_beta$term), max(abs(beta - saved_beta$beta_RT100)) < 1e-12,
          identical(rownames(V), saved_beta$term), identical(colnames(V), saved_beta$term),
          all(dim(V) == dim(saved_V)), max(abs(V - saved_V)) < 1e-12, isREML(fit))
fixed_formula <- ~ condition * X * D + condition:run2
mf <- model.frame(fit)
stopifnot(nrow(mf) == 10995L, length(unique(mf$subject)) == 33L,
          identical(colnames(model.matrix(fixed_formula, mf)), names(beta)))

# Model-matrix construction: response absent is subtracted within group/run; runs receive equal 1/2 weight.
levels_condition <- levels(mf$condition)
row_x <- function(condition, D, X, run2) {
  nd <- data.frame(condition = factor(condition, levels = levels_condition), X = X, D = D, run2 = run2)
  z <- model.matrix(fixed_formula, data = nd)
  stopifnot(identical(colnames(z), names(beta)))
  as.numeric(z[1, ])
}
named_row_x <- function(condition, D, X, run2) setNames(row_x(condition, D, X, run2), names(beta))
cost_vector <- function(condition, D) {
  0.5 * (named_row_x(condition, D, 0, 0) - named_row_x("Absent", D, 0, 0)) +
    0.5 * (named_row_x(condition, D, 0, 1) - named_row_x("Absent", D, 0, 1))
}
coupling_vector <- function(condition, D) {
  # Exact finite difference across one standardized X unit equals the derivative because the model is linear in X.
  0.5 * ((named_row_x(condition, D, 1, 0) - named_row_x("Absent", D, 1, 0)) -
           (named_row_x(condition, D, 0, 0) - named_row_x("Absent", D, 0, 0))) +
    0.5 * ((named_row_x(condition, D, 1, 1) - named_row_x("Absent", D, 1, 1)) -
           (named_row_x(condition, D, 0, 1) - named_row_x("Absent", D, 0, 1)))
}

# Physical mapping frozen by the specification: SS orientation=B50/color=U50; DS orientation=U50/color=B50.
cost_cells <- list(
  orientation_SS_B50 = cost_vector("BF", 0) / 2 + cost_vector("BR", 0) / 2,
  orientation_DS_U50 = cost_vector("UF", 1) / 2 + cost_vector("UR", 1) / 2,
  color_SS_U50 = cost_vector("UF", 0) / 2 + cost_vector("UR", 0) / 2,
  color_DS_B50 = cost_vector("BF", 1) / 2 + cost_vector("BR", 1) / 2
)
coupling_cells <- list(
  orientation_SS_B50 = coupling_vector("BF", 0) / 2 + coupling_vector("BR", 0) / 2,
  orientation_DS_U50 = coupling_vector("UF", 1) / 2 + coupling_vector("UR", 1) / 2,
  color_SS_U50 = coupling_vector("UF", 0) / 2 + coupling_vector("UR", 0) / 2,
  color_DS_B50 = coupling_vector("BF", 1) / 2 + coupling_vector("BR", 1) / 2
)
behavior_contrasts <- list(
  orientation_SS_minus_DS_regular_first = cost_cells$orientation_SS_B50 - cost_cells$orientation_DS_U50,
  color_DS_minus_SS_regular_first = cost_cells$color_DS_B50 - cost_cells$color_SS_U50
)
coupling_contrasts <- list(
  orientation_SS_minus_DS_regular_first = coupling_cells$orientation_SS_B50 - coupling_cells$orientation_DS_U50,
  color_DS_minus_SS_regular_first = coupling_cells$color_DS_B50 - coupling_cells$color_SS_U50
)

one_d <- function(v, id, estimand, family = NA_character_) {
  z <- as.data.frame(contest1D(fit, as.numeric(v), confint = TRUE, ddf = "Satterthwaite"))
  lo <- names(z)[grepl("lower", names(z), ignore.case = TRUE)][1]
  hi <- names(z)[grepl("upper", names(z), ignore.case = TRUE)][1]
  data.frame(contrast_id = id, estimand = estimand, estimate_RT100 = z$Estimate, SE_RT100 = z[["Std. Error"]],
             df = z$df, t = z[["t value"]], ci_low_RT100 = z[[lo]], ci_high_RT100 = z[[hi]],
             p_raw = z[["Pr(>|t|)"]], Holm_family = family, stringsAsFactors = FALSE)
}
convert_ms <- function(x) {
  for (nm in c("estimate", "SE", "ci_low", "ci_high")) x[[paste0(nm, "_ms")]] <- x[[paste0(nm, "_RT100")]] * 100
  x$unit <- "ms per standardized group-centered DMN connectivity unit"
  x
}
cell_labels_cost <- c(
  orientation_SS_B50 = "Orientation, SS: B50 cost relative to Absent at X=0; equal mean across runs",
  orientation_DS_U50 = "Orientation, DS: U50 cost relative to Absent at X=0; equal mean across runs",
  color_SS_U50 = "Color, SS: U50 cost relative to Absent at X=0; equal mean across runs",
  color_DS_B50 = "Color, DS: B50 cost relative to Absent at X=0; equal mean across runs"
)
cell_labels_coupling <- sub("cost relative to Absent at X=0; equal mean across runs", "DMN coupling derivative of the cost relative to Absent, per X unit", cell_labels_cost, fixed = TRUE)
rt_cells <- convert_ms(do.call(rbind, Map(function(v, id, e) one_d(v, id, e), cost_cells, names(cost_cells), unname(cell_labels_cost))))
coupling_cells_out <- convert_ms(do.call(rbind, Map(function(v, id, e) one_d(v, id, e), coupling_cells, names(coupling_cells), unname(cell_labels_coupling))))
behavior_out <- convert_ms(do.call(rbind, Map(function(v, id) one_d(v, id, ifelse(grepl("orientation", id), "Orientation: SS B50 minus DS U50 (regularity-carrying group first)", "Color: DS B50 minus SS U50 (regularity-carrying group first)"), "Holm-2 behavioral group contrasts"), behavior_contrasts, names(behavior_contrasts))))
coupling_out <- convert_ms(do.call(rbind, Map(function(v, id) one_d(v, id, ifelse(grepl("orientation", id), "Orientation coupling: SS B50 minus DS U50 (regularity-carrying group first)", "Color coupling: DS B50 minus SS U50 (regularity-carrying group first)"), "Holm-2 coupling group contrasts"), coupling_contrasts, names(coupling_contrasts))))
# Behavioral costs are milliseconds, not connectivity derivatives.
rt_cells$unit <- "ms"
behavior_out$unit <- "ms"
behavior_out$p_Holm2 <- p.adjust(behavior_out$p_raw, method = "holm")
coupling_out$p_Holm2 <- p.adjust(coupling_out$p_raw, method = "holm")

# Joint two-df profiles: both prescribed within-physical-dimension group differences together.
md_one <- function(L, id, estimand) {
  z <- as.data.frame(contestMD(fit, L = unname(L), ddf = "Satterthwaite"))
  data.frame(test_id = id, estimand = estimand, statistic = z$`F value`, num_df = z$NumDF, den_df = z$DenDF,
             p_raw = z$`Pr(>F)`, matrix_rank = qr(L)$rank, stringsAsFactors = FALSE)
}
behavior_omnibus <- md_one(do.call(rbind, behavior_contrasts), "behavior_group_profile", "Joint 2-df test: orientation SS−DS and color DS−SS RT-cost group differences")
coupling_omnibus <- md_one(do.call(rbind, coupling_contrasts), "coupling_group_profile", "Joint 2-df test: orientation SS−DS and color DS−SS DMN-coupling group differences")
omnibus <- rbind(behavior_omnibus, coupling_omnibus)
omnibus$p_Holm2 <- p.adjust(omnibus$p_raw, method = "holm")
omnibus$Holm_family <- "Holm-2 across behavioral and coupling omnibus tests"

vecs <- function(x, domain) do.call(rbind, lapply(seq_along(x), function(i) data.frame(domain = domain, contrast_id = names(x)[i], coefficient = names(x[[i]]), weight_RT100 = as.numeric(x[[i]]), stringsAsFactors = FALSE)))
all_vectors <- rbind(vecs(cost_cells, "RT_cost_cell"), vecs(coupling_cells, "DMN_coupling_cell"), vecs(behavior_contrasts, "behavior_group_contrast"), vecs(coupling_contrasts, "coupling_group_contrast"))
all_vectors$weight_ms <- all_vectors$weight_RT100 * 100
all_L <- rbind(do.call(rbind, behavior_contrasts), do.call(rbind, coupling_contrasts))
rownames(all_L) <- c("behavior_orientation", "behavior_color", "coupling_orientation", "coupling_color")
contrast_cov <- all_L %*% V %*% t(all_L)
write.csv(rt_cells, file.path(OUT, "rt_cost_cells.csv"), row.names = FALSE)
write.csv(coupling_cells_out, file.path(OUT, "dmn_coupling_cells.csv"), row.names = FALSE)
write.csv(behavior_out, file.path(OUT, "behavior_group_contrasts.csv"), row.names = FALSE)
write.csv(coupling_out, file.path(OUT, "coupling_group_contrasts.csv"), row.names = FALSE)
write.csv(omnibus, file.path(OUT, "omnibus_group_profiles.csv"), row.names = FALSE)
write.csv(all_vectors, file.path(OUT, "contrast_vectors_model_matrix.csv"), row.names = FALSE)
write.csv(data.frame(contrast_id = rownames(contrast_cov), as.matrix(contrast_cov), check.names = FALSE), file.path(OUT, "cross_group_contrast_covariance_RT100.csv"), row.names = FALSE)

formulas <- c(
  "# Same-physical-dimension hierarchical contrasts: formula and code contract",
  "",
  "Frozen model: `RT100 ~ condition * X * D + condition:run2 + (1 + BF + BR + UF + UR | subject)` (REML; lmerTest Satterthwaite). `D=0` SS and `D=1` DS. `X` is group-centered DMN connectivity divided by the common pooled participant-level SD.",
  "",
  "For group `g` and condition `c`, the run-averaged RT cost vector is:",
  "`v_cost(c,g) = 0.5 [x(c,g,X=0,run2=0)-x(Absent,g,X=0,run2=0)] + 0.5 [x(c,g,X=0,run2=1)-x(Absent,g,X=0,run2=1)]`.",
  "The equal-region vector is `v_B50=0.5 v(BF,g)+0.5 v(BR,g)` or `v_U50=0.5 v(UF,g)+0.5 v(UR,g)`.",
  "",
  "The coupling vector is the exact one-unit derivative of this linear model: `v_coupling(c,g)=v_cost(c,g;X=1)-v_cost(c,g;X=0)`. The `condition:run2` terms cancel algebraically, but the generator still constructs the prescribed equal run average.",
  "",
  "Physical mapping: orientation=(SS B50, DS U50); color=(SS U50, DS B50). Group contrasts are orientation `SS B50 - DS U50` and color `DS B50 - SS U50`; thus the regularity-carrying group is first. Each is a cross-group physical-dimension comparison, not an isolated causal regularity effect.",
  "",
  "`contest1D(..., ddf='Satterthwaite', confint=TRUE)` supplies pointwise t intervals and one-df tests. `contestMD(..., ddf='Satterthwaite')` jointly tests the two ordered group contrasts (rank 2) in each domain. Holm-2 is applied independently to the two behavioral contrasts, the two coupling contrasts, and the two omnibus tests. No prior contrast family is changed.",
  "",
  "Exact machine-readable coefficient weights are in `contrast_vectors_model_matrix.csv`; their cross-group covariance matrix is in `cross_group_contrast_covariance_RT100.csv`. The executable model-matrix generator is `derive_same_dimension_hierarchical_contrasts.R`; the separately implemented beta/vcov verifier is `verify_same_dimension_hierarchical_contrasts.R`."
)
writeLines(formulas, file.path(OUT, "FORMULAS_AND_CODE.md"))

input_hashes <- list(model_sha256 = sha(MODEL), beta_sha256 = sha(BETA), vcov_sha256 = sha(VCOV), prior_verifier_sha256 = sha(PRIOR_VERIFY), upstream_manifest_sha256 = sha(MANIFEST), analysis_spec_sha256 = sha(SPEC))
run_record <- list(status = "PASS", no_refit = TRUE, upstream_model = normalizePath(MODEL), source_hashes = input_hashes,
                   prerequisites = list(prior_independent_verifier_status = prior$status, all_prior_checks_pass = all(unlist(prior$checks)), selected_model = man$model$selected_model, REML = isREML(fit)),
                   counts = list(model_rows = nrow(mf), participants = length(unique(mf$subject)), rt_cost_cells = nrow(rt_cells), coupling_cells = nrow(coupling_cells_out), behavioral_group_contrasts = nrow(behavior_out), coupling_group_contrasts = nrow(coupling_out), omnibus_tests = nrow(omnibus)),
                   correction_families = list(behavior_contrasts = "Holm-2", coupling_contrasts = "Holm-2", omnibus = "Holm-2"))
write_json(run_record, file.path(OUT, "run_record.json"), pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("PASS derived contrasts: 4 RT cells, 4 coupling cells, 2+2 group contrasts, 2 omnibus tests; no refit.\n"))
