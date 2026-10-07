#!/usr/bin/env Rscript
# Independent verifier: no source/import of the model-matrix generator and no refit.
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(jsonlite) })
options(stringsAsFactors = FALSE)
ROOT <- Sys.getenv("DMN_PROJECT_ROOT", unset = normalizePath(getwd(), mustWork = TRUE))
OUT <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/same_dimension_hierarchical_v1")
UP <- file.path(ROOT, "manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/location_standardized_capture_v1/trial_model")
MODEL <- file.path(UP, "final_selected_model.rds")
BETA <- file.path(UP, "final_selected_beta.csv")
VCOV <- file.path(UP, "final_selected_vcov_RT100.csv")
sha <- function(path) unname(strsplit(system2("sha256sum", path, stdout = TRUE), " +")[[1]][1])
stopifnot(all(file.exists(c(MODEL, BETA, VCOV,
  file.path(OUT, "rt_cost_cells.csv"), file.path(OUT, "dmn_coupling_cells.csv"),
  file.path(OUT, "behavior_group_contrasts.csv"), file.path(OUT, "coupling_group_contrasts.csv"),
  file.path(OUT, "omnibus_group_profiles.csv"), file.path(OUT, "contrast_vectors_model_matrix.csv")))))
fit <- readRDS(MODEL)
beta_file <- read.csv(BETA, check.names = FALSE)
V_raw <- read.csv(VCOV, check.names = FALSE)
V_file <- as.matrix(V_raw[, -1])
rownames(V_file) <- V_raw[[1]]
beta <- setNames(beta_file$beta_RT100, beta_file$term)
stopifnot(identical(names(beta), rownames(V_file)), identical(names(beta), colnames(V_file)), isREML(fit))

# Independent algebraic coefficient map. These terms are not generated from model.matrix.
zero <- function() setNames(rep(0, length(beta)), names(beta))
weighted <- function(x) { v <- zero(); v[names(x)] <- as.numeric(x); v }
SS_B50_cost <- weighted(c(conditionBF=.5, conditionBR=.5, `conditionAbsent:run2`=-.5, `conditionBF:run2`=.25, `conditionBR:run2`=.25))
DS_U50_cost <- weighted(c(conditionUF=.5, conditionUR=.5, `conditionUF:D`=.5, `conditionUR:D`=.5, `conditionAbsent:run2`=-.5, `conditionUF:run2`=.25, `conditionUR:run2`=.25))
SS_U50_cost <- weighted(c(conditionUF=.5, conditionUR=.5, `conditionAbsent:run2`=-.5, `conditionUF:run2`=.25, `conditionUR:run2`=.25))
DS_B50_cost <- weighted(c(conditionBF=.5, conditionBR=.5, `conditionBF:D`=.5, `conditionBR:D`=.5, `conditionAbsent:run2`=-.5, `conditionBF:run2`=.25, `conditionBR:run2`=.25))
SS_B50_coup <- weighted(c(`conditionBF:X`=.5, `conditionBR:X`=.5))
DS_U50_coup <- weighted(c(`conditionUF:X`=.5, `conditionUR:X`=.5, `conditionUF:X:D`=.5, `conditionUR:X:D`=.5))
SS_U50_coup <- weighted(c(`conditionUF:X`=.5, `conditionUR:X`=.5))
DS_B50_coup <- weighted(c(`conditionBF:X`=.5, `conditionBR:X`=.5, `conditionBF:X:D`=.5, `conditionBR:X:D`=.5))
rt_cells <- list(orientation_SS_B50=SS_B50_cost, orientation_DS_U50=DS_U50_cost, color_SS_U50=SS_U50_cost, color_DS_B50=DS_B50_cost)
coup_cells <- list(orientation_SS_B50=SS_B50_coup, orientation_DS_U50=DS_U50_coup, color_SS_U50=SS_U50_coup, color_DS_B50=DS_B50_coup)
behavior <- list(orientation_SS_minus_DS_regular_first=SS_B50_cost-DS_U50_cost, color_DS_minus_SS_regular_first=DS_B50_cost-SS_U50_cost)
coupling <- list(orientation_SS_minus_DS_regular_first=SS_B50_coup-DS_U50_coup, color_DS_minus_SS_regular_first=DS_B50_coup-SS_U50_coup)

manual_1d <- function(v) {
  est <- sum(v * beta); se <- sqrt(as.numeric(t(v) %*% V_file %*% v))
  z <- as.data.frame(contest1D(fit, as.numeric(v), confint=TRUE, ddf="Satterthwaite"))
  lo <- names(z)[grepl("lower", names(z), ignore.case=TRUE)][1]; hi <- names(z)[grepl("upper", names(z), ignore.case=TRUE)][1]
  c(estimate_RT100=est, SE_RT100=se, df=z$df, t=est/se, ci_low_RT100=z[[lo]], ci_high_RT100=z[[hi]], p_raw=2*pt(abs(est/se), df=z$df, lower.tail=FALSE))
}
make_table <- function(xs) {
  out <- do.call(rbind, lapply(xs, manual_1d))
  data.frame(contrast_id=rownames(out), out, row.names=NULL, check.names=FALSE)
}
ind_rt <- make_table(rt_cells); ind_coup <- make_table(coup_cells)
ind_behavior <- make_table(behavior); ind_coupling <- make_table(coupling)
ind_behavior$p_Holm2 <- p.adjust(ind_behavior$p_raw, method="holm")
ind_coupling$p_Holm2 <- p.adjust(ind_coupling$p_raw, method="holm")
manual_md <- function(L, id) {
  z <- as.data.frame(contestMD(fit, L=unname(L), ddf="Satterthwaite"))
  data.frame(test_id=id, statistic=z$`F value`, num_df=z$NumDF, den_df=z$DenDF, p_raw=z$`Pr(>F)`, matrix_rank=qr(L)$rank)
}
ind_omnibus <- rbind(manual_md(do.call(rbind, behavior), "behavior_group_profile"), manual_md(do.call(rbind, coupling), "coupling_group_profile"))
ind_omnibus$p_Holm2 <- p.adjust(ind_omnibus$p_raw, method="holm")

# Direct comparisons to all generator artifacts; differences are stored rather than silently rounded.
read_result <- function(name) read.csv(file.path(OUT, name), check.names=FALSE)
maxdiff <- function(a, b) max(abs(as.numeric(a)-as.numeric(b)))
compare_1d <- function(saved, independent, p_holm=FALSE) {
  stopifnot(nrow(saved)==nrow(independent), identical(saved$contrast_id, independent$contrast_id))
  cols <- c("estimate_RT100", "SE_RT100", "df", "t", "ci_low_RT100", "ci_high_RT100", "p_raw")
  ans <- setNames(vapply(cols, function(x) maxdiff(saved[[x]], independent[[x]]), numeric(1)), cols)
  if (p_holm) ans <- c(ans, p_Holm2=maxdiff(saved$p_Holm2, independent$p_Holm2))
  ans
}
rt_diff <- compare_1d(read_result("rt_cost_cells.csv"), ind_rt)
coup_diff <- compare_1d(read_result("dmn_coupling_cells.csv"), ind_coup)
behavior_diff <- compare_1d(read_result("behavior_group_contrasts.csv"), ind_behavior, TRUE)
coupling_diff <- compare_1d(read_result("coupling_group_contrasts.csv"), ind_coupling, TRUE)
saved_omnibus <- read_result("omnibus_group_profiles.csv")
stopifnot(nrow(saved_omnibus)==2L, identical(saved_omnibus$test_id, ind_omnibus$test_id))
omnibus_cols <- c("statistic", "num_df", "den_df", "p_raw", "matrix_rank", "p_Holm2")
omnibus_diff <- setNames(vapply(omnibus_cols, function(x) maxdiff(saved_omnibus[[x]], ind_omnibus[[x]]), numeric(1)), omnibus_cols)

saved_vectors <- read_result("contrast_vectors_model_matrix.csv")
ind_vectors <- rbind(
  do.call(rbind, lapply(names(rt_cells), function(id) data.frame(domain="RT_cost_cell", contrast_id=id, coefficient=names(rt_cells[[id]]), weight_RT100=as.numeric(rt_cells[[id]])))),
  do.call(rbind, lapply(names(coup_cells), function(id) data.frame(domain="DMN_coupling_cell", contrast_id=id, coefficient=names(coup_cells[[id]]), weight_RT100=as.numeric(coup_cells[[id]])))),
  do.call(rbind, lapply(names(behavior), function(id) data.frame(domain="behavior_group_contrast", contrast_id=id, coefficient=names(behavior[[id]]), weight_RT100=as.numeric(behavior[[id]])))),
  do.call(rbind, lapply(names(coupling), function(id) data.frame(domain="coupling_group_contrast", contrast_id=id, coefficient=names(coupling[[id]]), weight_RT100=as.numeric(coupling[[id]]))))
)
vector_contract_ok <- nrow(saved_vectors)==nrow(ind_vectors) && identical(saved_vectors$domain, ind_vectors$domain) && identical(saved_vectors$contrast_id, ind_vectors$contrast_id) && identical(saved_vectors$coefficient, ind_vectors$coefficient)
vector_max_abs_difference <- if (vector_contract_ok) max(abs(saved_vectors$weight_RT100-ind_vectors$weight_RT100)) else Inf
all_diffs <- c(rt_diff, coup_diff, behavior_diff, coupling_diff, omnibus_diff, vector_weights=vector_max_abs_difference)
pass <- vector_contract_ok && all(is.finite(all_diffs)) && max(all_diffs) < 1e-10 && identical(sha(MODEL), "afacc506fda7801db929df4c06f6dd033a507521907bbe7be2843394f86b983d")
result <- list(status=if (pass) "PASS" else "FAIL", independent_verifier=TRUE, no_refit=TRUE,
  inputs=list(model_sha256=sha(MODEL), beta_sha256=sha(BETA), vcov_sha256=sha(VCOV)),
  checks=list(beta_vcov_named_contract=TRUE, vector_contract=vector_contract_ok, exact_rank_behavior=qr(do.call(rbind, behavior))$rank, exact_rank_coupling=qr(do.call(rbind, coupling))$rank,
              rt_cell_rows=nrow(ind_rt), coupling_cell_rows=nrow(ind_coup), behavior_contrast_rows=nrow(ind_behavior), coupling_contrast_rows=nrow(ind_coupling), omnibus_rows=nrow(ind_omnibus)),
  max_absolute_differences=as.list(all_diffs), tolerance=1e-10,
  independently_replayed=list(rt_cost_cells=split(ind_rt[-1], ind_rt$contrast_id), coupling_cells=split(ind_coup[-1], ind_coup$contrast_id), behavioral_group_contrasts=split(ind_behavior[-1], ind_behavior$contrast_id), coupling_group_contrasts=split(ind_coupling[-1], ind_coupling$contrast_id), omnibus=split(ind_omnibus[-1], ind_omnibus$test_id)))
write_json(result, file.path(OUT, "numeric_validation.json"), pretty=TRUE, auto_unbox=TRUE, digits=16)
if (!pass) stop("Independent numeric validation failed")
cat("PASS independent beta/vcov and lmerTest replay: all requested cells, contrasts, and 2-df omnibus tests agree < 1e-10.\n")
