# Exploratory scanner-run moderation of the DMN–learning association

## Authorized scope
Test whether the DMN association with rare-minus-frequent spatial learning changes between scanner runs. Retain the existing location-by-run interaction; do not simplify it away. This is a separate author-requested exploratory sensitivity, not a replacement of the original selected fit or a preregistered analysis. Do not edit the manuscript, Supplement, original analysis sources, or historical correction families.

## Frozen inputs and one-axis change
Use the original exact_model_frame.rds and final_selected_model.rds in spatial_learning_dmn_v1/trial_model. Confirm identical model variables and row order, 6,721 eligible biased-dimension scanner trials, 33 participants (SS16, DS17), correct responses, inclusive RT 200–3,000 ms, all original rows retained, recorded location labels unchanged, and no imputation. Retain corrected primary Default91/common399 connectivity, group centering and its global participant-SD scaling; do not standardize across repeated trial rows.

Original formula:
`RT100 ~ L*X*D + run2 + L:run2 + (1+L|subject)`

New formula:
`RT100 ~ L*X*D + L*X*run2 + (1+L|subject)`

This adds X:run2 and L:X:run2 and preserves all lower-order terms. L=Rare1/Frequent0; D=DS1/SS0; run2=run2_1/run1_0; X is the frozen standardized group-centered predictor. REML, correlated participant random intercept/location slope, and Satterthwaite inference remain unchanged. This models a COMMON run-related change in neural–learning coupling across SS/DS, not group-specific temporal moderation or a four-way interaction. The original random-effects structure is retained; inference is conditional on that structure, not a claim that trial rows are independent participants.

## Contrast and multiplicity registry fixed before fitting
- Focal run-change contrast: L:X:run2. Run2 minus Run1 change in the DMN modulation of rare-minus-frequent RT, ms per global participant SD of corrected DMN. One targeted exploratory two-sided Satterthwaite test, unadjusted; pointwise 95% CI.
- Six run-specific neural–learning slopes: pooled, SS, DS in each run. Pooled weights SS16/33 and DS17/33. A NEW exploratory Holm-6 family; report raw and adjusted p values with pointwise 95% CIs. These are within-run slopes, not direct between-run tests.
- Four equally run-averaged companion contrasts: pooled, SS, DS neural–learning slopes and DS-minus-SS moderation. Weight both runs equally and participants by the original group sizes. A NEW exploratory Holm-4 companion family, separate from the preserved original Holm-4 family. Report old versus new estimates; do not interpret threshold crossing as evidence of a changed effect.
- Do not fit ML model comparisons or compare REML AIC between different fixed-effect formulas. The focal Wald/Satterthwaite contrast directly tests the question.

## Numerical and verification gates
Run sequentially: canonical bobyqa, alternative nloptwrap, bobyqa restart from the alternative theta, and an independent repeat of the alternative/restart sequence. Keep the same formula/random structure in all five fits. Save every fit and checkpoint diagnostics. Inspect singularity, optimizer codes/messages, raw/scaled gradients, Hessian eigenvalues, random covariance and rePCA. Require agreement in log likelihood <1e-6, fixed coefficients (RT100 units) <1e-6, and theta <1e-5 before calling the result reproducible. Do not change covariance structure if a boundary fit occurs without separately reporting the blocker.

Reopen saved fitted objects and exports. Independently reconstruct all contrast weights from model-matrix predictions on Location × Connectivity × Group × Run grids, verify estimates/SEs/dfs/CIs/raw and Holm p values, and confirm original source hashes remain unchanged. No new plots are requested.
