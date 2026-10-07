# New exploratory DMN–spatial-learning analysis

## Scope fixed before inspecting new associations
The author requests an actual rare-minus-frequent location analysis, distinct from present-minus-absent capture. Use the current corrected physical Default91/common399 predictor, exact N33 scanner behavioral cohort (SS16/DS17) and correct, finite RT200–3000ms inclusive. Retain initial trials. Preserve prior manuscript/data/model/figures and all previous correction families. All new analyses are exploratory; no causal or preregistered claim.

Primary temporal scope is scanner only, matching the measured task-run connectivity. Do not substitute the previous N35 training-plus-scanner group learning means or attach N35 behavior to N33 neural data. Existing combined behavioral learning model remains unchanged.

## Sources and coding
Neural source: `resolved_reanalysis/extraction_common_coverage/participant_scores_resolved.csv`, corrected `dmn_conn_primary` only; carried legacy X/gc/neural fields must not be used. Raw behavioral source: `revision-1/attached-v3-correct-rt/brain_behavior/correct_rt_trials_n33.csv`, exact10995 rows. Bias: SS Same=orientation, DS Different=color. Spatial learning uses only category=biased and recorded DisRegion in {Rare,Frequent}; L=1Rare/0Frequent. Independently audit recorded region coding and source event/task definitions before fitting; save counts and discrepancies. No trial imputation, rare/frequent count equalization, new subject exclusion or absence subtraction. If mapping cannot be verified, report blocker rather than inventing learning labels.

## Participant learning and observed-score association
Define spatial-learning score S_i = mean scanner RT(rare biased) − mean scanner RT(frequent biased), positive indicates faster responses in frequent region. Recompute from retained trials and save rare/frequent means/counts overall, by run, and by original recorded odd/even global_order parity (no renumbering). Verify correspondence of trial dimension and group. Calculate group-centered corrected-DMN versus S_i and within SS/DS correlations: dfN−3 pooled, N−2 within group; pointwise Fisher and 10k paired participant percentile intervals with seed20261004. These three associations form one new exploratory Holm3 family. Score means/one-sample tests by SS/DS form a separate Holm2 descriptive family; include d_z=t/sqrt(N) and mean/SD, and percentile CIs for d_z.

Directly compare the same-cohort pooled neural–capture and neural–learning correlations with the existing group-adjusted Williams approximation n_eff32/df29; one explicitly new exploratory comparison, raw p (one-test family), paired participant bootstrap delta-r CI. Do not infer a difference merely from separate significance or selectively prefer an interval over a test.

Reliability: odd/even internal consistency and Run1/Run2 rank stability of S_i, each half/run uses its own rare/frequent means. Report pooled group-adjusted, SS and DS (six tests) as one new exploratory Holm6 family with 10k paired participant percentile intervals. Record all count support; no empty-cell imputation. Negative half r and Spearman–Brown projections are not meaningful negative reliability. Reliability concerns limit null and trait interpretation.

## Primary new trial-level model
Fit biased-dimension scanner trials only:
`RT100 ~ L*X*D + run2 + L:run2 + (1+L|subject)`
RT100=RT_ms/100; D0SS/D1DS; X=(correctedFC−groupMean)/GLOBAL sampleSD of group-centered correctedFC across33 participants, NOT SD repeated by trial. run2=0run1/1run2. Location×run permits the behavioral learning contrast to change between runs. X interactions identify whether connectivity is associated with the rare-minus-frequent effect rather than generic RT/capture.

New neural–learning slope contrast family (Holm4):
1 pooled participant-weighted DMN modulation of rare-minus-frequent RT: L:X + (17/33)L:X:D;
2 SS modulation: L:X;
3 DS modulation: L:X + L:X:D;
4 DS−SS modulation: L:X:D.
Report ms per global corrected-DMN SD, estimate/SE/t/Satterthwaite df/pointwise95CI/raw+Holm4p. Save model RDS, exact model frame, complete fixed effects, full random covariance, diagnostic code/messages, singularity, gradient and Hessian. Model mean location effects are descriptive model contrasts, not extra independent neural findings.

REML estimation; start correlated intercept/location slope with bobyqa, maxfun200000. Run same-formula nloptwrap and bobyqa restart to diagnose convergence, preserve all fits and require reproducibility before selecting any no-warning, nonsingular optimum. No significance-based model selection. If all same-formula fits are singular or numerically unresolved, run a labeled independent intercept/location-slope sensitivity `(1+L||subject)` with numeric L; preserve the original boundary fit and state changed covariance. Do not silently drop location slope or inflate between-person information from trial counts. Do not fit a Bayesian/measurement-error model or new exposure model in this task.

## Interpretation boundaries
Group and physical biased dimension remain linked. Location learning is an actual rare-minus-frequent behavioral contrast but is not a causal mechanism measure. Negative/uncertain neural–learning coupling cannot establish absent learning or absent neural relation, especially with small N and score reliability limitations. This new model cannot assign causal percentages to physical dimension versus learning or separate dimension exposure from assigned bias. Preserve original capture results; do not claim capture indexes learning solely because both correlate with connectivity.

## Outputs
Separate participant and trial-model subdirectories with runnable code, source hashes, verification JSON, saved scores/frames and full CSVs. Parent independently verifies outputs before incorporating Methods/Results and supplement into new v8 manuscripts cloned from the latest v7 equation-purpose package. No response-letter update or new figures requested.
