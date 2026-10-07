# Exploratory physical-dimension versus assigned-bias comparison

## Status and scope
New author-requested exploratory analyses; preserve all existing sources, manuscript packages, figures and correction families. Do not describe the new comparisons as preregistered, causal decomposition, or increased independent sample size. No manuscript edits or figures.

## Inputs and gate
Use corrected physical Yeo-7 Default91/common399 `dmn_conn_primary`, NOT legacy `dmn_conn`, `X` or `dmn_conn_gc` columns. Freeze extracted participant scores and the exact N33 correct/RT-valid 10,995-row behavioral table by SHA-256. N33 = SS16 + DS17; RT 200–3,000 ms inclusive, correct==1, first trials retained. Verify physical Same=orientation, Different=color, and score reconstruction with each participant's absent baseline.

## Participant association comparisons
Keep the existing primary group-centered DMN–biased-capture association. Calculate group-centered DMN–orientation (`same_capture`) and DMN–color (`diff_capture`) associations on the exact same 33 participants; both physical scores mix assigned-biased and assigned-unbiased cells across groups. Ordinary within-group centering removes one group contrast, giving correlation inference df=N−3 and Fisher SE=1/sqrt(N−4). The two new physical-score associations form a new exploratory Holm-2 family, without altering the original primary/secondary families.

Compare biased versus orientation, biased versus color, and orientation versus color using dependent-correlation Williams tests with the existing project group-adjusted approximation n_eff=N−1, df=N−4=29. All three new comparisons form exploratory Holm-3. Also report paired SS/DS-stratified 10,000 participant percentile intervals for correlations and differences (seed 20261004), preserving common neural/behavior pairing and re-centering within each resample. Intervals are pointwise, not simultaneous.

## Descriptive covariance decomposition
Partition the primary centered DMN–biased-capture cross-product into SS orientation and DS color contributions. Normalize both contributions by the SAME full-sample centered neural norm × full-sample centered biased-capture norm so the contributions sum exactly to the primary r. Express their shares of the total cross-product as covariance shares, never variance explained, causal contribution, sensory-versus-learning weights, or independent evidence. Bootstrap these quantities using the same paired stratified participants and record possible unstable denominators.

## Joint-model recoding
Read the verified saved primary joint model (no refit or changed optimizer, data, random-effects structure or neural scale). The four capture slopes are SS orientation/biased, SS color/unbiased, DS orientation/unbiased, DS color/biased. Derive participant-weighted physical orientation/color slopes, their difference, participant-weighted bias-minus-unbiased slope, and the difference between those two contrast magnitudes. These five new contrast questions form one exploratory Holm-5 family, including the repeated bias contrast solely for within-panel comparison; preserve its original Holm-9 result separately. Use saved optimum, fixed covariance and lmerTest Satterthwaite inference; verify exact saved frame and coefficients. Label ms per global group-centered DMN SD.

## Identifiability boundaries
Physical dimension × assigned bias fills four between-group condition cells, but assigned group is deterministically tied to the physical-dimension/bias combination. The recoding is not a new manipulation or new information. Bias status remains confounded with greater dimension exposure. Capture (present minus absent) is not spatial learning (rare minus frequent); a learning-score model would be separately exploratory and must use actual location contrasts, consider reliability and collinearity, and cannot yield identifiable causal percentages for dimension versus learning here.
