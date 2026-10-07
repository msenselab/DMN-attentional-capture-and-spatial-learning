# Inference correction-family plan — set before empirical execution

**Status:** prepared before new empirical inference. This file supersedes no approved manuscript claim; it records the correction scopes implemented by `run_dependent_inference.py` so that output cannot silently inherit obsolete family sizes.

## Confirmatory participant association

- **One primary test:** pooled binary-group-adjusted correlation between the fresh physical Yeo-7 Default predictor and the subject-level **biased capture contrast** (biased RT minus absent RT), two-sided, df = 30. It is not multiplicity adjusted.
- **Primary behavioral contrast:** `biased_capture` is a within-participant condition contrast, not an absolute mean RT. `biased_priming` is separately defined only as biased-after-biased minus biased-after-nonbiased (including absent/unbiased prior trial); it is not used as a substitute for capture.

## Revision-stage exploratory participant family: Holm-8

The code applies Holm to exactly these eight predeclared non-primary tests:

1. pooled group-adjusted Default–unbiased-capture;
2. pooled group-adjusted Default–all-capture;
3. pooled partial Default–biased-capture adjusting unbiased capture and group;
4. pooled partial Default–biased-capture adjusting measured biased priming and group;
5. SS Default–biased-capture;
6. SS Default–biased-capture partialling unbiased capture;
7. DS Default–biased-capture;
8. DS Default–biased-capture partialling unbiased capture.

The former **group-only partial** is excluded: it estimates the same group-centered primary association and is not a distinct hypothesis. The former **planned-weight capture** is excluded: applying the SS 16/33 and DS 17/33 participant-group proportions as within-person biased/unbiased condition weights is not a valid within-participant estimand. It is not retained as a tenth member to preserve an old denominator.

## Separate declared families/scopes

- **Williams comparisons:** Holm-2 across full biased-vs-unbiased and count-matched-biased-vs-unbiased dependent-correlation comparisons (Williams approximate df = 29).
- **Physical Yeo-7 network comparator audit:** Holm-7 only within the seven physical network comparator rows. It is emitted as `AUDIT_ONLY_NOT_TABLE7`; it does not claim to be the parent-owned Table 7 family.
- **Common-coverage whole-edge enrichment:** hypergeometric O/E rows are descriptive conditional counts only, not independent-edge enrichment inference. The separate within-SS/DS behavioral-label permutation suite rebuilds the 79,401-edge raw screen per draw and uses one-sided selected-count-surplus statistics, Holm-56 across 28 pair categories × positive/negative sign and Holm-14 across seven overlapping endpoint categories × sign.
- **Preprocessing variants:** Holm-5 across the five stated preprocessing variants. The mean-FD-adjusted model is explicitly descriptive/pointwise.
- **Conditional resampling:** direct distributions for matched biased minus fixed unbiased, and matched biased minus bootstrap-unbiased, have percentile intervals only. No fraction of nominal draw-level significance is interpreted as a p value. Any archived nominal-draw fraction is exported only in the file name-marked audit output.
- **Subject-level history coefficients / S30 audit:** descriptive diagnostic correlations only; no confirmatory family is claimed. The joint 144-column history LMM remains the inferential history model.

## Reconciliation required in parent submission materials

Any old “Holm-10” language or table must be changed to the declared Holm-8 family above, or retained solely as historical audit with explicit exclusion from fresh empirical interpretation. The analysis contract's instruction to preserve 16/33 and 17/33 cell weights applies to the specified group-weighted contrasts, not to a fabricated weighted within-person behavioral capture score.
