# Reproduction guide

## Start with an external workspace

The code keeps the original study-relative file layout to preserve real producer/consumer dependencies. The launcher copies **only the released source** into a separate directory. This does not fetch, fabricate or publish empirical data.

```bash
W=/absolute/path/to/dmn-workspace
python run_analysis.py --workspace "$W" --prepare-only
python run_analysis.py --list
```

`DMN_PROJECT_ROOT` is exported by the launcher. Scripts with no output argument write beside their copied source inside `$W`, not inside the Git checkout. Use a fresh workspace and preserve the original inputs. Do not override input hashes, convergence flags, fixed cohort checks, or geometry assertions to make a job run.

## Required external inputs

The source-data DOI is https://doi.org/10.12751/g-node.zfrem9. The following are **not bundled**; that DOI does not establish availability of every derived intermediate:

- Raw BIDS scanner event tables, original MATLAB behavioral trials and the BIDS-to-original mapping in `participants.tsv` for correctness linkage and training/scanner reconstruction.
- `derivatives/behavior/vigilance_state/merged_events_with_correct.csv`: correctness-linked, original-chronology scanner events.
- `manuscript/paper2/code_and_data/dmn_capture_unified.csv`: the frozen N33 cohort/legacy-score sidecar consumed by the original trial-table producer. Its legacy connectivity/X fields are **not** the current neural predictor.
- fMRIPrep task BOLD/confound derivatives for the authorized 33 participants and two runs per participant, plus Schaefer-2018 physical Yeo-7/Yeo-17 atlas resources.
- Frozen scientific metadata `resolved_reanalysis/analysis_contract.json`, `analysis_contract_coverage_amendment.json`, coverage-QC/acceptance manifests and the optimizer-recovery execution plan. These contain data/source bindings and are external metadata, not included private snapshots.
- Corrected common399 matrices, their participant order, physical parcel metadata and resolved connectivity-score tables, or the inputs required to generate them.
- Frozen count-matching/resampling arrays and source tables for dependent inference, or the retained RNG checkpoint producer and its required inputs.
- Saved model frames/RDS fits and derived CSV/JSON outputs for entry points explicitly labeled **readback**, **saved-model recoding**, **optimizer integration** or **figure generation**. Those entry points are not independent fitting pipelines.
- The accepted baseline Figure 2 bitmap and acceptance/readback metadata for exact presentation-replay checks. These are external figure inputs, not manuscript files shipped with this repository.

Paths in each script are relative to `$W` or its copied source directory. The original revision-prefix is `manuscript/paper2_dmn-connectivity/revision-1/`; current analysis families are primarily under its `R1/review_round2/`. Consult the actual read statements and `analysis_catalog.json` for complete per-entry input paths. Input JSON that contains absolute paths must be generated/bound to the new workspace through the relevant producer/adapter; copied private-machine checkpoint metadata cannot simply be treated as a fresh run.

## Required external contracts and common399 route

See [INPUT_CONTRACTS.md](INPUT_CONTRACTS.md) for the exact two external frozen JSON paths, the retained all66-mask coverage-QC producer, and the current common399 adapter → CPM metadata route. These are required external inputs, not bundled or fabricated scientific configuration. The CPM `--check-inputs-only` mode validates that real source chain without permutations.

## Execution order and distinct families

1. **Behavioral linkage and tables:** `behavior_link_correctness` joins scanner events to MATLAB correctness; `behavior_prepare_n33` produces the frozen correct/finite 200–3000-ms N35/N33 trial tables. `behavior_n35_refresh` fits the separate training/scanner behavioral models and generates their tables. It does not supply the current neural predictor.
2. **Connectivity:** `connectivity_extract`, then `connectivity_verify` and `connectivity_handoff`, using the frozen contract/amendment. Common399 excludes native parcel 323 uniformly; matrix indices 0–398 are not native atlas IDs. Primary Yeo-7 Default includes 91 parcels. The complementary Yeo-17 DefaultA/B/C set contains 79 parcels and is not an identical partition of Default91.
3. **Participant inference and parent joint-model recovery:** `participant_inference` generates associations/resampling and invokes the paired R fitter. `capture_joint_optimizer_recovery` retains the actual same-formula recovery; `capture_joint_integrate` re-exports the selected saved optimum and runs readback checks. Integration is not a refit. Keep the selected, optimizer-resolved v2 result distinct from the earlier diagnostic-flagged directory.
4. **Main hierarchical capture:** `capture_standardized_lmm`, then `capture_standardized_verify`. Equal-region contrasts differ from participant correlations and do not equalize trial precision.
5. **Main hierarchical learning:** `learning_current_lmm` is the principal **run-moderated** model, despite its historical `run_moderation_sensitivity_*` directory name. Its fitting script requires an output-directory argument. `learning_archived_lmm` supplies the separate archived run-invariant fit/frame prerequisite; the current source defines it as archived but no longer reports a separate archived-learning result table. It is not a current principal result or independent replication.
6. **Participant and physical-dimension companions:** `learning_participant`, `capture_equal_region_participant`, physical-score checks/covariance decomposition, same-dimension hierarchical contrasts and reliability readers. The archived physical-model recoding readers and removed physical-score-superiority result panels are excluded. Some retained covariance producer routines also export historical comparison products; those extra products are not current manuscript claims. Preserve each retained declared correction family rather than pooling results across entry points.
7. **Network context:** `network_context_capture` and `network_context_learning` provide the 28 network-pair association screens and descriptive raw signed-edge counts. Network-summary associations are participant-level tests; counts/selected-to-expected ratios are descriptive. The capture runner also exports historical hypergeometric audit columns; these are **not valid participant-level inferential evidence**. The separate dependency-preserving permutation code is retained for the reported category tests, including the same-algorithm archived-null replay, not as a new independent analysis.
8. **Exploratory CPM:** `nested_cpm_capture`; `learning_network_prepare`, then `nested_cpm_learning`; and the separately labeled intact-network/supportive/specificity extensions. Inferential runs use 10,000 complete nested within-group permutations, training-fold centering and selection, with the declared tail definition and family. Ordinary pooled-LOOCV Pearson p-values are not inferential values. The older authorized CPM file is retained as the actually imported **algorithm library**, not exposed as a current standalone empirical runner.
9. **Figures:** `figure2`, `figure3`, `figure4`, `figure_s1_and_cpm_verify`, `figure_s2`, `figure_s3`. Historical filenames do not establish current numbering: S1 is exploratory CPM, S2 capture counts and S3 learning counts. The current live S1 is a preserved parent-managed image; the retained analytical verifier render is provenance-linked, not pixel-identical to that embedded image.

Original behavioral/task source references and the frozen count-matched RNG producer may contain historical output/reporting routines. Their required source/provenance roles are recorded in the manifest; do not substitute their legacy neural estimates or plots for current Default91/common399 results. Prose/live-document updaters, reviewer correspondence, earlier invalid ordinary-LOOCV reporting scripts and the halted atlas-misaligned refresh are not released.

## Tested CLI examples

A data-free algorithm/atlas crosswalk check, explicitly **not empirical evidence**:

```bash
python run_analysis.py --workspace "$W" --analysis nested_cpm_capture -- --self-test
```

Inspect a gated empirical interface without running a fit:

```bash
python run_analysis.py --workspace "$W" --analysis participant_inference -- --help
```

After providing the required real inputs, use the exact documented defaults or arguments, e.g. `--input-manifest`/`--output` for `participant_inference` and `--n-perm 10000` for `nested_cpm_capture`. Do not copy a partially completed or differently fingerprinted permutation checkpoint into a new run. Some readback scripts have no argument parser and execute immediately; `--help` is **not** a safe discovery method for those scripts.

## What this release does not claim

This publication makes the selected analysis implementations available. It does not assert an automatically runnable raw-data-to-submission pipeline from a bare clone, availability of every private intermediate at the data DOI, a fresh rerun of all model fits/permutations, or reproduction of every embedded bitmap. See `VALIDATION.md` for the checks actually executed.
