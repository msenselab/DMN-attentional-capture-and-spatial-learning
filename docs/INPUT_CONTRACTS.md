# Required external frozen inputs and common399 route

The source code is public; study configuration, participant data and MRI derivatives are not bundled. A bare clone cannot execute empirical extraction/CPM until these authorized external files are provided.

## Frozen configuration files

At the external workspace prefix `manuscript/paper2_dmn-connectivity/revision-1/R1/review_round2/resolved_reanalysis/`, provide:

- `analysis_contract.json`
- `analysis_contract_coverage_amendment.json`

These are **external frozen study-configuration inputs**, not outputs of a fitting script and not included in this code-only release. Obtain them with the authorized study-input handoff; the data DOI alone does not supply this internal configuration. The amendment binds the parent contract hash. Do not invent contracts or bypass the checks. Retargeting authorized configuration/provenance to a new workspace must preserve scientific settings and bind the new file hashes, invalidating incompatible old checkpoints.

## Coverage provenance producer

`connectivity_coverage_audit` is the retained, geometry-only source that produces `coverage_qc/coverage_audit.json` at that same prefix, using the approved N33 membership and all 66 fMRIPrep run masks. It does not inspect outcome associations. Its report contains participant/run provenance and stays outside Git. Alternatively provide the exact authorized frozen coverage-QC report at this external path.

## Active path (not the obsolete 400-parcel finisher)

1. Provide authorized cohort, behavior, atlas and fMRIPrep inputs plus the frozen contracts.
2. Run `connectivity_coverage_audit`, `connectivity_extract`, then `connectivity_verify`.
3. Run `connectivity_handoff` to validate the immutable common399 extraction and produce `common399_analysis_ready/physical_yeo7_common399_parcels.csv`.
4. `nested_cpm_capture` uses **that adapter metadata as its default**. Scores, matrices and matrix-subject order still come from `extraction_common_coverage/`; the adapter does not duplicate them.

No 400-parcel finisher is required by this active route. Native label 323 is excluded uniformly, local matrix indices are 0–398 and Yeo-7 Default contains 91 parcels.

With those real external files installed, the actual source can verify this route without empirical permutations:

```bash
python run_analysis.py --workspace /absolute/path/to/dmn-workspace --analysis nested_cpm_capture -- --check-inputs-only
```

This mode validates real input hashes, N33/10995-trial alignment, matrix symmetry/diagonal/finiteness and physical common399/7-to-17 crosswalk geometry. It reports **no empirical CPM inference**. The subsequent full run still requires 10,000 full nested within-group permutations and the original declared seed/correction families.
