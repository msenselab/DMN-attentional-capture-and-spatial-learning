# Release checks

This is a **code-only source publication**, not a claim of full raw-data-to-manuscript reproduction. Participant data, MRI derivatives, saved empirical fits and permutation archives were not uploaded.

## Checks executed before the first upload

- Positive source allowlist and SHA-256 verification: **80 source/specification files**: 50 Python, 14 R, 4 MATLAB task-provenance references and 12 Markdown specifications/plans.
- Python compilation and R parsing: all 50 scientific Python sources and all 14 R sources passed; the launcher/checker also compiled. MATLAB task execution was not tested.
- Scientific source equivalence: all 50 Python algorithm ASTs and all 14 parsed R expression trees matched their study originals after the declared I/O-path normalization. The one pre-existing moved-N35 script-root defect was repaired explicitly; formulas, numerical constants, exclusions, seeds and correction families were not altered.
- Synthetic nested CPM: the actual released vectorized implementation agreed with its independent scalar training-fold implementation; training-only group centering passed. The actual launcher/workspace `--self-test` also passed the physical 7/17-network crosswalk: nominal 400, common 399, omitted native label 323, Default91 and complementary DefaultABC79.
- **Narrow empirical spot check:** the released behavioral producer regenerated the exact 10,995-row N33 trial CSV byte-identically from existing inputs. The released association routines reproduced all nine current participant-association rows, including raw/Holm p-values and intervals, within absolute tolerance **1e-12** against the selected optimizer-resolved saved outputs. This did not refit models or repeat empirical permutations.
- **Figure-source spot check:** the released plotting implementations regenerated current Figure 4 and Supplement Figures S2/S3 from existing saved inputs in an external workspace. Their PNGs matched the approved saved PNGs byte-for-byte and pixel-for-pixel. Inputs and generated images remain outside the public repository.
- Gated CLI discovery succeeded for extraction, participant inference, nested capture CPM, capture/learning network screens and accepted Figure 3 plotting.
- Private machine-path/credential patterns and forbidden data/document extensions were checked on the allowed source tree. The public repository starts with a new root history; the private project's history is not imported.

## Boundaries

These checks do **not** establish a fresh rerun of every mixed model, every 10,000-permutation family, MRI preprocessing/extraction or all figures. Saved-RDS/readback entry points require compatible fits/versions and their documented external inputs. Full empirical reproduction is conditional on those inputs, not on a passing syntax or synthetic check.

Current live S1 is a preserved parent-managed CPM bitmap rather than a pixel-identical copy of the analytical verifier plot. This release includes the analytical plotting implementation and documents that distinction.

The committed archive is independently checked before push using `python tests/check_release.py`, workspace preparation and the actual nested-CPM synthetic self-test. Remote publication is verified against the final commit and file allowlist separately; local testing alone does not establish upload success.
