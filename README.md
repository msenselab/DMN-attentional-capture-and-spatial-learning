# DMN, attentional capture and spatial learning

Analysis-code release for the task-fMRI study of default-mode-network connectivity, attentional capture and spatial learning.

## Scientific scope

- The main neural measure is atlas-defined mean within-DMN connectivity. The current common-coverage extraction retains 399 parcels, including 91 Default-network parcels; earlier atlas-column assignments are not the primary analysis.
- Behavioral measures distinguish capture by the dimension assigned a spatial bias (orientation in SS; color in DS), capture by the alternative dimension, and spatial learning.
- The main connectivity sample contains 33 participants. Full scanner behavioral summaries use 35 participants where specified; these are not interchangeable samples.
- Primary participant-level associations, dependent-correlation comparisons, trial-level mixed models and reported robustness analyses are distinct analysis families.
- Whole-brain selected-edge counts and selected/expected ratios are **descriptive**. They are not independent-edge inferential evidence.
- DefaultA/B/C connectome-based analyses are **exploratory**. Inferential LOOCV values require complete nested within-group permutation, not ordinary Pearson p-values on pooled held-out predictions.

The release retains the current analysis producers and required source helpers, including explicitly reported Supplement sensitivities. It does not expose the private research repository or its history.

## Repository layout

- `code_source/`: selected scientific scripts and required text specifications, retaining their original study-relative paths so that imports and producer/consumer paths remain traceable. Names under `manuscript/` identify the original analysis workspace; manuscript drafts are **not** included.
- `analysis_catalog.json`: entry points and their analysis roles.
- `release_manifest.json`: source-to-release mapping, original and released SHA-256 hashes, and portability-change records.
- `docs/`: execution order, external input requirements and dependency notes. Start with [Reproduction](docs/REPRODUCTION.md), [Source index](docs/SOURCE_INDEX.md), and [Validation](docs/VALIDATION.md).
- `run_analysis.py`: copies the released source into an **external** data/results workspace and runs a chosen entry point.
- `tests/`: source-integrity and explicitly synthetic algorithm checks; no empirical participant data are bundled.

## Data and reproduction

Source data: **https://doi.org/10.12751/g-node.zfrem9**. Consult the dataset's own access and licensing information. This repository links to the data rather than redistributing them.

**This is a code-only release, not a preprocessed-data distribution.** MRI derivatives, atlas resources, analysis-ready trial tables, participant matrices, saved mixed-model fits and resampling arrays are external inputs. The data DOI is not a claim that every intermediate file required by a postprocessing script is deposited there. See `docs/REPRODUCTION.md` and the catalog before starting a job. Some postprocessing entry points deliberately require an earlier fitting/extraction step and retain exact frozen-input checks.

Use Python and R with the dependencies in `requirements.txt` and `docs/ENVIRONMENT.md`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py --list
python tests/check_release.py
```

Prepare an external workspace (choose a new directory, not this checkout):

```bash
python run_analysis.py --workspace /path/to/dmn-workspace --prepare-only
```

Place the documented external inputs at their study-relative paths inside that workspace before empirical execution. The following tested entry point instead runs an explicitly synthetic algorithm/atlas check (not an empirical result):

```bash
python run_analysis.py --workspace /path/to/dmn-workspace --analysis nested_cpm_capture -- --self-test
```

The launcher exports `DMN_PROJECT_ROOT` and does not overwrite differing source files in an existing workspace. Published scripts replace the private machine's study-root defaults with this configurable root; formulas, contrasts, exclusions, random seeds and correction families are retained. Generated outputs stay in the external workspace. Nested permutation analyses are substantial compute jobs and are not automatically launched by the source tests.

## Verification and limits

`docs/VALIDATION.md` records the checks actually executed for this code release. Syntax/source checks and synthetic tests are not a substitute for reproducing empirical results with the required inputs. No empirical result is inferred from the synthetic checks.

## Citation and licensing

Please cite the accompanying study and the source dataset when using this pipeline. `CITATION.cff` identifies this code repository; it does not invent a publication DOI for the manuscript.

A new reuse license has **not** been assigned by this publication. Copyright remains with the respective source authors; any existing source-file license notices remain applicable. Public visibility alone does not grant a new software reuse license. Third-party Python/R packages and atlas resources retain their own licenses.
