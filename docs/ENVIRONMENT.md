# Environment

Source-release checks used Python **3.13.11** and R **4.5.2**. Python packages are recorded in `requirements.txt`; this is the environment used for the release checks, not a claim that every historical fit was made with these versions.

R packages present during checking:

- `lme4` 1.1-38
- `lmerTest` 3.2.0
- `jsonlite` 2.0.0
- `digest` 0.6.39
- `Matrix` 1.7-4

Install the core R dependencies, using a compatible R version:

```r
install.packages(c("lme4", "lmerTest", "jsonlite", "digest"))
```

Retained saved-RDS readers are version-sensitive. Reusing saved `lmer` objects across package versions is not equivalent to refitting or selecting a new optimizer. Keep the selected formula, REML setting, convergence/singularity checks and exact correction families.

Linux tools: `Rscript`, `bash` and `sha256sum`. The four MATLAB task-source files are provenance references read by the learning audit; they are not statistical entry points, and MATLAB task execution is not part of the source-release checks.

MRI extraction also requires the applicable fMRIPrep derivatives and Schaefer-2018 400-parcel Yeo-7/Yeo-17 atlas resources. Set `DMN_ATLAS_CACHE` to a Nilearn cache directory if needed; the default is `~/nilearn_data`. The atlas's own license applies. No MRI derivative, atlas image or package environment is bundled here.

The original `lmm_behavioral_original.py` provenance module imports `pingouin`; current nested CPM checks do not execute that module.
