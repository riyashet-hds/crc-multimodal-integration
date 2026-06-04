# Practical analysis pipeline

Reproducible integrative multimodal data analytics pipeline (colorectal cancer).
Primary integration: Metabolomics + Blood Biochemistry (n = 4596).
Secondary/sensitivity: Metabolomics + Diet.
Proteomics excluded (disjoint cohort, severe per-protein missingness).

Seed: 2026 throughout.

## Subfolders
- `00_eda/` — distributions, missingness audit, outlier flags.
- `01_preprocessing/` — cleaned, imputed, log/z-scaled matrices (`.parquet`).
- `02_intermediate/` — DIABLO (mixOmics, R) + sCCA outputs.
- `03_late/` — per-modality RF + Boruta + stacked classifier + igraph network.
- `04_compare/` — common vs distinct features (Venn + ranked tables).
- `05_xai/` — SHAP, permutation importance, surrogate tree.
- `06_clinical/` — marker → pathway → PMID mapping.
- `testing/` — sensitivity addenda (e.g. 2-modality metab+biochem stack; see `SUMMARY.md §1b`).
- `figures/` — final 300 dpi `.png`/`.pdf` result figures.
- `tables/` — `.csv` and `.xlsx` outputs.

Entry points are numbered Python scripts in each subfolder, run in order.
