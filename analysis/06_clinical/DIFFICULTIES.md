# Difficulties encountered

Honest list of issues that shape interpretation of the results.

## 1. Cohort mismatch — proteomics excluded from integration
Proteomics (n=518 rows; 509 unique Samples; 8 duplicate rows) shares **zero** Sample-IDs with Metabolomics / Biochem / Diet (n=4596). Integration requires sample-level alignment, so Proteomics was excluded from the multimodal pipeline. The disjoint cohort is a property of how the data were supplied, not an analyst choice.

## 2. Severe per-protein missingness in Proteomics
Even if alignment were available, several Olink proteins are essentially missing in this assay batch: GLIPR1 99.8%, NPM1 73.6%, PCOLCE 67.2%, CST1 32.2%, ENDOU 29.5%. Any naive imputation would invent signal. A separate proteomics-only exploratory arm would require a stringent missingness filter (<20-30%) followed by KNN/PMM imputation.

## 3. UK Biobank sentinel-code masking in Diet
Diet items use UKBB ordinal coding with `-3` ("Prefer not to answer") and `-1` ("Do not know") as in-range sentinels. **143 cells across 8 items** were recoded to NaN before median imputation. Failing to recode would have introduced a fictitious "negative intake" category that confounds downstream effect estimates.

## 4. Heavy collinearity in NMR lipid subfractions
The 50 NMR metabolites are not independent — many describe sub-particle composition of the same lipoprotein class. This is exactly why an **unsupervised** strategy (rCCA, MOFA-family) is informative: comp 1 alone captures the dominant HDL/ApoA-I axis with canonical correlation 0.884 between blocks. Justifies KNN-k=10 imputation (preserves correlation structure) over per-feature median.

## 5. Right-skewed metabolite distributions
12 of 50 NMR metabolites had skew > 1 after imputation (primarily triglyceride and large-VLDL features). `log1p` transformation was applied before z-scaling so linear-projection methods (PLS-DA, CCA) operate on approximately Gaussian inputs.

## 6. Tukey outlier flags are high on Biochem (33 %)
Long-tailed clinical features (Lipoprotein_A, ALT) inflate the IQR-based flag rate. **No samples were dropped.** Multivariate Mahalanobis (chi-squared p<0.001) flags only 2.3% of biochem samples and 7.6% of metab samples — a more defensible single-sample outlier rate. This is reported in `analysis/01_preprocessing/outlier_summary.json`.

## 7. Imputation introduces uncertainty in Blood Biochemistry
Lipoprotein_A (24% missing) and Direct_bilirubin (17.3%) are partially imputed by per-class median. Their importance ranks in the RF survive this perturbation but the *magnitude* of their effect size should be interpreted with care; a sensitivity analysis swapping the imputation strategy is a sensible follow-up.

## 8. Binary outcome only — no stage information
The supplied label is `Control` / `Colon Cancer` with no TNM stage, location (right vs left), MSI status, or KRAS mutation. The model speaks to discrimination only — not to clinically actionable subtyping. Any "biomarker discovery vs subtype identification" framing should reflect this.

## 9. Cross-sectional design — association only
There is no temporal information about whether biomarkers were measured before/after diagnosis. Therefore: no causal inference, no claims about predictive risk in healthy populations. This is the key caveat when contrasting with prospective studies such as Yao & Tian 2014 [DOI](https://doi.org/10.1007/s10552-014-0507-y).

## 10. Metabolomics and Diet are weak standalone predictors
RF AUC of 0.565 (metab) and 0.512 (diet) means the per-modality classifiers are barely above chance. Boruta confirmed 0/50 metabolite features and 0/8 diet features against shadow noise — strict null at the per-feature level. The integration narrative therefore pivots on *complementarity*: stacked LR AUC 0.827 vs biochem-alone 0.824 = **+0.004** AUC, a clinically tiny but statistically detectable gain. This is honest reporting; do not over-claim integration value. **Sensitivity follow-up** (`analysis/testing/01_stacked_2modality.py`): re-running the stack on only the omic + non-omic primary pair (metabolomics + biochemistry, dropping the inert Diet block) gives AUC 0.8274 — within −0.00005 of the 3-modality stack. Diet was not diluting the fusion; it was simply non-contributory, and the small gain is metabolomics refining the biochemistry decision either way.

## 11. DIABLO predictive performance is modest
DIABLO weighted-vote BER (comp 3) ~0.43 (~AUC 0.59) is well below the late-RF stack. Reason: DIABLO is a linear sparse PLS-DA constrained to 5 metab + 4 biochem features per component, while RF on full biochem (8 features) captures non-linear interactions and gets AUC 0.82. **DIABLO's value here is biomarker discovery (interpretable joint loadings), not prediction.** The biomarker-discovery vs subtype-identification distinction applies directly here.

## 12. Class label has trailing-whitespace inconsistency
The proteomics file labels CRC as `'Colon Cancer '` (trailing space) where the other three files use `'Colon Cancer'`. Stripping is required when concatenating any cross-modality label vectors.

## 13. Reproducibility
Seed = 2026 set in Python (numpy, sklearn) and R (set.seed). However, `RandomForestClassifier(n_jobs=-1)` permits non-deterministic micro-variation across runs; differences are well within 0.001 AUC and do not affect feature ranking conclusions.
