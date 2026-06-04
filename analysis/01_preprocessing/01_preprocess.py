"""
A1.2 — Preprocessing for Metabolomics + Blood Biochem + Diet (n = 4596).

Steps:
  1. Load, clean column names, inner-join on Sample.
  2. Recode UK Biobank sentinels (-3, -1) in Diet to NaN.
  3. Imputation:
     - Biochem: median per class (Control / Colon Cancer) per feature.
     - Diet: median (overall) — small missingness, ordinal codes.
     - Metabolomics: KNN (k = 10) preserves correlation structure of NMR lipid subfractions.
  4. Outliers: Tukey 1.5xIQR flag + Mahalanobis (chi^2 p<0.001) flag. Flagged, NOT dropped.
  5. Distributions: Shapiro-Wilk on subsample, log1p transform right-skewed metabolites (|skew|>1).
  6. Z-scale all numeric features (StandardScaler).
  7. Save .parquet matrices per modality + a combined long-form .parquet for downstream R / Python.

Seed: 2026.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = ROOT / "analysis" / "01_preprocessing"
EDA  = ROOT / "analysis" / "00_eda"
OUT.mkdir(parents=True, exist_ok=True)


def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    return df


def short_name(col, suffix_tag):
    """Strip the trailing '-Modality  .0.0' UKBB-style suffix; keep human-readable core."""
    base = col.split("-")[0].strip()
    return f"{base}__{suffix_tag}"


metab  = clean_cols(pd.read_excel(DATA / "Metabolomics.xlsx"))
biochem = clean_cols(pd.read_excel(DATA / "Blood Bio Chemistry.xlsx"))
diet   = clean_cols(pd.read_excel(DATA / "Diet.xlsx"))

# Standardise label
for df in (metab, biochem, diet):
    df["casecontrol"] = df["casecontrol"].astype(str).str.strip()

# Rename features with modality suffix
metab_feats   = [c for c in metab.columns   if c not in ("Sample", "casecontrol")]
biochem_feats = [c for c in biochem.columns if c not in ("Sample", "casecontrol")]
diet_feats    = [c for c in diet.columns    if c not in ("Sample", "casecontrol")]

metab   = metab.rename(  columns={c: short_name(c, "met")  for c in metab_feats})
biochem = biochem.rename(columns={c: short_name(c, "bio")  for c in biochem_feats})
diet    = diet.rename(   columns={c: short_name(c, "diet") for c in diet_feats})

metab_feats   = [c for c in metab.columns   if c not in ("Sample", "casecontrol")]
biochem_feats = [c for c in biochem.columns if c not in ("Sample", "casecontrol")]
diet_feats    = [c for c in diet.columns    if c not in ("Sample", "casecontrol")]

# ---- Diet sentinel recoding ----
n_recoded = 0
for c in diet_feats:
    mask = diet[c].isin([-3, -1])
    n_recoded += int(mask.sum())
    diet.loc[mask, c] = np.nan
print(f"[A1.2] Diet UKBB sentinels recoded to NaN: {n_recoded} cells")

# ---- Inner-join ----
joined = metab.merge(
    biochem.drop(columns=["casecontrol"]), on="Sample", how="inner"
).merge(
    diet.drop(columns=["casecontrol"]), on="Sample", how="inner"
)
assert joined.shape[0] == 4596, f"Expected 4596 joined rows, got {joined.shape[0]}"
print(f"[A1.2] joined shape: {joined.shape}")

label = joined["casecontrol"].copy()
y = (label == "Colon Cancer").astype(int).to_numpy()

# ---- Imputation ----
# Biochem: median per class per feature
biochem_block = joined[biochem_feats].copy()
for c in biochem_feats:
    for cls in ("Control", "Colon Cancer"):
        m = (label == cls)
        med = biochem_block.loc[m, c].median()
        biochem_block.loc[m & biochem_block[c].isna(), c] = med

# Diet: overall median
diet_block = joined[diet_feats].copy()
diet_block = diet_block.fillna(diet_block.median(numeric_only=True))

# Metabolomics: KNN (k=10)
metab_block = joined[metab_feats].copy()
if metab_block.isna().any().any():
    knn = KNNImputer(n_neighbors=10, weights="distance")
    metab_block = pd.DataFrame(
        knn.fit_transform(metab_block.to_numpy()),
        columns=metab_feats,
        index=metab_block.index,
    )

# Sanity: no NaN remaining
for name, blk in (("metab", metab_block), ("biochem", biochem_block), ("diet", diet_block)):
    assert blk.isna().sum().sum() == 0, f"{name} still has NaN"
print("[A1.2] imputation complete; no NaN remain.")

# ---- Outliers ----
def tukey_flags(df):
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags[c] = (df[c] < lo) | (df[c] > hi)
    return flags

tukey_metab   = tukey_flags(metab_block)
tukey_biochem = tukey_flags(biochem_block)

def mahalanobis_flag(df, alpha=0.001):
    X = df.to_numpy()
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    diff = X - mu
    md2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    cutoff = stats.chi2.ppf(1 - alpha, df=X.shape[1])
    return md2 > cutoff, md2

mh_metab_flag,   md2_metab   = mahalanobis_flag(metab_block)
mh_biochem_flag, md2_biochem = mahalanobis_flag(biochem_block)

outlier_summary = {
    "n_samples": int(joined.shape[0]),
    "tukey_any_flag_pct": {
        "metab":   float(tukey_metab.any(axis=1).mean() * 100),
        "biochem": float(tukey_biochem.any(axis=1).mean() * 100),
    },
    "mahalanobis_flag_pct_alpha_0.001": {
        "metab":   float(mh_metab_flag.mean() * 100),
        "biochem": float(mh_biochem_flag.mean() * 100),
    },
    "policy": "flag-only; no rows dropped (samples retained per analysis plan)",
}
with open(OUT / "outlier_summary.json", "w") as fp:
    json.dump(outlier_summary, fp, indent=2)
print(f"[A1.2] outlier summary: {outlier_summary}")

# ---- Distributions & log transform for metabolites ----
# Use skewness instead of Shapiro (Shapiro on n=4596 will reject everything;
# we want a practical right-skew filter).
skew_metab = metab_block.skew()
log_targets = skew_metab[skew_metab > 1.0].index.tolist()
# All metab values >= 0 in principle, but guard:
metab_logged = metab_block.copy()
for c in log_targets:
    if (metab_block[c] < 0).any():
        # shift then log1p to keep finite
        shift = -metab_block[c].min() + 1e-6
        metab_logged[c] = np.log1p(metab_block[c] + shift)
    else:
        metab_logged[c] = np.log1p(metab_block[c])
print(f"[A1.2] log1p applied to {len(log_targets)}/{len(metab_feats)} metabolite features (skew>1).")

# ---- Z-scale all numeric features ----
def zscale(df):
    sc = StandardScaler()
    out = pd.DataFrame(sc.fit_transform(df.to_numpy()), index=df.index, columns=df.columns)
    return out, {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}

metab_z,   metab_sc   = zscale(metab_logged)
biochem_z, biochem_sc = zscale(biochem_block)
diet_z,    diet_sc    = zscale(diet_block)

# ---- Save ----
def save_block(block_z, name, raw_block, log_targets=None):
    out = block_z.copy()
    out.insert(0, "Sample", joined["Sample"].to_numpy())
    out.insert(1, "casecontrol", label.to_numpy())
    out.insert(2, "y", y)
    out.to_parquet(OUT / f"{name}_z.parquet", index=False)
    raw_out = raw_block.copy()
    raw_out.insert(0, "Sample", joined["Sample"].to_numpy())
    raw_out.insert(1, "casecontrol", label.to_numpy())
    raw_out.to_parquet(OUT / f"{name}_imputed_raw.parquet", index=False)

save_block(metab_z,   "metabolomics", metab_logged)
save_block(biochem_z, "biochem",      biochem_block)
save_block(diet_z,    "diet",         diet_block)

# Save outlier flag indicators alongside (compact)
flag_df = pd.DataFrame({
    "Sample": joined["Sample"].to_numpy(),
    "mahalanobis_metab":   mh_metab_flag,
    "mahalanobis_biochem": mh_biochem_flag,
    "md2_metab":   md2_metab,
    "md2_biochem": md2_biochem,
    "tukey_metab_any":   tukey_metab.any(axis=1).to_numpy(),
    "tukey_biochem_any": tukey_biochem.any(axis=1).to_numpy(),
})
flag_df.to_parquet(OUT / "outlier_flags.parquet", index=False)

# Persist metadata
meta = {
    "seed": SEED,
    "n_samples": int(joined.shape[0]),
    "label_col": "casecontrol",
    "positive_class": "Colon Cancer",
    "n_features": {
        "metab": len(metab_feats),
        "biochem": len(biochem_feats),
        "diet": len(diet_feats),
    },
    "metabolite_log_transformed": log_targets,
    "diet_sentinels_recoded": n_recoded,
    "imputation": {
        "metab": "KNN (k=10, distance-weighted)",
        "biochem": "median per class per feature",
        "diet": "median overall (post UKBB sentinel recoding)",
    },
    "scaling": "z-score (StandardScaler) per feature after log transform",
    "outlier_policy": "flag-only; Tukey 1.5xIQR + Mahalanobis chi^2 p<0.001",
}
with open(OUT / "preprocess_meta.json", "w") as fp:
    json.dump(meta, fp, indent=2)

print("[A1.2] done. Wrote parquet files + metadata to analysis/01_preprocessing/")
