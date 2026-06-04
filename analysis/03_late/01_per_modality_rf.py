"""
A3.1 - Per-modality Random Forest with 5-fold stratified CV (3 repeats), seed=2026.
Saves out-of-fold predicted probabilities for downstream stacking.

For each modality (metabolomics, biochem, diet) using its z-scaled matrix:
  - RandomForestClassifier, n_estimators=1000, mtry tuned via small grid.
  - Stratified 5-fold x 3 repeats, recording per-fold AUC and OOF probabilities.

Outputs:
  analysis/03_late/oof_probs_<modality>.parquet      # Sample, y, p_crc per repeat
  analysis/03_late/rf_perf.csv                       # AUC mean+SD per modality
  analysis/03_late/rf_best_params.csv                # tuned mtry per modality
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
PRE  = ROOT / "analysis" / "01_preprocessing"
OUT  = ROOT / "analysis" / "03_late"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = ["metabolomics", "biochem", "diet"]
N_REPEATS = 3
N_FOLDS = 5

def load(name):
    df = pd.read_parquet(PRE / f"{name}_z.parquet")
    y = df["y"].to_numpy()
    X = df.drop(columns=["Sample", "casecontrol", "y"]).to_numpy()
    feat_names = [c for c in df.columns if c not in ("Sample","casecontrol","y")]
    sample = df["Sample"].to_numpy()
    return sample, X, y, feat_names

perf_rows = []
best_params = {}
for mod in MODALITIES:
    sample, X, y, feats = load(mod)
    p = X.shape[1]
    mtry_grid = sorted({max(1, int(round(np.sqrt(p)))),
                        max(2, int(round(p/3))),
                        max(2, int(round(p/2)))})
    # Inner CV on a single 80/20 split to keep cost down: simple search
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
        param_grid={"max_features": mtry_grid},
        scoring="roc_auc", cv=inner, n_jobs=1, refit=False,
    )
    grid.fit(X, y)
    best_mtry = grid.best_params_["max_features"]
    best_params[mod] = {"max_features": best_mtry, "grid": mtry_grid, "inner_cv_auc": float(grid.best_score_)}
    print(f"[A3.1] {mod}: tuned max_features={best_mtry} (grid={mtry_grid}) inner AUC={grid.best_score_:.4f}")

    # Outer 5-fold x 3-repeat for honest OOF probabilities
    oof_records = []
    fold_aucs = []
    for rep in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            rf = RandomForestClassifier(
                n_estimators=1000, max_features=best_mtry,
                random_state=SEED + rep*10 + fold_idx, n_jobs=-1
            )
            rf.fit(X[tr], y[tr])
            p_te = rf.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], p_te)
            fold_aucs.append(auc)
            for s_idx, sid, yt, pp in zip(te, sample[te], y[te], p_te):
                oof_records.append({"Sample": int(sid), "y": int(yt), "p_crc": float(pp),
                                    "repeat": rep, "fold": fold_idx})
    oof = pd.DataFrame(oof_records)
    oof.to_parquet(OUT / f"oof_probs_{mod}.parquet", index=False)

    perf_rows.append({
        "modality": mod,
        "n_features": p,
        "best_max_features": best_mtry,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_sd":   float(np.std(fold_aucs, ddof=1)),
        "n_folds_evaluated": len(fold_aucs),
    })
    print(f"[A3.1] {mod}: outer AUC = {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs, ddof=1):.4f}")

pd.DataFrame(perf_rows).to_csv(OUT / "rf_perf.csv", index=False)
pd.DataFrame([
    {"modality": m, **best_params[m]} for m in MODALITIES
]).to_csv(OUT / "rf_best_params.csv", index=False)
print("[A3.1] done.")
