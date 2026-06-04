"""
A3.2 - Late-fusion stacked classifier.
Meta-learner: logistic regression on per-fold out-of-fold predicted probabilities
from the three per-modality random forests (A3.1).
Reports: stacked AUC (mean +/- SD across repeats x folds), Brier, calibration table.
Outputs:
  analysis/03_late/stacked_perf.csv
  analysis/03_late/stacked_oof.parquet
  analysis/03_late/calibration_bins.csv
  analysis/figures/stacked_roc.png
  analysis/figures/stacked_calibration.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "analysis" / "03_late"
FIG  = ROOT / "analysis" / "figures"

MODALITIES = ["metabolomics", "biochem", "diet"]

# Load per-modality OOF prob tables and pivot to wide format keyed by (Sample, repeat, fold)
dfs = []
for m in MODALITIES:
    d = pd.read_parquet(OUT / f"oof_probs_{m}.parquet")
    d = d.rename(columns={"p_crc": f"p_{m}"})
    dfs.append(d)

base = dfs[0][["Sample","y","repeat","fold"]].copy()
wide = base.merge(dfs[0][["Sample","repeat","fold","p_metabolomics"]],
                  on=["Sample","repeat","fold"], how="left")
wide = wide.merge(dfs[1][["Sample","repeat","fold","p_biochem"]],
                  on=["Sample","repeat","fold"], how="left")
wide = wide.merge(dfs[2][["Sample","repeat","fold","p_diet"]],
                  on=["Sample","repeat","fold"], how="left")

assert wide[["p_metabolomics","p_biochem","p_diet"]].isna().sum().sum() == 0, \
    "Per-modality OOF predictions misaligned across CV partitions; check seeds."

print(f"[A3.2] stacked-input rows: {len(wide)} across {wide['repeat'].nunique()} repeats")

# Stack: refit the meta-learner using leave-one-fold-out within each repeat to keep it honest.
records = []
roc_curves = []
for rep in sorted(wide["repeat"].unique()):
    rep_df = wide[wide["repeat"] == rep].copy()
    fold_aucs = []
    fold_briers = []
    for fold in sorted(rep_df["fold"].unique()):
        tr = rep_df[rep_df["fold"] != fold]
        te = rep_df[rep_df["fold"] == fold]
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(tr[["p_metabolomics","p_biochem","p_diet"]].to_numpy(), tr["y"].to_numpy())
        p = meta.predict_proba(te[["p_metabolomics","p_biochem","p_diet"]].to_numpy())[:, 1]
        auc = roc_auc_score(te["y"], p)
        brier = brier_score_loss(te["y"], p)
        fold_aucs.append(auc); fold_briers.append(brier)
        for sid, yt, pp in zip(te["Sample"].to_numpy(), te["y"].to_numpy(), p):
            records.append({"Sample": int(sid), "y": int(yt), "p_stack": float(pp),
                            "repeat": int(rep), "fold": int(fold)})
    # Aggregate ROC for last repeat for the plot
    if rep == sorted(wide["repeat"].unique())[-1]:
        # full-repeat OOF curve
        rep_records = [r for r in records if r["repeat"] == int(rep)]
        rr = pd.DataFrame(rep_records)
        fpr, tpr, _ = roc_curve(rr["y"], rr["p_stack"])
        roc_curves.append((rep, fpr, tpr, np.mean(fold_aucs)))

stack_oof = pd.DataFrame(records)
stack_oof.to_parquet(OUT / "stacked_oof.parquet", index=False)

# Final stacked metrics: average per (repeat, fold)
agg = stack_oof.groupby(["repeat","fold"]).apply(
    lambda g: pd.Series({
        "auc": roc_auc_score(g["y"], g["p_stack"]),
        "brier": brier_score_loss(g["y"], g["p_stack"]),
    })
).reset_index()

# Also: per-modality "best single" AUC (overall pooled) for comparison
per_mod_pooled = {}
for m in MODALITIES:
    d = pd.read_parquet(OUT / f"oof_probs_{m}.parquet")
    aucs = []
    for (rep, fold), g in d.groupby(["repeat","fold"]):
        aucs.append(roc_auc_score(g["y"], g["p_crc"]))
    per_mod_pooled[m] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1))}

summary = pd.DataFrame([
    {"model": "RF metabolomics", **per_mod_pooled["metabolomics"]},
    {"model": "RF biochem",       **per_mod_pooled["biochem"]},
    {"model": "RF diet",          **per_mod_pooled["diet"]},
    {"model": "Stacked LR (3 mods)",
     "auc_mean": float(agg["auc"].mean()),
     "auc_sd":   float(agg["auc"].std(ddof=1))},
])
summary["brier_mean"] = [
    np.nan, np.nan, np.nan, float(agg["brier"].mean())
]
summary.to_csv(OUT / "stacked_perf.csv", index=False)
print("[A3.2] perf summary:")
print(summary.to_string(index=False))

# Calibration bins on stacked OOF (10 quantile bins)
stack_oof["bin"] = pd.qcut(stack_oof["p_stack"], q=10, labels=False, duplicates="drop")
cal = stack_oof.groupby("bin").agg(
    n=("y","size"),
    mean_pred=("p_stack","mean"),
    obs_rate=("y","mean"),
).reset_index()
cal.to_csv(OUT / "calibration_bins.csv", index=False)

# Plots
fig, ax = plt.subplots(figsize=(5,5))
for rep, fpr, tpr, mauc in roc_curves:
    ax.plot(fpr, tpr, label=f"Stacked (rep {rep}) AUC={mauc:.3f}")
ax.plot([0,1],[0,1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Stacked classifier ROC (pooled OOF)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stacked_roc.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(cal["mean_pred"], cal["obs_rate"], "o-", label="Stacked")
ax.plot([0,1],[0,1], "k--", alpha=0.4)
ax.set_xlabel("Mean predicted P(CRC)"); ax.set_ylabel("Observed CRC rate")
ax.set_title("Calibration (10 quantile bins)")
ax.legend(); fig.tight_layout()
fig.savefig(FIG / "stacked_calibration.png", dpi=200); plt.close(fig)

print("[A3.2] done.")
