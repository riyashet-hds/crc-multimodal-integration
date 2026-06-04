"""
TESTING ADDENDUM - 2-modality late-fusion stacked classifier.

Sensitivity check: does dropping the near-chance Diet block (RF AUC 0.512)
sharpen the late-fusion gain?

This mirrors analysis/03_late/02_stacked_classifier.py exactly - same meta-learner
(LogisticRegression, max_iter=2000, C=1.0), same leave-one-fold-out-within-repeat
honest stacking, same SEED=2026 - the ONLY change is the meta feature matrix:
    3-modality stack  ->  [p_metabolomics, p_biochem, p_diet]
    2-modality stack  ->  [p_metabolomics, p_biochem]              <-- this script

Pairing rationale: Metabolomics is the omic block; Blood Biochemistry is the
non-omic routine-clinical-chemistry block. This is the primary integration
pair (an omic + a non-omic modality).

Inputs  (reused, not re-derived):
  analysis/03_late/oof_probs_metabolomics.parquet
  analysis/03_late/oof_probs_biochem.parquet
  analysis/03_late/oof_probs_diet.parquet        (for the 3-way comparison print only)
Outputs (analysis/testing/):
  stacked_2mod_perf.csv
  stacked_2mod_oof.parquet
  stacked_2mod_roc.png
  stacked_2mod_calibration.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold  # noqa: F401  (kept for parity w/ source)
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
LATE = ROOT / "analysis" / "03_late"
OUT  = ROOT / "analysis" / "testing"
OUT.mkdir(parents=True, exist_ok=True)

# Meta-learner consumes only the omic + non-omic pair (diet dropped).
STACK_MODS = ["metabolomics", "biochem"]

# ---- Load per-modality OOF prob tables, pivot to wide keyed by (Sample, repeat, fold) ----
dfs = {}
for m in ["metabolomics", "biochem", "diet"]:
    d = pd.read_parquet(LATE / f"oof_probs_{m}.parquet")
    dfs[m] = d.rename(columns={"p_crc": f"p_{m}"})

base = dfs["metabolomics"][["Sample", "y", "repeat", "fold"]].copy()
wide = base.copy()
for m in ["metabolomics", "biochem", "diet"]:
    wide = wide.merge(dfs[m][["Sample", "repeat", "fold", f"p_{m}"]],
                      on=["Sample", "repeat", "fold"], how="left")

stack_cols = [f"p_{m}" for m in STACK_MODS]
assert wide[stack_cols].isna().sum().sum() == 0, \
    "Per-modality OOF predictions misaligned across CV partitions; check seeds."

print(f"[TEST] 2-modality stack inputs: {len(wide)} rows across "
      f"{wide['repeat'].nunique()} repeats; meta features = {stack_cols}")

# ---- Honest stacking: refit the LR meta-learner leave-one-fold-out within each repeat ----
records = []
roc_curves = []
for rep in sorted(wide["repeat"].unique()):
    rep_df = wide[wide["repeat"] == rep].copy()
    fold_aucs = []
    for fold in sorted(rep_df["fold"].unique()):
        tr = rep_df[rep_df["fold"] != fold]
        te = rep_df[rep_df["fold"] == fold]
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(tr[stack_cols].to_numpy(), tr["y"].to_numpy())
        p = meta.predict_proba(te[stack_cols].to_numpy())[:, 1]
        fold_aucs.append(roc_auc_score(te["y"], p))
        for sid, yt, pp in zip(te["Sample"].to_numpy(), te["y"].to_numpy(), p):
            records.append({"Sample": int(sid), "y": int(yt), "p_stack": float(pp),
                            "repeat": int(rep), "fold": int(fold)})
    if rep == sorted(wide["repeat"].unique())[-1]:
        rr = pd.DataFrame([r for r in records if r["repeat"] == int(rep)])
        fpr, tpr, _ = roc_curve(rr["y"], rr["p_stack"])
        roc_curves.append((rep, fpr, tpr, float(np.mean(fold_aucs))))

stack_oof = pd.DataFrame(records)
stack_oof.to_parquet(OUT / "stacked_2mod_oof.parquet", index=False)

# ---- Aggregate stacked metrics per (repeat, fold) ----
agg = stack_oof.groupby(["repeat", "fold"]).apply(
    lambda g: pd.Series({
        "auc":   roc_auc_score(g["y"], g["p_stack"]),
        "brier": brier_score_loss(g["y"], g["p_stack"]),
    })
).reset_index()

# ---- Per-modality pooled AUC (same recipe as the 3-mod script) ----
per_mod_pooled = {}
for m in ["metabolomics", "biochem", "diet"]:
    d = dfs[m]
    aucs = [roc_auc_score(g["y"], g[f"p_{m}"]) for _, g in d.groupby(["repeat", "fold"])]
    per_mod_pooled[m] = {"auc_mean": float(np.mean(aucs)),
                         "auc_sd":   float(np.std(aucs, ddof=1))}

summary = pd.DataFrame([
    {"model": "RF metabolomics",                 **per_mod_pooled["metabolomics"]},
    {"model": "RF biochem",                      **per_mod_pooled["biochem"]},
    {"model": "Stacked LR (2 mods: metab+biochem)",
     "auc_mean": float(agg["auc"].mean()),
     "auc_sd":   float(agg["auc"].std(ddof=1))},
])
summary["brier_mean"] = [np.nan, np.nan, float(agg["brier"].mean())]
summary.to_csv(OUT / "stacked_2mod_perf.csv", index=False)

# ---- Calibration bins (10 quantile bins) ----
stack_oof["bin"] = pd.qcut(stack_oof["p_stack"], q=10, labels=False, duplicates="drop")
cal = stack_oof.groupby("bin").agg(
    n=("y", "size"), mean_pred=("p_stack", "mean"), obs_rate=("y", "mean"),
).reset_index()

# ---- Plots ----
fig, ax = plt.subplots(figsize=(5, 5))
for rep, fpr, tpr, mauc in roc_curves:
    ax.plot(fpr, tpr, label=f"2-mod stack (rep {rep}) AUC={mauc:.3f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("2-modality stacked classifier ROC (metab + biochem)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "stacked_2mod_roc.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(cal["mean_pred"], cal["obs_rate"], "o-", label="2-mod stack")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("Mean predicted P(CRC)"); ax.set_ylabel("Observed CRC rate")
ax.set_title("Calibration (10 quantile bins)")
ax.legend(); fig.tight_layout()
fig.savefig(OUT / "stacked_2mod_calibration.png", dpi=200); plt.close(fig)

# ---- 3-way comparison print ----
biochem_alone = per_mod_pooled["biochem"]["auc_mean"]
stack_2mod    = float(agg["auc"].mean())
stack_2mod_sd = float(agg["auc"].std(ddof=1))
perf3 = pd.read_csv(LATE / "stacked_perf.csv")
stack_3mod = float(perf3.loc[perf3["model"] == "Stacked LR (3 mods)", "auc_mean"].iloc[0])

print("\n[TEST] perf summary (2-modality stack):")
print(summary.to_string(index=False))
print("\n[TEST] ===== 3-way comparison =====")
print(f"  biochem RF alone         AUC = {biochem_alone:.3f}")
print(f"  stacked LR (3 mods)      AUC = {stack_3mod:.3f}   "
      f"(gain over biochem: {stack_3mod - biochem_alone:+.3f})")
print(f"  stacked LR (2 mods)      AUC = {stack_2mod:.3f} +/- {stack_2mod_sd:.3f}   "
      f"(gain over biochem: {stack_2mod - biochem_alone:+.3f})")
print(f"  effect of dropping diet  : {stack_2mod - stack_3mod:+.3f} AUC")
print("[TEST] done.")
