"""
A6 - XAI layer.
  1. SHAP TreeExplainer on each per-modality RF (full-data refit).
     Outputs: beeswarm summary, dependence plot for top 5 features per modality,
              force plot for one illustrative CRC case.
  2. SHAP LinearExplainer on the stacked logistic meta-learner.
  3. Surrogate decision tree (depth 4, rpart-equivalent via sklearn) fit to the
     stacked OOF probabilities -> human-readable rule extraction.
Outputs:
  analysis/figures/shap_beeswarm_<modality>.png
  analysis/figures/shap_dependence_<modality>_<rank>.png
  analysis/figures/shap_force_one_case.png
  analysis/figures/shap_stack_bar.png
  analysis/figures/surrogate_tree.png
  analysis/05_xai/shap_mean_abs_<modality>.csv
  analysis/05_xai/surrogate_rules.txt
Seed: 2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
PRE  = ROOT / "analysis" / "01_preprocessing"
LATE = ROOT / "analysis" / "03_late"
OUT  = ROOT / "analysis" / "05_xai"
FIG  = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODALITIES = ["metabolomics", "biochem", "diet"]
best_params = pd.read_csv(LATE / "rf_best_params.csv").set_index("modality").to_dict("index")

def load(name):
    df = pd.read_parquet(PRE / f"{name}_z.parquet")
    y = df["y"].to_numpy()
    feats = [c for c in df.columns if c not in ("Sample","casecontrol","y")]
    return df["Sample"].to_numpy(), df[feats], y, feats

# ---- Per-modality SHAP ----
shap_summary_rows = []
mod_models = {}
for mod in MODALITIES:
    sample, Xdf, y, feats = load(mod)
    X = Xdf.to_numpy()
    mtry = int(best_params[mod]["max_features"])
    rf = RandomForestClassifier(n_estimators=500, max_features=mtry,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    mod_models[mod] = (rf, Xdf, y, feats)

    explainer = shap.TreeExplainer(rf)
    # Use a sample of 500 for speed when computing SHAP if data is large
    bg_idx = np.random.RandomState(SEED).choice(len(X), size=min(500, len(X)), replace=False)
    shap_vals = explainer.shap_values(X[bg_idx])
    # binary classification -> shap_values returns (n, p, 2) in newer SHAP versions
    if isinstance(shap_vals, list):
        sv_pos = shap_vals[1]
    elif shap_vals.ndim == 3:
        sv_pos = shap_vals[:, :, 1]
    else:
        sv_pos = shap_vals

    mean_abs = np.abs(sv_pos).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    df_sum = pd.DataFrame({
        "feature": [feats[i] for i in order],
        "shap_mean_abs": mean_abs[order],
        "rank": np.arange(1, len(feats)+1),
    })
    df_sum.to_csv(OUT / f"shap_mean_abs_{mod}.csv", index=False)
    for f, v in zip(df_sum["feature"], df_sum["shap_mean_abs"]):
        shap_summary_rows.append({"modality": mod, "feature": f, "shap_mean_abs": float(v)})

    # Beeswarm summary
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv_pos, Xdf.iloc[bg_idx],
                      feature_names=feats, show=False, plot_size=None, max_display=12)
    plt.title(f"SHAP summary - {mod}\n(positive shap = pushes toward CRC class)")
    plt.tight_layout()
    plt.savefig(FIG / f"shap_beeswarm_{mod}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Dependence plots for top-5
    for rk, idx in enumerate(order[:5], start=1):
        try:
            plt.figure(figsize=(5,4))
            shap.dependence_plot(idx, sv_pos, Xdf.iloc[bg_idx], feature_names=feats,
                                 show=False, interaction_index=None)
            plt.title(f"{mod} - rank {rk}: {feats[idx][:30]}")
            plt.tight_layout()
            plt.savefig(FIG / f"shap_dependence_{mod}_{rk}.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[A6] dependence_plot failed for {mod} rank {rk}: {e}")

# ---- One illustrative force plot: a true-positive CRC case from biochem RF ----
rf_b, Xdf_b, y_b, feats_b = mod_models["biochem"]
preds = rf_b.predict_proba(Xdf_b.to_numpy())[:, 1]
# Pick the highest-confidence true CRC
crc_idx = np.where(y_b == 1)[0]
top_crc = crc_idx[np.argmax(preds[crc_idx])]
explainer_b = shap.TreeExplainer(rf_b)
sv_one = explainer_b.shap_values(Xdf_b.iloc[[top_crc]].to_numpy())
if isinstance(sv_one, list):
    sv_one_pos = sv_one[1][0]
    base_val = explainer_b.expected_value[1]
elif sv_one.ndim == 3:
    sv_one_pos = sv_one[0, :, 1]
    base_val = explainer_b.expected_value[1]
else:
    sv_one_pos = sv_one[0]
    base_val = explainer_b.expected_value

# Use a horizontal bar instead of force plot (force needs ipython display)
order_one = np.argsort(np.abs(sv_one_pos))[::-1][:8]
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d62728" if sv_one_pos[i] > 0 else "#1f77b4" for i in order_one]
ax.barh([feats_b[i][:35] for i in order_one[::-1]],
        [sv_one_pos[i] for i in order_one[::-1]],
        color=colors[::-1])
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("SHAP value (toward CRC = positive)")
ax.set_title(f"Per-feature SHAP for highest-confidence CRC case\n"
             f"P(CRC) = {preds[top_crc]:.3f}; baseline E[f] = {base_val:.3f}")
fig.tight_layout()
fig.savefig(FIG / "shap_force_one_case.png", dpi=200)
plt.close(fig)

# ---- Stacked model SHAP (LinearExplainer on the meta) ----
stack_oof = pd.read_parquet(LATE / "stacked_oof.parquet")
# Refit a single LR on the union of per-modality OOF probs (all rows)
# Need the OOF prob columns aligned with stack_oof rows. Easier: refit on per-mod OOF means.
# We'll reconstruct the meta-input matrix from per-mod oof parquet files.
def load_oof(m):
    d = pd.read_parquet(LATE / f"oof_probs_{m}.parquet")
    # Average over repeats per (Sample) to keep one row per sample
    return d.groupby("Sample")["p_crc"].mean().rename(f"p_{m}")
pm = pd.concat([load_oof(m) for m in MODALITIES], axis=1).reset_index()
y_full = pd.read_parquet(PRE / "biochem_z.parquet")[["Sample","y"]]
pm = pm.merge(y_full, on="Sample")
meta = LogisticRegression(max_iter=2000)
meta.fit(pm[[f"p_{m}" for m in MODALITIES]].to_numpy(), pm["y"].to_numpy())
expl = shap.LinearExplainer(meta, pm[[f"p_{m}" for m in MODALITIES]].to_numpy())
sv_meta = expl.shap_values(pm[[f"p_{m}" for m in MODALITIES]].to_numpy())
mean_abs_meta = np.abs(sv_meta).mean(axis=0)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.barh([f"p_{m}" for m in MODALITIES], mean_abs_meta, color=["#1f77b4","#d62728","#2ca02c"])
ax.set_xlabel("Mean |SHAP| (meta-learner)")
ax.set_title("Stacked meta-learner: per-modality contribution\n(higher = modality drives the final decision more)")
for i, v in enumerate(mean_abs_meta):
    ax.text(v, i, f" {v:.3f}", va="center")
fig.tight_layout()
fig.savefig(FIG / "shap_stack_bar.png", dpi=200)
plt.close(fig)

# ---- Surrogate decision tree (depth 4) on stacked OOF probabilities ----
tree = DecisionTreeRegressor(max_depth=4, random_state=SEED, min_samples_leaf=50)
tree.fit(pm[[f"p_{m}" for m in MODALITIES]].to_numpy(), pm.merge(stack_oof.groupby("Sample")["p_stack"].mean(), on="Sample")["p_stack"].to_numpy())
rules = export_text(tree, feature_names=[f"p_{m}" for m in MODALITIES])
(OUT / "surrogate_rules.txt").write_text(rules)

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(tree, feature_names=[f"p_{m}" for m in MODALITIES],
          filled=True, impurity=False, fontsize=8, ax=ax)
ax.set_title("Surrogate decision tree (depth 4) approximating the stacked LR\n"
             "Leaf values = predicted P(CRC)")
fig.tight_layout()
fig.savefig(FIG / "surrogate_tree.png", dpi=180)
plt.close(fig)

print("[A6] done.")
