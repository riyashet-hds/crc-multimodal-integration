"""
A4 - Common vs distinct features between intermediate (DIABLO) and late (RF) strategies.

For each modality (metabolomics, biochem):
  - Intermediate set = union of top-20 |loadings| across DIABLO components.
  - Late set        = top-20 by RF impurity importance.
  - Produce Venn-style figure, ranked overlap table, and sensitivity tables at
    N = {10, 20, 30}.

Outputs:
  analysis/04_compare/overlap_<modality>.csv
  analysis/04_compare/sensitivity_overlap.csv
  analysis/figures/venn_<modality>.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2 as _venn2  # noqa - check available

SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
LATE = ROOT / "analysis" / "03_late"
INTM = ROOT / "analysis" / "02_intermediate"
OUT  = ROOT / "analysis" / "04_compare"
FIG  = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def intermediate_top_set(modality_block):
    """Union top-20 |loadings| across components from DIABLO output."""
    fn = INTM / f"diablo_top20_{modality_block}.csv"
    df = pd.read_csv(fn)
    df["abs_loading"] = df["loading"].abs()
    # Rank features by max |loading| across components
    feat_max = df.groupby("feature")["abs_loading"].max().sort_values(ascending=False)
    return feat_max

def late_top_set(modality):
    fn = LATE / f"rf_importance_{modality}.csv"
    df = pd.read_csv(fn)
    df = df.sort_values("importance_impurity", ascending=False)
    return df.set_index("feature")["importance_impurity"]

# Map between block names
MAP = {"metabolomics": ("metab", "metabolomics"), "biochem": ("biochem", "biochem")}

sens_records = []
overlap_summary = {}
for ui_name, (intm_name, late_name) in MAP.items():
    intm = intermediate_top_set(intm_name)
    late = late_top_set(late_name)
    intm_ranks = {f: r+1 for r, f in enumerate(intm.index)}
    late_ranks = {f: r+1 for r, f in enumerate(late.index)}

    for N in (10, 20, 30):
        A = set(intm.index[:min(N, len(intm))])
        B = set(late.index[:min(N, len(late))])
        inter = A & B
        sens_records.append({
            "modality": ui_name,
            "N": N,
            "n_intermediate": len(A),
            "n_late": len(B),
            "n_common": len(inter),
            "jaccard": len(inter) / len(A | B) if (A | B) else 0.0,
        })

    # Default N = 20 for the per-modality overlap table
    N = 20
    A = set(intm.index[:min(N, len(intm))])
    B = set(late.index[:min(N, len(late))])
    union = A | B
    rows = []
    for f in union:
        rows.append({
            "feature": f,
            "modality": ui_name,
            "in_intermediate_top20": f in A,
            "in_late_top20":          f in B,
            "rank_intermediate":      intm_ranks.get(f, np.nan),
            "rank_late":              late_ranks.get(f, np.nan),
            "abs_loading_max":        float(intm.get(f, np.nan)),
            "rf_importance":          float(late.get(f, np.nan)),
        })
    overlap_df = pd.DataFrame(rows)
    # Agreement score: lower mean rank across strategies (only where both ranked)
    both = overlap_df["in_intermediate_top20"] & overlap_df["in_late_top20"]
    overlap_df["agreement_score"] = np.where(
        both, (overlap_df["rank_intermediate"] + overlap_df["rank_late"]) / 2, np.nan
    )
    overlap_df = overlap_df.sort_values(
        ["in_intermediate_top20","in_late_top20","agreement_score"],
        ascending=[False, False, True]
    )
    overlap_df.to_csv(OUT / f"overlap_{ui_name}.csv", index=False)
    overlap_summary[ui_name] = {"n_common": int(both.sum()), "n_intm_only": int((overlap_df["in_intermediate_top20"] & ~overlap_df["in_late_top20"]).sum()),
                                 "n_late_only": int((~overlap_df["in_intermediate_top20"] & overlap_df["in_late_top20"]).sum())}
    print(f"[A4] {ui_name}: intermediate AND late (N=20) = {int(both.sum())} features")

    # Venn
    try:
        from matplotlib_venn import venn2
        fig, ax = plt.subplots(figsize=(6, 5))
        venn2([A, B], set_labels=("Intermediate\n(DIABLO top-20)", "Late\n(RF impurity top-20)"), ax=ax)
        ax.set_title(f"Feature overlap — {ui_name}\n(n=4596; seed=2026)")
        fig.tight_layout()
        fig.savefig(FIG / f"venn_{ui_name}.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        # Fallback: simple bar
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = ["intermediate only", "common", "late only"]
        vals = [len(A - B), len(A & B), len(B - A)]
        ax.bar(cats, vals, color=["#1f77b4","#9467bd","#d62728"])
        for c, v in zip(cats, vals):
            ax.text(c, v + 0.1, str(v), ha="center")
        ax.set_title(f"Feature overlap — {ui_name}")
        fig.tight_layout()
        fig.savefig(FIG / f"venn_{ui_name}.png", dpi=200)
        plt.close(fig)

pd.DataFrame(sens_records).to_csv(OUT / "sensitivity_overlap.csv", index=False)
print("[A4] sensitivity:")
print(pd.DataFrame(sens_records).to_string(index=False))
print("[A4] done.")
