"""
A3.3b - RF impurity + permutation importance per modality (full-data refit),
        then a cross-modality Spearman network on the top-K features per modality.

This complements 03_boruta_and_network.py (which gives a strict Boruta call but
yields no cross-modality edges because metab/diet individually do not exceed
shadow features). Here we use a top-K ranking so the cross-modality network is
populated and the late-integration feature lists are comparable with DIABLO's
top-20 loadings (for A4 common-vs-distinct).

Outputs:
  analysis/03_late/rf_importance_<modality>.csv  (impurity + permutation imp.)
  analysis/03_late/topK_features_per_modality.csv
  analysis/03_late/network_edges.csv  (overwrites earlier empty version)
  analysis/03_late/network_nodes.csv
  analysis/figures/network_cross_modality.png
Seed: 2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from statsmodels.stats.multitest import multipletests

SEED = 2026
np.random.seed(SEED)
TOP_K = 10  # per modality; biochem only has 8 so will use all

ROOT = Path(__file__).resolve().parents[2]
PRE  = ROOT / "analysis" / "01_preprocessing"
OUT  = ROOT / "analysis" / "03_late"
FIG  = ROOT / "analysis" / "figures"

MODALITIES = ["metabolomics", "biochem", "diet"]
best_params = pd.read_csv(OUT / "rf_best_params.csv").set_index("modality").to_dict("index")

def load(name):
    df = pd.read_parquet(PRE / f"{name}_z.parquet")
    y = df["y"].to_numpy()
    feats = [c for c in df.columns if c not in ("Sample","casecontrol","y")]
    X = df[feats].to_numpy()
    return df["Sample"].to_numpy(), X, y, feats

importance_tables = {}
topK = {}
feature_vectors = {}  # (mod, feat) -> array

for mod in MODALITIES:
    sample, X, y, feats = load(mod)
    feature_vectors[mod] = {f: X[:, i] for i, f in enumerate(feats)}
    mtry = int(best_params[mod]["max_features"])
    rf = RandomForestClassifier(n_estimators=1000, max_features=mtry,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    imp_impurity = rf.feature_importances_
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_perm = perm.importances_mean
    df_imp = pd.DataFrame({
        "feature": feats,
        "importance_impurity": imp_impurity,
        "importance_perm_mean": imp_perm,
        "importance_perm_sd":   perm.importances_std,
    }).sort_values("importance_impurity", ascending=False).reset_index(drop=True)
    df_imp["rank_impurity"] = df_imp.index + 1
    df_imp.to_csv(OUT / f"rf_importance_{mod}.csv", index=False)
    importance_tables[mod] = df_imp
    k = min(TOP_K, len(feats))
    topK[mod] = df_imp.head(k)["feature"].tolist()
    print(f"[A3.3b] {mod}: top-{k} = {topK[mod][:5]}{'...' if len(topK[mod])>5 else ''}")

# Persist top-K table
rows = []
for mod, lst in topK.items():
    for r, f in enumerate(lst, start=1):
        rows.append({"modality": mod, "rank": r, "feature": f})
pd.DataFrame(rows).to_csv(OUT / "topK_features_per_modality.csv", index=False)

# ---- Cross-modality Spearman network ----
selected = []
for mod in MODALITIES:
    for f in topK[mod]:
        selected.append((mod, f, feature_vectors[mod][f]))
n = len(selected)
print(f"[A3.3b] selected nodes = {n}")

pairs = []
pvals = []
for i in range(n):
    for j in range(i+1, n):
        mi, fi, vi = selected[i]
        mj, fj, vj = selected[j]
        if mi == mj:
            continue
        r, p = stats.spearmanr(vi, vj)
        pairs.append((mi, fi, mj, fj, float(r)))
        pvals.append(float(p))

if pvals:
    _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
else:
    p_adj = np.array([])

edge_records = []
for k, ((mi, fi, mj, fj, r), padj) in enumerate(zip(pairs, p_adj)):
    if abs(r) > 0.3 and padj < 0.05:
        edge_records.append({
            "src_modality": mi, "src": fi,
            "dst_modality": mj, "dst": fj,
            "spearman_rho": r, "p_bh": float(padj),
        })
edges_df = pd.DataFrame(edge_records)
edges_df.to_csv(OUT / "network_edges.csv", index=False)
print(f"[A3.3b] cross-modality edges (|rho|>0.3, BH<0.05): {len(edges_df)} / {len(pairs)}")

# Build graph
G = nx.Graph()
for mod, f, _ in selected:
    G.add_node(f, modality=mod)
for r in edge_records:
    G.add_edge(r["src"], r["dst"], weight=abs(r["spearman_rho"]), rho=r["spearman_rho"])

deg = dict(G.degree())
btw = nx.betweenness_centrality(G, weight="weight") if G.number_of_edges() else {n: 0.0 for n in G.nodes()}
node_rows = []
for node in G.nodes():
    node_rows.append({
        "feature": node,
        "modality": G.nodes[node]["modality"],
        "degree": int(deg[node]),
        "betweenness": float(btw[node]),
    })
pd.DataFrame(node_rows).sort_values(["degree","betweenness"], ascending=[False, False]).to_csv(
    OUT / "network_nodes.csv", index=False)

# Render
color_map = {"metabolomics": "#1f77b4", "biochem": "#d62728", "diet": "#2ca02c"}
fig, ax = plt.subplots(figsize=(11, 9))
pos = nx.spring_layout(G, seed=SEED, k=0.7)
node_colors = [color_map[G.nodes[n]["modality"]] for n in G.nodes()]
node_sizes  = [120 + 35 * deg[n] for n in G.nodes()]
pos_edges = [(u,v) for u,v,d in G.edges(data=True) if d["rho"] > 0]
neg_edges = [(u,v) for u,v,d in G.edges(data=True) if d["rho"] < 0]
nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="#444",
                      alpha=0.5, width=[abs(G[u][v]["rho"])*3 for u,v in pos_edges], ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color="#cc6677",
                      style="dashed", alpha=0.6,
                      width=[abs(G[u][v]["rho"])*3 for u,v in neg_edges], ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                      edgecolors="black", linewidths=0.6, ax=ax)
labels = {n: n.split("__")[0][:24] for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
handles = [mpatches.Patch(color=c, label=m) for m, c in color_map.items()]
handles += [mlines.Line2D([0], [0], color="#444",
                          label="positive Spearman (|rho|>0.3, BH<0.05)"),
            mlines.Line2D([0], [0], color="#cc6677", linestyle="dashed",
                          label="negative Spearman")]
ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92)
ax.set_title("Cross-modality Spearman network on top-K RF-importance features\n"
             "(n=4596; nodes coloured by modality; edge width = |rho|; dashed = negative correlation)")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "network_cross_modality.png", dpi=220)
plt.close(fig)
print("[A3.3b] done.")
