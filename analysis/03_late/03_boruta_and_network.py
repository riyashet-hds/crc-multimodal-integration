"""
A3.3 - Boruta feature confirmation per modality + cross-modality Spearman network.
Steps:
  1. Run Boruta on each z-scaled modality block (seed=2026, max_iter=100).
  2. Save Confirmed / Tentative / Rejected lists per modality.
  3. On the union of Confirmed features across all three blocks, build a
     cross-modality Spearman correlation graph (|rho|>0.3, BH-adjusted p<0.05).
  4. Compute degree and betweenness centrality; save edge + node tables;
     render igraph-style network figure (matplotlib + networkx).
Seed: 2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from statsmodels.stats.multitest import multipletests

SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
PRE  = ROOT / "analysis" / "01_preprocessing"
OUT  = ROOT / "analysis" / "03_late"
FIG  = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODALITIES = ["metabolomics", "biochem", "diet"]

def load(name):
    df = pd.read_parquet(PRE / f"{name}_z.parquet")
    y = df["y"].to_numpy()
    feats = [c for c in df.columns if c not in ("Sample","casecontrol","y")]
    X = df[feats].to_numpy()
    return df["Sample"].to_numpy(), X, y, feats

all_confirmed = {}
boruta_rows = []
feature_data = {}  # modality -> {feat: column array}

for mod in MODALITIES:
    sample, X, y, feats = load(mod)
    feature_data[mod] = {f: X[:, i] for i, f in enumerate(feats)}
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")
    boruta = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=SEED, verbose=0)
    boruta.fit(X, y)
    confirmed = [feats[i] for i, s in enumerate(boruta.support_)         if s]
    tentative = [feats[i] for i, s in enumerate(boruta.support_weak_)    if s]
    rejected  = [feats[i] for i in range(len(feats))
                 if not boruta.support_[i] and not boruta.support_weak_[i]]
    all_confirmed[mod] = confirmed
    print(f"[A3.3] {mod}: confirmed={len(confirmed)}, tentative={len(tentative)}, rejected={len(rejected)}")
    for i, f in enumerate(feats):
        status = ("confirmed" if boruta.support_[i]
                  else "tentative" if boruta.support_weak_[i]
                  else "rejected")
        boruta_rows.append({
            "modality": mod, "feature": f, "status": status,
            "rank": int(boruta.ranking_[i]),
        })

pd.DataFrame(boruta_rows).to_csv(OUT / "boruta_results.csv", index=False)

# ---- Cross-modality Spearman network ----
selected = []  # list of (mod, feat, vector)
for mod in MODALITIES:
    for f in all_confirmed[mod]:
        selected.append((mod, f, feature_data[mod][f]))

print(f"[A3.3] union confirmed features = {len(selected)}")

n = len(selected)
edges = []
pvals_all = []
pairs = []
for i in range(n):
    for j in range(i+1, n):
        mi, fi, vi = selected[i]
        mj, fj, vj = selected[j]
        if mi == mj:
            continue  # only cross-modality edges
        r, p = stats.spearmanr(vi, vj)
        pairs.append((i, j, mi, fi, mj, fj, float(r)))
        pvals_all.append(float(p))

reject, p_adj, _, _ = multipletests(pvals_all, alpha=0.05, method="fdr_bh") if pvals_all else (np.array([]),np.array([]),None,None)

edge_records = []
for k, (i, j, mi, fi, mj, fj, r) in enumerate(pairs):
    keep = (abs(r) > 0.3) and (p_adj[k] < 0.05)
    if keep:
        edge_records.append({
            "src_modality": mi, "src": fi,
            "dst_modality": mj, "dst": fj,
            "spearman_rho": r, "p_bh": float(p_adj[k]),
        })
edges_df = pd.DataFrame(edge_records)
edges_df.to_csv(OUT / "network_edges.csv", index=False)
print(f"[A3.3] cross-modality edges kept (|rho|>0.3, BH<0.05): {len(edges_df)} / {len(pairs)}")

# Build graph, compute centrality
G = nx.Graph()
for mod, f, _ in selected:
    G.add_node(f, modality=mod)
for r in edge_records:
    G.add_edge(r["src"], r["dst"], weight=abs(r["spearman_rho"]), rho=r["spearman_rho"])

# Drop nodes with degree 0 from centrality summary (kept in graph for plotting)
deg = dict(G.degree())
btw = nx.betweenness_centrality(G, weight="weight")
node_rows = []
for node in G.nodes():
    node_rows.append({
        "feature": node,
        "modality": G.nodes[node]["modality"],
        "degree": int(deg[node]),
        "betweenness": float(btw[node]),
    })
nodes_df = pd.DataFrame(node_rows).sort_values(["degree","betweenness"], ascending=[False, False])
nodes_df.to_csv(OUT / "network_nodes.csv", index=False)

# ---- Render network ----
color_map = {"metabolomics": "#1f77b4", "biochem": "#d62728", "diet": "#2ca02c"}
fig, ax = plt.subplots(figsize=(11, 9))
pos = nx.spring_layout(G, seed=SEED, k=0.8)
node_colors = [color_map[G.nodes[n]["modality"]] for n in G.nodes()]
node_sizes  = [80 + 25 * deg[n] for n in G.nodes()]
# Edge styling by sign
pos_edges = [(u,v) for u,v,d in G.edges(data=True) if d["rho"] > 0]
neg_edges = [(u,v) for u,v,d in G.edges(data=True) if d["rho"] < 0]
nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="#888", alpha=0.6, width=1.2, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color="#cc6677", style="dashed", alpha=0.6, width=1.2, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.5, ax=ax)
# Label only the top-degree nodes to avoid clutter
top_nodes = nodes_df.head(15)["feature"].tolist()
labels = {n: (n.split("__")[0][:20]) for n in G.nodes() if n in top_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
# Legend
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
handles = [mpatches.Patch(color=c, label=m) for m, c in color_map.items()]
handles += [mlines.Line2D([0], [0], color="#888", label="positive Spearman (|rho|>0.3, BH<0.05)"),
            mlines.Line2D([0], [0], color="#cc6677", linestyle="dashed", label="negative Spearman")]
ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("Cross-modality Spearman network on Boruta-confirmed features\n(n=4596; nodes coloured by modality; edge style = sign of rho)")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "network_cross_modality.png", dpi=220)
plt.close(fig)

print("[A3.3] done.")
