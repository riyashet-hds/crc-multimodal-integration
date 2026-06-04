"""
A1.1 — Load all four xlsx files, audit shapes, missingness, class balance.
Writes:
  analysis/00_eda/audit_shapes.csv
  analysis/00_eda/audit_missingness_per_feature.csv
  analysis/00_eda/audit_class_balance.csv
  analysis/00_eda/id_overlap.csv
  analysis/figures/missingness_heatmap_<modality>.png
Seed: 2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT  = ROOT / "analysis" / "00_eda"
FIG  = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FILES = {
    "metabolomics": DATA / "Metabolomics.xlsx",
    "biochem":      DATA / "Blood Bio Chemistry.xlsx",
    "diet":         DATA / "Diet.xlsx",
    "proteomics":   DATA / "Proteomics.xlsx",
}

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\xa0", " ").strip()
        for c in df.columns
    ]
    return df

dfs = {}
shapes = []
class_counts = []
miss_rows = []

for name, fp in FILES.items():
    df = clean_cols(pd.read_excel(fp))
    dfs[name] = df
    n_rows, n_cols = df.shape
    feat_cols = [c for c in df.columns if c not in ("Sample", "casecontrol")]
    miss = df[feat_cols].isna().mean() * 100
    overall = float(miss.mean())
    shapes.append({
        "modality": name,
        "file": fp.name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_features": len(feat_cols),
        "missing_pct_overall": round(overall, 3),
        "n_unique_sample_ids": int(df["Sample"].nunique()),
        "n_duplicate_sample_rows": int(df["Sample"].duplicated().sum()),
    })
    counts = df["casecontrol"].astype(str).str.strip().value_counts().to_dict()
    class_counts.append({"modality": name, **counts})
    for feat, pct in miss.items():
        miss_rows.append({"modality": name, "feature": feat, "missing_pct": round(float(pct), 3)})

pd.DataFrame(shapes).to_csv(OUT / "audit_shapes.csv", index=False)
pd.DataFrame(class_counts).to_csv(OUT / "audit_class_balance.csv", index=False)
miss_df = pd.DataFrame(miss_rows).sort_values(["modality", "missing_pct"], ascending=[True, False])
miss_df.to_csv(OUT / "audit_missingness_per_feature.csv", index=False)

# Sample-ID overlap
ids = {k: set(v["Sample"].astype(str)) for k, v in dfs.items()}
overlap = []
keys = list(ids)
for i, a in enumerate(keys):
    for b in keys[i+1:]:
        overlap.append({"a": a, "b": b, "intersection": len(ids[a] & ids[b])})
triple = ids["metabolomics"] & ids["biochem"] & ids["diet"]
overlap.append({"a": "metab_AND_biochem_AND_diet", "b": "", "intersection": len(triple)})
pd.DataFrame(overlap).to_csv(OUT / "id_overlap.csv", index=False)

# Missingness heatmap per modality (binary mask)
for name, df in dfs.items():
    feat = [c for c in df.columns if c not in ("Sample", "casecontrol")]
    if not feat:
        continue
    mask = df[feat].isna().to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(min(12, 0.04 * len(feat) + 4), 6))
    ax.imshow(mask, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_title(f"Missingness pattern — {name} ({mask.shape[0]} samples × {mask.shape[1]} features)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Sample index")
    fig.tight_layout()
    fig.savefig(FIG / f"missingness_heatmap_{name}.png", dpi=200)
    plt.close(fig)

print("[A1.1] audit complete. Shapes:")
print(pd.DataFrame(shapes).to_string(index=False))
print("\n[A1.1] class balance:")
print(pd.DataFrame(class_counts).to_string(index=False))
print("\n[A1.1] ID overlap:")
print(pd.DataFrame(overlap).to_string(index=False))
