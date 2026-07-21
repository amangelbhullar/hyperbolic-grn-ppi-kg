import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon
import itertools

lineage = pd.read_csv("/storage/bhull113/scKG/kg/thymus_lineage_tree.csv")
pseudobulk = pd.read_csv("/storage/bhull113/scKG/kg/pseudobulk_thymus_celltype_gene.csv")

lineage = lineage[lineage["depth"] >= 0].reset_index(drop=True)
valid_types = set(lineage["cell_type"])
pseudobulk = pseudobulk[pseudobulk["cell_type"].isin(valid_types)]

cell_types = sorted(valid_types)
ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
n_types = len(cell_types)
print(f"Cell types (hematopoietic lineage only): {n_types}")

pivot = pseudobulk.pivot_table(index="cell_type", columns="gene_symbol", values="mean_expression", fill_value=0)
pivot = pivot.loc[cell_types]
features_base = torch.tensor(pivot.values, dtype=torch.float32)
features_base = F.normalize(features_base, dim=1)

edges = []
for _, row in lineage.iterrows():
    if pd.notna(row["parent"]) and row["parent"] in ct_to_idx:
        edges.append((ct_to_idx[row["parent"]], ct_to_idx[row["cell_type"]]))
        edges.append((ct_to_idx[row["cell_type"]], ct_to_idx[row["parent"]]))
edge_index = torch.tensor(edges, dtype=torch.long).T
print(f"Tree edges (symmetrized): {edge_index.shape[1]}")

true_depth = torch.tensor(lineage.set_index("cell_type").loc[cell_types, "depth"].values, dtype=torch.float32)

adj_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
all_pairs = [(i, j) for i in range(n_types) for j in range(n_types) if i != j]
neg_pairs_list = [p for p in all_pairs if p not in adj_set]
neg_pairs = torch.tensor(neg_pairs_list, dtype=torch.long).T

def mobius_add(x, y, c=1.0, eps=1e-5):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / (denom + eps)

def proj_to_ball(x, c=1.0, eps=1e-5):
    norm = x.norm(dim=-1, keepdim=True)
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def poincare_dist(x, y, c=1.0, eps=1e-5):
    sq_c = c ** 0.5
    mob = mobius_add(-x, y, c)
    return 2 / sq_c * torch.atanh((sq_c * mob.norm(dim=-1) + eps).clamp(max=1 - eps))

class HyperbolicEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=16, c_init=1.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb_dim)
        self.log_c = nn.Parameter(torch.tensor(float(np.log(c_init))))
    def forward(self, x, edge_index, n_nodes, n_layers=2):
        c = torch.exp(self.log_c)
        h = torch.tanh(self.lin(x)) * 0.1
        h = proj_to_ball(h, c=c.item())
        for _ in range(n_layers):
            row, col = edge_index
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])
            deg = torch.zeros(n_nodes).index_add_(0, row, torch.ones_like(row, dtype=torch.float)).clamp(min=1).unsqueeze(1)
            agg = agg / deg
            h = proj_to_ball(mobius_add(h, 0.1 * agg, c=c.item()), c=c.item())
        return h, c

class EuclideanEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=16):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb_dim)
    def forward(self, x, edge_index, n_nodes, n_layers=2):
        h = torch.tanh(self.lin(x)) * 0.1
        for _ in range(n_layers):
            row, col = edge_index
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])
            deg = torch.zeros(n_nodes).index_add_(0, row, torch.ones_like(row, dtype=torch.float)).clamp(min=1).unsqueeze(1)
            agg = agg / deg
            h = h + 0.1 * agg
        return h, torch.tensor(0.0)

def euclidean_dist(x, y):
    return (x - y).norm(dim=-1)

def train_and_eval(model_type, emb_dim, seed, n_epochs=200, margin=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_type == "hyperbolic":
        model = HyperbolicEncoder(in_dim=features_base.shape[1], emb_dim=emb_dim)
        dist_fn = lambda a, b, c: poincare_dist(a, b, c=c.item())
    else:
        model = EuclideanEncoder(in_dim=features_base.shape[1], emb_dim=emb_dim)
        dist_fn = lambda a, b, c: euclidean_dist(a, b)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    row_pos, col_pos = edge_index
    row_neg, col_neg = neg_pairs
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        emb, c = model(features_base, edge_index, n_types)
        d_pos = dist_fn(emb[row_pos], emb[col_pos], c)
        d_neg = dist_fn(emb[row_neg], emb[col_neg], c)
        loss = d_pos.mean() + F.relu(margin - d_neg).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        emb, c = model(features_base, edge_index, n_types)
        if model_type == "hyperbolic":
            origin = torch.zeros_like(emb[0]).unsqueeze(0).expand(n_types, -1)
            dist_from_origin = poincare_dist(emb, origin, c=c.item()).numpy()
        else:
            centroid = emb.mean(dim=0, keepdim=True)
            dist_from_origin = euclidean_dist(emb, centroid.expand(n_types, -1)).numpy()
    rho, p = spearmanr(dist_from_origin, true_depth.numpy())
    return rho, p

seeds = list(range(10))
dims = [4, 8, 16, 32]
results = []
for dim, model_type, seed in itertools.product(dims, ["hyperbolic", "euclidean"], seeds):
    rho, p = train_and_eval(model_type, dim, seed)
    results.append({"embedding_dim": dim, "model_type": model_type, "seed": seed, "rho": rho, "p_value": p})

results_df = pd.DataFrame(results)
results_df.to_csv("/storage/bhull113/scKG/kg/thymus_robustness_sweep_results.csv", index=False)

print("=== THYMUS SUMMARY (mean ± std across 10 seeds) ===")
summary = results_df.groupby(["embedding_dim", "model_type"])["rho"].agg(["mean", "std"]).reset_index()
print(summary.to_string())

print("\n=== Paired Wilcoxon test (hyperbolic vs euclidean rho, per dim) ===")
for dim in dims:
    h_rhos = results_df[(results_df["embedding_dim"] == dim) & (results_df["model_type"] == "hyperbolic")]["rho"].values
    e_rhos = results_df[(results_df["embedding_dim"] == dim) & (results_df["model_type"] == "euclidean")]["rho"].values
    stat, p = wilcoxon(h_rhos, e_rhos)
    print(f"dim={dim}: hyperbolic mean={h_rhos.mean():.4f}, euclidean mean={e_rhos.mean():.4f}, wilcoxon p={p:.4f}")
