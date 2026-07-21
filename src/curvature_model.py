import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

edges = pd.read_csv("/storage/bhull113/scKG/kg/final_edges_bonemarrow.csv")
reg_edges = edges[edges["edge_type"] == "regulates"]
ppi_edges = edges[edges["edge_type"] == "interacts_with"]
gene_map = pd.read_csv("/storage/bhull113/scKG/kg/final_gene_id_map_bonemarrow.csv")
n_genes = len(gene_map)

reg_fanout_mean = reg_edges.groupby("source_id").size().mean()
ppi_deg_mean = pd.concat([ppi_edges["source_id"], ppi_edges["target_id"]]).value_counts().mean()

pos_src_np = pd.concat([reg_edges["source_id"], ppi_edges["source_id"]]).values
pos_tgt_np = pd.concat([reg_edges["target_id"], ppi_edges["target_id"]]).values
pos_rel_np = np.array([0]*len(reg_edges) + [1]*len(ppi_edges))
n_edges_total = len(pos_src_np)

def mobius_add(x, y, c, eps=1e-5):
    c = c.unsqueeze(-1) if c.dim() == 1 else c
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / (denom + eps)

def proj_to_ball(x, c, eps=1e-5):
    c_ = c.unsqueeze(-1) if c.dim() == 1 else c
    norm = x.norm(dim=-1, keepdim=True)
    maxnorm = (1 - eps) / (c_.clamp(min=1e-6) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def poincare_dist(x, y, c, eps=1e-5):
    sq_c = c.clamp(min=1e-6) ** 0.5
    mob = mobius_add(-x, y, c)
    return 2 / sq_c * torch.atanh((sq_c * mob.norm(dim=-1) + eps).clamp(max=1 - eps))

class MultiRelationalHyperbolic(nn.Module):
    def __init__(self, n_nodes, emb_dim=16, n_relations=2, c_init=1.0):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        self.log_c = nn.Parameter(torch.full((n_relations,), float(np.log(c_init))))
    def get_curvatures(self):
        return torch.exp(self.log_c)
    def forward(self, src, tgt, rel):
        c = torch.exp(self.log_c[rel])
        x = proj_to_ball(self.embedding(src), c)
        y = proj_to_ball(self.embedding(tgt), c)
        return poincare_dist(x, y, c), c

def run_seed(seed, n_epochs=150):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos_src = torch.tensor(pos_src_np, dtype=torch.long)
    pos_tgt = torch.tensor(pos_tgt_np, dtype=torch.long)
    pos_rel = torch.tensor(pos_rel_np, dtype=torch.long)

    model = MultiRelationalHyperbolic(n_genes, n_relations=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        d_pos, _ = model(pos_src, pos_tgt, pos_rel)
        neg_tgt = torch.randint(0, n_genes, (n_edges_total,))
        d_neg, _ = model(pos_src, neg_tgt, pos_rel)
        loss = d_pos.mean() + F.relu(1.0 - d_neg).mean()
        loss.backward()
        optimizer.step()
    return model.get_curvatures().detach().numpy()

results = []
for seed in range(10):
    c_reg, c_ppi = run_seed(seed)
    results.append({"seed": seed, "c_regulates": c_reg, "c_interacts_with": c_ppi})
    print(f"seed={seed}: c_regulates={c_reg:.4f}, c_interacts_with={c_ppi:.4f}")

df = pd.DataFrame(results)
df.to_csv("/storage/bhull113/scKG/kg/bonemarrow_curvature_multiseed.csv", index=False)

print(f"\n=== SUMMARY (mean ± std across 10 seeds) ===")
print(f"regulates:       c_r = {df['c_regulates'].mean():.4f} ± {df['c_regulates'].std():.4f}  (fan-out={reg_fanout_mean:.2f})")
print(f"interacts_with:  c_r = {df['c_interacts_with'].mean():.4f} ± {df['c_interacts_with'].std():.4f}  (fan-out={ppi_deg_mean:.2f})")

from scipy.stats import wilcoxon
stat, p = wilcoxon(df["c_regulates"], df["c_interacts_with"])
print(f"\nPaired Wilcoxon test (regulates vs interacts_with curvature): stat={stat:.2f}, p={p:.4f}")
