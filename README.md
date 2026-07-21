# hyperbolic-grn-ppi-kg

Code and constructed knowledge graphs accompanying:

**"Structural Selectivity and Learned Hyperbolic Geometry in Pancreas, Bone Marrow, and Thymus Gene Regulatory Knowledge Graphs"**
Amangel Bhullar, Ziad Kobti — ALIFE 2026 Late Breaking Abstract

## Overview

This repository shows that **fan-out (branching factor), not tree-likeness (Gromov δ-hyperbolicity), drives the curvature a model learns to assign to a relation** — a pattern first identified in temporal knowledge graphs, replicated here across three independent human gene regulatory and protein-protein interaction (PPI) knowledge graphs built from single-cell transcriptomic data. We further show that low-dimensional hyperbolic embeddings recover known cell-differentiation lineage hierarchies better than dimension-matched Euclidean embeddings in two of the three tissues.

## Contribution

This repository provides:

1. **Three constructed heterogeneous knowledge graphs** (pancreas, bone marrow, thymus), each combining a transcription factor regulatory network (`regulates`) and a high-confidence protein-protein interaction network (`interacts_with`), filtered to tissue-specific highly variable genes derived from single-cell RNA-seq data.
2. **Manually-encoded ground-truth cell-differentiation lineage trees** for bone marrow and thymus, built from established developmental biology.
3. **Code** for multi-relational Poincaré-ball curvature learning and hyperbolic vs. Euclidean hierarchy-recovery evaluation.

The knowledge graphs are derived from publicly available data sources (see **Data Sources** below). The specific construction, filtering, integration, and lineage annotations are original to this work; the underlying raw data is not.

## Repository Structure

```text
hyperbolic-grn-ppi-kg/
|-- data/kg/
|   |-- pancreas/
|   |   |-- gene_nodes.csv
|   |   |-- edges_regulatory.csv       (CollecTRI-derived, HVG-filtered)
|   |   |-- edges_ppi.csv              (STRING-derived, HVG-filtered)
|   |   `-- pseudobulk_celltype_gene.csv
|   |-- bone_marrow/
|   |   |-- gene_nodes.csv
|   |   |-- edges_regulatory.csv
|   |   |-- edges_ppi.csv
|   |   |-- pseudobulk_celltype_gene.csv
|   |   `-- lineage_tree.csv           (manually encoded ground truth)
|   `-- thymus/
|       |-- gene_nodes.csv
|       |-- edges_regulatory.csv
|       |-- edges_ppi.csv
|       |-- pseudobulk_celltype_gene.csv
|       `-- lineage_tree.csv
|-- src/
|   |-- curvature_model.py             (multi-relational Poincare-ball model, fan-out, Gromov delta)
|   `-- hierarchy_recovery.py          (hyperbolic vs Euclidean lineage recovery sweep)
|-- results/
|   |-- pancreas_curvature_multiseed.csv
|   |-- bonemarrow_curvature_multiseed.csv
|   |-- thymus_curvature_multiseed.csv
|   |-- hierarchy_recovery_sweep_bonemarrow.csv
|   |-- hierarchy_recovery_sweep_thymus.csv
|   `-- figures/
|-- requirements.txt
|-- LICENSE
`-- README.md
```


## Data Sources

Raw single-cell data is not redistributed here due to size and licensing. The knowledge graphs in `data/kg/` were constructed from:

- **Tabula Sapiens** single-cell RNA-seq atlas — Tabula Sapiens Consortium (2022), *Science*, 376(6594):eabl4896. https://doi.org/10.1126/science.abl4896
- **CollecTRI** transcription factor-target regulatory network — Müller-Dott et al. (2023), *Nucleic Acids Research*, 51(20):10934–10949. https://doi.org/10.1093/nar/gkad841
- **STRING** protein-protein interaction network v12.0 — Szklarczyk et al. (2023), *Nucleic Acids Research*, 51(D1):D638–D646. https://doi.org/10.1093/nar/gkac1000

## Setup

```bash
git clone https://github.com/amangelbhullar/hyperbolic-grn-ppi-kg.git
cd hyperbolic-grn-ppi-kg
pip install -r requirements.txt
```

## Key Results

### Fan-out and learned curvature per relation and tissue

| Dataset | Relation | Fan-out | Curvature |
|---|---|---|---|
| Pancreas | regulates | 15.06 | 0.602 ± 0.007 |
| Pancreas | interacts_with | 20.89 | 0.842 ± 0.009 |
| Bone Marrow | regulates | 15.65 | 0.648 ± 0.009 |
| Bone Marrow | interacts_with | 18.84 | 0.873 ± 0.015 |
| Thymus | regulates | 13.89 | 0.580 ± 0.006 |
| Thymus | interacts_with | 18.61 | 0.880 ± 0.021 |

Paired Wilcoxon *p* = 0.002 for `regulates` vs. `interacts_with` in all three tissues (10 seeds). In every tissue, `interacts_with` has **higher** fan-out and **higher** learned curvature, despite also having **higher** Gromov δ (i.e., being *less* tree-like).

### Hierarchy recovery (Spearman ρ, embedding distance-from-origin vs. true lineage depth)

| Dim. | Dataset | Hyperbolic ρ | Euclidean ρ |
|---|---|---|---|
| 4 | Bone Marrow | 0.660 ± 0.068 | 0.574 ± 0.077 |
| 4 | Thymus | 0.325 ± 0.060 | 0.123 ± 0.062 |
| 32 | Bone Marrow | 0.400 ± 0.023 | 0.614 ± 0.027 |
| 32 | Thymus | 0.355 ± 0.130 | 0.598 ± 0.034 |

Full dimension sweep ({4, 8, 16, 32}) available in `results/hierarchy_recovery_sweep_bonemarrow.csv` and `results/hierarchy_recovery_sweep_thymus.csv`.

## Citation

```bibtex
@inproceedings{bhullar2026structural,
  title={Structural Selectivity and Learned Hyperbolic Geometry in Pancreas, Bone Marrow, and Thymus Gene Regulatory Knowledge Graphs},
  author={Bhullar, Amangel and Kobti, Ziad},
  booktitle={ALIFE 2026 Late Breaking Abstracts},
  year={2026}
}
```

## License

MIT License (see `LICENSE`). Note: constructed knowledge graph files are derived from CollecTRI and STRING data, which have their own licensing terms.

## Contact

amangelbhullar@gmail.com

Amangel Bhullar — School of Computer Science, University of Windsor
