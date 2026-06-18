# Combining Text Embeddings and Citation-Graph Structure for Systematic-Review Screening

**MSc Thesis — Beliz Pekkan | Utrecht University | 2026**

> Does fusing citation-graph structure with contextual text embeddings improve citation screening for systematic reviews? This project implements and evaluates a GNN+NLP fusion model against text-only and graph-only baselines across 18 SYNERGY+ datasets using a static, one-shot ranking paradigm.

---

## Overview

Systematic-review screening is an extreme class-imbalance problem: a reviewer may read tens of thousands of abstracts to find fewer than 1% that are relevant. This thesis investigates whether citation-graph structure (how papers cite one another within a candidate pool) provides complementary signal to text-based representations for ranking relevant papers early.

Five experimental conditions are compared under **Leave-One-Dataset-Out (LODO) cross-validation** using **Normalised Loss (NL)** as the primary metric (lower = better):

| Condition | Description |
|-----------|-------------|
| **C1** | TF-IDF + LinearSVC (ASReview ELAS-Ultra baseline) |
| **C2a** | Frozen SciBERT + LinearSVC |
| **C2b** | Frozen SciBERT + Logistic Regression |
| **C3** | Graph Attention Network (GAT) only |
| **C4** | GNN + NLP Fusion (SciBERT + GAT, concatenated) |

**Main result:** C4 achieves the best pool-mean NL (0.282 vs 0.338 for C1, ΔNL = −0.056), outperforming C1 on 8/10 RQ1 datasets and ranking first on 4/10. The improvement is driven primarily by text features (~80%), with the graph branch contributing a consistent but smaller secondary gain (~16%). No measured structural graph property predicts C4 performance; the strongest predictor is the TF-IDF baseline's own NL (ρ = 0.71, p = 0.001).

---

## Data

Experiments use a curated subset of **SYNERGY+**, an extended collection of openly available systematic-review datasets in OpenAlex format. Each dataset contains:
- Binary inclusion labels (1 = included, 0 = excluded)
- Title and abstract text
- `referenced_works` field used to construct within-dataset citation graphs

**Selection:** 18 unique datasets (22 selections across pools, counting shared datasets) were chosen from 119 candidates using a composite 10-point scoring procedure (size, graph density, inclusion rate, giant-component coverage). Datasets range from 228–7,394 records; inclusion rates from 0.2%–23.7%.

**Pools:**
- **RQ1** — 10 cross-domain datasets (all five conditions)
- **RQ2-A** — 6 osteoarthritis datasets (within-domain generalisation)
- **RQ2-B** — 6 PTSD/trauma datasets (within-domain generalisation)

Four datasets appear in both RQ1 and an RQ2 pool (Pijls\_2018, Pinos-Cisneros\_2023, Tumkaya\_2018, Rinne\_2021). The analysis notebooks handle pool overlap explicitly to avoid data leakage.

---

## Models

### C1 — TF-IDF + LinearSVC
Replicates ASReview's ELAS-Ultra configuration loaded via `get_ai_config("elas_u4")`. Produces a static ranking via `decision_function`. Serves as the primary baseline.

### C2a / C2b — Frozen SciBERT
Uses `allenai/scibert_scivocab_uncased` as a frozen feature extractor. The 768-dimensional `[CLS]` embedding is L2-normalised and passed to LinearSVC (C2a) or Logistic Regression (C2b). Embeddings are cached as `.npy` files and shared across C2a, C2b, and C4.

### C3 — Graph Attention Network (GAT)
Builds a directed within-dataset citation graph (nodes = papers, edges = citations). Node features are 132-dimensional: 128-dim LSA-compressed TF-IDF + 4 structural features (in-degree, out-degree, PageRank, normalised publication year). The two-layer GAT uses 8 attention heads → 512-dim hidden → 128-dim output. Trained with Focal Loss (γ=2.0, α=0.75) under LODO; runs are fully inductive (test graph unseen during training).

### C4 — GNN + NLP Fusion
Concatenates the 768-dim SciBERT embedding (C2a) and the 128-dim GAT embedding (C3) into a 896-dim vector, passed to a two-layer MLP classification head (896→256→1). The asymmetric dimensionality is deliberate: the head receives more text signal by default, degrading gracefully toward C2a when the graph branch is uninformative.

---

## Evaluation

- **Metric:** Normalised Loss (NL) — area between the model's recall curve and the ideal curve, normalised by worst-case area. NL = 0 is perfect; NL ≈ 0.5 is random. Chosen over WSS@95 for cross-dataset comparability across wildly varying inclusion rates.
- **Protocol:** 5 seeds per condition per dataset (42–46). Dataset-level mean NL across seeds is the unit of analysis.
- **Statistics:** Friedman test (omnibus) + pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction. All tests are reported with rank-biserial effect sizes.

---

## Key Results

| Comparison | ΔNL | p (adj) | r |
|------------|-----|---------|---|
| C4 vs C1 | −0.056 | 0.580 | 0.491 |
| C4 vs C2a | −0.011 | 0.767 | 0.273 |
| C4 vs C2b | −0.009 | 0.750 | 0.345 |
| C4 vs C3 | −0.087 | 0.336 | 0.636 |

Friedman omnibus: χ²(4) = 5.688, p = 0.224 (underpowered at N=10; ~22–25 datasets needed for 80% power).

**Ablation decomposition (C1 → C4, ΔNL = −0.056 total):**
- Feature effect (TF-IDF → SciBERT): ΔNL = −0.045 (~80%)
- Classifier effect (SVM → LR): ΔNL = −0.002 (~4%)
- Graph fusion effect (C2b → C4): ΔNL = −0.009 (~16%)

**RQ2 domain results:**

| Domain | C1 NL | C4 NL | ΔNL |
|--------|-------|-------|-----|
| Osteoarthritis | 0.309 | 0.297 | −0.012 |
| PTSD/Trauma | 0.399 | 0.297 | −0.102 |

---

## Installation

```bash
git clone https://github.com/belizpekkan/citation-graph-screening.git
cd citation-graph-screening
pip install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0
torch-geometric
transformers>=4.30
scikit-learn
asreview
scipy
pandas
numpy
matplotlib
seaborn
```

SciBERT inference requires ~2GB RAM per dataset; GAT training requires a GPU for datasets >3,000 nodes (8–12 hours on a single GPU at the 200-epoch ceiling for the largest datasets).

---

## Reproducing Analysis

Open `notebooks/post_review_analysis.ipynb` and run all cells. The notebook:
1. Loads `all_results.csv` and decontaminates the RQ1/RQ2 pools (handles overlapping datasets correctly)
2. Computes pool means, SDs, and condition rankings
3. Runs Friedman + Wilcoxon + Holm–Bonferroni tests
4. Computes Spearman correlations for structural predictor analysis (N=18 unique datasets)
5. Exports `thesis_stats_clean.csv` — a tidy summary of all reported statistics

Figures are generated by `notebooks/thesis_visualisations_CLEAN.ipynb`.

---

## Structural Predictor Analysis

A key finding is that no measured citation-graph structural property (density, connectivity, pool size, average degree, isolated-node fraction) significantly predicts C4 performance across the 18 unique datasets. The full Spearman ρ matrix (12 structural features × 5 conditions) is shown in the thesis (Figure 10). The strongest predictor of C4 NL is C1 NL (ρ = 0.71, p = 0.001) — the TF-IDF baseline's own performance on the same dataset.

Within the 12 RQ2 within-domain datasets, the **fusion benefit** (C2b NL − C4 NL) scales with citation density (ρ = 0.70, p = 0.011), consistent with the graph contributing more when the citation network is richly connected. See Appendix C of the thesis for the full within-domain density analysis.

---

## Limitations

- **Static ranking only.** All conditions produce a single pre-screening ranking; ASReview's iterative active-learning loop (which exploits human feedback) substantially outperforms any static ranker and is not directly compared here.
- **Small dataset sample.** N=10 for RQ1, N=6 per domain for RQ2. The Friedman test is underpowered; results should be read as suggestive rather than confirmatory.
- **Selection bias.** The 22 datasets were chosen to favour larger, denser, higher-inclusion reviews. Performance may not generalise to the sparse, low-inclusion reviews most common in practice.
- **Structural leakage.** Citation graphs are built over the full candidate set including test papers. This is contained (no cross-dataset leakage under LODO) but may modestly inflate GAT-condition scores.

---

## Citation

```bibtex
@thesis{pekkan2026citation,
  author  = {Beliz Pekkan},
  title   = {Combining Text Embeddings and Citation-Graph Structure for Systematic-Review Screening},
  school  = {Utrecht University},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## Acknowledgements

Experiments use the [SYNERGY+](https://github.com/asreview/synergy-dataset) dataset collection. The TF-IDF + LinearSVC baseline replicates the [ASReview ELAS-Ultra](https://github.com/asreview/asreview) configuration. SciBERT embeddings use the [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) checkpoint from HuggingFace. GAT implementation uses [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).
