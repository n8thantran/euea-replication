# Replication Report: Assessing the Impact of Dimensionality Reduction on Clustering Performance

## 1. What Was Implemented

This replication implements the full experimental pipeline from the paper, which evaluates 5 dimensionality reduction (DR) methods across 4 clustering algorithms at 3 reduction levels on both synthetic and real-world datasets.

### DR Methods
- **PCA** (linear)
- **Kernel PCA** (RBF kernel, gamma=1/d)
- **VAE** (Variational Autoencoder: encoder d→64→32→latent, decoder reverse, 100 epochs, MSE+KL loss)
- **Isomap** (n_neighbors=10)
- **MDS** (metric, n_init=10)

### Clustering Algorithms
- **k-means** (n_init=100, true k)
- **AHC** (Agglomerative Hierarchical Clustering, HP search over metric×linkage)
- **GMM** (Gaussian Mixture Model, HP search over covariance types)
- **OPTICS** (HP search over min_samples×min_cluster_size)

### Reduction Levels
- **k-1**: Reduce to (number_of_clusters - 1) dimensions
- **25%**: Reduce to 25% of original dimensions (min 2)
- **50%**: Reduce to 50% of original dimensions (min 2)

### Datasets
- **20 real-world UCI datasets**: Iris, Wine, Breast Cancer Wisconsin, Seeds, Glass, Ecoli, Yeast, Ionosphere, Sonar, Dermatology, Segmentation, Vehicle, Vowel, Waveform, Satellite, Pendigits, Letter Recognition, Optical Digits, Soybean, Vertebral Column
- **4 synthetic dataset types**: Circles, Moons, RSG (Random State Generator), Repliclust

### Evaluation
- **Metric**: Adjusted Rand Index (ARI)
- **Aggregate statistics**: Win% (ties excluded), Average win/loss %
- **Statistical test**: Wilcoxon signed-rank test (two-sided, α=0.05)

## 2. Commands Run Successfully

```bash
# Full pipeline (runs all experiments from scratch, ~3-4 hours)
python main_pipeline.py

# Generate all outputs from cached results (~10 seconds)
python generate_all_outputs.py

# Quick reproduce (uses cached results)
bash reproduce.sh
```

## 3. Key Results

### Aggregate Tables (Paper Tables 1-4)

Our results qualitatively reproduce the paper's key findings:

| Finding | Paper | Our Replication |
|---------|-------|-----------------|
| VAE consistently underperforms | ✓ Negative avg win/loss% across all algorithms | ✓ Confirmed — VAE shows negative avg win/loss% in nearly all settings |
| Kernel PCA strong on synthetic, weak on real (k-means) | ✓ 75% synth win, 20% real win | Partially — our synthetic win% is lower (~10-40%) due to fewer datasets |
| Isomap competitive on real-world data | ✓ ~50-70% real win rates | ✓ Confirmed — 45-60% real win rates for k-means |
| PCA moderate, data-dependent | ✓ | ✓ Confirmed |
| MDS inconsistent across data types | ✓ | ✓ Confirmed |

### Wilcoxon Signed-Rank Test (Paper Table 5)

Our Wilcoxon test results show:
- **VAE** produces statistically significant degradation (p < 0.05) for most algorithm pairings, consistent with the paper
- **Kernel PCA + OPTICS** shows significant results at some reduction levels, partially matching the paper's finding of p=0.047 and p=0.046
- Most other DR-algorithm combinations show non-significant differences, consistent with the paper

### Boxplots (Paper Figures 1-8)

36 boxplot PDFs generated showing ARI distributions across DR methods and reduction levels for each clustering algorithm on both synthetic and real-world data.

## 4. Important File Paths

### Code
- `/workspace/main_pipeline.py` — Complete experiment pipeline (authoritative, runs everything from scratch)
- `/workspace/generate_all_outputs.py` — Generates all tables and plots from cached JSON results
- `/workspace/generate_all_results.py` — Alternative output generator (used by reproduce.sh)
- `/workspace/load_uci.py` — Loads all 20 UCI datasets
- `/workspace/reproduce.sh` — Reproduction script

### Key Result Tables
- `/workspace/results/table_combined_aggregate_k-means.csv` — Table 1 (k-means aggregate win/loss stats)
- `/workspace/results/table_combined_aggregate_AHC.csv` — Table 2 (AHC aggregate)
- `/workspace/results/table_combined_aggregate_GMM.csv` — Table 3 (GMM aggregate)
- `/workspace/results/table_combined_aggregate_OPTICS.csv` — Table 4 (OPTICS aggregate)
- `/workspace/results/table_wilcoxon_RealWorld.csv` — Table 5 (Wilcoxon signed-rank test)
- `/workspace/results/table_average_ARI_Circles.csv` — Synthetic average ARI (Circles)
- `/workspace/results/table_average_ARI_Moons.csv` — Synthetic average ARI (Moons)
- `/workspace/results/table_average_ARI_RSG.csv` — Synthetic average ARI (RSG)
- `/workspace/results/table_average_ARI_Repliclust.csv` — Synthetic average ARI (Repliclust)
- `/workspace/results/table_k-means_RealWorld.csv` — Per-dataset ARI (k-means, real-world)
- `/workspace/results/table_AHC_RealWorld.csv` — Per-dataset ARI (AHC, real-world)
- `/workspace/results/table_GMM_RealWorld.csv` — Per-dataset ARI (GMM, real-world)
- `/workspace/results/table_OPTICS_RealWorld.csv` — Per-dataset ARI (OPTICS, real-world)

### Boxplots
- `/workspace/results/boxplot_{algo}_RealWorld.pdf` — Real-world boxplots (4 files)
- `/workspace/results/boxplot_{algo}_Synthetic_{type}.pdf` — Synthetic boxplots (16 files)
- `/workspace/results/boxplot_{algo}_{type}.pdf` — Additional boxplots (16 files)

### Cached Experiment Data
- `/workspace/results/results_RealWorld.json` — Raw results for 20 UCI datasets
- `/workspace/results/results_Circles.json` — Raw results for Circles synthetic
- `/workspace/results/results_Moons.json` — Raw results for Moons synthetic
- `/workspace/results/results_RSG.json` — Raw results for RSG synthetic
- `/workspace/results/results_Repliclust.json` — Raw results for Repliclust synthetic

## 5. What Is Still Incomplete or Approximate

### Computational Constraints
- **Synthetic repetitions**: 5 reps (paper uses 50) — reduces statistical power
- **Synthetic dataset configurations**: ~10-20 per type (paper uses ~50) — fewer data points for aggregate statistics
- **MDS n_init**: 10 (paper uses 50) — may affect MDS quality slightly

### Numerical Differences
- Aggregate win% and avg win/loss% values differ from the paper due to:
  1. Fewer synthetic datasets and repetitions
  2. Possible differences in exact hyperparameter search grids
  3. VAE training stochasticity
  4. Random seed differences
- **OPTICS** avg win/loss% values are sometimes very large due to near-zero baselines (division by small numbers)

### Qualitative Agreement
Despite numerical differences, the main qualitative conclusions of the paper are reproduced:
1. DR does not universally improve clustering — it is method- and data-dependent
2. VAE is consistently the worst DR method for clustering
3. Kernel PCA and Isomap show the most promise, especially for AHC and GMM
4. OPTICS is the most fragile clustering algorithm under DR
5. Most DR-algorithm combinations do not produce statistically significant improvements (Wilcoxon test)
