# Dimensionality Reduction for Clustering - Implementation Progress

## Current Phase: Starting fresh implementation of the CORRECT paper

## Paper Summary
**Title**: "Assessing the impact of dimensionality reduction on clustering performance — a systematic study"

**Goal**: Systematically evaluate 5 DR methods × 4 clustering algorithms × 3 reduction levels on synthetic + real-world data, measuring ARI.

### DR Methods
1. PCA (sklearn, default settings)
2. Kernel PCA (sklearn, kernel='rbf', default gamma)
3. VAE (custom PyTorch: encoder 64→32→latent, decoder 32→64→d, BatchNorm, Dropout=0.4, Adam, MSE loss, 100 epochs, batch=64, 70/30 split)
4. Isomap (sklearn, default settings)
5. MDS (sklearn, random_state=10, n_init=50)

### Clustering Algorithms
1. k-means (k-means++ init, n_init=100)
2. AHC (various affinity/linkage, best per dataset type)
3. GMM (various covariance types, best per dataset type)
4. OPTICS (xi method, min_samples 5-10, min_cluster_size 0-1)

### Reduction Levels
1. k-1 dimensions (min 2)
2. 25% of original dimensions
3. 50% of original dimensions

### Datasets
**Synthetic** (1165 total):
- Circles: 2-cluster and 5-cluster, embedded in 10/50/200 dims, 50 datasets per config (300 total)
- Moons: 2-cluster and 5-cluster, embedded in 10/50/200 dims, 50 datasets per config (300 total)
- RSG (Rodriguez): k∈{2,10,50}, d∈{10,50,200}, Nc∈{5,50,100}, 265 datasets
- Repliclust: 2-cluster and 5-cluster, in 10/50/200 dims, 50 datasets per config (300 total)
- Noise injection: 75% features get Gaussian noise (25% σ=1, 25% σ=0.5, 25% σ=0.25, 25% clean)

**Real-world** (20 UCI datasets):
Breast tissue, Breast Wisconsin, Ecoli, Glass, Haberman, Ionosphere, Iris, Movement libras, Musk, Parkinsons, Segmentation, Sonar all, Spectf, Transfusion, Vehicle, Vertebral column, Vowel context, Wine, Wine quality red, Yeast

### Evaluation
- ARI (Adjusted Rand Index)
- Compare: No reduction baseline vs. each DR method at each level
- Aggregate: % wins over baseline, average win/loss ARI change

### Key Tables to Reproduce
1. Tables A.1-A.4: Average ARI for synthetic data (Circles, Moons, RSG, Repliclust)
2. Tables for real-world data: k-means, AHC, GMM, OPTICS (Tables in appendix)
3. Aggregate tables: Tables 1-4 (% wins, avg win/loss for each clustering algo)
4. Wilcoxon signed-rank test table
5. Boxplot figures

## Implementation Plan
- [ ] 1. Set up environment and install packages
- [ ] 2. Download/generate all datasets
  - [ ] 2a. Generate Circles synthetic data
  - [ ] 2b. Generate Moons synthetic data
  - [ ] 2c. Generate/download RSG data
  - [ ] 2d. Generate Repliclust data
  - [ ] 2e. Download 20 UCI datasets
- [ ] 3. Implement preprocessing (z-score normalization, noise injection)
- [ ] 4. Implement DR methods (PCA, KPCA, VAE, Isomap, MDS)
- [ ] 5. Implement clustering algorithms (k-means, AHC, GMM, OPTICS)
- [ ] 6. Run experiments on synthetic data
- [ ] 7. Run experiments on real-world data
- [ ] 8. Compute aggregate statistics
- [ ] 9. Generate boxplots and tables
- [ ] 10. Create reproduce.sh and REPORT.md

## Key Decisions
- Will use sklearn for most methods
- VAE implemented in PyTorch
- For AHC/GMM/OPTICS: need to determine best hyperparams per dataset type
- RSG data: need to find Rodriguez et al. datasets or implement generator

## Completed Work
(none yet for correct paper)

## Failed Approaches
- Previously implemented wrong paper (AdaCD for LLM safety) - completely wrong
