# Dimensionality Reduction for Clustering - Implementation Progress

## Current Phase: Building main experiment pipeline

## Status Summary
- **UCI data loading**: DONE and TESTED - all 20 datasets load correctly matching paper Table 1
- **Synthetic data generation**: Code written (generate_data.py) but NOT YET RUN
- **Experiment pipeline**: NOT YET BUILT - this is the main task now
- **Results**: None generated yet

## Paper Summary
**Title**: "Assessing the impact of dimensionality reduction on clustering performance"

**Goal**: Evaluate 5 DR methods × 4 clustering algorithms × 3 reduction levels on synthetic + real-world data, measuring ARI.

### DR Methods (all sklearn except VAE)
1. PCA (default settings)
2. Kernel PCA (kernel='rbf', default gamma)
3. VAE (PyTorch: encoder d→64→32→latent, decoder latent→32→64→d, BatchNorm, Dropout=0.4, Adam, MSE, 100 epochs, batch=64, 70/30 split, sigmoid output)
4. Isomap (default settings)
5. MDS (random_state=10, n_init=50)

### Clustering Algorithms
1. k-means (k-means++ init, n_init=100)
2. AHC (best affinity/linkage per dataset TYPE - search over euclidean/l1/l2/manhattan/cosine × complete/average/single/ward)
3. GMM (best covariance_type per dataset TYPE - search over spherical/tied/diag/full)
4. OPTICS (xi method, search min_samples 5-10, min_cluster_size 0-1 step 0.05)

### Reduction Levels
1. k-1 dimensions (min 2)
2. 25% of original dimensions (round, min 2)
3. 50% of original dimensions (round, min 2)

### Key: AHC/GMM/OPTICS hyperparameters are chosen per DATASET TYPE
- For synthetic: best params chosen per type (Circles, Moons, RSG, Repliclust)
- For real-world: best params chosen across all 20 datasets together
- "Best" = highest average ARI across all datasets of that type

### Tables to Reproduce
- Tables A.1-A.4: Average ARI for synthetic data (Circles, Moons, RSG, Repliclust)
- Tables for real-world: k-means, AHC, GMM, OPTICS ARI per dataset (Tables in appendix)
- Tables 1-4: Aggregate (% wins, avg win/loss for each clustering algo)
- Wilcoxon signed-rank test table
- Boxplot figures (8 total: 4 synthetic + 4 real-world)

## Implementation Plan
- [x] 1. UCI data loading (load_uci.py) - tested, all 20 datasets correct
- [x] 2. Synthetic data generation code (generate_data.py) - written, not run
- [ ] 3. Build main experiment pipeline (experiment.py)
  - [ ] 3a. DR methods (PCA, KPCA, VAE, Isomap, MDS)
  - [ ] 3b. Clustering with hyperparameter search
  - [ ] 3c. Evaluation (ARI computation)
- [ ] 4. Run on real-world data first (faster, has ground truth tables to compare)
- [ ] 5. Run on synthetic data
- [ ] 6. Generate aggregate tables and plots
- [ ] 7. Wilcoxon test
- [ ] 8. Create reproduce.sh and REPORT.md

## Key Design Decisions
- Run real-world experiments FIRST since we have exact ARI values to compare against
- For computational efficiency: may reduce synthetic dataset count if needed
- VAE uses sigmoid output activation + MSE loss (paper specifies this)
- z-score normalization applied BEFORE DR (paper: "all datasets were preprocessed using z-score normalization")
- For synthetic data: noise injection happens during generation, then z-score before DR

## Files
- `/workspace/load_uci.py` - UCI dataset loader, TESTED
- `/workspace/generate_data.py` - Synthetic data generator, NOT YET RUN
- `/workspace/uci_data/` - Downloaded UCI data files

## Failed Approaches
- Previous implementation was for wrong paper (AdaCD/LLM safety paper)
- Cleaned workspace and started fresh for DR+clustering paper
