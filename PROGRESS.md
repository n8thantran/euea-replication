# Dimensionality Reduction for Clustering - Implementation Progress

## Current Phase: Fixing experiment code and running real-world experiments

## Paper Summary
- **Goal**: Systematic comparison of 5 DR methods × 4 clustering algorithms × 3 reduction levels
- **DR Methods**: PCA, Kernel PCA, VAE, Isomap, MDS
- **Clustering**: k-means, AHC (Agglomerative), GMM, OPTICS
- **Reduction levels**: k-1 dims, 25% of features, 50% of features
- **Metric**: Adjusted Rand Index (ARI)
- **Datasets**: 1165 synthetic (Circles, Moons, RSG, Repliclust) + 20 UCI real-world

## Implementation Plan
- [x] 1. Read paper thoroughly
- [x] 2. Implement UCI dataset loader (load_uci_datasets.py) - 20 datasets
- [x] 3. Download UCI data files
- [x] 4. Implement DR methods (PCA, KernelPCA, VAE, Isomap, MDS)
- [x] 5. Implement clustering (k-means, AHC, GMM, OPTICS)
- [x] 6. Implement evaluation pipeline (ARI, aggregate stats, Wilcoxon)
- [x] 7. Write experiment.py with full pipeline
- [ ] 8. Fix AHC (try all affinity/linkage combos, pick best)
- [ ] 9. Fix GMM (try all covariance types, pick best)
- [ ] 10. Fix OPTICS (min_samples 5-10 step 1, xi 0-1 step 0.05)
- [ ] 11. Run real-world experiments
- [ ] 12. Run synthetic experiments (at least representative subset)
- [ ] 13. Generate boxplot figures
- [ ] 14. Compare results to paper's Tables A.5-A.8 (real-world ARI)
- [ ] 15. Compare aggregate stats to paper's Tables 1-4
- [ ] 16. Compare Wilcoxon test to paper's Table A.9
- [ ] 17. Create reproduce.sh and REPORT.md

## Key Paper Details
- **Preprocessing**: z-score normalization
- **k-means**: k-means++ init, n_init=100
- **AHC**: Try affinity (euclidean, l1, l2, manhattan, cosine) × linkage (complete, average, single, ward). Ward only with euclidean. Pick best per dataset type.
- **GMM**: Try covariance types (spherical, tied, diag, full), n_init=10. Pick best per dataset type.
- **OPTICS**: xi method, min_samples=[5,6,7,8,9,10], min_cluster_size=[0.0,0.05,...,1.0], best per dataset type
- **MDS**: random_state=10, n_init=50
- **Kernel PCA**: rbf kernel
- **VAE**: encoder [64,32]+BN+Dropout(0.4), latent, decoder [32,64]+sigmoid, MSE+KL, Adam, 100 epochs, batch=64, 70/30 split
- **Reduction dims**: k-1: max(k-1, 2); 25%: round(0.25*d); 50%: round(0.50*d)

## Completed Work
- /workspace/load_uci_datasets.py - Loads 19/20 UCI datasets (Segmentation handled separately in experiment.py)
- /workspace/experiment.py - Full experiment pipeline (needs AHC/GMM/OPTICS fixes)
- /workspace/uci_data/ - Downloaded UCI data files

## Key Issues to Fix
1. AHC: Need to try all affinity/linkage combos and pick best (currently using defaults)
2. GMM: Need to try all covariance types and pick best (currently using defaults)
3. OPTICS: Need full parameter grid (min_samples 5-10, xi 0-1 step 0.05)
4. For real-world data, paper picks best hyperparams per dataset (not per dataset type)

## Files
- /workspace/experiment.py - Main experiment code
- /workspace/load_uci_datasets.py - UCI dataset loader
- /workspace/uci_data/ - Data files
