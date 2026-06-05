# Dimensionality Reduction for Clustering - Implementation Progress

## Current Phase: Running real-world and synthetic experiments

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
- [ ] 8. Run real-world experiments
- [ ] 9. Run synthetic experiments
- [ ] 10. Generate boxplot figures
- [ ] 11. Compare results to paper's Tables A.5-A.8 (real-world ARI)
- [ ] 12. Compare aggregate stats to paper's Tables 1-4
- [ ] 13. Compare Wilcoxon test to paper's Table A.9
- [ ] 14. Create reproduce.sh and REPORT.md

## Key Paper Details
- **Preprocessing**: z-score normalization
- **k-means**: k-means++ init, n_init=100
- **AHC**: Best affinity/linkage per dataset type (including ward, complete, average, single)
- **GMM**: Best covariance type per dataset, n_init=10
- **OPTICS**: xi method, min_samples=[5-10], min_cluster_size=[0-1 step 0.05], best per dataset
- **MDS**: random_state=10, n_init=50
- **Kernel PCA**: rbf kernel
- **VAE**: encoder [64,32]+BN+Dropout(0.4), latent, decoder [32,64]+sigmoid, MSE+KL, Adam, 100 epochs, batch=64, 70/30 split
- **Reduction dims**: k-1: max(k-1, 2); 25%: round(0.25*d); 50%: round(0.50*d)

## Completed Work
- /workspace/load_uci_datasets.py - Loads all 20 UCI datasets
- /workspace/experiment.py - Full experiment pipeline (DR + clustering + evaluation)
- /workspace/uci_data/ - Downloaded UCI data files

## Files to Clean Up
- Old AdaCD files: run_all.py, inference.py, evaluate.py, evaluate_llm.py, run_inference.py, run_fast.py, data_utils.py, download_datasets.py
- data/ directory with text files from previous paper
- outputs/ directory from previous project
