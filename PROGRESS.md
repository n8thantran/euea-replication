# Dimensionality Reduction for Clustering - Implementation Progress

## Current Phase: Final deliverables (reproduce.sh, REPORT.md, verification)

## Status Summary
- **All experiments completed**: Real-world (20 UCI datasets) + Synthetic (Circles, Moons, RSG, Repliclust)
- **All outputs generated**: 71 CSV tables, 36 PDF boxplots, Wilcoxon tests
- **Key scripts**:
  - `main_pipeline.py`: Full experiment pipeline (can run from scratch)
  - `generate_all_outputs.py`: Generates all tables/boxplots from cached JSON results
  - `load_uci.py`: Loads all 20 UCI datasets
- **Result files**: All in `results/` directory
  - `results_RealWorld.json`, `results_Circles.json`, `results_Moons.json`, `results_RSG.json`, `results_Repliclust.json`
  - `table_*_RealWorld.csv` (4 algos), `table_*_{synth_type}.csv` (4×4)
  - `table_aggregate_*.csv` (4 algos - combined synth+real)
  - `table_wilcoxon.csv` (combined)
  - `boxplot_*_RealWorld.pdf`, `boxplot_*_Synthetic_*.pdf`

## Remaining TODO
- [ ] Write clean `reproduce.sh`
- [ ] Write `REPORT.md`
- [ ] Verify reproduce.sh runs
- [ ] Final commit + end_task

## Paper Summary
**Title**: "Assessing the impact of dimensionality reduction on clustering performance"
- 5 DR methods × 4 clustering algorithms × 3 reduction levels
- Evaluated on synthetic (Circles, Moons, RSG, Repliclust) + 20 real-world UCI datasets
- Key metric: ARI (Adjusted Rand Index)
- Aggregate stats: Win% (ties excluded), Avg win/loss %
- Wilcoxon signed-rank test for statistical significance

## Key Implementation Decisions
- N_SYNTH_REPS=5 (paper uses 50 - computational constraint)
- Fewer synthetic dataset configs than paper (10-20 vs 50 per type)
- MDS n_init=10 (paper uses 50) for speed
- VAE: 100 epochs, batch=64, encoder d→64→32→latent, MSE+KL loss, sigmoid output
- z-score normalization before DR; min-max for VAE internally
- HP search: AHC over metric×linkage; GMM over cov types; OPTICS over min_samples×min_cluster_size

## File Map
- `/workspace/main_pipeline.py` - Complete experiment pipeline (authoritative)
- `/workspace/generate_all_outputs.py` - Generate tables/plots from cached results
- `/workspace/load_uci.py` - UCI dataset loading (all 20 datasets)
- `/workspace/results/` - All output files
- `/workspace/reproduce.sh` - To be updated
- `/workspace/REPORT.md` - To be written

## Known Differences from Paper
- Aggregate numbers differ due to fewer synthetic reps and datasets
- Some qualitative trends match (VAE consistently bad, Kernel PCA mixed)
- OPTICS avg win/loss% values are very large due to near-zero baselines
