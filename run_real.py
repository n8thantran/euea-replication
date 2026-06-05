#!/usr/bin/env python3
"""Run real-world experiments in stages with DR cache persistence."""
import os
import sys
import json
import pickle
import time
import numpy as np

# Stage 1: Precompute DR and save cache
# Stage 2: Run all clustering algorithms
# Stage 3: Generate tables, stats, plots

def stage1_dr():
    """Precompute DR transformations and save to disk."""
    from load_uci import load_all_uci
    from experiment import precompute_all_dr
    
    print("=" * 60)
    print("STAGE 1: LOADING DATA + PRECOMPUTING DR")
    print("=" * 60)
    
    dataset_list = load_all_uci()
    datasets = {}
    for name, X, y, k in dataset_list:
        datasets[name] = (X, y, k)
    print(f"Loaded {len(datasets)} datasets")
    
    t0 = time.time()
    dr_cache = precompute_all_dr(datasets)
    print(f"DR precomputation took {time.time()-t0:.1f}s")
    
    # Save
    with open('results/dr_cache_real.pkl', 'wb') as f:
        pickle.dump(dr_cache, f)
    with open('results/datasets_real.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    print("Saved DR cache and datasets to disk")


def stage2_clustering():
    """Run all clustering algorithms using cached DR."""
    from experiment import (
        run_kmeans_experiments, run_ahc_experiments, run_gmm_experiments,
        run_optics_experiments, find_best_ahc_params, find_best_gmm_params,
        find_best_optics_params
    )
    
    print("=" * 60)
    print("STAGE 2: CLUSTERING")
    print("=" * 60)
    
    with open('results/dr_cache_real.pkl', 'rb') as f:
        dr_cache = pickle.load(f)
    with open('results/datasets_real.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    all_results = {}
    
    # k-means
    print("\n--- K-MEANS ---")
    t0 = time.time()
    all_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
    print(f"k-means took {time.time()-t0:.1f}s")
    
    # AHC
    print("\n--- AHC ---")
    t0 = time.time()
    ahc_metric, ahc_linkage = find_best_ahc_params(datasets, dr_cache)
    all_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_metric, ahc_linkage)
    print(f"AHC took {time.time()-t0:.1f}s")
    
    # GMM
    print("\n--- GMM ---")
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    all_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    print(f"GMM took {time.time()-t0:.1f}s")
    
    # OPTICS
    print("\n--- OPTICS ---")
    t0 = time.time()
    optics_ms, optics_mcs = find_best_optics_params(datasets, dr_cache)
    all_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, optics_ms, optics_mcs)
    print(f"OPTICS took {time.time()-t0:.1f}s")
    
    # Save
    with open('results/real_world_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    hyperparams = {
        'AHC': {'metric': ahc_metric, 'linkage': ahc_linkage},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(optics_ms), 'min_cluster_size': float(optics_mcs)},
    }
    with open('results/chosen_hyperparams_real.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print("\nAll clustering done!")
    return all_results


def stage3_analysis():
    """Generate tables, stats, plots."""
    from experiment import (
        format_results_table, compute_aggregate_stats, compute_wilcoxon_tests,
        generate_boxplots, DR_METHODS, REDUCTION_LEVELS
    )
    
    print("=" * 60)
    print("STAGE 3: ANALYSIS")
    print("=" * 60)
    
    with open('results/real_world_results.json', 'r') as f:
        all_results = json.load(f)
    
    # Format tables
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        df = format_results_table(all_results[algo], algo)
        df.to_csv(f'results/table_{algo}_real.csv')
        print(f"\n{algo} results:")
        print(df.to_string())
    
    # Aggregate stats
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    agg_stats = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        agg_stats[algo] = compute_aggregate_stats(all_results[algo])
        print(f"\n{algo}:")
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                s = agg_stats[algo][method][level]
                print(f"  {method:12s} {level:4s}: win={s['win_pct']:5.1f}%  loss={s['loss_pct']:5.1f}%  avg_win={s['avg_win_change']:+7.2f}%  avg_loss={s['avg_loss_change']:+7.2f}%")
    with open('results/aggregate_stats_real.json', 'w') as f:
        json.dump(agg_stats, f, indent=2)
    
    # Wilcoxon test
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 60)
    wilcoxon_results = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        wilcoxon_results[algo] = compute_wilcoxon_tests(all_results[algo])
        print(f"\n{algo}:")
        for method in DR_METHODS:
            vals = [wilcoxon_results[algo][method][l] for l in REDUCTION_LEVELS]
            sig = ['*' if v < 0.05 else ' ' for v in vals]
            print(f"  {method:12s}: k-1={vals[0]:.4f}{sig[0]}  25%={vals[1]:.4f}{sig[1]}  50%={vals[2]:.4f}{sig[2]}")
    with open('results/wilcoxon_results_real.json', 'w') as f:
        json.dump(wilcoxon_results, f, indent=2)
    
    # Boxplots
    generate_boxplots(all_results, 'RealWorld', './results')
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    if len(sys.argv) > 1:
        stage = sys.argv[1]
    else:
        stage = 'all'
    
    if stage in ['1', 'dr', 'all']:
        stage1_dr()
    if stage in ['2', 'cluster', 'all']:
        stage2_clustering()
    if stage in ['3', 'analysis', 'all']:
        stage3_analysis()
