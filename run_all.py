#!/usr/bin/env python3
"""
Complete experiment runner for the DR+Clustering paper replication.
Runs both real-world and synthetic experiments, generates all tables and figures.
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

# Ensure results directory exists
os.makedirs('./results', exist_ok=True)

from experiment import (
    DR_METHODS, REDUCTION_LEVELS, get_all_conditions,
    precompute_all_dr, 
    run_kmeans_experiments, run_ahc_experiments, run_gmm_experiments, run_optics_experiments,
    find_best_ahc_params, find_best_gmm_params, find_best_optics_params,
    compute_aggregate_stats, compute_wilcoxon_tests, generate_boxplots,
    format_results_table, format_aggregate_table, format_wilcoxon_table
)
from load_uci import load_all_uci
from generate_data import generate_all_synthetic_datasets


def load_all_uci_datasets():
    """Load all UCI datasets and return as dict {name: (X, y, k)}."""
    datasets_list = load_all_uci()
    datasets = {}
    for name, X, y, k in datasets_list:
        datasets[name] = (X, y, k)
    return datasets


def run_real_world_experiments():
    """Run all experiments on 20 UCI real-world datasets."""
    print("=" * 70)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 70)
    
    # Load datasets
    print("\n[1/6] Loading UCI datasets...")
    datasets = load_all_uci_datasets()
    print(f"  Loaded {len(datasets)} datasets")
    
    # Precompute DR
    print("\n[2/6] Precomputing DR transformations...")
    cache_file = './results/dr_cache_real_final.pkl'
    if os.path.exists(cache_file):
        print(f"  Loading cached DR from {cache_file}")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
        # Check if all datasets are in cache
        missing = [k for k in datasets if k not in dr_cache]
        if missing:
            print(f"  Missing {len(missing)} datasets in cache, recomputing all...")
            dr_cache = precompute_all_dr(datasets)
            with open(cache_file, 'wb') as f:
                pickle.dump(dr_cache, f)
    else:
        dr_cache = precompute_all_dr(datasets)
        with open(cache_file, 'wb') as f:
            pickle.dump(dr_cache, f)
    
    # Run clustering experiments
    all_results = {}
    
    print("\n[3/6] Running k-means experiments...")
    t0 = time.time()
    all_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    print("\n[4/6] Running AHC experiments...")
    t0 = time.time()
    ahc_metric, ahc_linkage = find_best_ahc_params(datasets, dr_cache)
    all_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_metric, ahc_linkage)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    print("\n[5/6] Running GMM experiments...")
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    all_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    print("\n[6/6] Running OPTICS experiments...")
    t0 = time.time()
    optics_ms, optics_mcs = find_best_optics_params(datasets, dr_cache)
    all_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, optics_ms, optics_mcs)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Save raw results
    with open('./results/real_world_results_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save hyperparameters
    hyperparams = {
        'AHC': {'metric': ahc_metric, 'linkage': ahc_linkage},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(optics_ms), 'min_cluster_size': float(optics_mcs)},
    }
    with open('./results/hyperparams_real.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    return all_results


def run_synthetic_experiments():
    """Run all experiments on synthetic datasets (4 types × N datasets each)."""
    print("=" * 70)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 70)
    
    # Paper uses many datasets per config; use 10 per config for feasibility
    n_datasets = 10
    
    synthetic_types = ['Circles', 'Moons', 'RSG', 'Repliclust']
    all_type_results = {}
    
    for dtype in synthetic_types:
        print(f"\n{'='*50}")
        print(f"Synthetic type: {dtype}")
        print(f"{'='*50}")
        
        print(f"\n  Generating {n_datasets} {dtype} datasets per config...")
        datasets = generate_all_synthetic_datasets(dtype, n_datasets=n_datasets)
        print(f"  Generated {len(datasets)} datasets")
        
        if len(datasets) == 0:
            print(f"  WARNING: No datasets generated for {dtype}, skipping")
            continue
        
        print(f"\n  Precomputing DR transformations...")
        cache_file = f'./results/dr_cache_{dtype}_final.pkl'
        if os.path.exists(cache_file):
            print(f"  Loading cached DR from {cache_file}")
            with open(cache_file, 'rb') as f:
                dr_cache = pickle.load(f)
            missing = [k for k in datasets if k not in dr_cache]
            if missing:
                print(f"  Missing {len(missing)} datasets, recomputing...")
                dr_cache = precompute_all_dr(datasets, timeout_sec=60)
                with open(cache_file, 'wb') as f:
                    pickle.dump(dr_cache, f)
        else:
            dr_cache = precompute_all_dr(datasets, timeout_sec=60)
            with open(cache_file, 'wb') as f:
                pickle.dump(dr_cache, f)
        
        type_results = {}
        
        print(f"\n  Running k-means...")
        t0 = time.time()
        type_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
        print(f"  k-means done in {time.time()-t0:.1f}s")
        
        print(f"\n  Running AHC...")
        t0 = time.time()
        ahc_metric, ahc_linkage = find_best_ahc_params(datasets, dr_cache)
        type_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_metric, ahc_linkage)
        print(f"  AHC done in {time.time()-t0:.1f}s")
        
        print(f"\n  Running GMM...")
        t0 = time.time()
        gmm_cov = find_best_gmm_params(datasets, dr_cache)
        type_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
        print(f"  GMM done in {time.time()-t0:.1f}s")
        
        print(f"\n  Running OPTICS...")
        t0 = time.time()
        optics_ms, optics_mcs = find_best_optics_params(datasets, dr_cache)
        type_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, optics_ms, optics_mcs)
        print(f"  OPTICS done in {time.time()-t0:.1f}s")
        
        all_type_results[dtype] = type_results
        
        # Save per-type results
        with open(f'./results/synthetic_{dtype}_results.json', 'w') as f:
            json.dump(type_results, f, indent=2)
        
        print(f"\n  {dtype} complete!")
    
    return all_type_results


def generate_all_outputs(real_results, synthetic_results):
    """Generate all tables, figures, and statistics."""
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)
    
    conditions = get_all_conditions()
    
    # ============================================================
    # REAL-WORLD TABLES (Appendix: per-dataset ARI tables)
    # ============================================================
    print("\n--- Real-world per-dataset tables ---")
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo not in real_results:
            continue
        df = format_results_table(real_results[algo], algo)
        fname = f'./results/table_{algo}_real.csv'
        df.to_csv(fname)
        print(f"  Saved {fname}")
    
    # ============================================================
    # SYNTHETIC TABLES (Appendix: average ARI per data type)
    # ============================================================
    print("\n--- Synthetic average ARI tables ---")
    for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        if dtype not in synthetic_results:
            continue
        type_res = synthetic_results[dtype]
        # Compute average ARI across all datasets of this type
        avg_table = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            if algo not in type_res:
                continue
            algo_res = type_res[algo]
            avg_row = {}
            for cond in conditions:
                vals = [algo_res[ds].get(cond, 0.0) for ds in algo_res]
                avg_row[cond] = round(np.mean(vals), 3) if vals else 0.0
            avg_table[algo] = avg_row
        
        df = pd.DataFrame(avg_table).T
        df.columns = conditions
        fname = f'./results/table_avg_ARI_{dtype}.csv'
        df.to_csv(fname)
        print(f"  Saved {fname}")
    
    # ============================================================
    # AGGREGATE TABLES (Main body: Tables 1-4)
    # ============================================================
    print("\n--- Aggregate statistics tables ---")
    
    # Combine all synthetic results into one pool per algorithm
    synthetic_combined = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        combined = {}
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            if dtype in synthetic_results and algo in synthetic_results[dtype]:
                for ds_name, row in synthetic_results[dtype][algo].items():
                    combined[f"{dtype}_{ds_name}"] = row
        synthetic_combined[algo] = combined
    
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        # Real-world aggregate
        if algo in real_results:
            real_stats = compute_aggregate_stats(real_results[algo])
        else:
            real_stats = None
        
        # Synthetic aggregate
        if algo in synthetic_combined and synthetic_combined[algo]:
            synth_stats = compute_aggregate_stats(synthetic_combined[algo])
        else:
            synth_stats = None
        
        # Build combined table matching paper format
        rows = []
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                row = {'Method': method, 'Reduction': level}
                if synth_stats:
                    s = synth_stats[method][level]
                    row['Synth Win %'] = s['win_pct']
                    row['Synth Avg Win/Loss %'] = s['avg_win_loss_pct']
                if real_stats:
                    r = real_stats[method][level]
                    row['Real Win %'] = r['win_pct']
                    row['Real Avg Win/Loss %'] = r['avg_win_loss_pct']
                rows.append(row)
        
        df = pd.DataFrame(rows)
        fname = f'./results/table_aggregate_{algo}.csv'
        df.to_csv(fname, index=False)
        print(f"  Saved {fname}")
    
    # ============================================================
    # WILCOXON TESTS (Real-world only, paper Table 5)
    # ============================================================
    print("\n--- Wilcoxon signed-rank tests ---")
    wilcoxon_all = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo in real_results:
            wilcoxon_all[algo] = compute_wilcoxon_tests(real_results[algo])
    
    with open('./results/wilcoxon_final.json', 'w') as f:
        json.dump(wilcoxon_all, f, indent=2)
    
    # Format as table
    rows = []
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo not in wilcoxon_all:
            continue
        row = {'Algorithm': algo}
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                row[key] = wilcoxon_all[algo][method][level]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv('./results/table_wilcoxon.csv', index=False)
    print(f"  Saved table_wilcoxon.csv")
    
    # ============================================================
    # BOXPLOTS
    # ============================================================
    print("\n--- Boxplots ---")
    
    # Real-world boxplots
    if real_results:
        generate_boxplots(real_results, 'RealWorld', './results')
    
    # Synthetic boxplots (combine all types)
    synth_boxplot_data = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        combined = {}
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            if dtype in synthetic_results and algo in synthetic_results[dtype]:
                for ds_name, row in synthetic_results[dtype][algo].items():
                    combined[f"{dtype}_{ds_name}"] = row
        if combined:
            synth_boxplot_data[algo] = combined
    
    if synth_boxplot_data:
        generate_boxplots(synth_boxplot_data, 'Synthetic', './results')
    
    print("\n--- All outputs generated! ---")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-only', action='store_true', help='Run only real-world experiments')
    parser.add_argument('--synth-only', action='store_true', help='Run only synthetic experiments')
    parser.add_argument('--outputs-only', action='store_true', help='Only generate outputs from saved results')
    args = parser.parse_args()
    
    real_results = None
    synthetic_results = None
    
    if args.outputs_only:
        # Load saved results
        if os.path.exists('./results/real_world_results_final.json'):
            with open('./results/real_world_results_final.json') as f:
                real_results = json.load(f)
        
        synthetic_results = {}
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            fname = f'./results/synthetic_{dtype}_results.json'
            if os.path.exists(fname):
                with open(fname) as f:
                    synthetic_results[dtype] = json.load(f)
    else:
        if not args.synth_only:
            real_results = run_real_world_experiments()
        
        if not args.real_only:
            synthetic_results = run_synthetic_experiments()
        
        # Load any missing results from files
        if real_results is None and os.path.exists('./results/real_world_results_final.json'):
            with open('./results/real_world_results_final.json') as f:
                real_results = json.load(f)
        
        if synthetic_results is None:
            synthetic_results = {}
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            if dtype not in synthetic_results:
                fname = f'./results/synthetic_{dtype}_results.json'
                if os.path.exists(fname):
                    with open(fname) as f:
                        synthetic_results[dtype] = json.load(f)
    
    # Generate all outputs
    if real_results is None:
        real_results = {}
    if synthetic_results is None:
        synthetic_results = {}
    
    generate_all_outputs(real_results, synthetic_results)
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
