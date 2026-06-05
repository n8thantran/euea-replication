#!/usr/bin/env python3
"""
Generate all final tables and summary statistics from existing JSON results.
This script consolidates results from prior experiment runs into the paper's table formats.
"""

import json
import os
import numpy as np
from scipy.stats import wilcoxon
import csv

RESULTS_DIR = '/workspace/results'

ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']
DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
LEVELS = ['k-1', '25%', '50%']

SYNTH_TYPES = ['Circles', 'Moons', 'RSG', 'Repliclust']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_results_json(data):
    """Parse results JSON into structured format.
    
    Input format (from run_experiments.py):
    {
        "k-means": {"dataset_name": {"No Reduction": 0.65, "PCA_k-1": 0.66, ...}},
        "AHC": {...},
        ...
    }
    
    Returns:
    - baseline[algo] = list of ARI scores
    - dr_scores[algo][method][level] = list of ARI scores
    - dataset_names = list
    """
    baseline = {a: [] for a in ALGOS}
    dr_scores = {a: {dr: {lv: [] for lv in LEVELS} for dr in DR_METHODS} for a in ALGOS}
    per_dataset = {}
    
    # Get list of datasets from first algo
    first_algo = None
    for a in ALGOS:
        if a in data and not a.startswith('_'):
            ds_data = data[a]
            ds_names = [k for k in ds_data.keys() if not k.startswith('_')]
            if ds_names:
                first_algo = a
                break
    
    if first_algo is None:
        return baseline, dr_scores, per_dataset, []
    
    ds_names = sorted([k for k in data[first_algo].keys() if not k.startswith('_')])
    
    for name in ds_names:
        per_dataset[name] = {}
        for algo in ALGOS:
            if algo not in data or name not in data[algo]:
                continue
            
            ds = data[algo][name]
            base_val = ds.get('No Reduction', 0.0)
            baseline[algo].append(base_val)
            per_dataset[name][algo] = {'baseline': base_val, 'dr': {}}
            
            for dr in DR_METHODS:
                for lv in LEVELS:
                    key = f"{dr}_{lv}"
                    val = ds.get(key, base_val)
                    dr_scores[algo][dr][lv].append(val)
                    per_dataset[name][algo]['dr'][(dr, lv)] = val
    
    return baseline, dr_scores, per_dataset, ds_names


def compute_win_stats(dr_vals, base_vals):
    """Compute win% and avg win/loss for DR values vs baseline."""
    dr_arr = np.array(dr_vals)
    base_arr = np.array(base_vals)
    n = len(dr_arr)
    if n == 0:
        return 0, 0
    
    wins = np.sum(dr_arr > base_arr + 1e-6)
    losses = np.sum(dr_arr < base_arr - 1e-6)
    non_ties = wins + losses
    win_pct = (wins / non_ties * 100) if non_ties > 0 else 50.0
    
    # Average relative difference
    diffs = dr_arr - base_arr
    rel_diffs = []
    for d_val, b_val in zip(diffs, base_arr):
        if abs(b_val) > 1e-6:
            rel_diffs.append(d_val / abs(b_val) * 100)
        else:
            rel_diffs.append(d_val * 100)
    avg_diff = np.mean(rel_diffs) if rel_diffs else 0.0
    
    return round(win_pct, 1), round(avg_diff, 1)


def compute_wilcoxon_test(dr_vals, base_vals):
    """One-sided Wilcoxon signed-rank test: H1: DR > baseline."""
    dr_arr = np.array(dr_vals)
    base_arr = np.array(base_vals)
    diffs = dr_arr - base_arr
    non_zero = diffs[np.abs(diffs) > 1e-10]
    
    if len(non_zero) < 2:
        return 1.0
    try:
        _, p = wilcoxon(dr_arr, base_arr, alternative='greater')
        return round(p, 3)
    except:
        return 1.0


# ============================================================
# Generate Tables
# ============================================================

def generate_synthetic_ari_tables(all_synth_data):
    """Generate Tables A.1-A.4: Average ARI per synthetic type."""
    for dtype in SYNTH_TYPES:
        if dtype not in all_synth_data:
            print(f"  Skipping {dtype} - no data")
            continue
        
        baseline, dr_scores, _, ds_names = all_synth_data[dtype]
        
        fname = f"{RESULTS_DIR}/table_synthetic_{dtype}.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Algorithm', 'No Reduction']
            for dr in DR_METHODS:
                for lv in LEVELS:
                    header.append(f'{dr}_{lv}')
            writer.writerow(header)
            
            for algo in ALGOS:
                row = [algo]
                base_mean = np.mean(baseline[algo]) if baseline[algo] else 0
                row.append(f"{base_mean:.3f}")
                
                for dr in DR_METHODS:
                    for lv in LEVELS:
                        vals = dr_scores[algo][dr][lv]
                        mean_val = np.mean(vals) if vals else 0
                        row.append(f"{mean_val:.3f}")
                
                writer.writerow(row)
        
        print(f"  Saved {fname}")


def generate_combined_aggregate_tables(synth_data_all, real_data):
    """Generate Tables 1-4: Combined synthetic+real aggregate stats."""
    # Combine all synthetic data
    synth_baseline = {a: [] for a in ALGOS}
    synth_dr = {a: {dr: {lv: [] for lv in LEVELS} for dr in DR_METHODS} for a in ALGOS}
    
    for dtype in SYNTH_TYPES:
        if dtype not in synth_data_all:
            continue
        baseline, dr_scores, _, _ = synth_data_all[dtype]
        for a in ALGOS:
            synth_baseline[a].extend(baseline[a])
            for dr in DR_METHODS:
                for lv in LEVELS:
                    synth_dr[a][dr][lv].extend(dr_scores[a][dr][lv])
    
    real_baseline, real_dr_scores, _, _ = real_data
    
    for algo in ALGOS:
        fname = f"{RESULTS_DIR}/table_aggregate_{algo}.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Reduction', 'Win% Synthetic', 'Win% Real', 
                           'Avg Win/Loss Synthetic', 'Avg Win/Loss Real'])
            
            for dr in DR_METHODS:
                for lv in LEVELS:
                    s_win, s_avg = compute_win_stats(
                        synth_dr[algo][dr][lv], synth_baseline[algo])
                    r_win, r_avg = compute_win_stats(
                        real_dr_scores[algo][dr][lv], real_baseline[algo])
                    
                    writer.writerow([dr, lv, s_win, r_win, s_avg, r_avg])
        
        print(f"  Saved {fname}")


def generate_combined_wilcoxon_table(synth_data_all, real_data):
    """Generate Table 5: Combined Wilcoxon test results."""
    # Combine all synthetic data
    synth_baseline = {a: [] for a in ALGOS}
    synth_dr = {a: {dr: {lv: [] for lv in LEVELS} for dr in DR_METHODS} for a in ALGOS}
    
    for dtype in SYNTH_TYPES:
        if dtype not in synth_data_all:
            continue
        baseline, dr_scores, _, _ = synth_data_all[dtype]
        for a in ALGOS:
            synth_baseline[a].extend(baseline[a])
            for dr in DR_METHODS:
                for lv in LEVELS:
                    synth_dr[a][dr][lv].extend(dr_scores[a][dr][lv])
    
    real_baseline, real_dr_scores, _, _ = real_data
    
    fname = f"{RESULTS_DIR}/table_wilcoxon_combined.csv"
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Algorithm']
        for dr in DR_METHODS:
            for lv in LEVELS:
                header.append(f'{dr}_{lv}_Synth')
                header.append(f'{dr}_{lv}_Real')
        writer.writerow(header)
        
        for algo in ALGOS:
            row = [algo]
            for dr in DR_METHODS:
                for lv in LEVELS:
                    p_synth = compute_wilcoxon_test(
                        synth_dr[algo][dr][lv], synth_baseline[algo])
                    p_real = compute_wilcoxon_test(
                        real_dr_scores[algo][dr][lv], real_baseline[algo])
                    row.append(f"{p_synth:.3f}")
                    row.append(f"{p_real:.3f}")
            writer.writerow(row)
    
    print(f"  Saved {fname}")


def generate_summary_report(synth_data_all, real_data):
    """Generate a text summary of key findings."""
    real_baseline, real_dr_scores, _, _ = real_data
    
    fname = f"{RESULTS_DIR}/summary.txt"
    with open(fname, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SUMMARY OF RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Real-world summary
        f.write("REAL-WORLD DATASETS (20 UCI datasets)\n")
        f.write("-" * 40 + "\n")
        for algo in ALGOS:
            f.write(f"\n{algo}:\n")
            f.write(f"  Baseline avg ARI: {np.mean(real_baseline[algo]):.3f}\n")
            best_dr, best_lv, best_win = None, None, 0
            for dr in DR_METHODS:
                for lv in LEVELS:
                    win_pct, avg_diff = compute_win_stats(
                        real_dr_scores[algo][dr][lv], real_baseline[algo])
                    if win_pct > best_win:
                        best_win = win_pct
                        best_dr = dr
                        best_lv = lv
            f.write(f"  Best DR: {best_dr} at {best_lv} (Win%={best_win})\n")
        
        # Synthetic summary
        f.write("\n\nSYNTHETIC DATASETS\n")
        f.write("-" * 40 + "\n")
        for dtype in SYNTH_TYPES:
            if dtype not in synth_data_all:
                continue
            baseline, dr_scores, _, ds_names = synth_data_all[dtype]
            f.write(f"\n{dtype} ({len(ds_names)} datasets):\n")
            for algo in ALGOS:
                base_mean = np.mean(baseline[algo]) if baseline[algo] else 0
                best_dr, best_lv, best_val = None, None, base_mean
                for dr in DR_METHODS:
                    for lv in LEVELS:
                        mean_val = np.mean(dr_scores[algo][dr][lv]) if dr_scores[algo][dr][lv] else 0
                        if mean_val > best_val:
                            best_val = mean_val
                            best_dr = dr
                            best_lv = lv
                if best_dr:
                    f.write(f"  {algo}: baseline={base_mean:.3f}, best={best_dr}/{best_lv} ({best_val:.3f})\n")
                else:
                    f.write(f"  {algo}: baseline={base_mean:.3f}, no improvement\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  Saved {fname}")


def main():
    print("=" * 60)
    print("GENERATING FINAL RESULTS FROM EXISTING DATA")
    print("=" * 60)
    
    # Load real-world results
    print("\nLoading real-world results...")
    real_json = load_json(f"{RESULTS_DIR}/real_results_final.json")
    real_data = parse_results_json(real_json)
    baseline, dr_scores, per_dataset, ds_names = real_data
    print(f"  Loaded {len(ds_names)} datasets: {ds_names[:5]}...")
    
    # Load synthetic results
    print("\nLoading synthetic results...")
    all_synth_data = {}
    for dtype in SYNTH_TYPES:
        fpath = f"{RESULTS_DIR}/synth_{dtype}_final.json"
        if os.path.exists(fpath):
            synth_json = load_json(fpath)
            all_synth_data[dtype] = parse_results_json(synth_json)
            _, _, _, names = all_synth_data[dtype]
            print(f"  {dtype}: {len(names)} datasets")
        else:
            print(f"  {dtype}: NOT FOUND")
    
    # Generate synthetic ARI tables (Tables A.1-A.4)
    print("\nGenerating synthetic ARI tables...")
    generate_synthetic_ari_tables(all_synth_data)
    
    # Generate combined aggregate tables (Tables 1-4)
    print("\nGenerating combined aggregate tables...")
    generate_combined_aggregate_tables(all_synth_data, real_data)
    
    # Generate combined Wilcoxon table (Table 5)
    print("\nGenerating combined Wilcoxon table...")
    generate_combined_wilcoxon_table(all_synth_data, real_data)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(all_synth_data, real_data)
    
    print("\n" + "=" * 60)
    print("ALL TABLES GENERATED")
    print("=" * 60)


if __name__ == '__main__':
    main()
