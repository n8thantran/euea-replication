"""
Generate final combined aggregate tables (Tables 1-4 from paper) and Wilcoxon test table (Table 5).
These combine synthetic and real-world results into single tables per clustering algorithm.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import os
import warnings
warnings.filterwarnings('ignore')

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']
SYNTH_TYPES = ['Circles', 'Moons', 'RSG', 'Repliclust']

def load_results(dtype):
    """Load results JSON for a data type."""
    path = f'results/results_{dtype}.json'
    with open(path) as f:
        return json.load(f)

def compute_aggregate_stats(results, algo):
    """Compute win% and avg win/loss% for each DR method and level."""
    datasets = list(results[algo].keys())
    stats = {}
    for method in DR_METHODS:
        stats[method] = {}
        for level in LEVELS:
            key = f'{method}_{level}'
            wins = 0
            total = 0
            diffs = []
            for ds in datasets:
                baseline = results[algo][ds].get('No Reduction', 0)
                reduced = results[algo][ds].get(key, None)
                if reduced is None:
                    continue
                total += 1
                diff_pct = (reduced - baseline) * 100 if baseline != 0 else (reduced * 100)
                # Win = reduced ARI > baseline ARI
                if reduced > baseline + 1e-6:
                    wins += 1
                diffs.append(diff_pct)
            if total > 0:
                win_pct = (wins / total) * 100
                avg_diff = np.mean(diffs)
            else:
                win_pct = 0
                avg_diff = 0
            stats[method][level] = {
                'win_pct': round(win_pct, 2),
                'avg_diff': round(avg_diff, 2),
                'n_datasets': total,
                'n_wins': wins
            }
    return stats

def compute_wilcoxon_tests(results, algo):
    """Compute Wilcoxon signed-rank tests for each DR method/level."""
    datasets = list(results[algo].keys())
    tests = {}
    for method in DR_METHODS:
        tests[method] = {}
        for level in LEVELS:
            key = f'{method}_{level}'
            baselines = []
            reduced_vals = []
            for ds in datasets:
                baseline = results[algo][ds].get('No Reduction', 0)
                reduced = results[algo][ds].get(key, None)
                if reduced is not None:
                    baselines.append(baseline)
                    reduced_vals.append(reduced)
            baselines = np.array(baselines)
            reduced_vals = np.array(reduced_vals)
            diffs = reduced_vals - baselines
            # Remove zero diffs for Wilcoxon test
            nonzero = np.abs(diffs) > 1e-10
            if np.sum(nonzero) >= 5:
                try:
                    stat, pval = wilcoxon(diffs[nonzero], alternative='greater')
                    tests[method][level] = round(pval, 4)
                except:
                    tests[method][level] = 1.0
            else:
                tests[method][level] = 1.0
    return tests


def main():
    os.makedirs('results', exist_ok=True)
    
    # Load all results
    all_results = {}
    for dtype in SYNTH_TYPES + ['RealWorld']:
        all_results[dtype] = load_results(dtype)['results']
    
    # ===== COMBINED AGGREGATE TABLES (Tables 1-4) =====
    for algo in ALGOS:
        # Compute synthetic aggregate (across all 4 types)
        synth_datasets_results = {}
        for stype in SYNTH_TYPES:
            for ds, vals in all_results[stype][algo].items():
                synth_datasets_results[f'{stype}_{ds}'] = vals
        
        # Create a fake combined results dict
        synth_combined = {algo: synth_datasets_results}
        real_combined = {algo: all_results['RealWorld'][algo]}
        
        synth_stats = compute_aggregate_stats(synth_combined, algo)
        real_stats = compute_aggregate_stats(real_combined, algo)
        
        # Build table
        rows = []
        for method in DR_METHODS:
            for level in LEVELS:
                s = synth_stats[method][level]
                r = real_stats[method][level]
                rows.append({
                    'Method': method,
                    'Reduction': level,
                    'Synth Win %': s['win_pct'],
                    'Real Win %': r['win_pct'],
                    'Synth Avg Win/Loss %': s['avg_diff'],
                    'Real Avg Win/Loss %': r['avg_diff'],
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(f'results/table_combined_aggregate_{algo}.csv', index=False)
        print(f"\n{'='*80}")
        print(f"Table: Combined Aggregate for {algo} (Paper Table format)")
        print(f"{'='*80}")
        print(df.to_string(index=False))
    
    # ===== WILCOXON TESTS (Table 5 format) =====
    # Paper Table 5: Wilcoxon on real-world data
    print(f"\n{'='*80}")
    print("Wilcoxon Signed-Rank Tests (Real-World Data)")
    print(f"{'='*80}")
    
    wilcoxon_rows = []
    for algo in ALGOS:
        real_combined = {algo: all_results['RealWorld'][algo]}
        tests = compute_wilcoxon_tests(real_combined, algo)
        for method in DR_METHODS:
            for level in LEVELS:
                pval = tests[method][level]
                sig = '✓' if pval < 0.05 else ''
                wilcoxon_rows.append({
                    'Algorithm': algo,
                    'DR Method': method,
                    'Reduction': level,
                    'p-value': pval,
                    'Significant (p<0.05)': sig
                })
    
    df_w = pd.DataFrame(wilcoxon_rows)
    df_w.to_csv('results/table_wilcoxon_combined.csv', index=False)
    print(df_w.to_string(index=False))
    
    # ===== SYNTHETIC PER-TYPE AVERAGE ARI TABLES (Tables A.1-A.4) =====
    for stype in SYNTH_TYPES:
        results = all_results[stype]
        print(f"\n{'='*80}")
        print(f"Table A: Average ARI for {stype}")
        print(f"{'='*80}")
        
        rows = []
        for algo in ALGOS:
            datasets = list(results[algo].keys())
            row = {'Algorithm': algo}
            conditions = ['No Reduction'] + [f'{m}_{l}' for m in DR_METHODS for l in LEVELS]
            for cond in conditions:
                vals = [results[algo][ds].get(cond, np.nan) for ds in datasets]
                row[cond] = round(np.nanmean(vals), 3)
            rows.append(row)
        
        df_synth = pd.DataFrame(rows)
        df_synth.to_csv(f'results/table_average_ARI_{stype}.csv', index=False)
        print(df_synth.to_string(index=False))
    
    # ===== Print comparison with paper values for k-means aggregate =====
    print(f"\n{'='*80}")
    print("COMPARISON: k-means aggregate (ours vs paper Table 1)")
    print(f"{'='*80}")
    paper_kmeans = {
        ('PCA', 'k-1'): (26.48, 55.55, -3.34, -0.15),
        ('PCA', '25%'): (28.07, 70.00, -1.99, 0.70),
        ('PCA', '50%'): (25.02, 45.45, 0.21, 0.09),
        ('Kernel PCA', 'k-1'): (75.18, 20.00, 13.66, -7.70),
        ('Kernel PCA', '25%'): (61.27, 23.52, 12.64, -9.80),
        ('Kernel PCA', '50%'): (73.30, 16.66, 18.19, -7.90),
        ('VAE', 'k-1'): (27.26, 25.00, -15.31, -4.75),
        ('VAE', '25%'): (27.30, 25.00, -9.03, -3.75),
        ('VAE', '50%'): (31.33, 29.41, -7.08, -2.50),
        ('Isomap', 'k-1'): (55.25, 53.84, 5.11, -0.40),
        ('Isomap', '25%'): (47.34, 40.00, -1.40, -1.30),
        ('Isomap', '50%'): (55.96, 50.00, 4.58, 0.59),
        ('MDS', 'k-1'): (51.76, 54.54, -16.41, -1.50),
        ('MDS', '25%'): (30.63, 53.84, -3.08, -0.30),
        ('MDS', '50%'): (29.18, 63.63, -0.27, 0.29),
    }
    
    # Get our values
    synth_datasets_results = {}
    for stype in SYNTH_TYPES:
        for ds, vals in all_results[stype]['k-means'].items():
            synth_datasets_results[f'{stype}_{ds}'] = vals
    synth_combined = {'k-means': synth_datasets_results}
    real_combined = {'k-means': all_results['RealWorld']['k-means']}
    our_synth = compute_aggregate_stats(synth_combined, 'k-means')
    our_real = compute_aggregate_stats(real_combined, 'k-means')
    
    print(f"{'Method':<15} {'Level':<6} | {'Paper S%':>8} {'Ours S%':>8} | {'Paper R%':>8} {'Ours R%':>8} | {'Paper SA':>8} {'Ours SA':>8} | {'Paper RA':>8} {'Ours RA':>8}")
    for (method, level), (ps, pr, psa, pra) in paper_kmeans.items():
        os_val = our_synth[method][level]
        or_val = our_real[method][level]
        print(f"{method:<15} {level:<6} | {ps:>8.2f} {os_val['win_pct']:>8.2f} | {pr:>8.2f} {or_val['win_pct']:>8.2f} | {psa:>8.2f} {os_val['avg_diff']:>8.2f} | {pra:>8.2f} {or_val['avg_diff']:>8.2f}")
    
    print("\nDone generating all tables!")

if __name__ == '__main__':
    main()
