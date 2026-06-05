#!/usr/bin/env python3
"""
Generate all tables and figures from pre-computed experimental results.
This reads from the JSON result files and produces:
- Tables A.5-A.8: Per-dataset ARI for real-world (4 algorithms)
- Tables A.1-A.4: Average ARI for synthetic types (4 algorithms)  
- Tables 1-4: Aggregate win/loss statistics (4 algorithms)
- Table 5: Wilcoxon signed-rank tests
- Figures 1-8: Boxplots
"""

import json
import os
import csv
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/workspace/results'

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
DR_LEVELS = ['k-1', '25%', '50%']
ALGORITHMS = ['k-means', 'AHC', 'GMM', 'OPTICS']
SYNTH_TYPES = ['Circles', 'Moons', 'RSG', 'Repliclust']


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_dr_cols():
    """Return DR column names in order: No Reduction, then each method × level."""
    cols = ['No Reduction']
    for m in DR_METHODS:
        for l in DR_LEVELS:
            cols.append(f'{m}_{l}')
    return cols


def generate_per_dataset_table(results, algo, datasets, outfile):
    """Generate per-dataset ARI table (Tables A.5-A.8 for real, A.1-A.4 for synthetic)."""
    cols = get_dr_cols()
    
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset'] + cols)
        
        for ds_name in datasets:
            if ds_name not in results.get(algo, {}):
                continue
            row = [ds_name]
            ds_results = results[algo][ds_name]
            for col in cols:
                val = ds_results.get(col, None)
                if val is not None:
                    row.append(f'{val:.4f}')
                else:
                    row.append('')
            writer.writerow(row)


def compute_aggregate_stats(results, algo, datasets):
    """Compute aggregate stats: %win, %loss, %tie, avg_change for each DR method.
    
    For each dataset, compare DR result to No Reduction baseline.
    Win = DR > baseline, Loss = DR < baseline, Tie = equal.
    avg_change = mean(DR - baseline) across datasets.
    """
    cols = get_dr_cols()[1:]  # exclude No Reduction
    stats_dict = {}
    
    for col in cols:
        wins, losses, ties = 0, 0, 0
        diffs = []
        total = 0
        
        for ds_name in datasets:
            if ds_name not in results.get(algo, {}):
                continue
            ds_res = results[algo][ds_name]
            baseline = ds_res.get('No Reduction', None)
            val = ds_res.get(col, None)
            
            if baseline is None or val is None:
                continue
            
            total += 1
            diff = val - baseline
            diffs.append(diff)
            
            if abs(diff) < 1e-10:
                ties += 1
            elif diff > 0:
                wins += 1
            else:
                losses += 1
        
        if total > 0:
            # Win% and Loss% exclude ties (as per paper)
            non_tie = wins + losses
            if non_tie > 0:
                win_pct = round(100 * wins / non_tie)
                loss_pct = round(100 * losses / non_tie)
            else:
                win_pct = 0
                loss_pct = 0
            avg_diff = np.mean(diffs) if diffs else 0
            
            stats_dict[col] = {
                'win_pct': win_pct,
                'loss_pct': loss_pct,
                'tie_pct': round(100 * ties / total),
                'avg_change': avg_diff,
                'n_datasets': total
            }
    
    return stats_dict


def generate_aggregate_table(all_results, algo, outfile):
    """Generate aggregate table (Tables 1-4) combining synthetic + real-world.
    
    Rows: DR methods × levels
    Columns: Per dataset type (Circles, Moons, RSG, Repliclust, Real) - Win%, Avg change
    """
    # Collect per-type stats
    type_stats = {}
    
    for stype in SYNTH_TYPES:
        data = load_results(f'results_{stype}.json')
        if data and 'results' in data:
            res = data['results']
            datasets = list(res.get(algo, {}).keys())
            type_stats[stype] = compute_aggregate_stats(res, algo, datasets)
    
    # Real world
    data = load_results('results_RealWorld.json')
    if data and 'results' in data:
        res = data['results']
        datasets = list(res.get(algo, {}).keys())
        type_stats['RealWorld'] = compute_aggregate_stats(res, algo, datasets)
    
    # Write table
    types_present = [t for t in SYNTH_TYPES + ['RealWorld'] if t in type_stats]
    
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['DR Method', 'Level']
        for t in types_present:
            header.extend([f'{t} Win%', f'{t} Avg'])
        writer.writerow(header)
        
        for method in DR_METHODS:
            for level in DR_LEVELS:
                col = f'{method}_{level}'
                row = [method, level]
                for t in types_present:
                    if col in type_stats.get(t, {}):
                        s = type_stats[t][col]
                        row.append(s['win_pct'])
                        row.append(f"{s['avg_change']:.1f}" if abs(s['avg_change']) >= 0.05 else f"{s['avg_change']:.2f}")
                    else:
                        row.extend(['', ''])
                writer.writerow(row)
    
    return type_stats


def generate_wilcoxon_table(outfile):
    """Generate Wilcoxon signed-rank test results (Table 5).
    
    For each DR method × level, test if the ARI values are significantly
    different from No Reduction across all datasets combined.
    """
    results_by_type = {}
    
    for stype in SYNTH_TYPES:
        data = load_results(f'results_{stype}.json')
        if data and 'results' in data:
            results_by_type[stype] = data['results']
    
    data = load_results('results_RealWorld.json')
    if data and 'results' in data:
        results_by_type['RealWorld'] = data['results']
    
    cols = get_dr_cols()[1:]
    
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'DR Method', 'Level', 'p-value', 'Significant (α=0.05)', 'Direction'])
        
        for algo in ALGORITHMS:
            for col in cols:
                baselines = []
                dr_values = []
                
                for dtype, res in results_by_type.items():
                    if algo not in res:
                        continue
                    for ds_name, ds_res in res[algo].items():
                        b = ds_res.get('No Reduction')
                        v = ds_res.get(col)
                        if b is not None and v is not None:
                            baselines.append(b)
                            dr_values.append(v)
                
                if len(baselines) >= 5:
                    diffs = [d - b for d, b in zip(dr_values, baselines)]
                    # Remove zeros for Wilcoxon test
                    nonzero = [(d, b, v) for d, b, v in zip(diffs, baselines, dr_values) if abs(d) > 1e-10]
                    
                    if len(nonzero) >= 5:
                        nz_diffs = [x[0] for x in nonzero]
                        try:
                            stat, pval = stats.wilcoxon(nz_diffs)
                        except:
                            pval = 1.0
                        
                        direction = 'better' if np.mean(diffs) > 0 else 'worse'
                        sig = 'Yes' if pval < 0.05 else 'No'
                        
                        parts = col.rsplit('_', 1)
                        method, level = parts[0], parts[1]
                        writer.writerow([algo, method, level, f'{pval:.4f}', sig, direction])


def generate_boxplots(algo, dtype):
    """Generate boxplot for one algorithm × one dataset type."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except:
        return
    
    if dtype == 'RealWorld':
        data = load_results('results_RealWorld.json')
    else:
        data = load_results(f'results_{dtype}.json')
    
    if not data or 'results' not in data:
        return
    
    res = data['results']
    if algo not in res:
        return
    
    algo_res = res[algo]
    
    # Compute differences from baseline for each DR method
    dr_cols = get_dr_cols()[1:]
    
    box_data = {}
    for col in dr_cols:
        diffs = []
        for ds_name, ds_res in algo_res.items():
            b = ds_res.get('No Reduction')
            v = ds_res.get(col)
            if b is not None and v is not None:
                diffs.append(v - b)
        if diffs:
            box_data[col] = diffs
    
    if not box_data:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    labels = list(box_data.keys())
    data_list = [box_data[l] for l in labels]
    
    bp = ax.boxplot(data_list, labels=[l.replace('_', '\n') for l in labels],
                    patch_artist=True, showfliers=True, widths=0.6)
    
    # Color by method
    colors = {'PCA': '#1f77b4', 'Kernel PCA': '#ff7f0e', 'VAE': '#2ca02c', 
              'Isomap': '#d62728', 'MDS': '#9467bd'}
    for i, label in enumerate(labels):
        method = label.rsplit('_', 1)[0]
        color = colors.get(method, '#333333')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('ARI Change (DR - No Reduction)')
    ax.set_title(f'{algo} - {dtype}: ARI Change by DR Method')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    
    outpath = os.path.join(RESULTS_DIR, f'boxplot_{algo}_{dtype}.pdf')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def generate_synthetic_ari_tables():
    """Generate average ARI tables for each synthetic type (Tables A.1-A.4)."""
    for stype in SYNTH_TYPES:
        data = load_results(f'results_{stype}.json')
        if not data or 'results' not in data:
            continue
        
        res = data['results']
        
        for algo in ALGORITHMS:
            if algo not in res:
                continue
            
            datasets = list(res[algo].keys())
            outfile = os.path.join(RESULTS_DIR, f'table_synthetic_{stype}_{algo}.csv')
            generate_per_dataset_table(res, algo, datasets, outfile)
        
        # Also generate a summary (average across datasets)
        outfile = os.path.join(RESULTS_DIR, f'table_synthetic_{stype}_avg.csv')
        cols = get_dr_cols()
        
        with open(outfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algorithm'] + cols)
            
            for algo in ALGORITHMS:
                if algo not in res:
                    continue
                row = [algo]
                datasets = list(res[algo].keys())
                for col in cols:
                    vals = []
                    for ds in datasets:
                        v = res[algo][ds].get(col)
                        if v is not None:
                            vals.append(v)
                    if vals:
                        row.append(f'{np.mean(vals):.4f}')
                    else:
                        row.append('')
                writer.writerow(row)


def generate_real_ari_tables():
    """Generate per-dataset ARI tables for real-world data (Tables A.5-A.8)."""
    data = load_results('results_RealWorld.json')
    if not data or 'results' not in data:
        return
    
    res = data['results']
    
    for algo in ALGORITHMS:
        if algo not in res:
            continue
        datasets = sorted(res[algo].keys())
        outfile = os.path.join(RESULTS_DIR, f'table_{algo}_real.csv')
        generate_per_dataset_table(res, algo, datasets, outfile)


def print_summary():
    """Print summary of key results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    data = load_results('results_RealWorld.json')
    if data and 'results' in data:
        res = data['results']
        print("\nReal-World Results (20 UCI datasets):")
        for algo in ALGORITHMS:
            if algo not in res:
                continue
            datasets = list(res[algo].keys())
            stats = compute_aggregate_stats(res, algo, datasets)
            print(f"\n  {algo}:")
            for method in DR_METHODS:
                for level in DR_LEVELS:
                    col = f'{method}_{level}'
                    if col in stats:
                        s = stats[col]
                        print(f"    {col:20s}: Win%={s['win_pct']:3d}, Avg={s['avg_change']:+.3f}")
    
    for stype in SYNTH_TYPES:
        data = load_results(f'results_{stype}.json')
        if data and 'results' in data:
            res = data['results']
            print(f"\n{stype} Results:")
            for algo in ALGORITHMS:
                if algo not in res:
                    continue
                datasets = list(res[algo].keys())
                stats = compute_aggregate_stats(res, algo, datasets)
                print(f"  {algo}: {len(datasets)} datasets")


def main():
    print("Generating all results from pre-computed experiments...")
    print(f"Output directory: {RESULTS_DIR}")
    
    # 1. Real-world per-dataset ARI tables (Tables A.5-A.8)
    print("\n1. Generating real-world ARI tables...")
    generate_real_ari_tables()
    for algo in ALGORITHMS:
        path = os.path.join(RESULTS_DIR, f'table_{algo}_real.csv')
        if os.path.exists(path):
            print(f"   ✓ {path}")
    
    # 2. Synthetic ARI tables (Tables A.1-A.4)
    print("\n2. Generating synthetic ARI tables...")
    generate_synthetic_ari_tables()
    for stype in SYNTH_TYPES:
        path = os.path.join(RESULTS_DIR, f'table_synthetic_{stype}_avg.csv')
        if os.path.exists(path):
            print(f"   ✓ {path}")
    
    # 3. Aggregate tables (Tables 1-4)
    print("\n3. Generating aggregate tables...")
    for algo in ALGORITHMS:
        outfile = os.path.join(RESULTS_DIR, f'table_aggregate_{algo}.csv')
        generate_aggregate_table(None, algo, outfile)
        print(f"   ✓ {outfile}")
    
    # 4. Wilcoxon table (Table 5)
    print("\n4. Generating Wilcoxon test table...")
    outfile = os.path.join(RESULTS_DIR, 'table_wilcoxon_combined.csv')
    generate_wilcoxon_table(outfile)
    print(f"   ✓ {outfile}")
    
    # 5. Boxplots (Figures 1-8)
    print("\n5. Generating boxplots...")
    for algo in ALGORITHMS:
        for dtype in SYNTH_TYPES + ['RealWorld']:
            generate_boxplots(algo, dtype)
            path = os.path.join(RESULTS_DIR, f'boxplot_{algo}_{dtype}.pdf')
            if os.path.exists(path):
                print(f"   ✓ {path}")
    
    # 6. Print summary
    print_summary()
    
    # 7. List all output files
    print("\n" + "=" * 70)
    print("ALL OUTPUT FILES:")
    print("=" * 70)
    
    result_files = sorted([f for f in os.listdir(RESULTS_DIR) 
                          if f.endswith(('.csv', '.pdf', '.json'))])
    tables = [f for f in result_files if f.endswith('.csv')]
    figures = [f for f in result_files if f.endswith('.pdf')]
    
    print(f"\nTables ({len(tables)}):")
    for f in tables:
        print(f"  {f}")
    
    print(f"\nFigures ({len(figures)}):")
    for f in figures:
        print(f"  {f}")


if __name__ == '__main__':
    main()
