"""
Generate all tables, boxplots, and analysis from existing experiment results.
This reads the JSON result files and produces all paper outputs.
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = './results'
DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']
CONDITIONS = ['No Reduction'] + [f'{m}_{l}' for m in DR_METHODS for l in REDUCTION_LEVELS]


def load_results(path):
    """Load results JSON, handling both old and new formats."""
    with open(path) as f:
        data = json.load(f)
    if 'results' in data:
        return data['results']
    return data


def make_ari_table(results_dict, dataset_names, round_digits=2):
    """Create ARI table: rows=datasets, cols=conditions."""
    rows = []
    for ds in dataset_names:
        if ds in results_dict:
            row = {}
            for cond in CONDITIONS:
                row[cond] = round(results_dict[ds].get(cond, np.nan), round_digits)
            rows.append(row)
        else:
            rows.append({c: np.nan for c in CONDITIONS})
    return pd.DataFrame(rows, index=dataset_names)


def make_synth_avg_table(results_dict, dataset_names, round_digits=2):
    """Create averaged synthetic table (average over repetitions with same config)."""
    # Group by base config (strip _repN suffix)
    from collections import defaultdict
    config_groups = defaultdict(list)
    for ds_name in results_dict:
        # Find base name (remove _rep0, _rep1, etc.)
        parts = ds_name.rsplit('_rep', 1)
        base = parts[0] if len(parts) > 1 else ds_name
        config_groups[base].append(ds_name)
    
    rows = []
    config_names = sorted(config_groups.keys())
    for config in config_names:
        ds_list = config_groups[config]
        row = {}
        for cond in CONDITIONS:
            vals = [results_dict[ds].get(cond, np.nan) for ds in ds_list if ds in results_dict]
            vals = [v for v in vals if not np.isnan(v)]
            row[cond] = round(np.mean(vals), round_digits) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=config_names)


def compute_aggregate_stats(synth_results, real_results):
    """Compute aggregate stats (win%, avg win/loss%) for each algo."""
    tables = {}
    
    for algo in ALGOS:
        rows = []
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f'{method}_{level}'
                row = {'Method': method, 'Reduction': level}
                
                for dtype, results in [('Synth', synth_results), ('Real', real_results)]:
                    if algo not in results:
                        row[f'{dtype} Win %'] = np.nan
                        row[f'{dtype} Avg Win/Loss %'] = np.nan
                        continue
                    
                    wins, losses, n_total = 0, 0, 0
                    ari_diffs = []
                    
                    for ds_name, ds_results in results[algo].items():
                        if ds_name.startswith('_'):
                            continue
                        base = ds_results.get('No Reduction', np.nan)
                        dr_val = ds_results.get(cond, np.nan)
                        
                        if np.isnan(base) or np.isnan(dr_val):
                            continue
                        
                        n_total += 1
                        diff = dr_val - base
                        ari_diffs.append(diff)
                        
                        if diff > 1e-9:
                            wins += 1
                        elif diff < -1e-9:
                            losses += 1
                        # else: tie
                    
                    total_wl = wins + losses
                    win_pct = (wins / total_wl * 100) if total_wl > 0 else 0.0
                    # Average win/loss %: mean ARI difference * 100 over ALL datasets
                    avg_pct = np.mean(ari_diffs) * 100 if ari_diffs else 0.0
                    
                    row[f'{dtype} Win %'] = round(win_pct, 2)
                    row[f'{dtype} Avg Win/Loss %'] = round(avg_pct, 2)
                
                rows.append(row)
        
        tables[algo] = pd.DataFrame(rows)
    
    return tables


def make_boxplot(results_dict, algo, label, outpath):
    """Create boxplot of ARI values across conditions."""
    # Collect ARI values per condition
    cond_data = {}
    for cond in CONDITIONS:
        vals = []
        for ds_name, ds_results in results_dict.items():
            if ds_name.startswith('_'):
                continue
            v = ds_results.get(cond, np.nan)
            if not np.isnan(v):
                vals.append(v)
        cond_data[cond] = vals
    
    # Create short labels
    short_labels = ['None']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            short_labels.append(f'{m[:3]}_{l}')
    
    data_list = [cond_data.get(c, []) for c in CONDITIONS]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(data_list, labels=short_labels, patch_artist=True)
    
    colors = ['lightgray'] + ['#AEC7E8']*3 + ['#FFBB78']*3 + ['#98DF8A']*3 + ['#FF9896']*3 + ['#C5B0D5']*3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(f'{algo} - {label}', fontsize=14)
    ax.set_ylabel('ARI', fontsize=12)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def run_wilcoxon_tests(results_dict):
    """Run Wilcoxon signed-rank tests comparing each DR condition to baseline."""
    wilcoxon_results = {}
    
    for algo in ALGOS:
        if algo not in results_dict:
            continue
        wilcoxon_results[algo] = {}
        
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f'{method}_{level}'
                base_vals = []
                dr_vals = []
                
                for ds_name, ds_results in results_dict[algo].items():
                    if ds_name.startswith('_'):
                        continue
                    b = ds_results.get('No Reduction', np.nan)
                    d = ds_results.get(cond, np.nan)
                    if not np.isnan(b) and not np.isnan(d):
                        base_vals.append(b)
                        dr_vals.append(d)
                
                if len(base_vals) >= 5:
                    diffs = np.array(dr_vals) - np.array(base_vals)
                    # Remove zeros for Wilcoxon
                    nonzero = diffs[np.abs(diffs) > 1e-10]
                    if len(nonzero) >= 5:
                        try:
                            stat, p = wilcoxon(nonzero, alternative='two-sided')
                            direction = 'better' if np.mean(diffs) > 0 else 'worse'
                            wilcoxon_results[algo][cond] = {
                                'p': float(p), 'stat': float(stat),
                                'n': len(nonzero), 'direction': direction,
                                'sig': 'yes' if p < 0.05 else 'no'
                            }
                        except Exception:
                            wilcoxon_results[algo][cond] = {'p': np.nan, 'sig': 'na'}
                    else:
                        wilcoxon_results[algo][cond] = {'p': np.nan, 'sig': 'na', 'n': len(nonzero)}
                else:
                    wilcoxon_results[algo][cond] = {'p': np.nan, 'sig': 'na'}
    
    return wilcoxon_results


def main():
    print("=" * 60)
    print("Generating all outputs from existing results")
    print("=" * 60)
    
    # ===== Load all results =====
    print("\nLoading results...")
    
    # Real-world
    rw_path = os.path.join(OUTPUT_DIR, 'results_RealWorld.json')
    if os.path.exists(rw_path):
        real_results = load_results(rw_path)
        print(f"  Real-world: loaded ({len(real_results.get('k-means', {}))} datasets for k-means)")
    else:
        print("  WARNING: No real-world results found!")
        real_results = {}
    
    # Synthetic
    synth_type_results = {}
    synth_all_results = {}
    for stype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        path = os.path.join(OUTPUT_DIR, f'results_{stype}.json')
        if os.path.exists(path):
            data = load_results(path)
            synth_type_results[stype] = data
            # Merge into synth_all
            for algo in ALGOS:
                if algo in data:
                    if algo not in synth_all_results:
                        synth_all_results[algo] = {}
                    synth_all_results[algo].update(data[algo])
            print(f"  {stype}: loaded ({len(data.get('k-means', {}))} datasets for k-means)")
        else:
            print(f"  WARNING: No {stype} results found!")
    
    # ===== Generate per-dataset ARI tables =====
    print("\nGenerating per-dataset ARI tables...")
    
    # Real-world tables
    for algo in ALGOS:
        if algo in real_results:
            ds_names = sorted([k for k in real_results[algo].keys() if not k.startswith('_')])
            df = make_ari_table(real_results[algo], ds_names)
            path = os.path.join(OUTPUT_DIR, f'table_{algo}_RealWorld.csv')
            df.to_csv(path)
            print(f"  {path} ({len(ds_names)} datasets)")
    
    # Synthetic tables (averaged over reps)
    for stype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        if stype in synth_type_results:
            for algo in ALGOS:
                if algo in synth_type_results[stype]:
                    ds_names = sorted([k for k in synth_type_results[stype][algo].keys() if not k.startswith('_')])
                    df = make_synth_avg_table(synth_type_results[stype][algo], ds_names)
                    path = os.path.join(OUTPUT_DIR, f'table_{algo}_{stype}.csv')
                    df.to_csv(path)
                    print(f"  {path}")
    
    # ===== Generate boxplots =====
    print("\nGenerating boxplots...")
    
    # Real-world boxplots
    for algo in ALGOS:
        if algo in real_results:
            outpath = os.path.join(OUTPUT_DIR, f'boxplot_{algo}_RealWorld.pdf')
            make_boxplot(real_results[algo], algo, 'Real-World', outpath)
            print(f"  {outpath}")
    
    # Synthetic boxplots
    for stype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        if stype in synth_type_results:
            for algo in ALGOS:
                if algo in synth_type_results[stype]:
                    outpath = os.path.join(OUTPUT_DIR, f'boxplot_{algo}_Synthetic_{stype}.pdf')
                    make_boxplot(synth_type_results[stype][algo], algo, f'Synthetic ({stype})', outpath)
                    print(f"  {outpath}")
    
    # ===== Aggregate tables (Tables 1-4) =====
    print("\nComputing aggregate statistics...")
    aggregate = compute_aggregate_stats(synth_all_results, real_results)
    
    for algo, df in aggregate.items():
        path = os.path.join(OUTPUT_DIR, f'table_aggregate_{algo}.csv')
        df.to_csv(path, index=False)
        print(f"\n  {algo} Aggregate Table:")
        print(df.to_string(index=False))
    
    # ===== Wilcoxon tests =====
    print("\n\nRunning Wilcoxon signed-rank tests...")
    
    # Real-world Wilcoxon
    wilcoxon_real = run_wilcoxon_tests(real_results)
    
    # Synthetic Wilcoxon (combined)
    wilcoxon_synth = run_wilcoxon_tests(synth_all_results)
    
    # Format combined Wilcoxon table
    wilcoxon_rows = []
    for algo in ALGOS:
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f'{method}_{level}'
                row = {'Algorithm': algo, 'DR Method': method, 'Reduction': level}
                
                # Real
                if algo in wilcoxon_real and cond in wilcoxon_real[algo]:
                    r = wilcoxon_real[algo][cond]
                    p = r.get('p', np.nan)
                    row['p_real'] = f'{p:.4f}' if not np.isnan(p) else 'NA'
                    row['sig_real'] = '✓' if r.get('sig') == 'yes' else ''
                    row['dir_real'] = r.get('direction', '')
                else:
                    row['p_real'] = 'NA'
                    row['sig_real'] = ''
                    row['dir_real'] = ''
                
                # Synthetic
                if algo in wilcoxon_synth and cond in wilcoxon_synth[algo]:
                    r = wilcoxon_synth[algo][cond]
                    p = r.get('p', np.nan)
                    row['p_synth'] = f'{p:.4f}' if not np.isnan(p) else 'NA'
                    row['sig_synth'] = '✓' if r.get('sig') == 'yes' else ''
                    row['dir_synth'] = r.get('direction', '')
                else:
                    row['p_synth'] = 'NA'
                    row['sig_synth'] = ''
                    row['dir_synth'] = ''
                
                wilcoxon_rows.append(row)
    
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_path = os.path.join(OUTPUT_DIR, 'table_wilcoxon.csv')
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"\n  Wilcoxon table saved to {wilcoxon_path}")
    
    # Print significant results
    print("\n  Significant Wilcoxon results (p < 0.05):")
    for _, row in wilcoxon_df.iterrows():
        for dtype in ['real', 'synth']:
            if row[f'sig_{dtype}'] == '✓':
                print(f"    {row['Algorithm']}/{row['DR Method']}/{row['Reduction']} ({dtype}): "
                      f"p={row[f'p_{dtype}']}, direction={row[f'dir_{dtype}']}")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    
    # Count files
    tables = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('table_') and f.endswith('.csv')]
    boxplots = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('boxplot_') and f.endswith('.pdf')]
    
    print(f"\n  Tables generated: {len(tables)}")
    for t in sorted(tables):
        print(f"    {t}")
    
    print(f"\n  Boxplots generated: {len(boxplots)}")
    for b in sorted(boxplots):
        print(f"    {b}")
    
    print(f"\n  All outputs in: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
