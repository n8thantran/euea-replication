"""
Run synthetic experiments efficiently.
Generates synthetic data, applies DR + clustering, saves results.
"""
import os
import sys
import json
import time
import signal
import warnings
import numpy as np
import pickle

warnings.filterwarnings('ignore')

# Import from experiment.py
from experiment import (
    precompute_all_dr, run_kmeans_experiments, 
    run_ahc_experiments, run_gmm_experiments, run_optics_experiments,
    find_best_ahc_params, find_best_gmm_params, find_best_optics_params,
    format_results_table, compute_aggregate_stats, compute_wilcoxon_tests,
    generate_boxplots, get_all_conditions, DR_METHODS, REDUCTION_LEVELS
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_synthetic_datasets_fast(random_state=42):
    """Generate synthetic datasets with reduced count for feasibility.
    Paper uses 50 per config; we use 5 per config."""
    from sklearn.datasets import make_circles, make_moons
    import repliclust
    
    rng = np.random.RandomState(random_state)
    synthetic = {}
    
    def add_noise_dims(X, target_dims, rng_local):
        """Add noisy dimensions per paper spec."""
        n_samples = X.shape[0]
        n_orig = X.shape[1]
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        n_per = n_extra // 4
        remainder = n_extra - 4 * n_per
        parts = []
        for sigma, _ in zip([1.0, 0.5, 0.25, 0.0], [0]*4):
            n_this = n_per + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            if sigma > 0:
                parts.append(rng_local.normal(0, sigma, (n_samples, n_this)))
            else:
                parts.append(np.zeros((n_samples, n_this)))
        noise = np.hstack(parts)[:, :n_extra]
        return np.hstack([X, noise])
    
    def embed_to_high_dim(X, target_dim, seed):
        """Embed 2D data into higher dimensions via random projection."""
        if target_dim <= X.shape[1]:
            return X
        r = np.random.RandomState(seed)
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        return X @ proj
    
    dims = [10, 50, 200]
    n_per_config = 5  # Reduced from 50 for feasibility
    
    # === CIRCLES ===
    print("Generating Circles datasets...")
    for k_val, n_samples in [(2, 2000), (5, 2000)]:
        for d in dims:
            for i in range(n_per_config):
                if k_val == 2:
                    X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, 
                                       random_state=random_state+i)
                    X = embed_to_high_dim(X, d, random_state+i+1000)
                else:
                    r = np.random.RandomState(random_state + i + 2000)
                    n_per = n_samples // k_val
                    X_list, y_list = [], []
                    for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                        theta = r.uniform(0, 2*np.pi, n_per)
                        rad = factor + r.normal(0, 0.05, n_per)
                        X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                        y_list.append(np.full(n_per, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = embed_to_high_dim(X, d, random_state+i+3000)
                X = add_noise_dims(X, d, np.random.RandomState(random_state+i+d))
                synthetic[f'Circles_k{k_val}_d{d}_t{i}'] = (X, y, k_val)
    
    # === MOONS ===
    print("Generating Moons datasets...")
    for k_val, n_samples in [(2, 2000), (5, 2000)]:
        for d in dims:
            for i in range(n_per_config):
                if k_val == 2:
                    X, y = make_moons(n_samples=n_samples, noise=0.1, 
                                     random_state=random_state+i+4000)
                    X = embed_to_high_dim(X, d, random_state+i+5000)
                else:
                    r = np.random.RandomState(random_state + i + 6000)
                    n_per = n_samples // k_val
                    X_list, y_list = [], []
                    stretches = [1.0, 1.5, 1.0, 1.5, 1.0]
                    rotations = [0, 160, -160, 10, 180]
                    x_shifts = [0, 3, -3, 2, -2]
                    y_shifts = [0, 1.0, 1.2, 1.5, 1.0]
                    for ci in range(5):
                        theta = np.linspace(0, np.pi, n_per)
                        x = np.cos(theta) * stretches[ci] + r.normal(0, 0.1, n_per)
                        yc = np.sin(theta) + r.normal(0, 0.1, n_per)
                        angle = np.radians(rotations[ci])
                        xr = x*np.cos(angle) - yc*np.sin(angle) + x_shifts[ci]
                        yr = x*np.sin(angle) + yc*np.cos(angle) + y_shifts[ci]
                        X_list.append(np.column_stack([xr, yr]))
                        y_list.append(np.full(n_per, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = embed_to_high_dim(X, d, random_state+i+7000)
                X = add_noise_dims(X, d, np.random.RandomState(random_state+i+d+100))
                synthetic[f'Moons_k{k_val}_d{d}_t{i}'] = (X, y, k_val)
    
    # === RSG ===
    print("Generating RSG datasets...")
    ks_rsg = [2, 10, 50]
    ds_rsg = [10, 50, 200]
    Ncs_rsg = [5, 50, 100]
    n_rsg_per = 3
    
    for k in ks_rsg:
        for d in ds_rsg:
            for Nc in Ncs_rsg:
                for i in range(n_rsg_per):
                    r = np.random.RandomState(random_state + k*1000 + d*100 + Nc + i)
                    alpha = max(0.1, min(0.9, 1.0 - k*0.01 - d*0.001))
                    centers = r.randn(k, d) * np.sqrt(d) * (1 + alpha)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * alpha
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples)
                        y_list.append(np.full(Nc, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = add_noise_dims(X, d, np.random.RandomState(random_state+k+d+Nc+i))
                    synthetic[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    # === REPLICLUST ===
    print("Generating Repliclust datasets...")
    for k in [2, 5]:
        Nc = 1000 if k == 2 else 400
        for d in dims:
            for i in range(n_per_config):
                try:
                    archetype = repliclust.Archetype(
                        n_clusters=k, dim=d, n_samples=k*Nc,
                        aspect_ref=3.0, radius=5.0
                    )
                    X, y, _ = repliclust.DataGenerator(archetype).synthesize(random_state=random_state+i)
                    X = add_noise_dims(X, d, np.random.RandomState(random_state+i+d+k))
                    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
                except Exception as e:
                    print(f"  Repliclust fallback for k={k}, d={d}, i={i}: {e}")
                    r = np.random.RandomState(random_state + i + 9000 + k*100 + d)
                    centers = r.randn(k, d) * np.sqrt(d)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * 0.5
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples)
                        y_list.append(np.full(Nc, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = add_noise_dims(X, d, np.random.RandomState(random_state+i+d+k))
                    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    
    return synthetic


def run_synthetic_type(dtype, datasets, output_dir):
    """Run all experiments for one synthetic type."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dtype.upper()} ({len(datasets)} datasets)")
    print(f"{'='*60}")
    
    # Check for cached DR
    cache_file = os.path.join(output_dir, f'dr_cache_{dtype}.pkl')
    if os.path.exists(cache_file):
        print(f"Loading cached DR from {cache_file}")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
        # Check if all datasets are in cache
        missing = [k for k in datasets if k not in dr_cache]
        if missing:
            print(f"  {len(missing)} datasets missing from cache, recomputing those")
            from experiment import precompute_all_dr
            missing_ds = {k: datasets[k] for k in missing}
            missing_cache = precompute_all_dr(missing_ds)
            dr_cache.update(missing_cache)
            with open(cache_file, 'wb') as f:
                pickle.dump(dr_cache, f)
    else:
        t0 = time.time()
        dr_cache = precompute_all_dr(datasets)
        print(f"DR precomputation took {time.time()-t0:.1f}s")
        with open(cache_file, 'wb') as f:
            pickle.dump(dr_cache, f)
    
    type_results = {}
    
    # k-means
    print(f"\n  Running k-means on {dtype}...")
    t0 = time.time()
    type_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
    print(f"  k-means took {time.time()-t0:.1f}s")
    
    # AHC
    print(f"\n  Running AHC on {dtype}...")
    t0 = time.time()
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache)
    type_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_m, ahc_l)
    type_results['AHC_params'] = {'metric': ahc_m, 'linkage': ahc_l}
    print(f"  AHC took {time.time()-t0:.1f}s")
    
    # GMM
    print(f"\n  Running GMM on {dtype}...")
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    type_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    type_results['GMM_params'] = {'covariance_type': gmm_cov}
    print(f"  GMM took {time.time()-t0:.1f}s")
    
    # OPTICS
    print(f"\n  Running OPTICS on {dtype}...")
    t0 = time.time()
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
    type_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, opt_ms, opt_mcs)
    type_results['OPTICS_params'] = {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)}
    print(f"  OPTICS took {time.time()-t0:.1f}s")
    
    # Save
    with open(os.path.join(output_dir, f'synthetic_{dtype}_results.json'), 'w') as f:
        json.dump(type_results, f, indent=2)
    
    # Print average ARI
    conditions = get_all_conditions()
    print(f"\nAverage ARI for {dtype}:")
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        avg_row = {}
        for cond in conditions:
            vals = [type_results[algo][ds].get(cond, 0.0) for ds in type_results[algo]]
            avg_row[cond] = round(np.mean(vals), 2) if vals else 0
        print(f"  {algo:8s}: No_Red={avg_row.get('No Reduction', 0):.2f}", end="")
        for m in DR_METHODS[:3]:  # Just show first 3 for brevity
            for l in REDUCTION_LEVELS[:1]:
                print(f"  {m}_{l}={avg_row.get(f'{m}_{l}', 0):.2f}", end="")
        print()
    
    return type_results


def main():
    print("=" * 60)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 60)
    
    # Generate all synthetic datasets
    all_synthetic = generate_synthetic_datasets_fast()
    print(f"\nGenerated {len(all_synthetic)} total synthetic datasets")
    
    # Group by type
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    all_type_results = {}
    
    for dtype in type_names:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets:
            print(f"No {dtype} datasets, skipping")
            continue
        
        result_file = os.path.join(OUTPUT_DIR, f'synthetic_{dtype}_results.json')
        if os.path.exists(result_file):
            print(f"\n{dtype} results already exist, loading...")
            with open(result_file) as f:
                all_type_results[dtype] = json.load(f)
        else:
            all_type_results[dtype] = run_synthetic_type(dtype, datasets, OUTPUT_DIR)
    
    # Save combined results
    with open(os.path.join(OUTPUT_DIR, 'synthetic_results_all.json'), 'w') as f:
        json.dump(all_type_results, f, indent=2)
    
    # Generate tables (average ARI per condition, matching paper Tables A.1-A.4)
    print("\n" + "=" * 60)
    print("SYNTHETIC AVERAGE ARI TABLES")
    print("=" * 60)
    conditions = get_all_conditions()
    
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        print(f"\n--- {dtype} ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype][algo]
            avg_row = {}
            for cond in conditions:
                vals = [results[ds].get(cond, 0.0) for ds in results]
                avg_row[cond] = round(np.mean(vals), 2) if vals else 0
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame([avg_row], index=[f'{dtype}_avg'], columns=conditions)
            csv_path = os.path.join(OUTPUT_DIR, f'table_{algo}_synthetic_{dtype}.csv')
            df.to_csv(csv_path)
            print(f"  {algo}: saved to {csv_path}")
    
    # Generate boxplots
    print("\n" + "=" * 60)
    print("GENERATING BOXPLOTS")
    print("=" * 60)
    for dtype in type_names:
        if dtype in all_type_results:
            generate_boxplots(all_type_results[dtype], f'Synthetic_{dtype}', OUTPUT_DIR)
    
    # Aggregate stats
    print("\n" + "=" * 60)
    print("SYNTHETIC AGGREGATE STATISTICS")
    print("=" * 60)
    synth_agg = {}
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        synth_agg[dtype] = {}
        print(f"\n--- {dtype} ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            stats = compute_aggregate_stats(all_type_results[dtype][algo])
            synth_agg[dtype][algo] = stats
            print(f"  {algo}:")
            for method in DR_METHODS:
                for level in REDUCTION_LEVELS:
                    s = stats[method][level]
                    print(f"    {method:12s} {level:4s}: win={s['win_pct']:5.1f}%  loss={s['loss_pct']:5.1f}%")
    
    with open(os.path.join(OUTPUT_DIR, 'aggregate_stats_synthetic.json'), 'w') as f:
        json.dump(synth_agg, f, indent=2)
    
    # Wilcoxon tests
    print("\n" + "=" * 60)
    print("WILCOXON TESTS (SYNTHETIC)")
    print("=" * 60)
    synth_wilcoxon = {}
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        synth_wilcoxon[dtype] = {}
        print(f"\n--- {dtype} ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            pvals = compute_wilcoxon_tests(all_type_results[dtype][algo])
            synth_wilcoxon[dtype][algo] = pvals
            print(f"  {algo}:")
            for method in DR_METHODS:
                vals = [pvals[method][l] for l in REDUCTION_LEVELS]
                sig = ['*' if v < 0.05 else ' ' for v in vals]
                print(f"    {method:12s}: k-1={vals[0]:.4f}{sig[0]}  25%={vals[1]:.4f}{sig[1]}  50%={vals[2]:.4f}{sig[2]}")
    
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results_synthetic.json'), 'w') as f:
        json.dump(synth_wilcoxon, f, indent=2)
    
    print("\n\nAll synthetic experiments complete!")


if __name__ == '__main__':
    main()
