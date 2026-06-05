"""
Fast synthetic experiments runner.
Generates synthetic data, applies DR + clustering, saves results.
Optimized for speed: fewer datasets, smaller samples, faster DR.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pickle
import pandas as pd

warnings.filterwarnings('ignore')

from experiment import (
    precompute_all_dr, run_kmeans_experiments, 
    run_ahc_experiments, run_gmm_experiments, run_optics_experiments,
    find_best_ahc_params, find_best_gmm_params, find_best_optics_params,
    format_results_table, compute_aggregate_stats, compute_wilcoxon_tests,
    generate_boxplots, get_all_conditions, DR_METHODS, REDUCTION_LEVELS
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_synthetic_datasets(random_state=42):
    """Generate synthetic datasets. 3 per config, 500 samples for speed."""
    from sklearn.datasets import make_circles, make_moons
    
    rng = np.random.RandomState(random_state)
    synthetic = {}
    
    def embed_to_high_dim(X, target_dim, seed):
        """Embed 2D data into higher dimensions via random projection + noise."""
        if target_dim <= X.shape[1]:
            return X
        r = np.random.RandomState(seed)
        # Random rotation into higher dim
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        X_proj = X @ proj
        # Add noise dimensions
        noise = r.normal(0, 0.1, (X.shape[0], target_dim))
        return X_proj + noise * 0.3
    
    dims = [10, 50, 200]
    n_per_config = 3  # Reduced from 50
    n_samples = 500   # Reduced from 2000
    
    # === CIRCLES ===
    print("Generating Circles datasets...")
    for k_val in [2, 5]:
        for d in dims:
            for i in range(n_per_config):
                seed = random_state + i + k_val*1000 + d*10
                if k_val == 2:
                    X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, 
                                       random_state=seed)
                    X = embed_to_high_dim(X, d, seed+100)
                else:
                    r = np.random.RandomState(seed)
                    n_per = n_samples // k_val
                    X_list, y_list = [], []
                    for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                        theta = r.uniform(0, 2*np.pi, n_per)
                        rad = factor + r.normal(0, 0.05, n_per)
                        X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                        y_list.append(np.full(n_per, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = embed_to_high_dim(X, d, seed+100)
                synthetic[f'Circles_k{k_val}_d{d}_t{i}'] = (X, y, k_val)
    
    # === MOONS ===
    print("Generating Moons datasets...")
    for k_val in [2, 5]:
        for d in dims:
            for i in range(n_per_config):
                seed = random_state + i + k_val*2000 + d*10
                if k_val == 2:
                    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
                    X = embed_to_high_dim(X, d, seed+200)
                else:
                    r = np.random.RandomState(seed)
                    n_per = n_samples // k_val
                    X_list, y_list = [], []
                    for ci in range(5):
                        theta = np.linspace(0, np.pi, n_per)
                        x = np.cos(theta) * (1 + ci*0.3) + r.normal(0, 0.1, n_per)
                        yc = np.sin(theta) + r.normal(0, 0.1, n_per)
                        angle = np.radians(ci * 72)
                        xr = x*np.cos(angle) - yc*np.sin(angle) + ci*2
                        yr = x*np.sin(angle) + yc*np.cos(angle) + ci
                        X_list.append(np.column_stack([xr, yr]))
                        y_list.append(np.full(n_per, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = embed_to_high_dim(X, d, seed+200)
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
                    synthetic[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    # === REPLICLUST ===
    print("Generating Repliclust datasets...")
    try:
        import repliclust
        for k in [2, 5]:
            Nc = 250 if k == 2 else 100  # Reduced sample size
            for d in dims:
                for i in range(n_per_config):
                    try:
                        repliclust.set_seed(random_state + i + k*100 + d)
                        archetype = repliclust.Archetype(
                            n_clusters=k, dim=d, n_samples=k*Nc,
                            aspect_ref=3.0, radius_maxmin=3
                        )
                        X, y, _ = repliclust.DataGenerator(archetype).synthesize()
                        synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
                    except Exception as e:
                        print(f"  Repliclust fallback for k={k}, d={d}, i={i}: {e}")
                        _make_gaussian_fallback(synthetic, k, d, Nc, i, random_state)
    except ImportError:
        print("Repliclust not available, using fallback Gaussian clusters")
        for k in [2, 5]:
            Nc = 250 if k == 2 else 100
            for d in dims:
                for i in range(n_per_config):
                    _make_gaussian_fallback(synthetic, k, d, Nc, i, random_state)
    
    return synthetic


def _make_gaussian_fallback(synthetic, k, d, Nc, i, random_state):
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
    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)


def run_synthetic_type(dtype, datasets, output_dir):
    """Run all experiments for one synthetic type."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dtype.upper()} ({len(datasets)} datasets)")
    print(f"{'='*60}")
    
    # Check for cached results
    result_file = os.path.join(output_dir, f'synthetic_{dtype}_results.json')
    if os.path.exists(result_file):
        print(f"Results already exist at {result_file}, loading...")
        with open(result_file) as f:
            return json.load(f)
    
    # Precompute DR
    cache_file = os.path.join(output_dir, f'dr_cache_{dtype}.pkl')
    if os.path.exists(cache_file):
        print(f"Loading cached DR from {cache_file}")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
        missing = [k for k in datasets if k not in dr_cache]
        if missing:
            print(f"  {len(missing)} datasets missing from cache, recomputing")
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
    print(f"  AHC (metric={ahc_m}, linkage={ahc_l}) took {time.time()-t0:.1f}s")
    
    # GMM
    print(f"\n  Running GMM on {dtype}...")
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    type_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    print(f"  GMM (cov={gmm_cov}) took {time.time()-t0:.1f}s")
    
    # OPTICS
    print(f"\n  Running OPTICS on {dtype}...")
    t0 = time.time()
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
    type_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, opt_ms, opt_mcs)
    print(f"  OPTICS (ms={opt_ms}, mcs={opt_mcs}) took {time.time()-t0:.1f}s")
    
    # Save hyperparams
    type_results['_hyperparams'] = {
        'AHC': {'metric': ahc_m, 'linkage': ahc_l},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)},
    }
    
    # Save results
    with open(result_file, 'w') as f:
        json.dump(type_results, f, indent=2)
    print(f"  Saved to {result_file}")
    
    return type_results


def main():
    print("=" * 60)
    print("SYNTHETIC EXPERIMENTS (FAST)")
    print("=" * 60)
    
    # Generate all synthetic datasets
    t0 = time.time()
    all_synthetic = generate_synthetic_datasets()
    print(f"\nGenerated {len(all_synthetic)} total synthetic datasets in {time.time()-t0:.1f}s")
    
    # Count by type
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    for tn in type_names:
        count = sum(1 for k in all_synthetic if k.startswith(tn))
        print(f"  {tn}: {count} datasets")
    
    all_type_results = {}
    
    for dtype in type_names:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets:
            print(f"No {dtype} datasets, skipping")
            continue
        all_type_results[dtype] = run_synthetic_type(dtype, datasets, OUTPUT_DIR)
    
    # Save combined results
    with open(os.path.join(OUTPUT_DIR, 'synthetic_results_all.json'), 'w') as f:
        json.dump(all_type_results, f, indent=2)
    
    # Generate tables
    print("\n" + "=" * 60)
    print("SYNTHETIC AVERAGE ARI TABLES")
    print("=" * 60)
    conditions = get_all_conditions()
    
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        print(f"\n--- {dtype} ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            avg_row = {}
            for cond in conditions:
                vals = [results[ds].get(cond, 0.0) for ds in results]
                avg_row[cond] = round(np.mean(vals), 2) if vals else 0
            
            df = pd.DataFrame([avg_row], index=[f'{dtype}_avg'], columns=conditions)
            csv_path = os.path.join(OUTPUT_DIR, f'table_{algo}_synthetic_{dtype}.csv')
            df.to_csv(csv_path)
            print(f"  {algo}: No_Red={avg_row.get('No Reduction',0):.2f} | saved to {csv_path}")
    
    # Generate boxplots
    print("\n" + "=" * 60)
    print("GENERATING BOXPLOTS")
    print("=" * 60)
    for dtype in type_names:
        if dtype in all_type_results:
            plot_data = {k: v for k, v in all_type_results[dtype].items() 
                        if k in ['k-means', 'AHC', 'GMM', 'OPTICS']}
            generate_boxplots(plot_data, f'Synthetic_{dtype}', OUTPUT_DIR)
    
    # Aggregate stats
    print("\n" + "=" * 60)
    print("SYNTHETIC AGGREGATE STATISTICS")
    print("=" * 60)
    synth_agg = {}
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        synth_agg[dtype] = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            stats = compute_aggregate_stats(results)
            synth_agg[dtype][algo] = stats
    
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
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            pvals = compute_wilcoxon_tests(results)
            synth_wilcoxon[dtype][algo] = pvals
    
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results_synthetic.json'), 'w') as f:
        json.dump(synth_wilcoxon, f, indent=2)
    
    print("\n\nAll synthetic experiments complete!")


if __name__ == '__main__':
    main()
