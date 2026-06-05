"""
Run synthetic experiments efficiently - one type at a time with caching.
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


def generate_all_synthetic(random_state=42):
    """Generate all synthetic datasets."""
    from sklearn.datasets import make_circles, make_moons
    
    rng = np.random.RandomState(random_state)
    synthetic = {}
    
    def add_noise_dims(X, target_dims, seed):
        rng_local = np.random.RandomState(seed)
        n_samples, n_orig = X.shape
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        n_per = n_extra // 4
        remainder = n_extra - 4 * n_per
        parts = []
        for sigma in [1.0, 0.5, 0.25, 0.0]:
            n_this = n_per + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            if sigma > 0:
                parts.append(rng_local.normal(0, sigma, (n_samples, n_this)))
            else:
                parts.append(np.zeros((n_samples, n_this)))
        noise = np.hstack(parts)[:, :n_extra]
        return np.hstack([X, noise])
    
    def embed_to_high_dim(X, target_dim, seed):
        if target_dim <= X.shape[1]:
            return X
        r = np.random.RandomState(seed)
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        return X @ proj
    
    dims = [10, 50, 200]
    n_per_config = 3
    
    # === CIRCLES ===
    print("Generating Circles datasets...")
    for k_val in [2, 5]:
        n_samples = 2000
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
                X = add_noise_dims(X, d, random_state+i+d+500)
                synthetic[f'Circles_k{k_val}_d{d}_t{i}'] = (X, y, k_val)
    
    # === MOONS ===
    print("Generating Moons datasets...")
    for k_val in [2, 5]:
        n_samples = 2000
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
                X = add_noise_dims(X, d, random_state+i+d+600)
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
            Nc = 1000 if k == 2 else 400
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
        print("Repliclust not available, using fallback")
        for k in [2, 5]:
            Nc = 1000 if k == 2 else 400
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


def run_one_type(dtype, datasets):
    """Run all experiments for one synthetic type."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dtype.upper()} ({len(datasets)} datasets)")
    print(f"{'='*60}")
    
    result_file = os.path.join(OUTPUT_DIR, f'synthetic_{dtype}_results.json')
    if os.path.exists(result_file):
        print(f"Results already exist at {result_file}, loading...")
        with open(result_file) as f:
            return json.load(f)
    
    # DR precomputation with caching
    cache_file = os.path.join(OUTPUT_DIR, f'dr_cache_{dtype}.pkl')
    if os.path.exists(cache_file):
        print(f"Loading cached DR from {cache_file}")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
        missing = [k for k in datasets if k not in dr_cache]
        if missing:
            print(f"  {len(missing)} missing, recomputing...")
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
    print(f"\n  Running k-means...")
    t0 = time.time()
    type_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
    print(f"  k-means took {time.time()-t0:.1f}s")
    
    # AHC
    print(f"\n  Running AHC...")
    t0 = time.time()
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache)
    type_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_m, ahc_l)
    print(f"  AHC took {time.time()-t0:.1f}s")
    
    # GMM
    print(f"\n  Running GMM...")
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    type_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    print(f"  GMM took {time.time()-t0:.1f}s")
    
    # OPTICS
    print(f"\n  Running OPTICS...")
    t0 = time.time()
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
    type_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, opt_ms, opt_mcs)
    print(f"  OPTICS took {time.time()-t0:.1f}s")
    
    type_results['_hyperparams'] = {
        'AHC': {'metric': ahc_m[0] if isinstance(ahc_m, tuple) else ahc_m, 
                'linkage': ahc_l if isinstance(ahc_m, str) else ahc_m[1] if isinstance(ahc_m, tuple) else 'ward'},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)},
    }
    
    with open(result_file, 'w') as f:
        json.dump(type_results, f, indent=2)
    print(f"Saved to {result_file}")
    
    return type_results


def main():
    # Parse args
    if len(sys.argv) > 1:
        types_to_run = sys.argv[1:]
    else:
        types_to_run = ['Circles', 'Moons', 'RSG', 'Repliclust']
    
    print("=" * 60)
    print(f"SYNTHETIC EXPERIMENTS: {types_to_run}")
    print("=" * 60)
    
    # Generate all synthetic datasets
    t0 = time.time()
    all_synthetic = generate_all_synthetic()
    print(f"\nGenerated {len(all_synthetic)} datasets in {time.time()-t0:.1f}s")
    
    all_type_results = {}
    for dtype in types_to_run:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets:
            print(f"No {dtype} datasets, skipping")
            continue
        all_type_results[dtype] = run_one_type(dtype, datasets)
    
    # Generate tables, boxplots, stats for completed types
    conditions = get_all_conditions()
    
    for dtype in types_to_run:
        if dtype not in all_type_results:
            continue
        print(f"\n--- {dtype} Tables ---")
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
            print(f"  {algo}: No_Red={avg_row.get('No Reduction',0):.2f}")
        
        # Boxplots
        plot_data = {k: v for k, v in all_type_results[dtype].items() 
                    if k in ['k-means', 'AHC', 'GMM', 'OPTICS']}
        generate_boxplots(plot_data, f'Synthetic_{dtype}', OUTPUT_DIR)
    
    # Aggregate stats
    synth_agg = {}
    for dtype in types_to_run:
        if dtype not in all_type_results:
            continue
        synth_agg[dtype] = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            synth_agg[dtype][algo] = compute_aggregate_stats(results)
    
    with open(os.path.join(OUTPUT_DIR, 'aggregate_stats_synthetic.json'), 'w') as f:
        json.dump(synth_agg, f, indent=2)
    
    # Wilcoxon tests
    synth_wilcoxon = {}
    for dtype in types_to_run:
        if dtype not in all_type_results:
            continue
        synth_wilcoxon[dtype] = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            synth_wilcoxon[dtype][algo] = compute_wilcoxon_tests(results)
    
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results_synthetic.json'), 'w') as f:
        json.dump(synth_wilcoxon, f, indent=2)
    
    print("\nAll synthetic experiments complete!")


if __name__ == '__main__':
    main()
