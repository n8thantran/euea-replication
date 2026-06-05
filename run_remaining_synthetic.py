#!/usr/bin/env python3
"""Run RSG and Repliclust synthetic experiments with faster OPTICS."""

import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, '/workspace')
from pipeline import (
    DR_METHODS, REDUCTION_LEVELS, ALGOS, get_all_conditions,
    precompute_dr, cluster_kmeans, cluster_ahc, cluster_gmm, cluster_optics,
    find_best_ahc, find_best_gmm,
    save_all_outputs, RESULTS_DIR
)


def find_best_optics_fast(datasets, dr_cache):
    """Faster OPTICS search: fewer combos, subsample datasets."""
    conds = get_all_conditions()
    ms_vals = [5, 10]
    mcs_vals = [0.05, 0.25, 0.5]
    
    # Subsample to max 10 datasets
    ds_names = sorted(datasets.keys())
    if len(ds_names) > 10:
        ds_names = ds_names[:10]
    
    print(f"  OPTICS fast search ({len(ms_vals)} ms × {len(mcs_vals)} mcs × {len(ds_names)} datasets)...")
    combo_scores = {(ms, mcs): [] for ms in ms_vals for mcs in mcs_vals}
    
    for name in ds_names:
        _, y, k = datasets[name]
        for c in ['No Reduction'] + [f'{m}_{l}' for m in ['PCA'] for l in REDUCTION_LEVELS]:
            X = dr_cache[name].get(c)
            if X is None: continue
            for ms in ms_vals:
                if ms >= X.shape[0]: continue
                try:
                    for mcs in mcs_vals:
                        labels = cluster_optics(X, ms, mcs)
                        combo_scores[(ms, mcs)].append(adjusted_rand_score(y, labels))
                except: pass
    
    best_score, best = -999, (5, 0.05)
    for combo, aris in combo_scores.items():
        if aris:
            avg = np.mean(aris)
            if avg > best_score:
                best_score, best = avg, combo
    print(f"    Best: ms={best[0]}, mcs={best[1]}, avg_ari={best_score:.4f}")
    return best


def run_clustering_fast(datasets, dr_cache, dtype):
    """Run all 4 clustering algorithms with faster OPTICS."""
    conds = get_all_conditions()
    all_results = {}
    params = {}
    
    # 1. k-means
    print(f"\n[1/4] k-means...")
    t0 = time.time()
    km_res = {}
    for name in sorted(datasets.keys()):
        _, y, k = datasets[name]
        row = {}
        for c in conds:
            X = dr_cache[name].get(c)
            if X is None: row[c] = 0.0; continue
            try: row[c] = round(adjusted_rand_score(y, cluster_kmeans(X, k)), 2)
            except: row[c] = 0.0
        km_res[name] = row
    all_results['k-means'] = km_res
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # 2. AHC
    print(f"\n[2/4] AHC...")
    t0 = time.time()
    ahc_params = find_best_ahc(datasets, dr_cache)
    params['ahc'] = list(ahc_params)
    ahc_res = {}
    for name in sorted(datasets.keys()):
        _, y, k = datasets[name]
        row = {}
        for c in conds:
            X = dr_cache[name].get(c)
            if X is None: row[c] = 0.0; continue
            try: row[c] = round(adjusted_rand_score(y, cluster_ahc(X, k, ahc_params[0], ahc_params[1])), 2)
            except: row[c] = 0.0
        ahc_res[name] = row
    all_results['AHC'] = ahc_res
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # 3. GMM
    print(f"\n[3/4] GMM...")
    t0 = time.time()
    gmm_cov = find_best_gmm(datasets, dr_cache)
    params['gmm'] = gmm_cov
    gmm_res = {}
    for name in sorted(datasets.keys()):
        _, y, k = datasets[name]
        row = {}
        for c in conds:
            X = dr_cache[name].get(c)
            if X is None: row[c] = 0.0; continue
            try: row[c] = round(adjusted_rand_score(y, cluster_gmm(X, k, gmm_cov)), 2)
            except: row[c] = 0.0
        gmm_res[name] = row
    all_results['GMM'] = gmm_res
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # 4. OPTICS (fast)
    print(f"\n[4/4] OPTICS...")
    t0 = time.time()
    optics_params = find_best_optics_fast(datasets, dr_cache)
    params['optics'] = list(optics_params)
    optics_res = {}
    for name in sorted(datasets.keys()):
        _, y, k = datasets[name]
        row = {}
        for c in conds:
            X = dr_cache[name].get(c)
            if X is None: row[c] = 0.0; continue
            try: row[c] = round(adjusted_rand_score(y, cluster_optics(X, optics_params[0], optics_params[1])), 2)
            except: row[c] = 0.0
        optics_res[name] = row
    all_results['OPTICS'] = optics_res
    print(f"  Done in {time.time()-t0:.1f}s")
    
    return all_results, params


def generate_rsg(n_datasets=5):
    rsg_ds = {}
    for i in range(n_datasets):
        for k in [3, 5]:
            for d in [10, 50]:
                rs = np.random.RandomState(i * 100 + k * 10 + d)
                n_per = 100
                X_list, y_list = [], []
                for c in range(k):
                    center = rs.randn(d) * 3
                    cov = np.eye(d) * (0.5 + rs.rand())
                    X_list.append(rs.multivariate_normal(center, cov, n_per))
                    y_list.append(np.full(n_per, c))
                X = np.vstack(X_list); y = np.concatenate(y_list)
                rsg_ds[f'RSG_{k}c_{d}d_{i}'] = (X, y, k)
    return rsg_ds


def generate_repliclust(n_datasets=5):
    repl_ds = {}
    for i in range(n_datasets):
        for k in [3, 5]:
            for d in [10, 50]:
                X, y = make_blobs(n_samples=k*100, n_features=d, centers=k,
                                  cluster_std=1.5, random_state=i*100+k*10+d)
                repl_ds[f'Repliclust_{k}c_{d}d_{i}'] = (X, y, k)
    return repl_ds


def main():
    for dtype, gen_fn in [('RSG', generate_rsg), ('Repliclust', generate_repliclust)]:
        print(f"\n{'='*60}")
        print(f"{dtype} EXPERIMENTS")
        print(f"{'='*60}")
        
        datasets = gen_fn(5)
        print(f"Generated {len(datasets)} datasets")
        
        # DR
        cache_path = os.path.join(RESULTS_DIR, f'dr_cache_{dtype}.pkl')
        if os.path.exists(cache_path):
            print(f"Loading DR cache from {cache_path}")
            with open(cache_path, 'rb') as f:
                dr_cache = pickle.load(f)
            missing = [d for d in datasets if d not in dr_cache]
            if missing:
                print(f"Computing {len(missing)} missing...")
                extra = precompute_dr({d: datasets[d] for d in missing})
                dr_cache.update(extra)
                with open(cache_path, 'wb') as f:
                    pickle.dump(dr_cache, f)
        else:
            print("Computing DR transformations...")
            dr_cache = precompute_dr(datasets)
            with open(cache_path, 'wb') as f:
                pickle.dump(dr_cache, f)
        
        # Clustering
        all_results, params = run_clustering_fast(datasets, dr_cache, dtype)
        save_all_outputs(all_results, params, dtype)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
