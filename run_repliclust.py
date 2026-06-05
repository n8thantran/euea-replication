#!/usr/bin/env python3
"""Run just Repliclust experiments with fast OPTICS."""
import os, sys, json, time, pickle, warnings
import numpy as np
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
    find_best_ahc, find_best_gmm, save_all_outputs, RESULTS_DIR
)

def generate_repliclust(n_datasets=5):
    repl_ds = {}
    for i in range(n_datasets):
        for k in [3, 5]:
            for d in [10, 50]:
                X, y = make_blobs(n_samples=k*100, n_features=d, centers=k,
                                  cluster_std=1.5, random_state=i*100+k*10+d)
                repl_ds[f'Repliclust_{k}c_{d}d_{i}'] = (X, y, k)
    return repl_ds

datasets = generate_repliclust(5)
print(f"Generated {len(datasets)} datasets")

cache_path = os.path.join(RESULTS_DIR, 'dr_cache_Repliclust.pkl')
with open(cache_path, 'rb') as f:
    dr_cache = pickle.load(f)
print(f"Loaded DR cache with {len(dr_cache)} entries")

conds = get_all_conditions()
all_results = {}
params = {}

# k-means
print("\n[1/4] k-means...")
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

# AHC
print("\n[2/4] AHC...")
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

# GMM
print("\n[3/4] GMM...")
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

# OPTICS - fixed params, no search
print("\n[4/4] OPTICS (fixed params)...")
t0 = time.time()
ms, mcs = 10, 0.25  # Use RSG best params
params['optics'] = [ms, mcs]
optics_res = {}
for name in sorted(datasets.keys()):
    _, y, k = datasets[name]
    row = {}
    for c in conds:
        X = dr_cache[name].get(c)
        if X is None: row[c] = 0.0; continue
        try: row[c] = round(adjusted_rand_score(y, cluster_optics(X, ms, mcs)), 2)
        except: row[c] = 0.0
    optics_res[name] = row
all_results['OPTICS'] = optics_res
print(f"  Done in {time.time()-t0:.1f}s")

save_all_outputs(all_results, params, 'Repliclust')
print("\nDone!")
