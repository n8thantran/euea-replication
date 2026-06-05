"""
Efficient synthetic experiment runner.
Runs all 4 types sequentially with optimized hyperparameter search.
"""
import os, sys, json, time, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, cluster_optics_xi
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from experiment import (
    precompute_all_dr, get_all_conditions, DR_METHODS, REDUCTION_LEVELS,
    generate_boxplots
)

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_all_synthetic(random_state=42):
    """Generate all synthetic datasets."""
    from sklearn.datasets import make_circles, make_moons
    
    rng = np.random.RandomState(random_state)
    synthetic = {}
    
    def embed_to_high_dim(X, target_dim, seed):
        if target_dim <= X.shape[1]:
            return X
        r = np.random.RandomState(seed)
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        return X @ proj
    
    def add_noise_dims(X, target_dims, seed):
        rng_local = np.random.RandomState(seed)
        n_samples, n_orig = X.shape
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        noise = rng_local.normal(0, 0.5, (n_samples, n_extra))
        return np.hstack([X, noise])
    
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
    
    for k in ks_rsg:
        for d in ds_rsg:
            for Nc in Ncs_rsg:
                for i in range(n_per_config):
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


# ============================================================
# FAST CLUSTERING FUNCTIONS
# ============================================================

def cluster_kmeans(X, k, random_state=42):
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state).fit_predict(X)

def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    if linkage == 'ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
    return AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage).fit_predict(X)

def cluster_gmm(X, k, covariance_type='full', random_state=42):
    return GaussianMixture(n_components=k, covariance_type=covariance_type, 
                          n_init=5, random_state=random_state).fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    model = OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05, 
                   min_cluster_size=min_cluster_size)
    return model.fit_predict(X)


# ============================================================
# FAST HYPERPARAMETER SEARCH (subsample for speed)
# ============================================================

def fast_ahc_search(datasets, dr_cache, max_ds=6):
    """Quick AHC search using subset of datasets."""
    combos = []
    for metric in ['euclidean', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))
    
    conditions = get_all_conditions()
    ds_names = sorted(datasets.keys())[:max_ds]  # subsample
    
    best_score, best_combo = -999, ('euclidean', 'ward')
    for metric, linkage in combos:
        total, count = 0.0, 0
        for ds_name in ds_names:
            _, y_true, k = datasets[ds_name]
            for cond in conditions:
                X = dr_cache[ds_name].get(cond)
                if X is None: continue
                try:
                    labels = cluster_ahc(X, k, metric=metric, linkage=linkage)
                    total += adjusted_rand_score(y_true, labels)
                    count += 1
                except: pass
        avg = total / count if count > 0 else -999
        if avg > best_score:
            best_score = avg
            best_combo = (metric, linkage)
    
    print(f"    Best AHC: {best_combo}, avg_ari={best_score:.4f}")
    return best_combo

def fast_gmm_search(datasets, dr_cache, max_ds=6):
    """Quick GMM search."""
    conditions = get_all_conditions()
    ds_names = sorted(datasets.keys())[:max_ds]
    
    best_score, best_cov = -999, 'full'
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        total, count = 0.0, 0
        for ds_name in ds_names:
            _, y_true, k = datasets[ds_name]
            for cond in conditions:
                X = dr_cache[ds_name].get(cond)
                if X is None: continue
                try:
                    labels = cluster_gmm(X, k, covariance_type=cov_type)
                    total += adjusted_rand_score(y_true, labels)
                    count += 1
                except: pass
        avg = total / count if count > 0 else -999
        if avg > best_score:
            best_score = avg
            best_cov = cov_type
    
    print(f"    Best GMM: {best_cov}, avg_ari={best_score:.4f}")
    return best_cov

def fast_optics_search(datasets, dr_cache, max_ds=6):
    """Quick OPTICS search."""
    conditions = get_all_conditions()
    ds_names = sorted(datasets.keys())[:max_ds]
    mcs_values = [0.05, 0.15, 0.25, 0.45, 0.65, 0.85]
    
    combo_scores = {}
    for ms in [5, 7, 10]:
        for mcs in mcs_values:
            combo_scores[(ms, mcs)] = []
    
    for ds_name in ds_names:
        _, y_true, k = datasets[ds_name]
        for cond in ['No Reduction', 'PCA_k-1', 'PCA_50%']:  # Only 3 conditions for speed
            X = dr_cache[ds_name].get(cond)
            if X is None: continue
            for ms in [5, 7, 10]:
                if ms >= X.shape[0]: continue
                try:
                    model = OPTICS(min_samples=ms, cluster_method='xi', xi=0.05, min_cluster_size=0.05)
                    model.fit(X)
                    for mcs in mcs_values:
                        try:
                            labels, _ = cluster_optics_xi(
                                reachability=model.reachability_,
                                predecessor=model.predecessor_,
                                ordering=model.ordering_,
                                min_samples=ms, min_cluster_size=mcs, xi=0.05
                            )
                            combo_scores[(ms, mcs)].append(adjusted_rand_score(y_true, labels))
                        except: pass
                except: pass
    
    best_score, best_combo = -999, (5, 0.05)
    for (ms, mcs), aris in combo_scores.items():
        if not aris: continue
        avg = np.mean(aris)
        if avg > best_score:
            best_score = avg
            best_combo = (ms, mcs)
    
    print(f"    Best OPTICS: ms={best_combo[0]}, mcs={best_combo[1]}, avg_ari={best_score:.4f}")
    return best_combo


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_clustering(datasets, dr_cache, algo, **kwargs):
    """Run one clustering algorithm on all datasets × conditions."""
    conditions = get_all_conditions()
    results = {}
    
    for ds_name in sorted(datasets.keys()):
        _, y_true, k = datasets[ds_name]
        results[ds_name] = {}
        for cond in conditions:
            X = dr_cache[ds_name].get(cond)
            if X is None:
                results[ds_name][cond] = 0.0
                continue
            try:
                if algo == 'k-means':
                    labels = cluster_kmeans(X, k)
                elif algo == 'AHC':
                    labels = cluster_ahc(X, k, metric=kwargs['metric'], linkage=kwargs['linkage'])
                elif algo == 'GMM':
                    labels = cluster_gmm(X, k, covariance_type=kwargs['covariance_type'])
                elif algo == 'OPTICS':
                    labels = cluster_optics(X, min_samples=kwargs['min_samples'], 
                                          min_cluster_size=kwargs['min_cluster_size'])
                ari = adjusted_rand_score(y_true, labels)
                results[ds_name][cond] = round(ari, 4)
            except:
                results[ds_name][cond] = 0.0
    
    return results


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
    
    # k-means (no search needed)
    print(f"\n  Running k-means...")
    t0 = time.time()
    type_results['k-means'] = run_clustering(datasets, dr_cache, 'k-means')
    print(f"  k-means took {time.time()-t0:.1f}s")
    
    # AHC
    print(f"\n  Running AHC...")
    t0 = time.time()
    metric, linkage = fast_ahc_search(datasets, dr_cache)
    type_results['AHC'] = run_clustering(datasets, dr_cache, 'AHC', metric=metric, linkage=linkage)
    print(f"  AHC took {time.time()-t0:.1f}s")
    
    # GMM
    print(f"\n  Running GMM...")
    t0 = time.time()
    cov_type = fast_gmm_search(datasets, dr_cache)
    type_results['GMM'] = run_clustering(datasets, dr_cache, 'GMM', covariance_type=cov_type)
    print(f"  GMM took {time.time()-t0:.1f}s")
    
    # OPTICS
    print(f"\n  Running OPTICS...")
    t0 = time.time()
    ms, mcs = fast_optics_search(datasets, dr_cache)
    type_results['OPTICS'] = run_clustering(datasets, dr_cache, 'OPTICS', min_samples=ms, min_cluster_size=mcs)
    print(f"  OPTICS took {time.time()-t0:.1f}s")
    
    type_results['_hyperparams'] = {
        'AHC': {'metric': metric, 'linkage': linkage},
        'GMM': {'covariance_type': cov_type},
        'OPTICS': {'min_samples': int(ms), 'min_cluster_size': float(mcs)},
    }
    
    with open(result_file, 'w') as f:
        json.dump(type_results, f, indent=2)
    print(f"Saved to {result_file}")
    
    return type_results


def compute_aggregate_stats(results_dict):
    """Compute % wins, avg win/loss for each DR method vs No Reduction."""
    conditions = get_all_conditions()
    no_red = 'No Reduction'
    dr_conditions = [c for c in conditions if c != no_red]
    
    stats = {}
    for dr_cond in dr_conditions:
        wins, losses, ties = 0, 0, 0
        win_margins, loss_margins = [], []
        for ds_name, ds_results in results_dict.items():
            if ds_name.startswith('_'): continue
            base = ds_results.get(no_red, 0)
            dr_val = ds_results.get(dr_cond, 0)
            if dr_val > base + 0.005:
                wins += 1
                win_margins.append(dr_val - base)
            elif dr_val < base - 0.005:
                losses += 1
                loss_margins.append(base - dr_val)
            else:
                ties += 1
        total = wins + losses + ties
        stats[dr_cond] = {
            'win_pct': round(100 * wins / total, 1) if total > 0 else 0,
            'loss_pct': round(100 * losses / total, 1) if total > 0 else 0,
            'tie_pct': round(100 * ties / total, 1) if total > 0 else 0,
            'avg_win': round(np.mean(win_margins), 4) if win_margins else 0,
            'avg_loss': round(np.mean(loss_margins), 4) if loss_margins else 0,
        }
    return stats


def compute_wilcoxon_tests(results_dict):
    """Wilcoxon signed-rank test: DR vs No Reduction."""
    conditions = get_all_conditions()
    no_red = 'No Reduction'
    dr_conditions = [c for c in conditions if c != no_red]
    
    tests = {}
    for dr_cond in dr_conditions:
        base_vals, dr_vals = [], []
        for ds_name, ds_results in results_dict.items():
            if ds_name.startswith('_'): continue
            base_vals.append(ds_results.get(no_red, 0))
            dr_vals.append(ds_results.get(dr_cond, 0))
        
        base_arr = np.array(base_vals)
        dr_arr = np.array(dr_vals)
        diff = dr_arr - base_arr
        
        if np.all(diff == 0):
            tests[dr_cond] = {'statistic': 0, 'p_value': 1.0, 'significant': False}
        else:
            try:
                stat, p = wilcoxon(dr_arr, base_arr, alternative='two-sided')
                tests[dr_cond] = {
                    'statistic': round(float(stat), 4),
                    'p_value': round(float(p), 6),
                    'significant': p < 0.05
                }
            except:
                tests[dr_cond] = {'statistic': 0, 'p_value': 1.0, 'significant': False}
    return tests


def main():
    if len(sys.argv) > 1:
        types_to_run = sys.argv[1:]
    else:
        types_to_run = ['Circles', 'Moons', 'RSG', 'Repliclust']
    
    print("=" * 60)
    print(f"SYNTHETIC EXPERIMENTS V2: {types_to_run}")
    print("=" * 60)
    
    # Generate all synthetic datasets
    t0 = time.time()
    all_synthetic = generate_all_synthetic()
    print(f"\nGenerated {len(all_synthetic)} datasets in {time.time()-t0:.1f}s")
    
    # Count per type
    for dtype in types_to_run:
        count = sum(1 for k in all_synthetic if k.startswith(dtype))
        print(f"  {dtype}: {count} datasets")
    
    all_type_results = {}
    for dtype in types_to_run:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets:
            print(f"No {dtype} datasets, skipping")
            continue
        all_type_results[dtype] = run_one_type(dtype, datasets)
    
    # Generate tables, boxplots, stats
    conditions = get_all_conditions()
    
    for dtype in types_to_run:
        if dtype not in all_type_results:
            continue
        print(f"\n--- {dtype} Tables ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            results = all_type_results[dtype].get(algo, {})
            if not results:
                continue
            # Average across all datasets of this type
            avg_row = {}
            ds_names = [k for k in results.keys() if not k.startswith('_')]
            for cond in conditions:
                vals = [results[ds].get(cond, 0.0) for ds in ds_names]
                avg_row[cond] = round(np.mean(vals), 2) if vals else 0
            df = pd.DataFrame([avg_row], index=[f'{dtype}_avg'], columns=conditions)
            csv_path = os.path.join(OUTPUT_DIR, f'table_{algo}_synthetic_{dtype}.csv')
            df.to_csv(csv_path)
            print(f"  {algo}: No_Red={avg_row.get('No Reduction',0):.2f}")
        
        # Boxplots
        plot_data = {k: v for k, v in all_type_results[dtype].items() 
                    if k in ['k-means', 'AHC', 'GMM', 'OPTICS']}
        try:
            generate_boxplots(plot_data, f'Synthetic_{dtype}', OUTPUT_DIR)
        except Exception as e:
            print(f"  Boxplot error for {dtype}: {e}")
    
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
    
    print("\n" + "="*60)
    print("ALL SYNTHETIC EXPERIMENTS COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
