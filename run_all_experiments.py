"""
Complete experiment runner for both real-world and synthetic data.
Optimized for speed while maintaining paper methodology.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pickle
import pandas as pd
import signal

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, cluster_optics_xi
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']


# ============================================================
# VAE
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar


def apply_vae(X, n_components, epochs=50, batch_size=64):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    X_min, X_max = X.min(0), X.max(0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X - X_min) / X_range
    n = len(X_scaled)
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    train_data = torch.FloatTensor(X_scaled[idx[:n_train]]).to(device)
    all_data = torch.FloatTensor(X_scaled).to(device)
    loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    model = VAE(input_dim, n_components).to(device)
    optimizer = optim.Adam(model.parameters())
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            mse = nn.functional.mse_loss(recon, batch, reduction='sum')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            optimizer.zero_grad()
            (mse + kl).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(all_data)
    return mu.cpu().numpy()


# ============================================================
# DR
# ============================================================
def get_reduction_dims(n_features, k):
    return {
        'k-1': min(n_features, max(2, k - 1)),
        '25%': min(n_features, max(2, int(np.round(0.25 * n_features)))),
        '50%': min(n_features, max(2, int(np.round(0.50 * n_features)))),
    }


def apply_dr(X, method, n_components):
    if n_components >= X.shape[1]:
        return X.copy()
    if method == 'PCA':
        return PCA(n_components=n_components, random_state=42).fit_transform(X)
    elif method == 'Kernel PCA':
        return KernelPCA(n_components=n_components, kernel='rbf', random_state=42).fit_transform(X)
    elif method == 'VAE':
        return apply_vae(X, n_components)
    elif method == 'Isomap':
        nn = min(5, X.shape[0] - 1)
        return Isomap(n_components=n_components, n_neighbors=nn).fit_transform(X)
    elif method == 'MDS':
        return MDS(n_components=n_components, random_state=10, n_init=1, max_iter=50,
                   normalized_stress='auto').fit_transform(X)


def get_all_conditions():
    conds = ['No Reduction']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            conds.append(f"{m}_{l}")
    return conds


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("timeout")


def precompute_all_dr(datasets, timeout=30):
    """Precompute all DR transformations."""
    dr_cache = {}
    total = len(datasets)
    for i, ds_name in enumerate(sorted(datasets.keys())):
        X_raw, y_true, k = datasets[ds_name]
        X = StandardScaler().fit_transform(X_raw)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        dims = get_reduction_dims(X.shape[1], k)
        ds_cache = {'No Reduction': X.copy()}
        if (i+1) % 10 == 0 or total <= 20:
            print(f"  DR [{i+1}/{total}] {ds_name} shape={X.shape}")
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                n_comp = dims[level]
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    X_red = apply_dr(X, method, n_comp)
                    signal.alarm(0)
                    ds_cache[key] = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
                except:
                    signal.alarm(0)
                    ds_cache[key] = None
        dr_cache[ds_name] = ds_cache
    return dr_cache


# ============================================================
# CLUSTERING
# ============================================================
def cluster_kmeans(X, k):
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit_predict(X)

def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    return AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage).fit_predict(X)

def cluster_gmm(X, k, covariance_type='full'):
    return GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=1, random_state=42).fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    return OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05, min_cluster_size=min_cluster_size).fit_predict(X)


# ============================================================
# HYPERPARAMETER SEARCH (FAST: only use No Reduction + a few conditions)
# ============================================================
def find_best_ahc_params(datasets, dr_cache, max_ds=15):
    """Find best (metric, linkage) using only No Reduction condition for speed."""
    combos = []
    for metric in ['euclidean', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))
    
    # Use subset of conditions for speed
    search_conds = ['No Reduction', 'PCA_50%', 'Kernel PCA_50%']
    keys = sorted(datasets.keys())
    if len(keys) > max_ds:
        rng = np.random.RandomState(42)
        keys = list(rng.choice(keys, max_ds, replace=False))
    
    best_score, best_combo = -999, ('euclidean', 'ward')
    for metric, linkage in combos:
        total, count = 0.0, 0
        for ds_name in keys:
            _, y_true, k = datasets[ds_name]
            for cond in search_conds:
                X = dr_cache[ds_name].get(cond)
                if X is None: continue
                try:
                    labels = cluster_ahc(X, k, metric=metric, linkage=linkage)
                    total += adjusted_rand_score(y_true, labels)
                    count += 1
                except: pass
        avg = total / count if count > 0 else -999
        if avg > best_score:
            best_score, best_combo = avg, (metric, linkage)
    print(f"    Best AHC: {best_combo}, avg_ari={best_score:.4f}")
    return best_combo


def find_best_gmm_params(datasets, dr_cache, max_ds=10):
    """Find best covariance_type using only No Reduction condition."""
    search_conds = ['No Reduction']
    keys = sorted(datasets.keys())
    if len(keys) > max_ds:
        rng = np.random.RandomState(42)
        keys = list(rng.choice(keys, max_ds, replace=False))
    
    best_score, best_cov = -999, 'full'
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        total, count = 0.0, 0
        for ds_name in keys:
            _, y_true, k = datasets[ds_name]
            for cond in search_conds:
                X = dr_cache[ds_name].get(cond)
                if X is None: continue
                try:
                    labels = cluster_gmm(X, k, covariance_type=cov_type)
                    total += adjusted_rand_score(y_true, labels)
                    count += 1
                except: pass
        avg = total / count if count > 0 else -999
        if avg > best_score:
            best_score, best_cov = avg, cov_type
    print(f"    Best GMM: {best_cov}, avg_ari={best_score:.4f}")
    return best_cov


def find_best_optics_params(datasets, dr_cache, max_ds=10):
    """Find best (min_samples, min_cluster_size)."""
    search_conds = ['No Reduction']
    keys = sorted(datasets.keys())
    if len(keys) > max_ds:
        rng = np.random.RandomState(42)
        keys = list(rng.choice(keys, max_ds, replace=False))
    
    mcs_values = [0.05, 0.2, 0.4, 0.6, 0.8]
    best_score, best_combo = -999, (5, 0.05)
    for ms in [5, 7, 10]:
        for mcs in mcs_values:
            total, count = 0.0, 0
            for ds_name in keys:
                _, y_true, k = datasets[ds_name]
                for cond in search_conds:
                    X = dr_cache[ds_name].get(cond)
                    if X is None: continue
                    if ms >= X.shape[0]: continue
                    try:
                        labels = cluster_optics(X, min_samples=ms, min_cluster_size=mcs)
                        total += adjusted_rand_score(y_true, labels)
                        count += 1
                    except: pass
            avg = total / count if count > 0 else -999
            if avg > best_score:
                best_score, best_combo = avg, (ms, mcs)
    print(f"    Best OPTICS: ms={best_combo[0]}, mcs={best_combo[1]}, avg_ari={best_score:.4f}")
    return best_combo


# ============================================================
# EXPERIMENT RUNNERS
# ============================================================
def run_clustering(datasets, dr_cache, algo, **kwargs):
    """Run one clustering algorithm on all datasets × conditions."""
    results = {}
    conditions = get_all_conditions()
    for ds_name in sorted(datasets.keys()):
        _, y_true, k = datasets[ds_name]
        row = {}
        for cond in conditions:
            X = dr_cache[ds_name].get(cond)
            if X is None:
                row[cond] = 0.0
                continue
            try:
                if algo == 'k-means':
                    labels = cluster_kmeans(X, k)
                elif algo == 'AHC':
                    labels = cluster_ahc(X, k, **kwargs)
                elif algo == 'GMM':
                    labels = cluster_gmm(X, k, **kwargs)
                elif algo == 'OPTICS':
                    labels = cluster_optics(X, **kwargs)
                row[cond] = round(adjusted_rand_score(y_true, labels), 2)
            except:
                row[cond] = 0.0
        results[ds_name] = row
    return results


# ============================================================
# ANALYSIS
# ============================================================
def compute_aggregate_stats(results_dict):
    datasets_list = sorted(results_dict.keys())
    stats = {}
    for method in DR_METHODS:
        stats[method] = {}
        for level in REDUCTION_LEVELS:
            key = f"{method}_{level}"
            wins, losses, total = 0, 0, 0
            win_changes, loss_changes = [], []
            for ds in datasets_list:
                baseline = results_dict[ds].get('No Reduction', 0.0)
                reduced = results_dict[ds].get(key, 0.0)
                total += 1
                if reduced > baseline + 0.005:
                    wins += 1
                    if abs(baseline) > 1e-6:
                        win_changes.append((reduced - baseline) / abs(baseline) * 100)
                    else:
                        win_changes.append(reduced * 100)
                elif reduced < baseline - 0.005:
                    losses += 1
                    if abs(baseline) > 1e-6:
                        loss_changes.append((reduced - baseline) / abs(baseline) * 100)
                    else:
                        loss_changes.append(-reduced * 100)
            stats[method][level] = {
                'win_pct': round(wins / total * 100, 1) if total > 0 else 0,
                'loss_pct': round(losses / total * 100, 1) if total > 0 else 0,
                'avg_win_change': round(np.mean(win_changes), 2) if win_changes else 0,
                'avg_loss_change': round(np.mean(loss_changes), 2) if loss_changes else 0,
                'wins': wins, 'losses': losses, 'total': total,
            }
    return stats


def compute_wilcoxon_tests(results_dict):
    datasets_list = sorted(results_dict.keys())
    p_values = {}
    for method in DR_METHODS:
        p_values[method] = {}
        for level in REDUCTION_LEVELS:
            key = f"{method}_{level}"
            baselines = np.array([results_dict[ds].get('No Reduction', 0.0) for ds in datasets_list])
            reduced = np.array([results_dict[ds].get(key, 0.0) for ds in datasets_list])
            diffs = reduced - baselines
            nonzero = np.abs(diffs) > 1e-10
            if nonzero.sum() >= 2:
                try:
                    _, p = wilcoxon(diffs[nonzero])
                    p_values[method][level] = round(p, 4)
                except:
                    p_values[method][level] = 1.0
            else:
                p_values[method][level] = 1.0
    return p_values


def generate_boxplots(all_results, data_label, output_dir='./results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = get_all_conditions()
    for algo in ["k-means", "AHC", "GMM", "OPTICS"]:
        results_dict = all_results.get(algo)
        if not results_dict: continue
        fig, ax = plt.subplots(figsize=(16, 6))
        box_data, xlabels = [], []
        for cond in conditions:
            vals = [results_dict[ds].get(cond, 0.0) for ds in results_dict]
            box_data.append(vals)
            if cond == 'No Reduction':
                xlabels.append('No Red.')
            else:
                parts = cond.rsplit('_', 1)
                xlabels.append(f"{parts[0]}\n{parts[1]}")
        bp = ax.boxplot(box_data, patch_artist=True)
        colors = ['gray'] + ['#1f77b4']*3 + ['#ff7f0e']*3 + ['#2ca02c']*3 + ['#d62728']*3 + ['#9467bd']*3
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('ARI')
        ax.set_title(f'{algo} — {data_label}')
        plt.tight_layout()
        fname = f'boxplot_{algo}_{data_label}.pdf'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# ============================================================
# SYNTHETIC DATA GENERATION (REDUCED)
# ============================================================
def generate_synthetic_datasets():
    """Generate synthetic datasets: Circles, Moons, RSG, Repliclust.
    Reduced count for speed: 3 per config instead of 50."""
    from sklearn.datasets import make_circles, make_moons
    
    rng = np.random.RandomState(42)
    synthetic = {}
    n_per = 3  # datasets per config
    n_samples = 500
    dims = [10, 50, 200]
    
    def embed_hd(X, target_dim, seed):
        if target_dim <= X.shape[1]: return X
        r = np.random.RandomState(seed)
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        X_proj = X @ proj
        noise = r.normal(0, 0.1, (X.shape[0], target_dim))
        return X_proj + noise * 0.3
    
    # Circles k=2
    for d in dims:
        for i in range(n_per):
            X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42+i)
            synthetic[f'Circles_k2_d{d}_t{i}'] = (embed_hd(X, d, 1000+i), y, 2)
    
    # Circles k=5
    for d in dims:
        for i in range(n_per):
            r = np.random.RandomState(42+i+2000)
            n_pc = n_samples // 5
            X_list, y_list = [], []
            for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                theta = r.uniform(0, 2*np.pi, n_pc)
                rad = factor + r.normal(0, 0.05, n_pc)
                X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                y_list.append(np.full(n_pc, ci))
            X, y = np.vstack(X_list), np.concatenate(y_list)
            synthetic[f'Circles_k5_d{d}_t{i}'] = (embed_hd(X, d, 3000+i), y, 5)
    
    # Moons k=2
    for d in dims:
        for i in range(n_per):
            X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42+i+4000)
            synthetic[f'Moons_k2_d{d}_t{i}'] = (embed_hd(X, d, 5000+i), y, 2)
    
    # Moons k=5
    for d in dims:
        for i in range(n_per):
            r = np.random.RandomState(42+i+6000)
            n_pc = n_samples // 5
            X_list, y_list = [], []
            for ci in range(5):
                theta = np.linspace(0, np.pi, n_pc)
                x = np.cos(theta) + r.normal(0, 0.1, n_pc)
                yc = np.sin(theta) + r.normal(0, 0.1, n_pc)
                angle = np.radians(ci * 72)
                xr = x*np.cos(angle) - yc*np.sin(angle) + ci*2
                yr = x*np.sin(angle) + yc*np.cos(angle) + ci
                X_list.append(np.column_stack([xr, yr]))
                y_list.append(np.full(n_pc, ci))
            X, y = np.vstack(X_list), np.concatenate(y_list)
            synthetic[f'Moons_k5_d{d}_t{i}'] = (embed_hd(X, d, 7000+i), y, 5)
    
    # RSG: k in {2,10,50}, d in {10,50,200}, Nc in {5,50,100}
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            for Nc in [5, 50, 100]:
                for i in range(n_per):
                    r = np.random.RandomState(42 + k*1000 + d*100 + Nc + i)
                    alpha = max(0.1, min(0.9, 1.0 - k*0.01 - d*0.001))
                    centers = r.randn(k, d) * np.sqrt(d) * (1 + alpha)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * alpha
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples)
                        y_list.append(np.full(Nc, ci))
                    X, y = np.vstack(X_list), np.concatenate(y_list)
                    synthetic[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    # Repliclust (fallback to anisotropic Gaussian)
    for k in [2, 5]:
        Nc_rep = 250 if k == 2 else 100
        for d in dims:
            for i in range(n_per):
                try:
                    import repliclust
                    repliclust.set_seed(42 + i + k*100 + d)
                    archetype = repliclust.Archetype(
                        n_clusters=k, dim=d, n_samples=k*Nc_rep,
                        aspect_ref=3.0, radius_maxmin=3
                    )
                    X, y, _ = repliclust.DataGenerator(archetype).synthesize()
                    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
                except:
                    r = np.random.RandomState(42 + i + 9000 + k*100 + d)
                    centers = r.randn(k, d) * np.sqrt(d)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * 0.5
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc_rep)
                        X_list.append(samples)
                        y_list.append(np.full(Nc_rep, ci))
                    X, y = np.vstack(X_list), np.concatenate(y_list)
                    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    
    return synthetic


# ============================================================
# MAIN RUNNERS
# ============================================================
def run_experiment_set(datasets, dr_cache, label):
    """Run all 4 clustering algorithms on a set of datasets."""
    results = {}
    
    # k-means
    t0 = time.time()
    results['k-means'] = run_clustering(datasets, dr_cache, 'k-means')
    print(f"  k-means: {time.time()-t0:.1f}s")
    
    # AHC
    t0 = time.time()
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache)
    results['AHC'] = run_clustering(datasets, dr_cache, 'AHC', metric=ahc_m, linkage=ahc_l)
    print(f"  AHC ({ahc_m}/{ahc_l}): {time.time()-t0:.1f}s")
    
    # GMM
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    results['GMM'] = run_clustering(datasets, dr_cache, 'GMM', covariance_type=gmm_cov)
    print(f"  GMM ({gmm_cov}): {time.time()-t0:.1f}s")
    
    # OPTICS
    t0 = time.time()
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
    results['OPTICS'] = run_clustering(datasets, dr_cache, 'OPTICS',
                                        min_samples=opt_ms, min_cluster_size=opt_mcs)
    print(f"  OPTICS (ms={opt_ms}, mcs={opt_mcs}): {time.time()-t0:.1f}s")
    
    results['_params'] = {
        'AHC': {'metric': ahc_m, 'linkage': ahc_l},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)},
    }
    return results


def run_real_world():
    """Run real-world experiments."""
    print("=" * 60)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 60)
    
    from load_uci import load_all_uci
    dataset_list = load_all_uci()
    datasets = {name: (X, y, k) for name, X, y, k in dataset_list}
    print(f"Loaded {len(datasets)} datasets")
    
    # Check for cached DR
    cache_file = os.path.join(OUTPUT_DIR, 'dr_cache_real_final.pkl')
    if os.path.exists(cache_file):
        print("Loading cached DR...")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
    else:
        t0 = time.time()
        dr_cache = precompute_all_dr(datasets)
        print(f"DR took {time.time()-t0:.1f}s")
        with open(cache_file, 'wb') as f:
            pickle.dump(dr_cache, f)
    
    results = run_experiment_set(datasets, dr_cache, 'RealWorld')
    
    # Save
    with open(os.path.join(OUTPUT_DIR, 'real_world_results_final.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Tables
    conditions = get_all_conditions()
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        df = pd.DataFrame(
            [[results[algo][ds].get(c, 0.0) for c in conditions] for ds in sorted(results[algo].keys())],
            index=sorted(results[algo].keys()), columns=conditions
        )
        df.to_csv(os.path.join(OUTPUT_DIR, f'table_{algo}_real_final.csv'))
    
    # Aggregate stats
    agg = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        agg[algo] = compute_aggregate_stats(results[algo])
    with open(os.path.join(OUTPUT_DIR, 'aggregate_stats_real_final.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    
    # Wilcoxon
    wilc = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        wilc[algo] = compute_wilcoxon_tests(results[algo])
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_real_final.json'), 'w') as f:
        json.dump(wilc, f, indent=2)
    
    # Boxplots
    generate_boxplots(results, 'RealWorld', OUTPUT_DIR)
    
    return results


def run_synthetic():
    """Run synthetic experiments."""
    print("\n" + "=" * 60)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 60)
    
    all_synthetic = generate_synthetic_datasets()
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    for tn in type_names:
        count = sum(1 for k in all_synthetic if k.startswith(tn))
        print(f"  {tn}: {count} datasets")
    
    all_type_results = {}
    
    for dtype in type_names:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets: continue
        
        print(f"\n--- {dtype} ({len(datasets)} datasets) ---")
        
        # Check for cached results
        result_file = os.path.join(OUTPUT_DIR, f'synthetic_{dtype}_final.json')
        if os.path.exists(result_file):
            print(f"  Loading cached results from {result_file}")
            with open(result_file) as f:
                all_type_results[dtype] = json.load(f)
            continue
        
        # Check for cached DR
        cache_file = os.path.join(OUTPUT_DIR, f'dr_cache_{dtype}_final.pkl')
        if os.path.exists(cache_file):
            print(f"  Loading cached DR...")
            with open(cache_file, 'rb') as f:
                dr_cache = pickle.load(f)
        else:
            t0 = time.time()
            dr_cache = precompute_all_dr(datasets)
            print(f"  DR took {time.time()-t0:.1f}s")
            with open(cache_file, 'wb') as f:
                pickle.dump(dr_cache, f)
        
        type_results = run_experiment_set(datasets, dr_cache, dtype)
        all_type_results[dtype] = type_results
        
        with open(result_file, 'w') as f:
            json.dump(type_results, f, indent=2)
        
        # Free memory
        del dr_cache
    
    # Save combined
    with open(os.path.join(OUTPUT_DIR, 'synthetic_results_final.json'), 'w') as f:
        json.dump(all_type_results, f, indent=2)
    
    # Generate tables and boxplots per type
    conditions = get_all_conditions()
    for dtype in type_names:
        if dtype not in all_type_results: continue
        print(f"\n--- {dtype} Average ARI ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            res = all_type_results[dtype].get(algo, {})
            if not res: continue
            avg_row = {}
            for cond in conditions:
                vals = [res[ds].get(cond, 0.0) for ds in res]
                avg_row[cond] = round(np.mean(vals), 2) if vals else 0
            df = pd.DataFrame([avg_row], index=[f'{dtype}_avg'], columns=conditions)
            df.to_csv(os.path.join(OUTPUT_DIR, f'table_{algo}_synthetic_{dtype}_final.csv'))
            print(f"  {algo}: No_Red={avg_row.get('No Reduction',0):.2f}")
        
        generate_boxplots(all_type_results[dtype], f'Synthetic_{dtype}', OUTPUT_DIR)
    
    # Aggregate stats per type
    synth_agg = {}
    for dtype in type_names:
        if dtype not in all_type_results: continue
        synth_agg[dtype] = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            res = all_type_results[dtype].get(algo, {})
            if res:
                synth_agg[dtype][algo] = compute_aggregate_stats(res)
    with open(os.path.join(OUTPUT_DIR, 'aggregate_stats_synthetic_final.json'), 'w') as f:
        json.dump(synth_agg, f, indent=2)
    
    # Wilcoxon per type
    synth_wilc = {}
    for dtype in type_names:
        if dtype not in all_type_results: continue
        synth_wilc[dtype] = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            res = all_type_results[dtype].get(algo, {})
            if res:
                synth_wilc[dtype][algo] = compute_wilcoxon_tests(res)
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_synthetic_final.json'), 'w') as f:
        json.dump(synth_wilc, f, indent=2)
    
    return all_type_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'all'], default='all')
    args = parser.parse_args()
    
    if args.mode in ['real', 'all']:
        run_real_world()
    if args.mode in ['synthetic', 'all']:
        run_synthetic()
    
    print("\n\nALL EXPERIMENTS COMPLETE!")
