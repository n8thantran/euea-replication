"""
Complete pipeline for replicating:
"Assessing the impact of dimensionality reduction on clustering performance"

This is the single authoritative script that produces ALL results.
"""
import os, sys, json, time, warnings, pickle, signal, traceback
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_circles, make_moons
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']

# Paper: 50 reps per synthetic config. We use fewer for computational feasibility.
N_SYNTH_REPS = 5

# ============================================================
# VAE  
# Paper: encoder d→64→32→latent, decoder latent→32→64→d
# BatchNorm, Dropout=0.4, Adam, MSE+KL, 100 epochs, batch=64, 70/30 split, sigmoid output
# ============================================================
class VAEModel(nn.Module):
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


def apply_vae(X, n_components, epochs=100, batch_size=64):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    # Min-max scale to [0,1] for sigmoid output
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
    model = VAEModel(input_dim, n_components).to(device)
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
# DR Methods
# ============================================================
def get_reduction_dims(n_features, k):
    return {
        'k-1': min(n_features - 1, max(2, k - 1)),
        '25%': min(n_features - 1, max(2, int(np.round(0.25 * n_features)))),
        '50%': min(n_features - 1, max(2, int(np.round(0.50 * n_features)))),
    }


def apply_dr(X, method, n_components):
    if n_components >= X.shape[1]:
        return X.copy()
    if method == 'PCA':
        return PCA(n_components=n_components).fit_transform(X)
    elif method == 'Kernel PCA':
        return KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X)
    elif method == 'VAE':
        return apply_vae(X, n_components)
    elif method == 'Isomap':
        nn = min(5, X.shape[0] - 1)
        return Isomap(n_components=n_components, n_neighbors=nn).fit_transform(X)
    elif method == 'MDS':
        # Paper: random_state=10, n_init=50. Use n_init=10 for speed.
        return MDS(n_components=n_components, random_state=10, n_init=10, max_iter=300,
                   normalized_stress='auto').fit_transform(X)
    return None


def get_all_conditions():
    conds = ['No Reduction']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            conds.append(f"{m}_{l}")
    return conds


def precompute_all_dr(datasets, timeout_sec=180, label=""):
    """Precompute DR transformations for all datasets."""
    dr_cache = {}
    total = len(datasets)
    for i, ds_name in enumerate(sorted(datasets.keys())):
        X_raw, y_true, k = datasets[ds_name]
        X = StandardScaler().fit_transform(X_raw)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        dims = get_reduction_dims(X.shape[1], k)
        ds_cache = {'No Reduction': X.copy()}
        
        if (i+1) % max(1, total//5) == 0 or total <= 30:
            print(f"  DR {label} [{i+1}/{total}] {ds_name} shape={X.shape}, dims={dims}")
        
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                n_comp = dims[level]
                try:
                    def _handler(signum, frame):
                        raise TimeoutError()
                    old = signal.signal(signal.SIGALRM, _handler)
                    signal.alarm(timeout_sec)
                    X_red = apply_dr(X, method, n_comp)
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
                    if X_red is not None:
                        ds_cache[key] = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        ds_cache[key] = None
                except:
                    signal.alarm(0)
                    ds_cache[key] = None
                    print(f"    TIMEOUT/ERROR: {ds_name}/{method}/{level}")
        dr_cache[ds_name] = ds_cache
    return dr_cache


# ============================================================
# Clustering
# ============================================================
def cluster_kmeans(X, k):
    return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)

def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    return AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage).fit_predict(X)

def cluster_gmm(X, k, covariance_type='full'):
    return GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42).fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    return OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05,
                  min_cluster_size=min_cluster_size).fit_predict(X)


# ============================================================
# Hyperparameter Search
# Best params chosen per dataset TYPE (average ARI over all datasets + all conditions)
# ============================================================
def find_best_ahc_params(datasets, dr_cache, subsample=None):
    combos = []
    for metric in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))
    
    conditions = get_all_conditions()
    keys = sorted(datasets.keys())
    if subsample and len(keys) > subsample:
        np.random.seed(42)
        keys = list(np.random.choice(keys, subsample, replace=False))
    
    best_score, best_combo = -999, ('euclidean', 'ward')
    for metric, linkage in combos:
        total, count = 0.0, 0
        for ds_name in keys:
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
            best_score, best_combo = avg, (metric, linkage)
    print(f"    Best AHC: {best_combo}, avg_ari={best_score:.4f}")
    return best_combo


def find_best_gmm_params(datasets, dr_cache, subsample=None):
    conditions = get_all_conditions()
    keys = sorted(datasets.keys())
    if subsample and len(keys) > subsample:
        np.random.seed(42)
        keys = list(np.random.choice(keys, subsample, replace=False))
    
    best_score, best_cov = -999, 'full'
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        total, count = 0.0, 0
        for ds_name in keys:
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
            best_score, best_cov = avg, cov_type
    print(f"    Best GMM: {best_cov}, avg_ari={best_score:.4f}")
    return best_cov


def find_best_optics_params(datasets, dr_cache, subsample=None):
    conditions = get_all_conditions()
    keys = sorted(datasets.keys())
    if subsample and len(keys) > subsample:
        np.random.seed(42)
        keys = list(np.random.choice(keys, subsample, replace=False))
    
    ms_range = range(5, 11)
    mcs_vals = [round(i*0.05, 2) for i in range(1, 21)]  # 0.05 to 1.0
    
    best_score, best_combo = -999, (5, 0.05)
    for ms in ms_range:
        for mcs in mcs_vals:
            total, count = 0.0, 0
            for ds_name in keys:
                _, y_true, k = datasets[ds_name]
                for cond in conditions:
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
# Run clustering on all conditions
# ============================================================
def run_clustering(datasets, dr_cache, algo, **kwargs):
    """Returns {ds_name: {condition: ARI_value}}"""
    results = {}
    conditions = get_all_conditions()
    for ds_name in sorted(datasets.keys()):
        _, y_true, k = datasets[ds_name]
        row = {}
        for cond in conditions:
            X = dr_cache[ds_name].get(cond)
            if X is None:
                row[cond] = np.nan
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
                row[cond] = adjusted_rand_score(y_true, labels)
            except:
                row[cond] = np.nan
        results[ds_name] = row
    return results


def run_all_algos(datasets, dr_cache, label, subsample_hp=None):
    """Run all 4 clustering algorithms with hyperparameter search."""
    all_results = {}
    params = {}
    
    t0 = time.time()
    print(f"  [{label}] Running k-means...")
    all_results['k-means'] = run_clustering(datasets, dr_cache, 'k-means')
    print(f"    Done in {time.time()-t0:.0f}s")
    
    t0 = time.time()
    print(f"  [{label}] Searching AHC params...")
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache, subsample=subsample_hp)
    all_results['AHC'] = run_clustering(datasets, dr_cache, 'AHC', metric=ahc_m, linkage=ahc_l)
    params['AHC'] = {'metric': ahc_m, 'linkage': ahc_l}
    print(f"    Done in {time.time()-t0:.0f}s")
    
    t0 = time.time()
    print(f"  [{label}] Searching GMM params...")
    gmm_cov = find_best_gmm_params(datasets, dr_cache, subsample=subsample_hp)
    all_results['GMM'] = run_clustering(datasets, dr_cache, 'GMM', covariance_type=gmm_cov)
    params['GMM'] = {'covariance_type': gmm_cov}
    print(f"    Done in {time.time()-t0:.0f}s")
    
    t0 = time.time()
    print(f"  [{label}] Searching OPTICS params...")
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache, subsample=subsample_hp)
    all_results['OPTICS'] = run_clustering(datasets, dr_cache, 'OPTICS',
                                            min_samples=opt_ms, min_cluster_size=opt_mcs)
    params['OPTICS'] = {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)}
    print(f"    Done in {time.time()-t0:.0f}s")
    
    all_results['_params'] = params
    return all_results


# ============================================================
# Data Loading
# ============================================================
def load_real_world_datasets():
    """Load all 20 UCI datasets."""
    from load_uci import load_all_uci
    raw = load_all_uci()
    datasets = {}
    for name, X, y, k in raw:
        datasets[name] = (np.array(X, dtype=float), np.array(y), k)
    return datasets


def inject_noise(X, rng):
    """Z-score normalize then add structured noise to 75% of features."""
    X = StandardScaler().fit_transform(X)
    d = X.shape[1]
    n = X.shape[0]
    perm = rng.permutation(d)
    q = d // 4
    for j in perm[:q]:
        X[:, j] += rng.normal(0, 1.0, n)
    for j in perm[q:2*q]:
        X[:, j] += rng.normal(0, 0.5, n)
    for j in perm[2*q:3*q]:
        X[:, j] += rng.normal(0, 0.25, n)
    return X


def embed_high_dim(X_2d, target_dim, rng):
    if target_dim <= X_2d.shape[1]:
        return X_2d
    proj = rng.randn(X_2d.shape[1], target_dim) / np.sqrt(target_dim)
    return X_2d @ proj


def generate_synthetic_datasets(synth_type):
    """Generate datasets for one synthetic type."""
    datasets = {}
    n_per = N_SYNTH_REPS
    dims = [10, 50, 200]
    
    if synth_type == 'Circles':
        # k=2
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(42 + i + d)
                X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42+i)
                X = embed_high_dim(X, d, rng)
                X = inject_noise(X, rng)
                datasets[f'Circles_k2_d{d}_t{i}'] = (X, y, 2)
        # k=5
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(1000 + i + d)
                X_list, y_list = [], []
                for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                    theta = rng.uniform(0, 2*np.pi, 100)
                    rad = factor + rng.normal(0, 0.05, 100)
                    X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                    y_list.append(np.full(100, ci))
                X = np.vstack(X_list); y = np.concatenate(y_list)
                X = embed_high_dim(X, d, rng)
                X = inject_noise(X, rng)
                datasets[f'Circles_k5_d{d}_t{i}'] = (X, y, 5)
    
    elif synth_type == 'Moons':
        # k=2
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(2000 + i + d)
                X, y = make_moons(n_samples=500, noise=0.1, random_state=2000+i)
                stretch = 1.0 + 0.5 * (i % 2)
                angle = np.radians(10 * (i - n_per//2))
                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                X = X @ R.T; X[:, 0] *= stretch
                X = embed_high_dim(X, d, rng)
                X = inject_noise(X, rng)
                datasets[f'Moons_k2_d{d}_t{i}'] = (X, y, 2)
        # k=5
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(3000 + i + d)
                n_pc = 100
                X_list, y_list = [], []
                angles_rot = [-160, -10, 0, 10, 180]
                x_shifts = [-4, -2, 0, 2, 4]
                y_shifts = [1.0, 1.2, 1.5, 1.0, 1.2]
                for ci in range(5):
                    X_moon, _ = make_moons(n_samples=n_pc*2, noise=0.1, random_state=3000+i+ci)
                    X_c = X_moon[:n_pc]
                    angle = np.radians(angles_rot[ci])
                    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    X_c = X_c @ R.T
                    X_c[:, 0] += x_shifts[ci]; X_c[:, 1] += y_shifts[ci]
                    X_list.append(X_c); y_list.append(np.full(n_pc, ci))
                X = np.vstack(X_list); y = np.concatenate(y_list)
                X = embed_high_dim(X, d, rng)
                X = inject_noise(X, rng)
                datasets[f'Moons_k5_d{d}_t{i}'] = (X, y, 5)
    
    elif synth_type == 'RSG':
        for k in [2, 10, 50]:
            for d in [10, 50, 200]:
                for Nc in [5, 50, 100]:
                    for i in range(min(n_per, 3)):
                        rng = np.random.RandomState(4000 + k*1000 + d*100 + Nc + i)
                        alpha = max(0.1, min(0.9, 1.0 - k*0.01 - d*0.001))
                        centers = rng.randn(k, d) * np.sqrt(d) * (1 + alpha)
                        X_list, y_list = [], []
                        for ci in range(k):
                            A = rng.randn(d, d) * alpha
                            cov = A @ A.T / d + np.eye(d) * 0.1
                            samples = rng.multivariate_normal(centers[ci], cov, size=Nc)
                            X_list.append(samples); y_list.append(np.full(Nc, ci))
                        X = np.vstack(X_list); y = np.concatenate(y_list)
                        X = inject_noise(X, rng)
                        datasets[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    elif synth_type == 'Repliclust':
        for k in [2, 5]:
            Nc = 1000 if k == 2 else 400
            for d in dims:
                for i in range(n_per):
                    rng = np.random.RandomState(5000 + k*100 + d + i)
                    centers = rng.randn(k, d) * np.sqrt(d) * 2
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = rng.randn(d, d) * 0.5
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = rng.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples); y_list.append(np.full(Nc, ci))
                    X = np.vstack(X_list); y = np.concatenate(y_list)
                    X = inject_noise(X, rng)
                    datasets[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    
    return datasets


# ============================================================
# Output: Tables
# ============================================================
def make_ari_table(results_dict, algo, dataset_names, round_digits=2):
    """Create a DataFrame with ARI values for one clustering algorithm."""
    conditions = get_all_conditions()
    rows = []
    for ds_name in dataset_names:
        if ds_name not in results_dict:
            continue
        row_data = {'Dataset': ds_name}
        for cond in conditions:
            val = results_dict[ds_name].get(cond, np.nan)
            row_data[cond] = round(val, round_digits) if not np.isnan(val) else np.nan
        rows.append(row_data)
    df = pd.DataFrame(rows)
    df.set_index('Dataset', inplace=True)
    return df


def make_synth_avg_table(results_dict, algo, dataset_names, round_digits=2):
    """Average ARI table for synthetic data (averaging over repetitions)."""
    conditions = get_all_conditions()
    # Group by config (remove trial suffix)
    configs = {}
    for name in dataset_names:
        # Remove _tN suffix
        parts = name.rsplit('_t', 1)
        config = parts[0]
        if config not in configs:
            configs[config] = []
        configs[config].append(name)
    
    rows = []
    for config in sorted(configs.keys()):
        row_data = {'Config': config}
        for cond in conditions:
            vals = []
            for ds_name in configs[config]:
                if ds_name in results_dict:
                    v = results_dict[ds_name].get(cond, np.nan)
                    if not np.isnan(v):
                        vals.append(v)
            row_data[cond] = round(np.mean(vals), round_digits) if vals else np.nan
        rows.append(row_data)
    df = pd.DataFrame(rows)
    df.set_index('Config', inplace=True)
    return df


# ============================================================
# Output: Aggregate Statistics (Tables 1-4 in paper)
# Win% = wins / (wins+losses) [ties excluded from denominator]
# Average win/loss% = mean relative ARI change for wins and losses
# ============================================================
def compute_aggregate_stats(results_by_algo, synth_results_all=None, real_results=None):
    """
    Compute Tables 1-4: for each DR method × level, compute:
    - % wins (synthetic + real-world)
    - avg win/loss % (synthetic + real-world)
    
    A "win" = DR ARI > baseline ARI (strict inequality)
    A "loss" = DR ARI < baseline ARI
    Win% = wins / (wins + losses) * 100  [ties excluded]
    Avg win/loss% = mean of ((DR_ARI - base_ARI) / max(|base_ARI|, 0.01)) * 100 over non-tied datasets
    """
    aggregate = {}  # algo -> list of rows
    
    for algo in ALGOS:
        rows_data = []
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f"{method}_{level}"
                row = {'Method': method, 'Reduction': level}
                
                for data_type, results_dict in [('Synthetic', synth_results_all), ('Real-world', real_results)]:
                    if results_dict is None or algo not in results_dict:
                        row[f'Win%_{data_type}'] = np.nan
                        row[f'AvgWL%_{data_type}'] = np.nan
                        continue
                    
                    algo_res = results_dict[algo]
                    wins, losses, changes = 0, 0, []
                    
                    for ds_name in algo_res:
                        base = algo_res[ds_name].get('No Reduction', np.nan)
                        dr = algo_res[ds_name].get(cond, np.nan)
                        if np.isnan(base) or np.isnan(dr):
                            continue
                        diff = dr - base
                        if abs(diff) < 1e-10:  # tie
                            continue
                        denom = max(abs(base), 0.01)
                        pct_change = (diff / denom) * 100
                        changes.append(pct_change)
                        if diff > 0:
                            wins += 1
                        else:
                            losses += 1
                    
                    total_non_tie = wins + losses
                    win_pct = (wins / total_non_tie * 100) if total_non_tie > 0 else np.nan
                    avg_change = np.mean(changes) if changes else np.nan
                    
                    row[f'Win%_{data_type}'] = round(win_pct, 2) if not np.isnan(win_pct) else np.nan
                    row[f'AvgWL%_{data_type}'] = round(avg_change, 2) if not np.isnan(avg_change) else np.nan
                
                rows_data.append(row)
        aggregate[algo] = pd.DataFrame(rows_data)
    
    return aggregate


# ============================================================
# Box plots
# ============================================================
def make_boxplot(results_dict, algo, label, outpath):
    """Create boxplot comparing DR methods, similar to paper's figures."""
    conditions = get_all_conditions()
    # Collect ARI differences (DR - baseline) for each condition
    data_for_plot = {}
    for cond in conditions:
        if cond == 'No Reduction':
            continue
        vals = []
        for ds_name in results_dict:
            base = results_dict[ds_name].get('No Reduction', np.nan)
            dr = results_dict[ds_name].get(cond, np.nan)
            if not np.isnan(base) and not np.isnan(dr):
                vals.append(dr)  # Plot absolute ARI, not difference
        data_for_plot[cond] = vals
    
    # Group by method
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Reorganize: for each method, 3 levels → 3 boxes
    positions = []
    tick_labels = []
    all_data = []
    pos = 1
    
    # Add no-reduction baseline
    baseline_vals = [results_dict[ds].get('No Reduction', np.nan) 
                     for ds in results_dict if not np.isnan(results_dict[ds].get('No Reduction', np.nan))]
    all_data.append(baseline_vals)
    positions.append(pos)
    tick_labels.append('No Red.')
    pos += 2
    
    for method in DR_METHODS:
        for level in REDUCTION_LEVELS:
            cond = f"{method}_{level}"
            vals = data_for_plot.get(cond, [])
            all_data.append(vals)
            positions.append(pos)
            tick_labels.append(f'{level}')
            pos += 1
        pos += 1  # gap between methods
    
    bp = ax.boxplot(all_data, positions=positions, widths=0.7, patch_artist=True)
    
    # Color by method
    colors = ['gray'] + ['#4472C4']*3 + ['#ED7D31']*3 + ['#A5A5A5']*3 + ['#FFC000']*3 + ['#5B9BD5']*3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ARI')
    ax.set_title(f'{algo} - {label}')
    
    # Add method labels
    method_positions = [positions[0]] + [np.mean(positions[1+i*3:1+(i+1)*3]) for i in range(5)]
    method_labels = ['Baseline'] + DR_METHODS
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(method_positions)
    ax2.set_xticklabels(method_labels, fontsize=9)
    
    ax.axhline(y=np.median(baseline_vals) if baseline_vals else 0, color='red', 
               linestyle='--', alpha=0.5, label='Baseline median')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Wilcoxon signed-rank test
# ============================================================
def run_wilcoxon_tests(results_dict):
    """
    For each algo × DR method × level, test if DR ARI differs from baseline.
    Returns dict with p-values and effect directions.
    """
    wilcoxon_results = {}
    for algo in ALGOS:
        if algo not in results_dict or algo.startswith('_'):
            continue
        algo_res = results_dict[algo]
        algo_wilc = {}
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f"{method}_{level}"
                baselines, dr_vals = [], []
                for ds_name in algo_res:
                    base = algo_res[ds_name].get('No Reduction', np.nan)
                    dr = algo_res[ds_name].get(cond, np.nan)
                    if not np.isnan(base) and not np.isnan(dr):
                        baselines.append(base)
                        dr_vals.append(dr)
                
                if len(baselines) >= 10:
                    diffs = np.array(dr_vals) - np.array(baselines)
                    # Remove zeros for Wilcoxon
                    nonzero = diffs[np.abs(diffs) > 1e-10]
                    if len(nonzero) >= 5:
                        try:
                            stat, pval = wilcoxon(nonzero)
                            direction = 'positive' if np.mean(nonzero) > 0 else 'negative'
                            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
                            algo_wilc[cond] = {
                                'stat': float(stat), 'p': float(pval),
                                'direction': direction, 'sig': sig,
                                'n': len(nonzero), 'mean_diff': float(np.mean(nonzero))
                            }
                        except:
                            algo_wilc[cond] = {'stat': np.nan, 'p': np.nan, 'sig': 'na'}
                    else:
                        algo_wilc[cond] = {'stat': np.nan, 'p': np.nan, 'sig': 'na', 'note': 'too few non-zero diffs'}
                else:
                    algo_wilc[cond] = {'stat': np.nan, 'p': np.nan, 'sig': 'na', 'note': 'too few datasets'}
        wilcoxon_results[algo] = algo_wilc
    return wilcoxon_results


def format_wilcoxon_table(wilcoxon_results):
    """Create a formatted table of Wilcoxon test results."""
    rows = []
    for algo in ALGOS:
        if algo not in wilcoxon_results:
            continue
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f"{method}_{level}"
                res = wilcoxon_results[algo].get(cond, {})
                pval = res.get('p', np.nan)
                sig = res.get('sig', 'na')
                direction = res.get('direction', '')
                mean_diff = res.get('mean_diff', np.nan)
                rows.append({
                    'Algorithm': algo,
                    'DR Method': method,
                    'Level': level,
                    'p-value': f'{pval:.4f}' if not np.isnan(pval) else 'NA',
                    'Significance': sig,
                    'Direction': direction,
                    'Mean Δ ARI': f'{mean_diff:.4f}' if not np.isnan(mean_diff) else 'NA'
                })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    
    # ===== REAL-WORLD DATA =====
    print("=" * 60)
    print("PHASE 1: REAL-WORLD DATASETS")
    print("=" * 60)
    
    print("\nLoading 20 UCI datasets...")
    real_datasets = load_real_world_datasets()
    print(f"  Loaded {len(real_datasets)} datasets")
    for name in sorted(real_datasets.keys()):
        X, y, k = real_datasets[name]
        print(f"    {name}: {X.shape}, k={k}")
    
    # Check for cached DR
    dr_cache_path = os.path.join(OUTPUT_DIR, 'dr_cache_real_v2.pkl')
    if os.path.exists(dr_cache_path):
        print("\nLoading cached DR transformations...")
        with open(dr_cache_path, 'rb') as f:
            dr_cache_real = pickle.load(f)
    else:
        print("\nComputing DR transformations for real-world data...")
        dr_cache_real = precompute_all_dr(real_datasets, timeout_sec=300, label="Real")
        with open(dr_cache_path, 'wb') as f:
            pickle.dump(dr_cache_real, f)
    
    print("\nRunning clustering on real-world data...")
    real_results = run_all_algos(real_datasets, dr_cache_real, "RealWorld")
    
    # Save real results
    real_ds_names = sorted(real_datasets.keys())
    for algo in ALGOS:
        if algo in real_results:
            df = make_ari_table(real_results[algo], algo, real_ds_names)
            csv_path = os.path.join(OUTPUT_DIR, f'table_{algo}_RealWorld.csv')
            df.to_csv(csv_path)
            print(f"  Saved {csv_path}")
    
    # Box plots for real data
    for algo in ALGOS:
        if algo in real_results:
            make_boxplot(real_results[algo], algo, 'Real-World',
                        os.path.join(OUTPUT_DIR, f'boxplot_{algo}_RealWorld.pdf'))
    print("  Saved real-world boxplots")
    
    # Wilcoxon tests on real data
    wilcoxon_real = run_wilcoxon_tests(real_results)
    wilcoxon_df = format_wilcoxon_table(wilcoxon_real)
    wilcoxon_df.to_csv(os.path.join(OUTPUT_DIR, 'wilcoxon_real.csv'), index=False)
    print("  Saved Wilcoxon test results")
    
    # Save raw results
    def convert_for_json(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    real_json = {}
    for algo in ALGOS:
        if algo in real_results:
            real_json[algo] = {ds: {k: convert_for_json(v) for k, v in row.items()} 
                              for ds, row in real_results[algo].items()}
    real_json['_params'] = {k: {kk: convert_for_json(vv) for kk, vv in v.items()} 
                           for k, v in real_results.get('_params', {}).items()}
    with open(os.path.join(OUTPUT_DIR, 'real_results.json'), 'w') as f:
        json.dump(real_json, f, indent=2, default=convert_for_json)
    
    # ===== SYNTHETIC DATA =====
    print("\n" + "=" * 60)
    print("PHASE 2: SYNTHETIC DATASETS")
    print("=" * 60)
    
    synth_all_results = {}  # {algo: {ds_name: {cond: ari}}} aggregated across all types
    synth_type_results = {}  # {type: {algo: {ds_name: {cond: ari}}}}
    
    for synth_type in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        print(f"\n--- Synthetic Type: {synth_type} ---")
        
        print(f"  Generating {synth_type} datasets...")
        synth_datasets = generate_synthetic_datasets(synth_type)
        print(f"  Generated {len(synth_datasets)} datasets")
        
        # DR cache
        cache_path = os.path.join(OUTPUT_DIR, f'dr_cache_{synth_type}_v2.pkl')
        if os.path.exists(cache_path):
            print(f"  Loading cached DR for {synth_type}...")
            with open(cache_path, 'rb') as f:
                dr_cache = pickle.load(f)
        else:
            print(f"  Computing DR for {synth_type}...")
            dr_cache = precompute_all_dr(synth_datasets, timeout_sec=120, label=synth_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(dr_cache, f)
        
        # Use subsampling for HP search if many datasets
        sub = min(30, len(synth_datasets)) if len(synth_datasets) > 30 else None
        results = run_all_algos(synth_datasets, dr_cache, synth_type, subsample_hp=sub)
        synth_type_results[synth_type] = results
        
        # Merge into synth_all
        for algo in ALGOS:
            if algo in results:
                if algo not in synth_all_results:
                    synth_all_results[algo] = {}
                synth_all_results[algo].update(results[algo])
        
        # Save per-type tables (averaged over repetitions)
        ds_names = sorted(synth_datasets.keys())
        for algo in ALGOS:
            if algo in results:
                df = make_synth_avg_table(results[algo], algo, ds_names)
                csv_path = os.path.join(OUTPUT_DIR, f'table_{algo}_{synth_type}.csv')
                df.to_csv(csv_path)
        
        # Per-type boxplots
        for algo in ALGOS:
            if algo in results:
                make_boxplot(results[algo], algo, f'Synthetic ({synth_type})',
                            os.path.join(OUTPUT_DIR, f'boxplot_{algo}_Synthetic_{synth_type}.pdf'))
        
        # Save raw results
        synth_json = {}
        for algo in ALGOS:
            if algo in results:
                synth_json[algo] = {ds: {k: convert_for_json(v) for k, v in row.items()} 
                                   for ds, row in results[algo].items()}
        synth_json['_params'] = {k: {kk: convert_for_json(vv) for kk, vv in v.items()} 
                                for k, v in results.get('_params', {}).items()}
        with open(os.path.join(OUTPUT_DIR, f'synth_{synth_type}_results.json'), 'w') as f:
            json.dump(synth_json, f, indent=2, default=convert_for_json)
        
        print(f"  Completed {synth_type}")
    
    # ===== AGGREGATE ANALYSIS =====
    print("\n" + "=" * 60)
    print("PHASE 3: AGGREGATE ANALYSIS")
    print("=" * 60)
    
    # Compute aggregate tables (Tables 1-4)
    aggregate = compute_aggregate_stats(
        None,
        synth_results_all=synth_all_results,
        real_results=real_results
    )
    
    for algo, df in aggregate.items():
        csv_path = os.path.join(OUTPUT_DIR, f'aggregate_{algo}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  {algo} Aggregate Table:")
        print(df.to_string(index=False))
    
    # Also run Wilcoxon on combined synthetic
    wilcoxon_synth = run_wilcoxon_tests(synth_all_results)
    wilcoxon_synth_df = format_wilcoxon_table(wilcoxon_synth)
    wilcoxon_synth_df.to_csv(os.path.join(OUTPUT_DIR, 'wilcoxon_synthetic.csv'), index=False)
    
    # Combined Wilcoxon table
    wilcoxon_all = {}
    for algo in ALGOS:
        wilcoxon_all[algo] = {}
        if algo in wilcoxon_real:
            for cond, res in wilcoxon_real[algo].items():
                wilcoxon_all[algo][f'{cond}_real'] = res
        if algo in wilcoxon_synth:
            for cond, res in wilcoxon_synth[algo].items():
                wilcoxon_all[algo][f'{cond}_synth'] = res
    
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_all.json'), 'w') as f:
        json.dump(wilcoxon_all, f, indent=2, default=convert_for_json)
    
    # Summary Wilcoxon table matching paper format
    wilcoxon_summary_rows = []
    for algo in ALGOS:
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                cond = f"{method}_{level}"
                row = {'Algorithm': algo, 'Method': method, 'Level': level}
                for dtype, wres in [('Real', wilcoxon_real), ('Synthetic', wilcoxon_synth)]:
                    if algo in wres and cond in wres[algo]:
                        r = wres[algo][cond]
                        pval = r.get('p', np.nan)
                        sig = r.get('sig', 'na')
                        direction = r.get('direction', '')
                        row[f'p_{dtype}'] = f'{pval:.4f}' if not np.isnan(pval) else 'NA'
                        row[f'sig_{dtype}'] = sig
                        row[f'dir_{dtype}'] = direction
                    else:
                        row[f'p_{dtype}'] = 'NA'
                        row[f'sig_{dtype}'] = 'na'
                        row[f'dir_{dtype}'] = ''
                wilcoxon_summary_rows.append(row)
    wilcoxon_summary_df = pd.DataFrame(wilcoxon_summary_rows)
    wilcoxon_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'wilcoxon_summary.csv'), index=False)
    
    # ===== DONE =====
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETED in {elapsed/60:.1f} minutes")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
