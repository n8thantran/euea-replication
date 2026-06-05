"""
Complete experiment runner for DR+Clustering paper replication.
Handles both real-world and synthetic data experiments.
"""
import os, sys, json, time, warnings, pickle
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
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import make_circles, make_moons
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']
N_PER_CONFIG = 3  # Paper uses 50 per config; we use 5 for speed

# ============================================================
# VAE
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
        'k-1': min(n_features, max(2, k - 1)),
        '25%': min(n_features, max(2, int(np.round(0.25 * n_features)))),
        '50%': min(n_features, max(2, int(np.round(0.50 * n_features)))),
    }


def apply_dr(X, method, n_components):
    if n_components >= X.shape[1]:
        return X.copy()
    try:
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
            return MDS(n_components=n_components, random_state=10, n_init=2, max_iter=200,
                       normalized_stress='auto').fit_transform(X)
    except:
        return None


def get_all_conditions():
    conds = ['No Reduction']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            conds.append(f"{m}_{l}")
    return conds


def precompute_all_dr(datasets, timeout_sec=60, label=""):
    """Precompute DR transformations with per-method timeout."""
    import signal
    dr_cache = {}
    total = len(datasets)
    for i, ds_name in enumerate(sorted(datasets.keys())):
        X_raw, y_true, k = datasets[ds_name]
        X = StandardScaler().fit_transform(X_raw)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        dims = get_reduction_dims(X.shape[1], k)
        ds_cache = {'No Reduction': X.copy()}
        
        if (i+1) % max(1, total//10) == 0 or total <= 30:
            print(f"  DR {label} [{i+1}/{total}] {ds_name} shape={X.shape}")
        
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
        dr_cache[ds_name] = ds_cache
    return dr_cache


# ============================================================
# Clustering
# ============================================================
def cluster_kmeans(X, k):
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit_predict(X)

def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    return AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage).fit_predict(X)

def cluster_gmm(X, k, covariance_type='full'):
    return GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42).fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    return OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05,
                  min_cluster_size=min_cluster_size).fit_predict(X)


# ============================================================
# Hyperparameter Search
# ============================================================
def find_best_ahc_params(datasets, dr_cache, fast=False):
    combos = []
    for metric in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))
    
    if fast:
        combos = [('euclidean','ward'),('euclidean','complete'),('euclidean','average'),
                   ('euclidean','single'),('manhattan','complete'),('manhattan','average'),
                   ('cosine','complete'),('cosine','average')]
    
    search_conds = ['No Reduction', 'PCA_50%']
    keys = sorted(datasets.keys())
    if fast and len(keys) > 20:
        keys = list(np.random.RandomState(42).choice(keys, 20, replace=False))
    
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


def find_best_gmm_params(datasets, dr_cache, fast=False):
    search_conds = ['No Reduction', 'PCA_50%']
    keys = sorted(datasets.keys())
    if fast and len(keys) > 20:
        keys = list(np.random.RandomState(42).choice(keys, 20, replace=False))
    
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


def find_best_optics_params(datasets, dr_cache, fast=False):
    search_conds = ['No Reduction']
    keys = sorted(datasets.keys())
    if fast and len(keys) > 15:
        keys = list(np.random.RandomState(42).choice(keys, 15, replace=False))
    
    ms_range = [5, 7, 10] if fast else range(5, 11)
    mcs_vals = [0.05, 0.1, 0.2, 0.3, 0.5] if fast else [i*0.05 for i in range(1, 21)]
    
    best_score, best_combo = -999, (5, 0.05)
    for ms in ms_range:
        for mcs in mcs_vals:
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
# Run clustering
# ============================================================
def run_clustering(datasets, dr_cache, algo, **kwargs):
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


def run_experiment_set(datasets, dr_cache, label, fast_search=True):
    results = {}
    params = {}
    
    t0 = time.time()
    print(f"  Running k-means...")
    results['k-means'] = run_clustering(datasets, dr_cache, 'k-means')
    print(f"  k-means: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print(f"  Searching AHC params...")
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache, fast=fast_search)
    results['AHC'] = run_clustering(datasets, dr_cache, 'AHC', metric=ahc_m, linkage=ahc_l)
    params['AHC'] = {'metric': ahc_m, 'linkage': ahc_l}
    print(f"  AHC: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print(f"  Searching GMM params...")
    gmm_cov = find_best_gmm_params(datasets, dr_cache, fast=fast_search)
    results['GMM'] = run_clustering(datasets, dr_cache, 'GMM', covariance_type=gmm_cov)
    params['GMM'] = {'covariance_type': gmm_cov}
    print(f"  GMM: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print(f"  Searching OPTICS params...")
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache, fast=fast_search)
    results['OPTICS'] = run_clustering(datasets, dr_cache, 'OPTICS',
                                        min_samples=opt_ms, min_cluster_size=opt_mcs)
    params['OPTICS'] = {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)}
    print(f"  OPTICS: {time.time()-t0:.1f}s")
    
    results['_params'] = params
    return results


# ============================================================
# Synthetic Data Generation
# ============================================================
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


def generate_synthetic_datasets():
    synthetic = {}
    n_per = N_PER_CONFIG
    dims = [10, 50, 200]
    
    # CIRCLES k=2
    for d in dims:
        for i in range(n_per):
            rng = np.random.RandomState(42 + i + d)
            X, y = make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=42+i)
            X = embed_high_dim(X, d, rng)
            X = inject_noise(X, rng)
            synthetic[f'Circles_k2_d{d}_t{i}'] = (X, y, 2)
    
    # CIRCLES k=5
    for d in dims:
        for i in range(n_per):
            rng = np.random.RandomState(1000 + i + d)
            X_list, y_list = [], []
            for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                theta = rng.uniform(0, 2*np.pi, 400)
                rad = factor + rng.normal(0, 0.05, 400)
                X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                y_list.append(np.full(400, ci))
            X = np.vstack(X_list); y = np.concatenate(y_list)
            X = embed_high_dim(X, d, rng)
            X = inject_noise(X, rng)
            synthetic[f'Circles_k5_d{d}_t{i}'] = (X, y, 5)
    
    # MOONS k=2
    for d in dims:
        for i in range(n_per):
            rng = np.random.RandomState(2000 + i + d)
            X, y = make_moons(n_samples=2000, noise=0.1, random_state=2000+i)
            stretch = 1.0 + 0.5 * (i % 2)
            angle = np.radians(10 * (i - n_per//2))
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            X = X @ R.T; X[:, 0] *= stretch
            X = embed_high_dim(X, d, rng)
            X = inject_noise(X, rng)
            synthetic[f'Moons_k2_d{d}_t{i}'] = (X, y, 2)
    
    # MOONS k=5
    for d in dims:
        for i in range(n_per):
            rng = np.random.RandomState(3000 + i + d)
            n_pc = 400
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
            synthetic[f'Moons_k5_d{d}_t{i}'] = (X, y, 5)
    
    # RSG: k∈{2,10,50}, d∈{10,50,200}, Nc∈{5,50,100}
    for k in [2, 10]:
        for d in [10, 50, 200]:
            for Nc in [50, 100]:
                for i in range(min(n_per, 3)):  # more configs = fewer reps
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
                    synthetic[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    # REPLICLUST: k∈{2,5}, d∈{10,50,200}
    for k in [2, 5]:
        Nc = 1000 if k == 2 else 400
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(5000 + k*100 + d + i)
                # Anisotropic Gaussian clusters
                centers = rng.randn(k, d) * np.sqrt(d) * 2
                X_list, y_list = [], []
                for ci in range(k):
                    A = rng.randn(d, d) * 0.5
                    cov = A @ A.T / d + np.eye(d) * 0.1
                    samples = rng.multivariate_normal(centers[ci], cov, size=Nc)
                    X_list.append(samples); y_list.append(np.full(Nc, ci))
                X = np.vstack(X_list); y = np.concatenate(y_list)
                X = inject_noise(X, rng)
                synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    
    return synthetic


# ============================================================
# Analysis
# ============================================================
def compute_aggregate_stats(results_dict):
    stats = {}
    ds_names = sorted(results_dict.keys())
    for method in DR_METHODS:
        stats[method] = {}
        for level in REDUCTION_LEVELS:
            key = f"{method}_{level}"
            wins, losses, total = 0, 0, 0
            win_ch, loss_ch = [], []
            for ds in ds_names:
                b = results_dict[ds].get('No Reduction', 0.0)
                r = results_dict[ds].get(key, 0.0)
                total += 1
                if r > b + 0.005:
                    wins += 1
                    win_ch.append((r - b) / max(abs(b), 0.01) * 100)
                elif r < b - 0.005:
                    losses += 1
                    loss_ch.append((r - b) / max(abs(b), 0.01) * 100)
            stats[method][level] = {
                'win_pct': round(wins/total*100, 1) if total else 0,
                'loss_pct': round(losses/total*100, 1) if total else 0,
                'avg_win': round(np.mean(win_ch), 1) if win_ch else 0,
                'avg_loss': round(np.mean(loss_ch), 1) if loss_ch else 0,
            }
    return stats


def compute_wilcoxon(results_dict):
    ds_names = sorted(results_dict.keys())
    pvals = {}
    for method in DR_METHODS:
        pvals[method] = {}
        for level in REDUCTION_LEVELS:
            key = f"{method}_{level}"
            b = np.array([results_dict[ds].get('No Reduction', 0.0) for ds in ds_names])
            r = np.array([results_dict[ds].get(key, 0.0) for ds in ds_names])
            d = r - b
            nz = np.abs(d) > 1e-10
            if nz.sum() >= 2:
                try:
                    _, p = wilcoxon(d[nz])
                    pvals[method][level] = round(p, 4)
                except:
                    pvals[method][level] = 1.0
            else:
                pvals[method][level] = 1.0
    return pvals


def generate_boxplots(all_results, data_label, output_dir=OUTPUT_DIR):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = get_all_conditions()
    for algo in ALGOS:
        res = all_results.get(algo)
        if not res: continue
        fig, ax = plt.subplots(figsize=(16, 6))
        box_data, xlabels = [], []
        for cond in conditions:
            vals = [res[ds].get(cond, 0.0) for ds in sorted(res.keys())]
            box_data.append(vals)
            if cond == 'No Reduction':
                xlabels.append('No Red.')
            else:
                parts = cond.rsplit('_', 1)
                xlabels.append(f"{parts[0]}\n{parts[1]}")
        bp = ax.boxplot(box_data, patch_artist=True)
        colors = ['gray']+['#1f77b4']*3+['#ff7f0e']*3+['#2ca02c']*3+['#d62728']*3+['#9467bd']*3
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('ARI'); ax.set_title(f'{algo} — {data_label}')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.tight_layout()
        fname = f'boxplot_{algo}_{data_label}.pdf'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


def make_result_table(results_dict, algo, label, output_dir=OUTPUT_DIR):
    conditions = get_all_conditions()
    ds_names = sorted(results_dict.keys())
    df = pd.DataFrame(
        [[results_dict[ds].get(c, 0.0) for c in conditions] for ds in ds_names],
        index=ds_names, columns=conditions
    )
    fname = f'table_{algo}_{label}.csv'
    df.to_csv(os.path.join(output_dir, fname))
    return df


# ============================================================
# Real-World
# ============================================================
def run_real_world():
    print("=" * 60)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 60)
    from load_uci import load_all_uci
    dataset_list = load_all_uci()
    datasets = {name: (X, y, k) for name, X, y, k in dataset_list}
    for name in sorted(datasets.keys()):
        X, y, k = datasets[name]
        print(f"  {name}: {X.shape}, k={k}")
    
    cache_file = os.path.join(OUTPUT_DIR, 'dr_cache_real_v5.pkl')
    if os.path.exists(cache_file):
        print("Loading cached DR...")
        with open(cache_file, 'rb') as f:
            dr_cache = pickle.load(f)
    else:
        t0 = time.time()
        dr_cache = precompute_all_dr(datasets, label="real")
        print(f"DR took {time.time()-t0:.1f}s")
        with open(cache_file, 'wb') as f:
            pickle.dump(dr_cache, f)
    
    results = run_experiment_set(datasets, dr_cache, 'RealWorld', fast_search=True)
    with open(os.path.join(OUTPUT_DIR, 'real_results_final.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    for algo in ALGOS:
        make_result_table(results[algo], algo, 'real')
    
    agg = {algo: compute_aggregate_stats(results[algo]) for algo in ALGOS}
    with open(os.path.join(OUTPUT_DIR, 'aggregate_real_final.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    
    wilc = {algo: compute_wilcoxon(results[algo]) for algo in ALGOS}
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_real_final.json'), 'w') as f:
        json.dump(wilc, f, indent=2)
    
    generate_boxplots(results, 'RealWorld')
    return results


# ============================================================
# Synthetic
# ============================================================
def run_synthetic():
    print("\n" + "=" * 60)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 60)
    
    all_synth = generate_synthetic_datasets()
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    for tn in type_names:
        count = sum(1 for k in all_synth if k.startswith(tn))
        print(f"  {tn}: {count} datasets")
    
    all_type_results = {}
    for dtype in type_names:
        datasets = {k: v for k, v in all_synth.items() if k.startswith(dtype)}
        if not datasets: continue
        print(f"\n--- {dtype} ({len(datasets)} datasets) ---")
        
        result_file = os.path.join(OUTPUT_DIR, f'synth_{dtype}_final.json')
        if os.path.exists(result_file):
            print(f"  Loading cached results...")
            with open(result_file) as f:
                all_type_results[dtype] = json.load(f)
            continue
        
        cache_file = os.path.join(OUTPUT_DIR, f'dr_cache_{dtype}_v5.pkl')
        if os.path.exists(cache_file):
            print(f"  Loading cached DR...")
            with open(cache_file, 'rb') as f:
                dr_cache = pickle.load(f)
        else:
            t0 = time.time()
            dr_cache = precompute_all_dr(datasets, label=dtype)
            print(f"  DR took {time.time()-t0:.1f}s")
            with open(cache_file, 'wb') as f:
                pickle.dump(dr_cache, f)
        
        type_results = run_experiment_set(datasets, dr_cache, dtype, fast_search=True)
        all_type_results[dtype] = type_results
        with open(result_file, 'w') as f:
            json.dump(type_results, f, indent=2)
        del dr_cache
    
    # Generate summary tables and boxplots
    conditions = get_all_conditions()
    synth_summary = {}
    for dtype in type_names:
        if dtype not in all_type_results: continue
        synth_summary[dtype] = {}
        for algo in ALGOS:
            res = all_type_results[dtype].get(algo, {})
            if not res: continue
            avg = {c: round(np.mean([res[ds].get(c, 0.0) for ds in res]), 3) for c in conditions}
            synth_summary[dtype][algo] = avg
            print(f"  {dtype}/{algo}: No_Red={avg.get('No Reduction',0):.3f}")
        generate_boxplots(all_type_results[dtype], f'Synthetic_{dtype}')
    
    # Save CSV tables
    for dtype in type_names:
        if dtype not in synth_summary: continue
        rows = []
        for algo in ALGOS:
            if algo in synth_summary[dtype]:
                r = synth_summary[dtype][algo]
                r2 = {'Algorithm': algo}; r2.update(r)
                rows.append(r2)
        if rows:
            df = pd.DataFrame(rows).set_index('Algorithm')
            df.to_csv(os.path.join(OUTPUT_DIR, f'table_synth_{dtype}.csv'))
    
    # Aggregate stats per type
    synth_agg = {}
    for dtype in type_names:
        if dtype not in all_type_results: continue
        synth_agg[dtype] = {}
        for algo in ALGOS:
            res = all_type_results[dtype].get(algo, {})
            if res: synth_agg[dtype][algo] = compute_aggregate_stats(res)
    with open(os.path.join(OUTPUT_DIR, 'aggregate_synth_final.json'), 'w') as f:
        json.dump(synth_agg, f, indent=2)
    
    synth_wilc = {}
    for dtype in type_names:
        if dtype not in all_type_results: continue
        synth_wilc[dtype] = {}
        for algo in ALGOS:
            res = all_type_results[dtype].get(algo, {})
            if res: synth_wilc[dtype][algo] = compute_wilcoxon(res)
    with open(os.path.join(OUTPUT_DIR, 'wilcoxon_synth_final.json'), 'w') as f:
        json.dump(synth_wilc, f, indent=2)
    
    return all_type_results


# ============================================================
# Combined aggregate tables (Tables 1-4)
# ============================================================
def generate_combined_tables():
    print("\n" + "=" * 60)
    print("COMBINED AGGREGATE TABLES")
    print("=" * 60)
    
    real_file = os.path.join(OUTPUT_DIR, 'real_results_final.json')
    if not os.path.exists(real_file):
        print("No real-world results!")
        return
    with open(real_file) as f:
        real_results = json.load(f)
    
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    synth_results = {}
    for dtype in type_names:
        f_path = os.path.join(OUTPUT_DIR, f'synth_{dtype}_final.json')
        if os.path.exists(f_path):
            with open(f_path) as f:
                synth_results[dtype] = json.load(f)
    
    for algo in ALGOS:
        print(f"\n--- {algo} ---")
        rows = []
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                all_b, all_r = [], []
                if algo in real_results:
                    for ds in real_results[algo]:
                        all_b.append(real_results[algo][ds].get('No Reduction', 0.0))
                        all_r.append(real_results[algo][ds].get(key, 0.0))
                for dtype in type_names:
                    if dtype in synth_results and algo in synth_results[dtype]:
                        for ds in synth_results[dtype][algo]:
                            all_b.append(synth_results[dtype][algo][ds].get('No Reduction', 0.0))
                            all_r.append(synth_results[dtype][algo][ds].get(key, 0.0))
                total = len(all_b)
                wins = sum(1 for b,r in zip(all_b,all_r) if r > b + 0.005)
                losses = sum(1 for b,r in zip(all_b,all_r) if r < b - 0.005)
                wch = [(r-b)/max(abs(b),0.01)*100 for b,r in zip(all_b,all_r) if r > b + 0.005]
                lch = [(r-b)/max(abs(b),0.01)*100 for b,r in zip(all_b,all_r) if r < b - 0.005]
                rows.append({
                    'Method': method, 'Level': level,
                    'Win%': round(wins/total*100,1) if total else 0,
                    'Loss%': round(losses/total*100,1) if total else 0,
                    'Avg Win%': round(np.mean(wch),1) if wch else 0,
                    'Avg Loss%': round(np.mean(lch),1) if lch else 0,
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f'table_combined_{algo}.csv'), index=False)
        print(df.to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'tables', 'all'], default='all')
    args = parser.parse_args()
    t_start = time.time()
    if args.mode in ['real', 'all']:
        run_real_world()
    if args.mode in ['synthetic', 'all']:
        run_synthetic()
    if args.mode in ['tables', 'all']:
        generate_combined_tables()
    print(f"\nTOTAL TIME: {time.time()-t_start:.0f}s")
