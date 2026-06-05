#!/usr/bin/env python3
"""
Complete experiment pipeline for:
"Assessing the impact of dimensionality reduction on clustering performance"

Reproduces: Tables A.1-A.4 (synthetic ARI), Tables A.5-A.8 (real-world ARI),
Tables 1-4 (aggregate stats), Table 5 (Wilcoxon), Figures 1-8 (boxplots).
"""

import numpy as np
import warnings
import json
import os
import time
import pickle
import sys
from itertools import product
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']

# ============================================================
# VAE Model
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


def train_vae_and_encode(X, latent_dim, epochs=100, batch_size=64):
    """Train VAE and return latent representations (mu)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n, d = X.shape
    
    # Min-max scale to [0,1] for sigmoid output
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_scaled = (X - X_min) / X_range
    
    # 70/30 split
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    train_data = torch.FloatTensor(X_scaled[idx[:n_train]]).to(device)
    all_data = torch.FloatTensor(X_scaled).to(device)
    
    loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    model = VAEModel(d, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            mse = nn.functional.mse_loss(recon, batch, reduction='mean')
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(all_data)
    return mu.cpu().numpy()


# ============================================================
# Dimensionality Reduction
# ============================================================
def compute_target_dims(n_features, n_clusters):
    """Compute target dimensions for k-1, 25%, 50% levels."""
    dims = {}
    dims['k-1'] = max(2, n_clusters - 1)
    dims['25%'] = max(2, round(n_features * 0.25))
    dims['50%'] = max(2, round(n_features * 0.50))
    return dims


def apply_dr(X_scaled, method, n_components, vae_epochs=100):
    """Apply a dimensionality reduction method. Returns reduced data or None on failure."""
    n, d = X_scaled.shape
    if n_components >= d:
        return X_scaled.copy()
    
    try:
        if method == 'PCA':
            return PCA(n_components=n_components).fit_transform(X_scaled)
        elif method == 'Kernel PCA':
            return KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X_scaled)
        elif method == 'VAE':
            return train_vae_and_encode(X_scaled, n_components, epochs=vae_epochs)
        elif method == 'Isomap':
            n_neighbors = min(5, n - 1)
            return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X_scaled)
        elif method == 'MDS':
            return MDS(n_components=n_components, random_state=10, n_init=4, max_iter=300, normalized_stress='auto').fit_transform(X_scaled)
    except Exception as e:
        print(f"    DR failed ({method}, {n_components}d): {e}")
        return None


# ============================================================
# Clustering
# ============================================================
def run_kmeans(X, k):
    """k-means with k-means++ init, n_init=100."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)


def run_ahc(X, k, affinity='euclidean', linkage='ward'):
    """Agglomerative Hierarchical Clustering."""
    if linkage == 'ward' and affinity != 'euclidean':
        return None
    try:
        ahc = AgglomerativeClustering(n_clusters=k, metric=affinity, linkage=linkage)
        return ahc.fit_predict(X)
    except:
        return None


def run_gmm(X, k, covariance_type='full'):
    """Gaussian Mixture Model."""
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, max_iter=200)
        return gmm.fit_predict(X)
    except:
        return None


def run_optics(X, min_samples=5, min_cluster_size=0.05):
    """OPTICS with xi method."""
    try:
        op = OPTICS(min_samples=min_samples, xi=0.05, cluster_method='xi',
                     min_cluster_size=min_cluster_size)
        return op.fit_predict(X)
    except:
        return None


# ============================================================
# Hyperparameter Search
# ============================================================
def find_best_ahc_params(datasets_list, verbose=False):
    """Find best AHC affinity/linkage combo across a set of datasets.
    datasets_list: list of (X_scaled, labels, k) tuples.
    Returns: (best_affinity, best_linkage)
    """
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    best_score = -999
    best_params = ('euclidean', 'ward')
    
    for aff, link in product(affinities, linkages):
        if link == 'ward' and aff != 'euclidean':
            continue
        scores = []
        for X, labels, k in datasets_list:
            pred = run_ahc(X, k, aff, link)
            if pred is not None:
                scores.append(adjusted_rand_score(labels, pred))
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_params = (aff, link)
    
    if verbose:
        print(f"  Best AHC: {best_params}, avg ARI={best_score:.3f}")
    return best_params


def find_best_gmm_params(datasets_list, verbose=False):
    """Find best GMM covariance type across a set of datasets."""
    cov_types = ['spherical', 'tied', 'diag', 'full']
    
    best_score = -999
    best_cov = 'full'
    
    for cov in cov_types:
        scores = []
        for X, labels, k in datasets_list:
            pred = run_gmm(X, k, cov)
            if pred is not None:
                scores.append(adjusted_rand_score(labels, pred))
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_cov = cov
    
    if verbose:
        print(f"  Best GMM: {best_cov}, avg ARI={best_score:.3f}")
    return best_cov


def find_best_optics_params(datasets_list, verbose=False):
    """Find best OPTICS min_samples and min_cluster_size across datasets."""
    best_score = -999
    best_params = (5, 0.05)
    
    min_samples_range = range(5, 11)
    min_cluster_sizes = np.arange(0.05, 1.01, 0.05)
    
    for ms in min_samples_range:
        for mcs in min_cluster_sizes:
            scores = []
            for X, labels, k in datasets_list:
                pred = run_optics(X, ms, mcs)
                if pred is not None:
                    scores.append(adjusted_rand_score(labels, pred))
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_params = (ms, round(mcs, 2))
    
    if verbose:
        print(f"  Best OPTICS: min_samples={best_params[0]}, min_cluster_size={best_params[1]}, avg ARI={best_score:.3f}")
    return best_params


# ============================================================
# Real-World Data Loading
# ============================================================
def load_real_world_datasets():
    """Load all 20 UCI datasets."""
    from load_uci import load_all_uci
    raw = load_all_uci()
    datasets = {}
    for name, (X, y) in raw.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        k = len(np.unique(y))
        datasets[name] = (X_scaled, y, k)
    return datasets


# ============================================================
# Synthetic Data Generation
# ============================================================
def add_noise(X):
    """Add structured noise: 25% σ=1, 25% σ=0.5, 25% σ=0.25, 25% clean."""
    n, d = X.shape
    # z-score normalize first
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    indices = np.random.permutation(d)
    q1 = d // 4
    q2 = d // 2
    q3 = 3 * d // 4
    
    X_noisy = X_norm.copy()
    X_noisy[:, indices[:q1]] += np.random.normal(0, 1.0, (n, len(indices[:q1])))
    X_noisy[:, indices[q1:q2]] += np.random.normal(0, 0.5, (n, len(indices[q1:q2])))
    X_noisy[:, indices[q2:q3]] += np.random.normal(0, 0.25, (n, len(indices[q2:q3])))
    # Last quarter: no noise
    
    return X_noisy


def generate_circles_dataset(n_samples, n_features, noise=0.05, factor=0.5):
    """Generate circles dataset embedded in n_features dimensions."""
    X_2d, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=None)
    if n_features > 2:
        extra = np.random.randn(n_samples, n_features - 2) * 0.1
        X = np.hstack([X_2d, extra])
    else:
        X = X_2d
    return add_noise(X), y


def generate_moons_dataset(n_samples, n_features, noise=0.1):
    """Generate moons dataset embedded in n_features dimensions."""
    X_2d, y = make_moons(n_samples=n_samples, noise=noise, random_state=None)
    if n_features > 2:
        extra = np.random.randn(n_samples, n_features - 2) * 0.1
        X = np.hstack([X_2d, extra])
    else:
        X = X_2d
    return add_noise(X), y


def generate_rsg_dataset(k, d, Nc, alpha=None):
    """Generate Rodriguez Structured Gaussian dataset."""
    if alpha is None:
        alpha = 0.5
    
    centers = np.random.randn(k, d) * alpha * np.sqrt(d)
    X_list, y_list = [], []
    
    for i in range(k):
        # Generate cluster-specific covariance
        A = np.random.randn(d, d) * 0.3
        cov = A @ A.T + np.eye(d) * 0.1
        X_cluster = np.random.multivariate_normal(centers[i], cov, size=Nc)
        X_list.append(X_cluster)
        y_list.append(np.full(Nc, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return add_noise(X), y


def generate_repliclust_dataset(k, d, n_total):
    """Generate Repliclust-style anisotropic clusters."""
    Nc = n_total // k
    centers = np.random.randn(k, d) * 3.0
    
    X_list, y_list = [], []
    for i in range(k):
        # Anisotropic: random rotation + scaling
        scales = np.random.exponential(1.0, d)
        scales[0] *= 3  # elongate along one axis
        A = np.random.randn(d, d)
        Q, _ = np.linalg.qr(A)
        cov = Q @ np.diag(scales) @ Q.T
        
        X_cluster = np.random.multivariate_normal(centers[i], cov, size=Nc)
        X_list.append(X_cluster)
        y_list.append(np.full(Nc, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return add_noise(X), y


def generate_synthetic_datasets(dtype, n_per_config=5):
    """Generate synthetic datasets for a given type.
    
    Returns list of (X_scaled, labels, k, config_str) tuples.
    """
    datasets = []
    
    if dtype == 'Circles':
        configs = [(2, d) for d in [10, 50, 200]]
        for k, d in configs:
            for rep in range(n_per_config):
                n_samples = 1000
                X, y = generate_circles_dataset(n_samples, d)
                X_s = StandardScaler().fit_transform(X)
                datasets.append((X_s, y, k, f"k{k}_d{d}_r{rep}"))
    
    elif dtype == 'Moons':
        configs = [(2, d) for d in [10, 50, 200]]
        for k, d in configs:
            for rep in range(n_per_config):
                n_samples = 1000
                X, y = generate_moons_dataset(n_samples, d)
                X_s = StandardScaler().fit_transform(X)
                datasets.append((X_s, y, k, f"k{k}_d{d}_r{rep}"))
    
    elif dtype == 'RSG':
        configs = list(product([2, 10, 50], [10, 50, 200], [5, 50, 100]))
        for k, d, Nc in configs:
            for rep in range(max(1, n_per_config // 3)):  # fewer reps for RSG (many configs)
                X, y = generate_rsg_dataset(k, d, Nc)
                X_s = StandardScaler().fit_transform(X)
                datasets.append((X_s, y, k, f"k{k}_d{d}_Nc{Nc}_r{rep}"))
    
    elif dtype == 'Repliclust':
        configs = [(2, d) for d in [10, 50, 200]] + [(5, d) for d in [10, 50, 200]]
        for k, d in configs:
            n_total = 2000
            for rep in range(n_per_config):
                X, y = generate_repliclust_dataset(k, d, n_total)
                X_s = StandardScaler().fit_transform(X)
                datasets.append((X_s, y, k, f"k{k}_d{d}_r{rep}"))
    
    return datasets


# ============================================================
# Main Experiment Runner
# ============================================================
def run_experiment_set(datasets, dataset_type, ahc_params, gmm_cov, optics_params,
                       vae_epochs=100, collect_per_dataset=False):
    """Run all DR methods × levels × clustering algorithms on a set of datasets.
    
    Args:
        datasets: dict or list of (X, labels, k, [name]) 
        dataset_type: string identifier
        ahc_params: (affinity, linkage)
        gmm_cov: covariance type string
        optics_params: (min_samples, min_cluster_size)
        vae_epochs: epochs for VAE training
        collect_per_dataset: if True, return per-dataset results (for real-world tables)
    
    Returns:
        results dict with ARI scores
    """
    ahc_aff, ahc_link = ahc_params
    optics_ms, optics_mcs = optics_params
    
    # Normalize input format
    if isinstance(datasets, dict):
        ds_list = [(name, X, y, k) for name, (X, y, k) in datasets.items()]
    else:
        ds_list = [(item[3] if len(item) > 3 else f"ds_{i}", item[0], item[1], item[2]) 
                   for i, item in enumerate(datasets)]
    
    n_ds = len(ds_list)
    print(f"\nRunning {dataset_type}: {n_ds} datasets")
    
    # Storage for all ARI scores
    # all_ari[algo][dr_method][level] = list of ARI scores
    all_ari = {a: {dr: {lv: [] for lv in LEVELS} for dr in DR_METHODS} for a in ALGOS}
    # baseline[algo] = list of ARI scores
    baseline_ari = {a: [] for a in ALGOS}
    
    # Per-dataset results (for real-world tables)
    per_dataset = {}
    
    for di, (name, X, labels, k) in enumerate(ds_list):
        if di % max(1, n_ds // 10) == 0:
            print(f"  [{di+1}/{n_ds}] {name} (n={X.shape[0]}, d={X.shape[1]}, k={k})")
        
        target_dims = compute_target_dims(X.shape[1], k)
        
        # Baseline (no reduction)
        base = {}
        base['k-means'] = adjusted_rand_score(labels, run_kmeans(X, k))
        
        pred_ahc = run_ahc(X, k, ahc_aff, ahc_link)
        base['AHC'] = adjusted_rand_score(labels, pred_ahc) if pred_ahc is not None else 0.0
        
        pred_gmm = run_gmm(X, k, gmm_cov)
        base['GMM'] = adjusted_rand_score(labels, pred_gmm) if pred_gmm is not None else 0.0
        
        pred_optics = run_optics(X, optics_ms, optics_mcs)
        base['OPTICS'] = adjusted_rand_score(labels, pred_optics) if pred_optics is not None else 0.0
        
        for a in ALGOS:
            baseline_ari[a].append(base[a])
        
        if collect_per_dataset:
            per_dataset[name] = {'baseline': base, 'dr': {}}
        
        # DR methods
        for dr in DR_METHODS:
            for lv in LEVELS:
                nd = target_dims[lv]
                X_red = apply_dr(X, dr, nd, vae_epochs=vae_epochs)
                
                if X_red is None:
                    for a in ALGOS:
                        all_ari[a][dr][lv].append(base[a])  # fallback to baseline
                    if collect_per_dataset:
                        per_dataset[name].setdefault('dr', {})
                        per_dataset[name]['dr'][(dr, lv)] = {a: base[a] for a in ALGOS}
                    continue
                
                dr_scores = {}
                dr_scores['k-means'] = adjusted_rand_score(labels, run_kmeans(X_red, k))
                
                pred_ahc = run_ahc(X_red, k, ahc_aff, ahc_link)
                dr_scores['AHC'] = adjusted_rand_score(labels, pred_ahc) if pred_ahc is not None else 0.0
                
                pred_gmm = run_gmm(X_red, k, gmm_cov)
                dr_scores['GMM'] = adjusted_rand_score(labels, pred_gmm) if pred_gmm is not None else 0.0
                
                pred_optics = run_optics(X_red, optics_ms, optics_mcs)
                dr_scores['OPTICS'] = adjusted_rand_score(labels, pred_optics) if pred_optics is not None else 0.0
                
                for a in ALGOS:
                    all_ari[a][dr][lv].append(dr_scores[a])
                
                if collect_per_dataset:
                    per_dataset[name]['dr'][(dr, lv)] = dr_scores
    
    return {
        'all_ari': all_ari,
        'baseline_ari': baseline_ari,
        'per_dataset': per_dataset,
        'n_datasets': n_ds,
        'dataset_type': dataset_type
    }


# ============================================================
# Analysis Functions
# ============================================================
def compute_aggregate_stats(results):
    """Compute % wins and avg win/loss for each algo/DR/level."""
    all_ari = results['all_ari']
    baseline_ari = results['baseline_ari']
    
    stats = {}
    for algo in ALGOS:
        stats[algo] = {}
        for dr in DR_METHODS:
            stats[algo][dr] = {}
            for lv in LEVELS:
                dr_scores = np.array(all_ari[algo][dr][lv])
                base_scores = np.array(baseline_ari[algo])
                
                n = len(dr_scores)
                if n == 0:
                    stats[algo][dr][lv] = {'win_pct': 0, 'avg_diff': 0}
                    continue
                
                # Win = DR score > baseline
                wins = np.sum(dr_scores > base_scores + 1e-6)
                # For percentage calculation, exclude ties
                non_ties = np.sum(np.abs(dr_scores - base_scores) > 1e-6)
                win_pct = (wins / non_ties * 100) if non_ties > 0 else 50.0
                
                # Average win/loss percentage
                diffs = dr_scores - base_scores
                # Relative to baseline (avoid div by 0)
                rel_diffs = []
                for d_val, b_val in zip(diffs, base_scores):
                    if abs(b_val) > 1e-6:
                        rel_diffs.append(d_val / abs(b_val) * 100)
                    else:
                        rel_diffs.append(d_val * 100)  # absolute diff as percentage
                avg_diff = np.mean(rel_diffs) if rel_diffs else 0.0
                
                stats[algo][dr][lv] = {
                    'win_pct': round(win_pct, 2),
                    'avg_diff': round(avg_diff, 2),
                    'n_datasets': n,
                    'n_wins': int(wins),
                    'n_ties': int(n - non_ties)
                }
    
    return stats


def compute_wilcoxon_tests(results):
    """Wilcoxon signed-rank test: H1: ARI_method > ARI_baseline."""
    all_ari = results['all_ari']
    baseline_ari = results['baseline_ari']
    
    wilc = {}
    for algo in ALGOS:
        wilc[algo] = {}
        for dr in DR_METHODS:
            wilc[algo][dr] = {}
            for lv in LEVELS:
                dr_scores = np.array(all_ari[algo][dr][lv])
                base_scores = np.array(baseline_ari[algo])
                
                diffs = dr_scores - base_scores
                non_zero = diffs[np.abs(diffs) > 1e-10]
                
                if len(non_zero) < 2:
                    wilc[algo][dr][lv] = 1.0
                else:
                    try:
                        _, p = wilcoxon(dr_scores, base_scores, alternative='greater')
                        wilc[algo][dr][lv] = round(p, 3)
                    except:
                        wilc[algo][dr][lv] = 1.0
    
    return wilc


# ============================================================
# Output Generation
# ============================================================
def generate_real_world_tables(results, output_dir):
    """Generate CSV tables matching paper Tables A.5-A.8."""
    per_dataset = results['per_dataset']
    
    for algo in ALGOS:
        rows = []
        for name in sorted(per_dataset.keys()):
            ds = per_dataset[name]
            row = {'Dataset': name, 'No Reduction': round(ds['baseline'][algo], 2)}
            for dr in DR_METHODS:
                for lv in LEVELS:
                    key = (dr, lv)
                    val = ds['dr'].get(key, {}).get(algo, 0.0)
                    row[f'{dr}_{lv}'] = round(val, 2)
            rows.append(row)
        
        # Write CSV
        cols = ['Dataset', 'No Reduction']
        for dr in DR_METHODS:
            for lv in LEVELS:
                cols.append(f'{dr}_{lv}')
        
        fname = f"{output_dir}/table_{algo}_real.csv"
        with open(fname, 'w') as f:
            f.write(','.join(cols) + '\n')
            for row in rows:
                vals = [str(row.get(c, '')) for c in cols]
                f.write(','.join(vals) + '\n')
        print(f"  Saved {fname}")


def generate_synthetic_table(results, dtype, output_dir):
    """Generate synthetic ARI table (average across all datasets of this type)."""
    all_ari = results['all_ari']
    baseline_ari = results['baseline_ari']
    
    fname = f"{output_dir}/table_synthetic_{dtype}.csv"
    with open(fname, 'w') as f:
        # Header
        cols = ['Algorithm', 'No Reduction']
        for dr in DR_METHODS:
            for lv in LEVELS:
                cols.append(f'{dr}_{lv}')
        f.write(','.join(cols) + '\n')
        
        for algo in ALGOS:
            row = [algo, f"{np.mean(baseline_ari[algo]):.3f}"]
            for dr in DR_METHODS:
                for lv in LEVELS:
                    row.append(f"{np.mean(all_ari[algo][dr][lv]):.3f}")
            f.write(','.join(row) + '\n')
    
    print(f"  Saved {fname}")


def generate_aggregate_table(stats_synth, stats_real, output_dir):
    """Generate aggregate tables matching paper Tables 1-4."""
    for algo in ALGOS:
        fname = f"{output_dir}/table_aggregate_{algo}.csv"
        with open(fname, 'w') as f:
            f.write('Method,Reduction,Win% Synthetic,Win% Real,AvgDiff Synthetic,AvgDiff Real\n')
            for dr in DR_METHODS:
                for lv in LEVELS:
                    ss = stats_synth[algo][dr][lv] if stats_synth else {'win_pct': 'N/A', 'avg_diff': 'N/A'}
                    sr = stats_real[algo][dr][lv] if stats_real else {'win_pct': 'N/A', 'avg_diff': 'N/A'}
                    f.write(f"{dr},{lv},{ss['win_pct']},{sr['win_pct']},{ss['avg_diff']},{sr['avg_diff']}\n")
        print(f"  Saved {fname}")


def generate_wilcoxon_table(wilc, output_dir):
    """Generate Wilcoxon test table matching paper Table 5."""
    fname = f"{output_dir}/table_wilcoxon.csv"
    with open(fname, 'w') as f:
        cols = ['Algorithm']
        for dr in DR_METHODS:
            for lv in LEVELS:
                cols.append(f'{dr}_{lv}')
        f.write(','.join(cols) + '\n')
        
        for algo in ALGOS:
            row = [algo]
            for dr in DR_METHODS:
                for lv in LEVELS:
                    p = wilc[algo][dr][lv]
                    row.append(f"{p:.3f}")
            f.write(','.join(row) + '\n')
    print(f"  Saved {fname}")


def generate_boxplots(results_list, data_label, output_dir):
    """Generate boxplot figures for each clustering algorithm.
    
    results_list: list of result dicts (one per synthetic type, or one for real-world)
    data_label: 'Synthetic' or 'RealWorld'
    """
    for algo in ALGOS:
        fig, axes = plt.subplots(1, 6, figsize=(24, 5), sharey=True)
        fig.suptitle(f'{algo} - {data_label}', fontsize=14)
        
        # Collect all ARI values: baseline + 5 DR methods × 3 levels
        categories = ['No\nReduction']
        all_data = []
        
        # Baseline
        baseline_vals = []
        for res in results_list:
            baseline_vals.extend(res['baseline_ari'][algo])
        all_data.append(baseline_vals)
        
        # Plot baseline in first subplot
        axes[0].boxplot([baseline_vals], positions=[1], widths=0.6)
        axes[0].set_title('No Reduction')
        axes[0].set_ylabel('ARI')
        
        for mi, dr in enumerate(DR_METHODS):
            box_data = []
            labels = []
            for lv in LEVELS:
                vals = []
                for res in results_list:
                    vals.extend(res['all_ari'][algo][dr][lv])
                box_data.append(vals)
                labels.append(lv)
            
            ax = axes[mi + 1]
            ax.boxplot(box_data, positions=[1, 2, 3], widths=0.6)
            ax.set_xticklabels(labels)
            ax.set_title(dr)
        
        plt.tight_layout()
        fname = f"{output_dir}/boxplot_{algo}_{data_label}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# ============================================================
# Main Pipeline
# ============================================================
def run_real_world_pipeline():
    """Run complete real-world experiment pipeline."""
    print("=" * 60)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 60)
    
    t0 = time.time()
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_real_world_datasets()
    print(f"  Loaded {len(datasets)} datasets")
    
    # Find best hyperparameters across all real-world datasets
    print("\nFinding best hyperparameters...")
    ds_tuples = [(X, y, k) for X, y, k in datasets.values()]
    
    ahc_params = find_best_ahc_params(ds_tuples, verbose=True)
    gmm_cov = find_best_gmm_params(ds_tuples, verbose=True)
    optics_params = find_best_optics_params(ds_tuples, verbose=True)
    
    print(f"\nHyperparameters: AHC={ahc_params}, GMM={gmm_cov}, OPTICS={optics_params}")
    
    # Run experiments
    results = run_experiment_set(
        datasets, 'RealWorld', ahc_params, gmm_cov, optics_params,
        vae_epochs=100, collect_per_dataset=True
    )
    
    # Generate outputs
    print("\nGenerating tables...")
    generate_real_world_tables(results, RESULTS_DIR)
    
    print("\nComputing aggregate statistics...")
    stats = compute_aggregate_stats(results)
    
    print("\nComputing Wilcoxon tests...")
    wilc = compute_wilcoxon_tests(results)
    generate_wilcoxon_table(wilc, RESULTS_DIR)
    
    print("\nGenerating boxplots...")
    generate_boxplots([results], 'RealWorld', RESULTS_DIR)
    
    # Save raw results
    with open(f"{RESULTS_DIR}/real_world_results_final.json", 'w') as f:
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, tuple): return list(obj)
            return obj
        
        save_data = {
            'hyperparams': {
                'ahc': list(ahc_params),
                'gmm': gmm_cov,
                'optics': list(optics_params)
            },
            'stats': stats,
            'wilcoxon': wilc,
            'per_dataset': {}
        }
        for name, ds in results['per_dataset'].items():
            save_data['per_dataset'][name] = {
                'baseline': {k: float(v) for k, v in ds['baseline'].items()},
                'dr': {f"{k[0]}|{k[1]}": {a: float(v) for a, v in vals.items()} 
                       for k, vals in ds['dr'].items()}
            }
        json.dump(save_data, f, indent=2, default=convert)
    
    elapsed = time.time() - t0
    print(f"\nReal-world experiments completed in {elapsed:.0f}s")
    
    return results, stats, wilc


def run_synthetic_pipeline(n_per_config=5, vae_epochs=50):
    """Run complete synthetic experiment pipeline."""
    print("=" * 60)
    print("SYNTHETIC EXPERIMENTS")
    print("=" * 60)
    
    t0 = time.time()
    all_synth_results = {}
    
    for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
        print(f"\n{'='*40}")
        print(f"Generating {dtype} datasets...")
        datasets = generate_synthetic_datasets(dtype, n_per_config=n_per_config)
        print(f"  Generated {len(datasets)} datasets")
        
        # Find best hyperparameters for this type
        print(f"  Finding best hyperparameters for {dtype}...")
        ds_tuples = [(X, y, k) for X, y, k, _ in datasets]
        
        ahc_params = find_best_ahc_params(ds_tuples, verbose=True)
        gmm_cov = find_best_gmm_params(ds_tuples, verbose=True)
        optics_params = find_best_optics_params(ds_tuples, verbose=True)
        
        # Run experiments
        results = run_experiment_set(
            [(X, y, k, name) for X, y, k, name in datasets],
            dtype, ahc_params, gmm_cov, optics_params,
            vae_epochs=vae_epochs, collect_per_dataset=False
        )
        
        all_synth_results[dtype] = results
        
        # Generate per-type table
        generate_synthetic_table(results, dtype, RESULTS_DIR)
    
    # Generate boxplots (all types combined)
    print("\nGenerating synthetic boxplots...")
    generate_boxplots(list(all_synth_results.values()), 'Synthetic', RESULTS_DIR)
    
    # Also per-type boxplots
    for dtype, res in all_synth_results.items():
        generate_boxplots([res], f'Synthetic_{dtype}', RESULTS_DIR)
    
    # Compute combined synthetic aggregate stats
    print("\nComputing combined synthetic aggregate statistics...")
    combined_ari = {a: {dr: {lv: [] for lv in LEVELS} for dr in DR_METHODS} for a in ALGOS}
    combined_baseline = {a: [] for a in ALGOS}
    
    for dtype, res in all_synth_results.items():
        for a in ALGOS:
            combined_baseline[a].extend(res['baseline_ari'][a])
            for dr in DR_METHODS:
                for lv in LEVELS:
                    combined_ari[a][dr][lv].extend(res['all_ari'][a][dr][lv])
    
    combined_results = {
        'all_ari': combined_ari,
        'baseline_ari': combined_baseline,
        'n_datasets': sum(r['n_datasets'] for r in all_synth_results.values()),
        'dataset_type': 'Synthetic_Combined'
    }
    
    synth_stats = compute_aggregate_stats(combined_results)
    
    # Save results
    save_data = {'stats': synth_stats, 'per_type': {}}
    for dtype, res in all_synth_results.items():
        type_stats = compute_aggregate_stats(res)
        avg_ari = {}
        for a in ALGOS:
            avg_ari[a] = {
                'baseline': float(np.mean(res['baseline_ari'][a])),
                'dr': {f"{dr}|{lv}": float(np.mean(res['all_ari'][a][dr][lv]))
                       for dr in DR_METHODS for lv in LEVELS}
            }
        save_data['per_type'][dtype] = {'stats': type_stats, 'avg_ari': avg_ari}
    
    with open(f"{RESULTS_DIR}/synthetic_results_final.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    
    elapsed = time.time() - t0
    print(f"\nSynthetic experiments completed in {elapsed:.0f}s")
    
    return all_synth_results, synth_stats


def main():
    """Run complete pipeline."""
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    real_results, real_stats, wilc = None, None, None
    synth_results, synth_stats = None, None
    
    if mode in ['all', 'real']:
        real_results, real_stats, wilc = run_real_world_pipeline()
    
    if mode in ['all', 'synthetic']:
        n_per_config = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        vae_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        synth_results, synth_stats = run_synthetic_pipeline(n_per_config=n_per_config, vae_epochs=vae_epochs)
    
    # Generate combined aggregate tables
    if real_stats and synth_stats:
        print("\nGenerating combined aggregate tables...")
        generate_aggregate_table(synth_stats, real_stats, RESULTS_DIR)
    elif real_stats:
        generate_aggregate_table(None, real_stats, RESULTS_DIR)
    elif synth_stats:
        generate_aggregate_table(synth_stats, None, RESULTS_DIR)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
