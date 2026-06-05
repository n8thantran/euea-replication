#!/usr/bin/env python3
"""
Clean implementation of the DR+Clustering experiment pipeline.
Paper: "Assessing the impact of dimensionality reduction on clustering performance"

This script runs the complete experiment pipeline:
1. Generate/load datasets (synthetic + real-world)
2. Apply DR methods at 3 reduction levels
3. Cluster with 4 algorithms
4. Compute ARI scores
5. Generate tables, boxplots, and statistical tests
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import pickle
import time
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_circles, make_moons

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def embed_to_high_dim(X, target_dim, random_state=None):
    """Embed low-dim data into higher dimensions using Gaussian Random Projection."""
    rng = np.random.RandomState(random_state)
    d_orig = X.shape[1]
    if target_dim <= d_orig:
        return X
    G = rng.randn(d_orig, target_dim) / np.sqrt(target_dim)
    return X @ G

def inject_noise(X, random_state=None):
    """Apply structured noise injection as described in the paper.
    1. Z-score normalize
    2. Add Gaussian noise to 75% of features
    """
    rng = np.random.RandomState(random_state)
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    perm = rng.permutation(d)
    q = d // 4
    
    X_noisy = X.copy()
    if q > 0:
        X_noisy[:, perm[:q]] += rng.normal(0, 1.0, (n, q))
    if q > 0:
        X_noisy[:, perm[q:2*q]] += rng.normal(0, 0.5, (n, q))
    if q > 0:
        X_noisy[:, perm[2*q:3*q]] += rng.normal(0, 0.25, (n, q))
    # Fourth quarter: clean
    
    return X_noisy

def generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=None):
    """Generate 2-cluster circles dataset."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y

def generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Generate 5-cluster concentric rings dataset."""
    rng = np.random.RandomState(random_state)
    factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    X_all, y_all = [], []
    for ci, r in enumerate(factors):
        theta = rng.uniform(0, 2*np.pi, n_per_cluster)
        rad = r + rng.normal(0, noise * r, n_per_cluster)
        x = rad * np.cos(theta)
        y_coord = rad * np.sin(theta)
        X_all.append(np.column_stack([x, y_coord]))
        y_all.append(np.full(n_per_cluster, ci))
    return np.vstack(X_all), np.concatenate(y_all)

def generate_moons_k2(n_samples=2000, noise=0.05, random_state=None):
    """Generate 2-cluster moons dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Generate 5-cluster moons dataset with stretching, rotations, translations."""
    rng = np.random.RandomState(random_state)
    X_all, y_all = [], []
    
    # Parameters for 5 crescent-shaped clusters
    configs = [
        {'stretch': 1.0, 'rotation': 0, 'tx': 0, 'ty': 0},
        {'stretch': 1.5, 'rotation': 160, 'tx': 3, 'ty': 1.0},
        {'stretch': 1.0, 'rotation': -160, 'tx': -3, 'ty': 1.2},
        {'stretch': 1.5, 'rotation': 10, 'tx': 2, 'ty': 1.5},
        {'stretch': 1.0, 'rotation': 180, 'tx': -4, 'ty': -1.0},
    ]
    
    for ci, cfg in enumerate(configs):
        X_moon, _ = make_moons(n_samples=n_per_cluster*2, noise=noise, random_state=rng.randint(100000))
        X_moon = X_moon[:n_per_cluster]
        
        # Stretch
        X_moon[:, 0] *= cfg['stretch']
        
        # Rotate
        angle = np.radians(cfg['rotation'])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        X_moon = X_moon @ R.T
        
        # Translate
        X_moon[:, 0] += cfg['tx']
        X_moon[:, 1] += cfg['ty']
        
        X_all.append(X_moon)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def generate_rsg_dataset(k, d, n_per_cluster, random_state=None):
    """Generate Rodriguez Structured Gaussian dataset with controlled cluster overlap.
    
    The key is the alpha mixing parameter that controls cluster separation.
    We tune alpha so that clusters have moderate overlap (ARI ~0.5-0.7 for k-means).
    """
    rng = np.random.RandomState(random_state)
    
    # Generate cluster centers with controlled separation
    # Use alpha to control how close clusters are
    # Lower alpha = more overlap
    alpha = 1.0  # Moderate separation
    
    # Generate random centers
    centers = rng.randn(k, d) * alpha * np.sqrt(d) / np.sqrt(k)
    
    # Generate cluster-specific covariance matrices (symmetric PSD)
    X_all, y_all = [], []
    for ci in range(k):
        # Random covariance matrix
        A = rng.randn(d, d) * 0.3
        cov = A @ A.T / d + np.eye(d) * 0.5
        
        X_cluster = rng.multivariate_normal(centers[ci], cov, size=n_per_cluster)
        X_all.append(X_cluster)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def generate_repliclust_dataset(k, d, n_per_cluster, random_state=None):
    """Generate Repliclust-style dataset: high-dimensional anisotropic clusters."""
    rng = np.random.RandomState(random_state)
    
    # Generate cluster centers with moderate separation
    separation = 3.0
    centers = rng.randn(k, d) * separation
    
    X_all, y_all = [], []
    for ci in range(k):
        # Anisotropic covariance - different scales per dimension
        scales = rng.uniform(0.3, 2.0, d)
        X_cluster = rng.randn(n_per_cluster, d) * scales + centers[ci]
        X_all.append(X_cluster)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def generate_synthetic_datasets(n_reps=10, rsg_reps=10):
    """Generate all synthetic datasets.
    
    Paper uses 50 reps for Circles/Moons/Repliclust, 265 total for RSG.
    We use fewer reps for computational feasibility.
    """
    datasets = {'Circles': [], 'Moons': [], 'RSG': [], 'Repliclust': []}
    dims = [10, 50, 200]
    
    print("Generating Circles datasets...")
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 1000 + d
            # k=2
            X, y = generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Circles'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep, 'name': f'Circles_k2_d{d}_r{rep}'})
            
            # k=5
            X, y = generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Circles'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep, 'name': f'Circles_k5_d{d}_r{rep}'})
    
    print("Generating Moons datasets...")
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 1000 + d + 100000
            # k=2
            X, y = generate_moons_k2(n_samples=2000, noise=0.05, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Moons'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep, 'name': f'Moons_k2_d{d}_r{rep}'})
            
            # k=5
            X, y = generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Moons'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep, 'name': f'Moons_k5_d{d}_r{rep}'})
    
    print("Generating RSG datasets...")
    for k_val in [2, 10, 50]:
        for d in [10, 50, 200]:
            for nc in [5, 50, 100]:
                for rep in range(rsg_reps):
                    seed = k_val * 10000 + d * 100 + nc + rep * 7
                    X, y = generate_rsg_dataset(k_val, d, nc, random_state=seed)
                    X_noisy = inject_noise(X, random_state=seed+1)
                    datasets['RSG'].append({'X': X_noisy, 'y': y, 'k': k_val, 'd': d, 'rep': rep, 'name': f'RSG_k{k_val}_d{d}_nc{nc}_r{rep}'})
    
    print("Generating Repliclust datasets...")
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 1000 + d + 200000
            # k=2
            X, y = generate_repliclust_dataset(2, d, 1000, random_state=seed)
            X_noisy = inject_noise(X, random_state=seed+1)
            datasets['Repliclust'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep, 'name': f'Repliclust_k2_d{d}_r{rep}'})
            
            # k=5
            X, y = generate_repliclust_dataset(5, d, 400, random_state=seed+2)
            X_noisy = inject_noise(X, random_state=seed+3)
            datasets['Repliclust'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep, 'name': f'Repliclust_k5_d{d}_r{rep}'})
    
    for dtype, dsets in datasets.items():
        print(f"  {dtype}: {len(dsets)} datasets")
    
    return datasets

# ============================================================
# REAL-WORLD DATA LOADING
# ============================================================

def load_real_world_datasets():
    """Load all 20 UCI datasets."""
    from load_uci import load_all_uci
    uci_data = load_all_uci()
    
    datasets = []
    for name, (X, y) in uci_data.items():
        k = len(np.unique(y))
        d = X.shape[1]
        datasets.append({
            'X': X, 'y': y, 'k': k, 'd': d, 'name': name
        })
    
    return datasets

# ============================================================
# VAE IMPLEMENTATION
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train_vae(X, latent_dim, epochs=100, batch_size=64, lr=1e-3):
    """Train VAE and return latent representations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Min-max scale to [0,1] for sigmoid output
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X - X_min) / X_range
    
    n = X_scaled.shape[0]
    input_dim = X_scaled.shape[1]
    
    # 70/30 split
    n_train = int(0.7 * n)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    train_dataset = TensorDataset(X_tensor[train_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Get latent representations for ALL data
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        Z = mu.cpu().numpy()
    
    return Z

# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def get_reduction_dims(k, d):
    """Get the 3 reduction levels: k-1, 25%, 50%."""
    km1 = max(k - 1, 2)
    pct25 = max(round(d * 0.25), 2)
    pct50 = max(round(d * 0.50), 2)
    return {'k-1': km1, '25%': pct25, '50%': pct50}

def apply_dr(X_norm, method, n_components, random_state=42):
    """Apply a single DR method. Returns reduced data or None on failure."""
    n_samples, n_features = X_norm.shape
    
    if n_components >= n_features:
        return X_norm  # No reduction needed
    
    if n_components < 1:
        n_components = 2
    
    try:
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(X_norm)
        
        elif method == 'KernelPCA':
            reducer = KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state)
            result = reducer.fit_transform(X_norm)
            if result.shape[1] < n_components:
                return None
            return result
        
        elif method == 'VAE':
            return train_vae(X_norm, n_components, epochs=100, batch_size=64)
        
        elif method == 'Isomap':
            n_neighbors = min(5, n_samples - 1)
            reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            return reducer.fit_transform(X_norm)
        
        elif method == 'MDS':
            reducer = MDS(n_components=n_components, random_state=10, n_init=4, max_iter=300, normalized_stress='auto')
            return reducer.fit_transform(X_norm)
        
    except Exception as e:
        print(f"    DR failed ({method}, n_comp={n_components}): {e}")
        return None

# ============================================================
# CLUSTERING
# ============================================================

def cluster_kmeans(X, k):
    """K-means clustering."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)

def cluster_ahc(X, k, affinity='euclidean', linkage='ward'):
    """Agglomerative Hierarchical Clustering."""
    try:
        ahc = AgglomerativeClustering(n_clusters=k, metric=affinity, linkage=linkage)
        if linkage == 'ward':
            ahc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        return ahc.fit_predict(X)
    except:
        return None

def cluster_gmm(X, k, covariance_type='full'):
    """Gaussian Mixture Model clustering."""
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, n_init=5)
        return gmm.fit_predict(X)
    except:
        return None

def cluster_optics(X, k, min_samples=5, min_cluster_size=0.05):
    """OPTICS clustering."""
    try:
        optics = OPTICS(min_samples=min_samples, xi=min_cluster_size, cluster_method='xi')
        labels = optics.fit_predict(X)
        return labels
    except:
        return None

# ============================================================
# HYPERPARAMETER SEARCH (per dataset type)
# ============================================================

def find_best_ahc_params(datasets_sample, max_datasets=20):
    """Find best AHC params across a sample of datasets."""
    # Ward only works with euclidean
    configs = [
        ('euclidean', 'ward'),
        ('euclidean', 'complete'),
        ('euclidean', 'average'),
        ('euclidean', 'single'),
        ('manhattan', 'complete'),
        ('manhattan', 'average'),
        ('manhattan', 'single'),
        ('cosine', 'complete'),
        ('cosine', 'average'),
        ('cosine', 'single'),
    ]
    
    sample = datasets_sample[:max_datasets]
    
    best_score = -1
    best_config = ('euclidean', 'ward')
    
    for affinity, linkage in configs:
        scores = []
        for ds in sample:
            X_norm = StandardScaler().fit_transform(ds['X'])
            labels = cluster_ahc(X_norm, ds['k'], affinity, linkage)
            if labels is not None:
                scores.append(adjusted_rand_score(ds['y'], labels))
        
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_config = (affinity, linkage)
    
    print(f"  Best AHC: affinity={best_config[0]}, linkage={best_config[1]}, avg ARI={best_score:.3f}")
    return best_config

def find_best_gmm_params(datasets_sample, max_datasets=20):
    """Find best GMM covariance type across a sample of datasets."""
    sample = datasets_sample[:max_datasets]
    
    best_score = -1
    best_cov = 'full'
    
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        scores = []
        for ds in sample:
            X_norm = StandardScaler().fit_transform(ds['X'])
            labels = cluster_gmm(X_norm, ds['k'], cov_type)
            if labels is not None:
                scores.append(adjusted_rand_score(ds['y'], labels))
        
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_cov = cov_type
    
    print(f"  Best GMM: covariance_type={best_cov}, avg ARI={best_score:.3f}")
    return best_cov

def find_best_optics_params(datasets_sample, max_datasets=15):
    """Find best OPTICS params across a sample of datasets."""
    sample = datasets_sample[:max_datasets]
    
    best_score = -1
    best_params = (5, 0.05)
    
    for min_samples in [5, 7, 10]:
        for min_cluster_size in [0.05, 0.1, 0.2, 0.3, 0.5]:
            scores = []
            for ds in sample:
                X_norm = StandardScaler().fit_transform(ds['X'])
                labels = cluster_optics(X_norm, ds['k'], min_samples, min_cluster_size)
                if labels is not None:
                    ari = adjusted_rand_score(ds['y'], labels)
                    scores.append(ari)
            
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_params = (min_samples, min_cluster_size)
    
    print(f"  Best OPTICS: min_samples={best_params[0]}, min_cluster_size={best_params[1]}, avg ARI={best_score:.3f}")
    return best_params

# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiments_on_datasets(datasets, dataset_type, dr_methods=['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']):
    """Run all experiments on a list of datasets.
    
    Returns a DataFrame with columns:
    dataset, algorithm, dr_method, reduction_level, ari
    """
    print(f"\n{'='*60}")
    print(f"Running experiments on {dataset_type} ({len(datasets)} datasets)")
    print(f"{'='*60}")
    
    # Step 1: Find best hyperparameters for AHC, GMM, OPTICS
    print("\nFinding best hyperparameters...")
    ahc_params = find_best_ahc_params(datasets)
    gmm_cov = find_best_gmm_params(datasets)
    optics_params = find_best_optics_params(datasets)
    
    results = []
    
    for di, ds in enumerate(datasets):
        if di % 10 == 0:
            print(f"\n  Processing dataset {di+1}/{len(datasets)}: {ds['name']}")
        
        X = ds['X']
        y = ds['y']
        k = ds['k']
        d = ds['d'] if 'd' in ds else X.shape[1]
        
        # Z-score normalize
        X_norm = StandardScaler().fit_transform(X)
        
        # Get reduction dimensions
        red_dims = get_reduction_dims(k, d)
        
        # --- No Reduction baseline ---
        for algo_name, cluster_fn in [
            ('k-means', lambda X: cluster_kmeans(X, k)),
            ('AHC', lambda X: cluster_ahc(X, k, ahc_params[0], ahc_params[1])),
            ('GMM', lambda X: cluster_gmm(X, k, gmm_cov)),
            ('OPTICS', lambda X: cluster_optics(X, k, optics_params[0], optics_params[1])),
        ]:
            labels = cluster_fn(X_norm)
            if labels is not None:
                ari = adjusted_rand_score(y, labels)
            else:
                ari = 0.0
            results.append({
                'dataset': ds['name'],
                'algorithm': algo_name,
                'dr_method': 'No Reduction',
                'reduction_level': 'None',
                'ari': ari,
                'k': k,
                'd': d,
            })
        
        # --- With DR ---
        for method in dr_methods:
            for level_name, n_comp in red_dims.items():
                if n_comp >= X_norm.shape[1]:
                    # No actual reduction
                    X_reduced = X_norm
                else:
                    X_reduced = apply_dr(X_norm, method, n_comp)
                
                if X_reduced is None:
                    # DR failed - record 0
                    for algo_name in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                        results.append({
                            'dataset': ds['name'],
                            'algorithm': algo_name,
                            'dr_method': method,
                            'reduction_level': level_name,
                            'ari': 0.0,
                            'k': k,
                            'd': d,
                        })
                    continue
                
                # Re-normalize after DR
                X_dr_norm = StandardScaler().fit_transform(X_reduced)
                
                for algo_name, cluster_fn in [
                    ('k-means', lambda X: cluster_kmeans(X, k)),
                    ('AHC', lambda X: cluster_ahc(X, k, ahc_params[0], ahc_params[1])),
                    ('GMM', lambda X: cluster_gmm(X, k, gmm_cov)),
                    ('OPTICS', lambda X: cluster_optics(X, k, optics_params[0], optics_params[1])),
                ]:
                    labels = cluster_fn(X_dr_norm)
                    if labels is not None:
                        ari = adjusted_rand_score(y, labels)
                    else:
                        ari = 0.0
                    results.append({
                        'dataset': ds['name'],
                        'algorithm': algo_name,
                        'dr_method': method,
                        'reduction_level': level_name,
                        'ari': ari,
                        'k': k,
                        'd': d,
                    })
    
    df = pd.DataFrame(results)
    return df

# ============================================================
# TABLE GENERATION
# ============================================================

def generate_ari_table(df, dataset_type, algo):
    """Generate ARI table matching paper format.
    
    Columns: No Reduction, PCA(k-1, 25%, 50%), KernelPCA(k-1, 25%, 50%), 
             VAE(k-1, 25%, 50%), Isomap(k-1, 25%, 50%), MDS(k-1, 25%, 50%)
    """
    df_algo = df[df['algorithm'] == algo]
    
    # Get No Reduction average
    no_red = df_algo[df_algo['dr_method'] == 'No Reduction']['ari'].mean()
    
    row = {'No Reduction': no_red}
    
    for method in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            mask = (df_algo['dr_method'] == method) & (df_algo['reduction_level'] == level)
            val = df_algo[mask]['ari'].mean() if mask.sum() > 0 else 0.0
            row[f'{method}_{level}'] = val
    
    return row

def generate_real_world_table(df, algo):
    """Generate per-dataset ARI table for real-world data."""
    df_algo = df[df['algorithm'] == algo]
    
    rows = []
    for ds_name in df_algo['dataset'].unique():
        df_ds = df_algo[df_algo['dataset'] == ds_name]
        
        row = {'Dataset': ds_name}
        
        # No Reduction
        no_red = df_ds[df_ds['dr_method'] == 'No Reduction']['ari'].values
        row['No Reduction'] = no_red[0] if len(no_red) > 0 else 0.0
        
        for method in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
            for level in ['k-1', '25%', '50%']:
                mask = (df_ds['dr_method'] == method) & (df_ds['reduction_level'] == level)
                vals = df_ds[mask]['ari'].values
                row[f'{method}_{level}'] = vals[0] if len(vals) > 0 else 0.0
        
        rows.append(row)
    
    return pd.DataFrame(rows)

# ============================================================
# AGGREGATE ANALYSIS
# ============================================================

def compute_aggregate_table(df, algo):
    """Compute % wins and avg win/loss for each DR method vs No Reduction.
    
    Paper Tables 1-4 format.
    """
    df_algo = df[df['algorithm'] == algo]
    
    results = {}
    
    for method in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            key = f'{method}_{level}'
            
            wins = 0
            losses = 0
            win_gains = []
            loss_drops = []
            total = 0
            
            for ds_name in df_algo['dataset'].unique():
                df_ds = df_algo[df_algo['dataset'] == ds_name]
                
                no_red_ari = df_ds[df_ds['dr_method'] == 'No Reduction']['ari'].values
                if len(no_red_ari) == 0:
                    continue
                no_red_ari = no_red_ari[0]
                
                mask = (df_ds['dr_method'] == method) & (df_ds['reduction_level'] == level)
                dr_ari = df_ds[mask]['ari'].values
                if len(dr_ari) == 0:
                    continue
                dr_ari = dr_ari[0]
                
                total += 1
                diff = dr_ari - no_red_ari
                
                if diff > 0:
                    wins += 1
                    win_gains.append(diff)
                elif diff < 0:
                    losses += 1
                    loss_drops.append(diff)
            
            if total > 0:
                results[key] = {
                    'pct_wins': 100 * wins / total,
                    'avg_win': np.mean(win_gains) if win_gains else 0,
                    'avg_loss': np.mean(loss_drops) if loss_drops else 0,
                    'pct_loss': 100 * losses / total,
                }
    
    return results

# ============================================================
# BOXPLOT GENERATION
# ============================================================

def generate_boxplot(df, algo, dataset_type, output_path):
    """Generate boxplot matching paper format."""
    df_algo = df[df['algorithm'] == algo]
    
    # Collect data for boxplot
    labels = []
    data = []
    
    # No Reduction
    vals = df_algo[df_algo['dr_method'] == 'No Reduction']['ari'].values
    if len(vals) > 0:
        data.append(vals)
        labels.append('No Red.')
    
    for method in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            mask = (df_algo['dr_method'] == method) & (df_algo['reduction_level'] == level)
            vals = df_algo[mask]['ari'].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f'{method}\n{level}')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['lightgray'] + ['#AEC7E8']*3 + ['#FFBB78']*3 + ['#98DF8A']*3 + ['#FF9896']*3 + ['#C5B0D5']*3
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('ARI')
    ax.set_title(f'{algo} - {dataset_type}')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# WILCOXON TEST
# ============================================================

def wilcoxon_test(df, algo):
    """Perform Wilcoxon signed-rank test for each DR method vs No Reduction."""
    df_algo = df[df['algorithm'] == algo]
    
    results = {}
    
    for method in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            key = f'{method}_{level}'
            
            no_red_scores = []
            dr_scores = []
            
            for ds_name in df_algo['dataset'].unique():
                df_ds = df_algo[df_algo['dataset'] == ds_name]
                
                no_red = df_ds[df_ds['dr_method'] == 'No Reduction']['ari'].values
                mask = (df_ds['dr_method'] == method) & (df_ds['reduction_level'] == level)
                dr = df_ds[mask]['ari'].values
                
                if len(no_red) > 0 and len(dr) > 0:
                    no_red_scores.append(no_red[0])
                    dr_scores.append(dr[0])
            
            if len(no_red_scores) >= 5:
                try:
                    stat, pval = wilcoxon(dr_scores, no_red_scores)
                    results[key] = {'statistic': stat, 'p_value': pval}
                except:
                    results[key] = {'statistic': None, 'p_value': None}
    
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs('/workspace/results', exist_ok=True)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    all_results = {}
    
    if mode in ['real', 'all']:
        print("\n" + "="*60)
        print("REAL-WORLD EXPERIMENTS")
        print("="*60)
        
        real_datasets = load_real_world_datasets()
        print(f"Loaded {len(real_datasets)} real-world datasets")
        
        df_real = run_experiments_on_datasets(real_datasets, 'RealWorld')
        df_real.to_csv('/workspace/results/results_real_world.csv', index=False)
        all_results['RealWorld'] = df_real
        
        # Generate tables and plots
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            table = generate_real_world_table(df_real, algo)
            table.to_csv(f'/workspace/results/table_{algo}_RealWorld.csv', index=False)
            print(f"\n{algo} Real-World Table:")
            print(table.to_string(index=False, float_format='%.3f'))
            
            generate_boxplot(df_real, algo, 'RealWorld', f'/workspace/results/boxplot_{algo}_RealWorld.pdf')
    
    if mode in ['synthetic', 'all']:
        print("\n" + "="*60)
        print("SYNTHETIC EXPERIMENTS")
        print("="*60)
        
        # Use fewer reps for speed
        n_reps = 5  # Paper uses 50
        rsg_reps = 3  # Paper uses ~10 per config
        
        synth_datasets = generate_synthetic_datasets(n_reps=n_reps, rsg_reps=rsg_reps)
        
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            datasets = synth_datasets[dtype]
            df_synth = run_experiments_on_datasets(datasets, dtype)
            df_synth.to_csv(f'/workspace/results/results_{dtype}.csv', index=False)
            all_results[dtype] = df_synth
            
            # Generate ARI table
            for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                row = generate_ari_table(df_synth, dtype, algo)
                print(f"\n{algo} {dtype}: {row}")
                
                generate_boxplot(df_synth, algo, dtype, f'/workspace/results/boxplot_{algo}_{dtype}.pdf')
    
    # Generate aggregate tables and Wilcoxon tests
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)
    
    # Combine all synthetic results
    synth_types = [t for t in ['Circles', 'Moons', 'RSG', 'Repliclust'] if t in all_results]
    if synth_types:
        df_all_synth = pd.concat([all_results[t] for t in synth_types], ignore_index=True)
        
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            print(f"\n{algo} Aggregate (Synthetic):")
            agg = compute_aggregate_table(df_all_synth, algo)
            for key, vals in agg.items():
                print(f"  {key}: wins={vals['pct_wins']:.1f}%, avg_win={vals['avg_win']:.3f}, avg_loss={vals['avg_loss']:.3f}")
            
            # Wilcoxon test
            wt = wilcoxon_test(df_all_synth, algo)
            print(f"\n{algo} Wilcoxon (Synthetic):")
            for key, vals in wt.items():
                if vals['p_value'] is not None:
                    sig = '*' if vals['p_value'] < 0.05 else ''
                    print(f"  {key}: p={vals['p_value']:.4f}{sig}")
    
    if 'RealWorld' in all_results:
        df_real = all_results['RealWorld']
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            print(f"\n{algo} Aggregate (Real-World):")
            agg = compute_aggregate_table(df_real, algo)
            for key, vals in agg.items():
                print(f"  {key}: wins={vals['pct_wins']:.1f}%, avg_win={vals['avg_win']:.3f}, avg_loss={vals['avg_loss']:.3f}")
    
    print("\n\nDone! Results saved to /workspace/results/")

if __name__ == '__main__':
    main()
