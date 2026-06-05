#!/usr/bin/env python3
"""
Final clean pipeline for replicating:
"Assessing the impact of dimensionality reduction on clustering performance"

Generates all results: synthetic + real-world ARI tables, aggregate tables,
boxplots, and Wilcoxon tests.
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import pickle
import time
import traceback
from collections import defaultdict
from itertools import product
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_circles, make_moons
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# ============================================================

def generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=None):
    """Two-cluster circles: 1000 per cluster."""
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=random_state)
    return X, y

def generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Five concentric rings with radial factors 1.0, 2.0, 3.5, 5.0, 7.0."""
    rng = np.random.RandomState(random_state)
    factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    X_all, y_all = [], []
    for ci, r in enumerate(factors):
        theta = rng.uniform(0, 2*np.pi, n_per_cluster)
        rad = r + rng.normal(0, noise, n_per_cluster)
        x = rad * np.cos(theta)
        y_coord = rad * np.sin(theta)
        X_all.append(np.column_stack([x, y_coord]))
        y_all.append(np.full(n_per_cluster, ci))
    return np.vstack(X_all), np.concatenate(y_all)

def generate_moons_k2(n_samples=2000, noise=0.05, stretch=1.0, rotation=0.0,
                       tx=0.0, ty=0.0, random_state=None):
    """Two-cluster moons with transformations."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # Apply stretch
    X[:, 0] *= stretch
    # Apply rotation
    theta = np.radians(rotation)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X = X @ R.T
    # Apply translation
    X[:, 0] += tx
    X[:, 1] += ty
    return X, y

def generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Five-cluster moons with different transformations."""
    rng = np.random.RandomState(random_state)
    configs = [
        {'stretch': 1.0, 'rotation': 0, 'tx': 0, 'ty': 0},
        {'stretch': 1.5, 'rotation': 160, 'tx': 4, 'ty': 1.0},
        {'stretch': 1.0, 'rotation': -160, 'tx': -4, 'ty': 1.2},
        {'stretch': 1.5, 'rotation': 10, 'tx': 2, 'ty': 1.5},
        {'stretch': 1.0, 'rotation': -10, 'tx': -2, 'ty': 1.0},
    ]
    X_all, y_all = [], []
    for ci, cfg in enumerate(configs):
        X, _ = make_moons(n_samples=n_per_cluster, noise=noise,
                          random_state=rng.randint(0, 100000))
        X[:, 0] *= cfg['stretch']
        theta = np.radians(cfg['rotation'])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        X = X @ R.T
        X[:, 0] += cfg['tx']
        X[:, 1] += cfg['ty']
        X_all.append(X)
        y_all.append(np.full(n_per_cluster, ci))
    return np.vstack(X_all), np.concatenate(y_all)

def embed_to_high_dim(X, target_dim, random_state=None):
    """Embed low-dim data into higher dimensions using Gaussian Random Projection."""
    rng = np.random.RandomState(random_state)
    d_orig = X.shape[1]
    if target_dim <= d_orig:
        return X
    # Use sklearn's GaussianRandomProjection in reverse:
    # We want to go FROM low-dim TO high-dim
    # Create a random matrix of shape (d_orig, target_dim)
    G = rng.randn(d_orig, target_dim) / np.sqrt(target_dim)
    return X @ G

def inject_noise(X, random_state=None):
    """Apply structured noise injection as described in the paper.
    1. Z-score normalize
    2. Add Gaussian noise to 75% of features:
       - 25% get N(0,1)
       - 25% get N(0,0.5)  
       - 25% get N(0,0.25)
       - 25% remain clean
    """
    rng = np.random.RandomState(random_state)
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    perm = rng.permutation(d)
    q = d // 4
    
    X_noisy = X.copy()
    # First quarter: sigma=1
    if q > 0:
        X_noisy[:, perm[:q]] += rng.normal(0, 1.0, (n, q))
    # Second quarter: sigma=0.5
    if q > 0:
        X_noisy[:, perm[q:2*q]] += rng.normal(0, 0.5, (n, q))
    # Third quarter: sigma=0.25
    if q > 0:
        X_noisy[:, perm[2*q:3*q]] += rng.normal(0, 0.25, (n, q))
    # Fourth quarter: clean
    
    return X_noisy

def generate_rsg_dataset(k, d, n_per_cluster, random_state=None):
    """Generate Rodriguez Structured Gaussian dataset."""
    rng = np.random.RandomState(random_state)
    
    # Generate cluster centers spread out
    centers = rng.randn(k, d) * np.sqrt(d)
    
    # Generate covariance matrices
    X_all, y_all = [], []
    for ci in range(k):
        # Random covariance: A @ A.T to ensure PSD
        A = rng.randn(d, d) * 0.5
        cov = A @ A.T / d + np.eye(d) * 0.1
        
        X_cluster = rng.multivariate_normal(centers[ci], cov, size=n_per_cluster)
        X_all.append(X_cluster)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def generate_repliclust_dataset(k, d, n_per_cluster, random_state=None):
    """Generate Repliclust-style dataset: high-dimensional anisotropic clusters."""
    rng = np.random.RandomState(random_state)
    
    # Generate well-separated cluster centers
    centers = np.zeros((k, d))
    for i in range(k):
        # Place centers along different dimensions
        dim_idx = i % d
        centers[i, dim_idx] = (i + 1) * 3.0
        # Add some random offset
        centers[i] += rng.randn(d) * 0.5
    
    X_all, y_all = [], []
    for ci in range(k):
        # Anisotropic covariance
        scales = rng.uniform(0.1, 2.0, d)
        X_cluster = rng.randn(n_per_cluster, d) * scales + centers[ci]
        X_all.append(X_cluster)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def generate_all_synthetic(n_reps=50, rsg_reps=30):
    """Generate all synthetic datasets.
    
    Circles: k=2 (Nc=1000) and k=5 (Nc=400), d in {10, 50, 200} → 6 configs × 50 reps = 300
    Moons: k=2 (Nc=1000) and k=5 (Nc=400), d in {10, 50, 200} → 6 configs × 50 reps = 300
    RSG: k in {2,10,50}, d in {10,50,200}, Nc in {5,50,100} → 27 configs × ~10 reps = 265
    Repliclust: k=2 (Nc=1000) and k=5 (Nc=400), d in {10, 50, 200} → 6 configs × 50 reps = 300
    """
    datasets = {'Circles': [], 'Moons': [], 'RSG': [], 'Repliclust': []}
    
    dims = [10, 50, 200]
    
    # Circles
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 100 + d
            # k=2
            X, y = generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Circles'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep})
            
            # k=5
            X, y = generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Circles'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep})
    
    # Moons
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 100 + d + 10000
            # k=2
            X, y = generate_moons_k2(n_samples=2000, noise=0.05, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Moons'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep})
            
            # k=5
            X, y = generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Moons'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep})
    
    # RSG: k in {2,10,50}, d in {10,50,200}, Nc in {5,50,100}
    # ~265 total datasets (paper uses 265 from Rodriguez et al.)
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            for nc in [5, 50, 100]:
                n_reps_rsg = rsg_reps
                for rep in range(n_reps_rsg):
                    seed = k * 10000 + d * 100 + nc + rep
                    X, y = generate_rsg_dataset(k, d, nc, random_state=seed)
                    X_noisy = inject_noise(X, random_state=seed+1)
                    datasets['RSG'].append({'X': X_noisy, 'y': y, 'k': k, 'd': d, 'rep': rep})
    
    # Repliclust
    for d in dims:
        for rep in range(n_reps):
            seed = rep * 100 + d + 20000
            # k=2
            X, y = generate_repliclust_dataset(2, d, 1000, random_state=seed)
            X_noisy = inject_noise(X, random_state=seed+1)
            datasets['Repliclust'].append({'X': X_noisy, 'y': y, 'k': 2, 'd': d, 'rep': rep})
            
            # k=5
            X, y = generate_repliclust_dataset(5, d, 400, random_state=seed+2)
            X_noisy = inject_noise(X, random_state=seed+3)
            datasets['Repliclust'].append({'X': X_noisy, 'y': y, 'k': 5, 'd': d, 'rep': rep})
    
    return datasets

# ============================================================
# SECTION 2: REAL-WORLD DATA LOADING
# ============================================================

def load_all_uci():
    """Load all 20 UCI datasets."""
    from load_uci import load_all_uci as _load
    return _load()

# ============================================================
# SECTION 3: DIMENSIONALITY REDUCTION
# ============================================================

class VAE(nn.Module):
    """Variational Autoencoder for dimensionality reduction."""
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
    X_range[X_range == 0] = 1.0
    X_scaled = (X - X_min) / X_range
    
    n = len(X_scaled)
    # 70/30 split
    n_train = int(0.7 * n)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    train_tensor = X_tensor[train_idx]
    
    input_dim = X.shape[1]
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            # MSE reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            # KL divergence
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

def apply_dr(X, method, n_components):
    """Apply dimensionality reduction method."""
    n_components = max(2, int(round(n_components)))
    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    
    if n_components >= X.shape[1]:
        return X  # No reduction needed
    
    try:
        if method == 'PCA':
            dr = PCA(n_components=n_components)
            return dr.fit_transform(X)
        
        elif method == 'Kernel PCA':
            dr = KernelPCA(n_components=n_components, kernel='rbf')
            return dr.fit_transform(X)
        
        elif method == 'VAE':
            return train_vae(X, n_components)
        
        elif method == 'Isomap':
            n_neighbors = min(5, X.shape[0] - 1)
            dr = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            return dr.fit_transform(X)
        
        elif method == 'MDS':
            dr = MDS(n_components=n_components, random_state=10, n_init=4, max_iter=300,
                     normalized_stress='auto')
            return dr.fit_transform(X)
        
        else:
            raise ValueError(f"Unknown DR method: {method}")
    except Exception as e:
        print(f"  DR failed ({method}, n_comp={n_components}): {e}")
        return None

def get_reduction_levels(d, k):
    """Get the three reduction levels: k-1, 25%, 50%."""
    levels = {}
    levels['k-1'] = max(2, k - 1)
    levels['25%'] = max(2, int(round(d * 0.25)))
    levels['50%'] = max(2, int(round(d * 0.50)))
    return levels

# ============================================================
# SECTION 4: CLUSTERING
# ============================================================

def cluster_kmeans(X, k):
    """K-means clustering."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)

def cluster_ahc(X, k, affinity='euclidean', linkage='ward'):
    """Agglomerative Hierarchical Clustering."""
    try:
        if linkage == 'ward':
            ahc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        else:
            ahc = AgglomerativeClustering(n_clusters=k, metric=affinity, linkage=linkage)
        return ahc.fit_predict(X)
    except:
        return np.zeros(len(X), dtype=int)

def cluster_gmm(X, k, covariance_type='full'):
    """Gaussian Mixture Model clustering."""
    try:
        gmm = GaussianMixture(n_clusters=k, covariance_type=covariance_type, random_state=42, n_init=1)
    except TypeError:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, n_init=1)
    return gmm.fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    """OPTICS clustering."""
    try:
        optics = OPTICS(min_samples=min_samples, cluster_method='xi',
                        min_cluster_size=min_cluster_size if min_cluster_size > 0 else None)
        labels = optics.fit_predict(X)
        return labels
    except:
        return -np.ones(len(X), dtype=int)

# ============================================================
# SECTION 5: HYPERPARAMETER SEARCH
# ============================================================

def find_best_ahc_params(datasets_list):
    """Find best AHC affinity/linkage by average ARI across datasets."""
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    best_score = -999
    best_params = ('euclidean', 'ward')
    
    # Sample datasets for speed
    sample = datasets_list
    if len(sample) > 50:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sample), 50, replace=False)
        sample = [datasets_list[i] for i in idx]
    
    for affinity in affinities:
        for linkage in linkages:
            if linkage == 'ward' and affinity != 'euclidean':
                continue
            scores = []
            for ds in sample:
                try:
                    labels = cluster_ahc(ds['X'], ds['k'], affinity, linkage)
                    scores.append(adjusted_rand_score(ds['y'], labels))
                except:
                    scores.append(0.0)
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_params = (affinity, linkage)
    
    print(f"  Best AHC params: affinity={best_params[0]}, linkage={best_params[1]}, avg ARI={best_score:.3f}")
    return best_params

def find_best_gmm_params(datasets_list):
    """Find best GMM covariance_type by average ARI across datasets."""
    cov_types = ['spherical', 'tied', 'diag', 'full']
    
    best_score = -999
    best_type = 'full'
    
    sample = datasets_list
    if len(sample) > 50:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sample), 50, replace=False)
        sample = [datasets_list[i] for i in idx]
    
    for ct in cov_types:
        scores = []
        for ds in sample:
            try:
                labels = cluster_gmm(ds['X'], ds['k'], ct)
                scores.append(adjusted_rand_score(ds['y'], labels))
            except:
                scores.append(0.0)
        avg = np.mean(scores)
        if avg > best_score:
            best_score = avg
            best_type = ct
    
    print(f"  Best GMM covariance_type: {best_type}, avg ARI={best_score:.3f}")
    return best_type

def find_best_optics_params(datasets_list):
    """Find best OPTICS min_samples and min_cluster_size."""
    min_samples_range = range(5, 11)
    min_cluster_sizes = np.arange(0, 1.05, 0.05)
    
    best_score = -999
    best_params = (5, 0.05)
    
    sample = datasets_list
    if len(sample) > 30:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sample), 30, replace=False)
        sample = [datasets_list[i] for i in idx]
    
    for ms in min_samples_range:
        for mcs in min_cluster_sizes:
            if mcs == 0:
                mcs_val = None
            else:
                mcs_val = mcs
            scores = []
            for ds in sample:
                try:
                    if ds['X'].shape[0] < ms:
                        scores.append(0.0)
                        continue
                    labels = cluster_optics(ds['X'], ms, mcs_val)
                    scores.append(adjusted_rand_score(ds['y'], labels))
                except:
                    scores.append(0.0)
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_params = (ms, mcs_val)
    
    print(f"  Best OPTICS params: min_samples={best_params[0]}, min_cluster_size={best_params[1]}, avg ARI={best_score:.3f}")
    return best_params

# ============================================================
# SECTION 6: MAIN EXPERIMENT RUNNER
# ============================================================

DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_NAMES = ['k-1', '25%', '50%']

def run_experiments_on_datasets(datasets_list, dataset_type_name, 
                                 ahc_params=None, gmm_cov=None, optics_params=None):
    """Run all experiments on a list of datasets.
    
    Returns: dict with structure:
        results[algo][dr_method][level] = list of ARI scores
        results[algo]['No Reduction'] = list of ARI scores
    """
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    
    results = {}
    for algo in algos:
        results[algo] = {'No Reduction': []}
        for dr in DR_METHODS:
            for level in REDUCTION_NAMES:
                results[algo][f'{dr}_{level}'] = []
    
    total = len(datasets_list)
    
    for idx, ds in enumerate(datasets_list):
        X_raw = ds['X']
        y_true = ds['y']
        k = ds['k']
        d = X_raw.shape[1]
        
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{dataset_type_name}] Processing dataset {idx+1}/{total} (k={k}, d={d}, n={len(X_raw)})")
        
        # Z-score normalize
        X = StandardScaler().fit_transform(X_raw)
        
        # Get reduction levels
        levels = get_reduction_levels(d, k)
        
        # --- No Reduction ---
        for algo in algos:
            try:
                if algo == 'k-means':
                    labels = cluster_kmeans(X, k)
                elif algo == 'AHC':
                    labels = cluster_ahc(X, k, ahc_params[0], ahc_params[1])
                elif algo == 'GMM':
                    labels = cluster_gmm(X, k, gmm_cov)
                elif algo == 'OPTICS':
                    labels = cluster_optics(X, optics_params[0], optics_params[1])
                ari = adjusted_rand_score(y_true, labels)
            except:
                ari = 0.0
            results[algo]['No Reduction'].append(ari)
        
        # --- With DR ---
        for dr_method in DR_METHODS:
            for level_name, n_comp in levels.items():
                if n_comp >= d:
                    # No actual reduction, use original
                    X_reduced = X
                else:
                    X_reduced = apply_dr(X, dr_method, n_comp)
                
                if X_reduced is None:
                    for algo in algos:
                        results[algo][f'{dr_method}_{level_name}'].append(0.0)
                    continue
                
                for algo in algos:
                    try:
                        if algo == 'k-means':
                            labels = cluster_kmeans(X_reduced, k)
                        elif algo == 'AHC':
                            labels = cluster_ahc(X_reduced, k, ahc_params[0], ahc_params[1])
                        elif algo == 'GMM':
                            labels = cluster_gmm(X_reduced, k, gmm_cov)
                        elif algo == 'OPTICS':
                            labels = cluster_optics(X_reduced, optics_params[0], optics_params[1])
                        ari = adjusted_rand_score(y_true, labels)
                    except:
                        ari = 0.0
                    results[algo][f'{dr_method}_{level_name}'].append(ari)
    
    return results

def compute_average_ari_table(results, algos=None):
    """Compute average ARI table from results."""
    if algos is None:
        algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    
    columns = ['No Reduction']
    for dr in DR_METHODS:
        for level in REDUCTION_NAMES:
            columns.append(f'{dr}_{level}')
    
    data = {}
    for algo in algos:
        row = {}
        for col in columns:
            vals = results[algo][col]
            row[col] = np.mean(vals) if len(vals) > 0 else 0.0
        data[algo] = row
    
    df = pd.DataFrame(data).T
    df.index.name = 'Algorithm'
    return df

# ============================================================
# SECTION 7: REAL-WORLD EXPERIMENTS
# ============================================================

def run_real_world_experiments():
    """Run experiments on all 20 UCI datasets."""
    print("=" * 60)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 60)
    
    uci_data = load_all_uci()
    
    # Convert to list format for HP search
    ds_list = []
    for name, (X, y) in uci_data.items():
        k = len(np.unique(y))
        ds_list.append({'X': X, 'y': y, 'k': k, 'name': name})
    
    # Find best hyperparameters across all real-world datasets
    print("\nFinding best AHC params for real-world data...")
    ahc_params = find_best_ahc_params(ds_list)
    
    print("Finding best GMM params for real-world data...")
    gmm_cov = find_best_gmm_params(ds_list)
    
    print("Finding best OPTICS params for real-world data...")
    optics_params = find_best_optics_params(ds_list)
    
    # Run experiments per dataset
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    
    # Store per-dataset results
    per_dataset_results = {}  # {algo: {dataset_name: {condition: ari}}}
    for algo in algos:
        per_dataset_results[algo] = {}
    
    for ds in ds_list:
        name = ds['name']
        X_raw = ds['X']
        y_true = ds['y']
        k = ds['k']
        d = X_raw.shape[1]
        
        print(f"\n  Processing {name} (n={len(X_raw)}, d={d}, k={k})")
        
        # Z-score normalize
        X = StandardScaler().fit_transform(X_raw)
        
        levels = get_reduction_levels(d, k)
        
        for algo in algos:
            per_dataset_results[algo][name] = {}
            
            # No reduction
            try:
                if algo == 'k-means':
                    labels = cluster_kmeans(X, k)
                elif algo == 'AHC':
                    labels = cluster_ahc(X, k, ahc_params[0], ahc_params[1])
                elif algo == 'GMM':
                    labels = cluster_gmm(X, k, gmm_cov)
                elif algo == 'OPTICS':
                    labels = cluster_optics(X, optics_params[0], optics_params[1])
                ari = adjusted_rand_score(y_true, labels)
            except:
                ari = 0.0
            per_dataset_results[algo][name]['No Reduction'] = round(ari, 2)
        
        # With DR
        for dr_method in DR_METHODS:
            for level_name, n_comp in levels.items():
                col_name = f'{dr_method}_{level_name}'
                
                if n_comp >= d:
                    X_reduced = X
                else:
                    X_reduced = apply_dr(X, dr_method, n_comp)
                
                if X_reduced is None:
                    for algo in algos:
                        per_dataset_results[algo][name][col_name] = 0.0
                    continue
                
                for algo in algos:
                    try:
                        if algo == 'k-means':
                            labels = cluster_kmeans(X_reduced, k)
                        elif algo == 'AHC':
                            labels = cluster_ahc(X_reduced, k, ahc_params[0], ahc_params[1])
                        elif algo == 'GMM':
                            labels = cluster_gmm(X_reduced, k, gmm_cov)
                        elif algo == 'OPTICS':
                            labels = cluster_optics(X_reduced, optics_params[0], optics_params[1])
                        ari = adjusted_rand_score(y_true, labels)
                    except:
                        ari = 0.0
                    per_dataset_results[algo][name][col_name] = round(ari, 2)
    
    return per_dataset_results, ahc_params, gmm_cov, optics_params

# ============================================================
# SECTION 8: OUTPUT GENERATION
# ============================================================

def save_real_world_tables(per_dataset_results):
    """Save per-dataset ARI tables for real-world data."""
    columns = ['No Reduction']
    for dr in DR_METHODS:
        for level in REDUCTION_NAMES:
            columns.append(f'{dr}_{level}')
    
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        rows = {}
        for name, vals in per_dataset_results[algo].items():
            rows[name] = {col: vals.get(col, 0.0) for col in columns}
        
        df = pd.DataFrame(rows).T
        df = df[columns]
        df.index.name = 'Dataset'
        
        fname = f'table_{algo}_RealWorld.csv'
        df.to_csv(os.path.join(RESULTS_DIR, fname))
        print(f"  Saved {fname}")

def save_synthetic_table(results, data_type):
    """Save average ARI table for synthetic data type."""
    df = compute_average_ari_table(results)
    df = df.round(3)
    fname = f'table_average_ARI_{data_type}.csv'
    df.to_csv(os.path.join(RESULTS_DIR, fname))
    print(f"  Saved {fname}")

def generate_boxplot(all_ari_values, algo_name, data_category, filename):
    """Generate boxplot figure.
    
    all_ari_values: dict mapping condition_name -> list of ARI values
    """
    conditions = ['No Reduction']
    for dr in DR_METHODS:
        for level in REDUCTION_NAMES:
            conditions.append(f'{dr}_{level}')
    
    data_for_plot = []
    labels = []
    for cond in conditions:
        if cond in all_ari_values:
            data_for_plot.append(all_ari_values[cond])
            labels.append(cond)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(data_for_plot, patch_artist=True, widths=0.6)
    
    # Color by method
    colors = {
        'No Reduction': '#808080',
        'PCA': '#1f77b4',
        'Kernel PCA': '#ff7f0e',
        'VAE': '#2ca02c',
        'Isomap': '#d62728',
        'MDS': '#9467bd',
    }
    
    for i, label in enumerate(labels):
        if label == 'No Reduction':
            color = colors['No Reduction']
        else:
            method = label.rsplit('_', 1)[0]
            color = colors.get(method, '#808080')
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ARI')
    ax.set_title(f'{algo_name} - {data_category}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

def compute_aggregate_table(synth_results_all, real_results):
    """Compute aggregate tables (% wins, avg win/loss) for each clustering algorithm.
    
    synth_results_all: dict of {data_type: results_dict}
    real_results: per_dataset_results from real-world experiments
    """
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    
    for algo in algos:
        rows = []
        
        for dr in DR_METHODS:
            for level in REDUCTION_NAMES:
                col = f'{dr}_{level}'
                
                # Synthetic: collect all ARI differences
                synth_diffs = []
                for dtype, res in synth_results_all.items():
                    baseline = res[algo]['No Reduction']
                    method_vals = res[algo][col]
                    for b, m in zip(baseline, method_vals):
                        synth_diffs.append(m - b)
                
                # Real-world: collect all ARI differences
                real_diffs = []
                for ds_name, vals in real_results[algo].items():
                    b = vals.get('No Reduction', 0)
                    m = vals.get(col, 0)
                    real_diffs.append(m - b)
                
                # Compute stats
                synth_diffs = np.array(synth_diffs)
                real_diffs = np.array(real_diffs)
                
                # % wins (strictly better than baseline)
                synth_wins = np.mean(synth_diffs > 0.005) * 100  # small threshold
                real_wins = np.mean(real_diffs > 0.005) * 100
                
                # Average win/loss percentage
                synth_avg_diff = np.mean(synth_diffs) * 100
                real_avg_diff = np.mean(real_diffs) * 100
                
                rows.append({
                    'Method': dr,
                    'Reduction': level,
                    'Synth Win%': round(synth_wins, 2),
                    'Real Win%': round(real_wins, 2),
                    'Synth Avg Win/Loss%': round(synth_avg_diff, 2),
                    'Real Avg Win/Loss%': round(real_avg_diff, 2),
                })
        
        df = pd.DataFrame(rows)
        fname = f'table_aggregate_{algo}.csv'
        df.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
        print(f"  Saved {fname}")

def compute_wilcoxon_test(real_results):
    """Compute Wilcoxon signed-rank test for real-world data."""
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    
    rows = []
    for algo in algos:
        row = {'Algorithm': algo}
        
        # Get baseline ARIs
        baselines = []
        ds_names = list(real_results[algo].keys())
        for name in ds_names:
            baselines.append(real_results[algo][name].get('No Reduction', 0))
        baselines = np.array(baselines)
        
        for dr in DR_METHODS:
            for level in REDUCTION_NAMES:
                col = f'{dr}_{level}'
                method_vals = []
                for name in ds_names:
                    method_vals.append(real_results[algo][name].get(col, 0))
                method_vals = np.array(method_vals)
                
                # One-sided Wilcoxon: H1: method > baseline
                diffs = method_vals - baselines
                
                # Remove zeros
                nonzero = diffs[diffs != 0]
                if len(nonzero) < 2:
                    p_val = 1.0
                else:
                    try:
                        stat, p_two = wilcoxon(nonzero, alternative='greater')
                        p_val = p_two
                    except:
                        p_val = 1.0
                
                row[f'{dr}_{level}'] = round(p_val, 3)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'table_Wilcoxon.csv'), index=False)
    print(f"  Saved table_Wilcoxon.csv")
    return df

# ============================================================
# SECTION 9: MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'all'], default='all')
    parser.add_argument('--synth-type', choices=['Circles', 'Moons', 'RSG', 'Repliclust', 'all'], default='all')
    parser.add_argument('--synth-reps', type=int, default=50)
    parser.add_argument('--rsg-reps', type=int, default=10)
    args = parser.parse_args()
    
    real_results = None
    synth_results_all = {}
    
    if args.mode in ['real', 'all']:
        per_dataset_results, ahc_params, gmm_cov, optics_params = run_real_world_experiments()
        real_results = per_dataset_results
        
        # Save tables
        save_real_world_tables(per_dataset_results)
        
        # Generate boxplots for real-world
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            all_ari = {}
            conditions = ['No Reduction']
            for dr in DR_METHODS:
                for level in REDUCTION_NAMES:
                    conditions.append(f'{dr}_{level}')
            
            for cond in conditions:
                vals = [per_dataset_results[algo][name].get(cond, 0) 
                        for name in per_dataset_results[algo]]
                all_ari[cond] = vals
            
            generate_boxplot(all_ari, algo, 'Real-World', f'boxplot_{algo}_RealWorld.pdf')
        
        # Wilcoxon test
        compute_wilcoxon_test(per_dataset_results)
        
        # Save real results
        with open(os.path.join(RESULTS_DIR, 'real_world_per_dataset.json'), 'w') as f:
            json.dump(per_dataset_results, f, indent=2)
    
    if args.mode in ['synthetic', 'all']:
        synth_types = ['Circles', 'Moons', 'RSG', 'Repliclust'] if args.synth_type == 'all' else [args.synth_type]
        
        for stype in synth_types:
            print(f"\n{'='*60}")
            print(f"SYNTHETIC EXPERIMENTS: {stype}")
            print(f"{'='*60}")
            
            # Generate data
            print(f"  Generating {stype} datasets...")
            all_synth = generate_all_synthetic(n_reps=args.synth_reps, rsg_reps=args.rsg_reps)
            datasets = all_synth[stype]
            print(f"  Generated {len(datasets)} datasets")
            
            # Find best hyperparameters for this type
            print(f"  Finding best AHC params for {stype}...")
            ahc_params = find_best_ahc_params(datasets)
            
            print(f"  Finding best GMM params for {stype}...")
            gmm_cov = find_best_gmm_params(datasets)
            
            print(f"  Finding best OPTICS params for {stype}...")
            optics_params = find_best_optics_params(datasets)
            
            # Run experiments
            results = run_experiments_on_datasets(
                datasets, stype, ahc_params, gmm_cov, optics_params
            )
            
            synth_results_all[stype] = results
            
            # Save average ARI table
            save_synthetic_table(results, stype)
            
            # Generate boxplots
            for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                all_ari = {'No Reduction': results[algo]['No Reduction']}
                for dr in DR_METHODS:
                    for level in REDUCTION_NAMES:
                        col = f'{dr}_{level}'
                        all_ari[col] = results[algo][col]
                
                generate_boxplot(all_ari, algo, f'Synthetic {stype}',
                               f'boxplot_{algo}_Synthetic_{stype}.pdf')
            
            # Save raw results
            # Convert lists to serializable format
            save_results = {}
            for algo in results:
                save_results[algo] = {}
                for key, vals in results[algo].items():
                    save_results[algo][key] = [float(v) for v in vals]
            
            with open(os.path.join(RESULTS_DIR, f'synth_results_{stype}.json'), 'w') as f:
                json.dump(save_results, f)
        
        # Aggregate tables (need both synth and real)
        if real_results is not None and len(synth_results_all) > 0:
            print("\nComputing aggregate tables...")
            compute_aggregate_table(synth_results_all, real_results)

if __name__ == '__main__':
    main()
