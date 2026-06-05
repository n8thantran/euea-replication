#!/usr/bin/env python3
"""
Clean, comprehensive pipeline for replicating:
"Assessing the impact of dimensionality reduction on clustering performance"

This script handles:
1. Synthetic data generation (Circles, Moons, RSG, Repliclust)
2. Real-world data loading (20 UCI datasets)
3. Dimensionality reduction (PCA, Kernel PCA, VAE, Isomap, MDS)
4. Clustering (k-means, AHC, GMM, OPTICS)
5. Evaluation (ARI)
6. Output generation (tables, boxplots, Wilcoxon tests)
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.random_projection import GaussianRandomProjection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

RESULTS_DIR = '/workspace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# ============================================================

def generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=None):
    """Generate 2-cluster circles dataset."""
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=random_state)
    return X, y

def generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Generate 5-cluster concentric rings."""
    rng = np.random.RandomState(random_state)
    factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    X_list, y_list = [], []
    for ci, r in enumerate(factors):
        theta = rng.uniform(0, 2*np.pi, n_per_cluster)
        rad = r + rng.normal(0, noise, n_per_cluster)
        X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
        y_list.append(np.full(n_per_cluster, ci))
    return np.vstack(X_list), np.concatenate(y_list)

def generate_moons_k2(n_samples=2000, noise=0.05, stretch=1.0, rotation=0.0, 
                       tx=0.0, ty=0.0, random_state=None):
    """Generate 2-cluster moons dataset with transformations."""
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # Apply stretch
    X[:, 0] *= stretch
    # Apply rotation
    angle = np.radians(rotation)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X = X @ R.T
    # Apply translation
    X[:, 0] += tx
    X[:, 1] += ty
    return X, y

def generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=None):
    """Generate 5-cluster moons dataset with different transformations."""
    from sklearn.datasets import make_moons
    rng = np.random.RandomState(random_state)
    
    # 5 different moon configurations with stretching, rotation, translation
    configs = [
        {'stretch': 1.0, 'rotation': 0, 'tx': 0, 'ty': 0},
        {'stretch': 1.5, 'rotation': 160, 'tx': 2, 'ty': 1.0},
        {'stretch': 1.0, 'rotation': -160, 'tx': -2, 'ty': 1.2},
        {'stretch': 1.5, 'rotation': 10, 'tx': 4, 'ty': 1.5},
        {'stretch': 1.0, 'rotation': -10, 'tx': -4, 'ty': 1.5},
    ]
    
    X_all, y_all = [], []
    for ci, cfg in enumerate(configs):
        # Generate a pair of moons
        X, _ = make_moons(n_samples=n_per_cluster, noise=noise, random_state=rng.randint(0, 100000))
        X[:, 0] *= cfg['stretch']
        angle = np.radians(cfg['rotation'])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        X = X @ R.T
        X[:, 0] += cfg['tx']
        X[:, 1] += cfg['ty']
        X_all.append(X)
        y_all.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_all), np.concatenate(y_all)

def embed_to_high_dim(X, target_dim, random_state=None):
    """Embed 2D data into higher dimensions using Gaussian Random Projection."""
    rng = np.random.RandomState(random_state)
    d_orig = X.shape[1]
    if target_dim <= d_orig:
        return X
    # Random Gaussian projection matrix (JL-style)
    G = rng.randn(d_orig, target_dim) / np.sqrt(target_dim)
    return X @ G

def inject_noise(X, random_state=None):
    """Apply noise injection as described in the paper.
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
    for j in perm[:q]:
        X_noisy[:, j] += rng.normal(0, 1.0, n)
    # Second quarter: sigma=0.5
    for j in perm[q:2*q]:
        X_noisy[:, j] += rng.normal(0, 0.5, n)
    # Third quarter: sigma=0.25
    for j in perm[2*q:3*q]:
        X_noisy[:, j] += rng.normal(0, 0.25, n)
    # Fourth quarter: clean
    
    return X_noisy

def generate_rsg_dataset(k, d, n_per_cluster, alpha=None, random_state=None):
    """Generate Rodriguez Structured Gaussian dataset.
    Based on the method from Rodriguez et al. (2019).
    """
    rng = np.random.RandomState(random_state)
    
    if alpha is None:
        # Choose alpha based on difficulty level
        alpha = rng.uniform(0.5, 2.0)
    
    n = k * n_per_cluster
    X_list, y_list = [], []
    
    # Generate cluster centers spread apart
    centers = rng.randn(k, d) * alpha * np.sqrt(d)
    
    for ci in range(k):
        # Generate random covariance matrix (SPD)
        A = rng.randn(d, d) * 0.5
        cov = A @ A.T / d + np.eye(d) * 0.1
        
        X_cluster = rng.multivariate_normal(centers[ci], cov, n_per_cluster)
        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_list), np.concatenate(y_list)

def generate_repliclust_dataset(k, d, n_per_cluster, random_state=None):
    """Generate Repliclust-style dataset: anisotropic Gaussian clusters."""
    rng = np.random.RandomState(random_state)
    
    n = k * n_per_cluster
    X_list, y_list = [], []
    
    # Generate well-separated cluster centers
    centers = rng.randn(k, d) * 3.0
    
    for ci in range(k):
        # Anisotropic covariance: random eigenvalues
        eigenvalues = rng.exponential(1.0, d)
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Random rotation
        Q, _ = np.linalg.qr(rng.randn(d, d))
        cov = Q @ np.diag(eigenvalues) @ Q.T
        
        X_cluster = rng.multivariate_normal(centers[ci], cov, n_per_cluster)
        X_list.append(X_cluster)
        y_list.append(np.full(n_per_cluster, ci))
    
    return np.vstack(X_list), np.concatenate(y_list)

def generate_all_synthetic(n_reps=10):
    """Generate all synthetic datasets.
    
    Returns dict: {type_name: [(X, y, config_str), ...]}
    
    Paper uses 50 reps per config, we use n_reps for computational feasibility.
    """
    datasets = {
        'Circles': [],
        'Moons': [],
        'RSG': [],
        'Repliclust': []
    }
    
    dims = [10, 50, 200]
    
    # === CIRCLES ===
    for d in dims:
        for rep in range(n_reps):
            seed = 1000 * d + rep
            # k=2
            X, y = generate_circles_k2(n_samples=2000, noise=0.05, factor=0.5, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Circles'].append((X_noisy, y, f'k2_d{d}_rep{rep}'))
            
            # k=5
            X, y = generate_circles_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Circles'].append((X_noisy, y, f'k5_d{d}_rep{rep}'))
    
    # === MOONS ===
    for d in dims:
        for rep in range(n_reps):
            seed = 2000 * d + rep
            # k=2
            X, y = generate_moons_k2(n_samples=2000, noise=0.05, random_state=seed)
            X_high = embed_to_high_dim(X, d, random_state=seed+1)
            X_noisy = inject_noise(X_high, random_state=seed+2)
            datasets['Moons'].append((X_noisy, y, f'k2_d{d}_rep{rep}'))
            
            # k=5
            X, y = generate_moons_k5(n_per_cluster=400, noise=0.05, random_state=seed+3)
            X_high = embed_to_high_dim(X, d, random_state=seed+4)
            X_noisy = inject_noise(X_high, random_state=seed+5)
            datasets['Moons'].append((X_noisy, y, f'k5_d{d}_rep{rep}'))
    
    # === RSG ===
    # Paper uses 265 datasets from Rodriguez et al.
    # We generate similar ones: k in {2,10,50}, d in {10,50,200}, Nc in {5,50,100}
    for k in [2, 10, 50]:
        for d in [10, 50, 200]:
            for nc in [5, 50, 100]:
                n_rsg_reps = min(n_reps, 10)  # Fewer reps for RSG
                for rep in range(n_rsg_reps):
                    seed = 3000 + k*1000 + d*10 + nc + rep*100
                    try:
                        X, y = generate_rsg_dataset(k, d, nc, random_state=seed)
                        X_noisy = inject_noise(X, random_state=seed+1)
                        datasets['RSG'].append((X_noisy, y, f'k{k}_d{d}_nc{nc}_rep{rep}'))
                    except:
                        pass
    
    # === REPLICLUST ===
    for d in dims:
        for rep in range(n_reps):
            seed = 4000 * d + rep
            # k=2
            X, y = generate_repliclust_dataset(2, d, 1000, random_state=seed)
            X_noisy = inject_noise(X, random_state=seed+1)
            datasets['Repliclust'].append((X_noisy, y, f'k2_d{d}_rep{rep}'))
            
            # k=5
            X, y = generate_repliclust_dataset(5, d, 400, random_state=seed+2)
            X_noisy = inject_noise(X, random_state=seed+3)
            datasets['Repliclust'].append((X_noisy, y, f'k5_d{d}_rep{rep}'))
    
    for dtype, dlist in datasets.items():
        print(f"  {dtype}: {len(dlist)} datasets")
    
    return datasets


# ============================================================
# SECTION 2: REAL-WORLD DATA LOADING
# ============================================================

def load_all_real_world():
    """Load all 20 UCI datasets. Returns list of (X, y, name, k)."""
    from load_uci import load_all_uci
    uci_datasets = load_all_uci()
    result = []
    for name, (X, y) in uci_datasets.items():
        k = len(np.unique(y))
        result.append((X, y, name, k))
    return result


# ============================================================
# SECTION 3: DIMENSIONALITY REDUCTION
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
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

def apply_vae(X, n_components, random_state=42):
    """Apply VAE for dimensionality reduction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Min-max scale to [0,1] for sigmoid output
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 70/30 split
    n = len(X_scaled)
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    
    X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
    X_val = torch.FloatTensor(X_scaled[val_idx]).to(device)
    X_all = torch.FloatTensor(X_scaled).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
    
    input_dim = X.shape[1]
    model = VAE(input_dim, n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            # MSE reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Extract z_mean for all data
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_all)
    
    return mu.cpu().numpy()

def compute_reduction_levels(d, k):
    """Compute the 3 reduction levels: k-1, 25%, 50%."""
    levels = {}
    levels['k-1'] = max(k - 1, 2)
    levels['25%'] = max(round(d * 0.25), 2)
    levels['50%'] = max(round(d * 0.50), 2)
    return levels

def apply_dr(X, method, n_components, random_state=42):
    """Apply a dimensionality reduction method."""
    n_samples, n_features = X.shape
    n_components = min(n_components, n_features)
    
    if n_components >= n_features:
        return X
    
    try:
        if method == 'PCA':
            dr = PCA(n_components=n_components, random_state=random_state)
            return dr.fit_transform(X)
        elif method == 'Kernel PCA':
            dr = KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state)
            return dr.fit_transform(X)
        elif method == 'VAE':
            return apply_vae(X, n_components, random_state=random_state)
        elif method == 'Isomap':
            n_neighbors = min(5, n_samples - 1)
            dr = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            return dr.fit_transform(X)
        elif method == 'MDS':
            dr = MDS(n_components=n_components, random_state=10, n_init=50, 
                     max_iter=300, normalized_stress='auto')
            return dr.fit_transform(X)
    except Exception as e:
        print(f"    DR failed ({method}, n_comp={n_components}): {e}")
        return None
    
    return None


# ============================================================
# SECTION 4: CLUSTERING
# ============================================================

def run_kmeans(X, k):
    """Run k-means clustering."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)

def run_ahc(X, k, affinity='euclidean', linkage='ward'):
    """Run Agglomerative Hierarchical Clustering."""
    try:
        ahc = AgglomerativeClustering(n_clusters=k, metric=affinity, linkage=linkage)
        if linkage == 'ward':
            ahc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        return ahc.fit_predict(X)
    except:
        return None

def run_gmm(X, k, covariance_type='full'):
    """Run Gaussian Mixture Model."""
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        return gmm.fit_predict(X)
    except:
        return None

def run_optics(X, min_samples=5, min_cluster_size=0.05):
    """Run OPTICS clustering."""
    try:
        optics = OPTICS(min_samples=min_samples, xi=min_cluster_size, 
                        cluster_method='xi')
        return optics.fit_predict(X)
    except:
        return None


# ============================================================
# SECTION 5: HYPERPARAMETER SEARCH
# ============================================================

def find_best_ahc_params(datasets_with_labels):
    """Find best AHC params (affinity, linkage) across a set of datasets.
    Returns the combination with highest average ARI.
    """
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    best_score = -1
    best_params = ('euclidean', 'ward')
    
    for aff in affinities:
        for link in linkages:
            if link == 'ward' and aff != 'euclidean':
                continue
            
            scores = []
            for X, y, k in datasets_with_labels:
                try:
                    pred = run_ahc(X, k, affinity=aff, linkage=link)
                    if pred is not None:
                        scores.append(adjusted_rand_score(y, pred))
                except:
                    pass
            
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_params = (aff, link)
    
    print(f"  Best AHC params: affinity={best_params[0]}, linkage={best_params[1]}, avg ARI={best_score:.3f}")
    return best_params

def find_best_gmm_params(datasets_with_labels):
    """Find best GMM covariance_type across a set of datasets."""
    cov_types = ['spherical', 'tied', 'diag', 'full']
    
    best_score = -1
    best_type = 'full'
    
    for ct in cov_types:
        scores = []
        for X, y, k in datasets_with_labels:
            try:
                pred = run_gmm(X, k, covariance_type=ct)
                if pred is not None:
                    scores.append(adjusted_rand_score(y, pred))
            except:
                pass
        
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_type = ct
    
    print(f"  Best GMM covariance_type: {best_type}, avg ARI={best_score:.3f}")
    return best_type

def find_best_optics_params(datasets_with_labels):
    """Find best OPTICS params across a set of datasets."""
    best_score = -1
    best_params = (5, 0.05)
    
    for ms in range(5, 11):
        for mcs_int in range(0, 21):  # 0 to 1 step 0.05
            mcs = mcs_int * 0.05
            if mcs == 0:
                mcs = 0.01  # Avoid 0
            
            scores = []
            for X, y, k in datasets_with_labels:
                try:
                    pred = run_optics(X, min_samples=ms, min_cluster_size=mcs)
                    if pred is not None:
                        scores.append(adjusted_rand_score(y, pred))
                except:
                    pass
            
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_params = (ms, mcs)
    
    print(f"  Best OPTICS params: min_samples={best_params[0]}, min_cluster_size={best_params[1]:.2f}, avg ARI={best_score:.3f}")
    return best_params


# ============================================================
# SECTION 6: MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiments_for_type(datasets, data_type, dr_methods=['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS'],
                             n_hp_search_datasets=20):
    """
    Run all experiments for a given data type.
    
    datasets: list of (X, y, config_str) or (X, y, name, k)
    data_type: 'Circles', 'Moons', 'RSG', 'Repliclust', 'RealWorld'
    
    Returns: DataFrame with all ARI results
    """
    print(f"\n{'='*60}")
    print(f"Running experiments for: {data_type}")
    print(f"Number of datasets: {len(datasets)}")
    print(f"{'='*60}")
    
    # Prepare datasets
    prepared = []
    for item in datasets:
        if len(item) == 4:
            X, y, name, k = item
        else:
            X, y, config = item
            name = config
            k = len(np.unique(y))
        prepared.append((X, y, name, k))
    
    # Step 1: Z-score normalize all datasets
    print("\nStep 1: Z-score normalization...")
    normalized = []
    for X, y, name, k in prepared:
        X_norm = StandardScaler().fit_transform(X)
        normalized.append((X_norm, y, name, k))
    
    # Step 2: Find best hyperparameters for AHC, GMM, OPTICS
    print("\nStep 2: Hyperparameter search...")
    # Use a subset for HP search to save time
    hp_datasets = normalized[:min(n_hp_search_datasets, len(normalized))]
    hp_data = [(X, y, k) for X, y, name, k in hp_datasets]
    
    ahc_params = find_best_ahc_params(hp_data)
    gmm_cov_type = find_best_gmm_params(hp_data)
    optics_params = find_best_optics_params(hp_data)
    
    # Step 3: Run clustering on all datasets with all DR methods
    print("\nStep 3: Running experiments...")
    
    all_results = []  # List of dicts
    
    for di, (X, y, name, k) in enumerate(normalized):
        if di % 10 == 0:
            print(f"  Dataset {di+1}/{len(normalized)}: {name} (n={X.shape[0]}, d={X.shape[1]}, k={k})")
        
        d = X.shape[1]
        levels = compute_reduction_levels(d, k)
        
        # Clustering on unreduced data
        for algo_name in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            try:
                if algo_name == 'k-means':
                    pred = run_kmeans(X, k)
                elif algo_name == 'AHC':
                    pred = run_ahc(X, k, affinity=ahc_params[0], linkage=ahc_params[1])
                elif algo_name == 'GMM':
                    pred = run_gmm(X, k, covariance_type=gmm_cov_type)
                elif algo_name == 'OPTICS':
                    pred = run_optics(X, min_samples=optics_params[0], 
                                     min_cluster_size=optics_params[1])
                
                if pred is not None:
                    ari = adjusted_rand_score(y, pred)
                else:
                    ari = np.nan
            except:
                ari = np.nan
            
            all_results.append({
                'dataset': name,
                'algorithm': algo_name,
                'dr_method': 'No Reduction',
                'dr_level': 'none',
                'n_components': d,
                'ari': ari
            })
        
        # DR + Clustering
        for dr_method in dr_methods:
            for level_name, n_comp in levels.items():
                if n_comp >= d:
                    # Skip if reduction doesn't actually reduce
                    for algo_name in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                        all_results.append({
                            'dataset': name,
                            'algorithm': algo_name,
                            'dr_method': dr_method,
                            'dr_level': level_name,
                            'n_components': n_comp,
                            'ari': np.nan
                        })
                    continue
                
                try:
                    X_dr = apply_dr(X, dr_method, n_comp, random_state=42)
                except Exception as e:
                    X_dr = None
                
                if X_dr is None:
                    for algo_name in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                        all_results.append({
                            'dataset': name,
                            'algorithm': algo_name,
                            'dr_method': dr_method,
                            'dr_level': level_name,
                            'n_components': n_comp,
                            'ari': np.nan
                        })
                    continue
                
                for algo_name in ['k-means', 'AHC', 'GMM', 'OPTICS']:
                    try:
                        if algo_name == 'k-means':
                            pred = run_kmeans(X_dr, k)
                        elif algo_name == 'AHC':
                            pred = run_ahc(X_dr, k, affinity=ahc_params[0], linkage=ahc_params[1])
                        elif algo_name == 'GMM':
                            pred = run_gmm(X_dr, k, covariance_type=gmm_cov_type)
                        elif algo_name == 'OPTICS':
                            pred = run_optics(X_dr, min_samples=optics_params[0],
                                             min_cluster_size=optics_params[1])
                        
                        if pred is not None:
                            ari = adjusted_rand_score(y, pred)
                        else:
                            ari = np.nan
                    except:
                        ari = np.nan
                    
                    all_results.append({
                        'dataset': name,
                        'algorithm': algo_name,
                        'dr_method': dr_method,
                        'dr_level': level_name,
                        'n_components': n_comp,
                        'ari': ari
                    })
    
    df = pd.DataFrame(all_results)
    
    # Save raw results
    df.to_csv(f'{RESULTS_DIR}/raw_results_{data_type}.csv', index=False)
    
    return df, {'ahc_params': ahc_params, 'gmm_cov_type': gmm_cov_type, 'optics_params': optics_params}


# ============================================================
# SECTION 7: TABLE GENERATION
# ============================================================

def generate_average_ari_table(df, data_type):
    """Generate average ARI table matching paper format.
    Rows: algorithms
    Columns: No Reduction, PCA(k-1, 25%, 50%), KPCA(...), VAE(...), Isomap(...), MDS(...)
    """
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    columns = ['No Reduction']
    for dr in dr_methods:
        for lev in levels:
            columns.append(f'{dr}_{lev}')
    
    table = pd.DataFrame(index=algos, columns=columns)
    
    for algo in algos:
        # No reduction
        mask = (df['algorithm'] == algo) & (df['dr_method'] == 'No Reduction')
        vals = df.loc[mask, 'ari'].dropna()
        table.loc[algo, 'No Reduction'] = round(vals.mean(), 3) if len(vals) > 0 else np.nan
        
        # DR methods
        for dr in dr_methods:
            for lev in levels:
                mask = (df['algorithm'] == algo) & (df['dr_method'] == dr) & (df['dr_level'] == lev)
                vals = df.loc[mask, 'ari'].dropna()
                col = f'{dr}_{lev}'
                table.loc[algo, col] = round(vals.mean(), 3) if len(vals) > 0 else np.nan
    
    table.to_csv(f'{RESULTS_DIR}/table_average_ARI_{data_type}.csv')
    print(f"\nAverage ARI table for {data_type}:")
    print(table.to_string())
    return table

def generate_per_dataset_table(df, data_type, algo):
    """Generate per-dataset ARI table for real-world data."""
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    columns = ['No Reduction']
    for dr in dr_methods:
        for lev in levels:
            columns.append(f'{dr}_{lev}')
    
    datasets = df['dataset'].unique()
    table = pd.DataFrame(index=datasets, columns=columns)
    
    for ds in datasets:
        # No reduction
        mask = (df['dataset'] == ds) & (df['algorithm'] == algo) & (df['dr_method'] == 'No Reduction')
        vals = df.loc[mask, 'ari'].dropna()
        table.loc[ds, 'No Reduction'] = round(vals.mean(), 2) if len(vals) > 0 else np.nan
        
        for dr in dr_methods:
            for lev in levels:
                mask = (df['dataset'] == ds) & (df['algorithm'] == algo) & \
                       (df['dr_method'] == dr) & (df['dr_level'] == lev)
                vals = df.loc[mask, 'ari'].dropna()
                col = f'{dr}_{lev}'
                table.loc[ds, col] = round(vals.mean(), 2) if len(vals) > 0 else np.nan
    
    table.to_csv(f'{RESULTS_DIR}/table_{algo}_{data_type}.csv')
    return table


# ============================================================
# SECTION 8: BOXPLOTS
# ============================================================

def generate_boxplots(df, data_type):
    """Generate boxplots for each clustering algorithm."""
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    for algo in algos:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        data_to_plot = []
        labels = []
        
        # No reduction
        mask = (df['algorithm'] == algo) & (df['dr_method'] == 'No Reduction')
        vals = df.loc[mask, 'ari'].dropna().values
        if len(vals) > 0:
            data_to_plot.append(vals)
            labels.append('No Red.')
        
        for dr in dr_methods:
            for lev in levels:
                mask = (df['algorithm'] == algo) & (df['dr_method'] == dr) & (df['dr_level'] == lev)
                vals = df.loc[mask, 'ari'].dropna().values
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    labels.append(f'{dr}\n{lev}')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color by DR method
            colors = ['gray'] + ['#1f77b4']*3 + ['#ff7f0e']*3 + ['#2ca02c']*3 + ['#d62728']*3 + ['#9467bd']*3
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f'{algo} - {data_type}')
        ax.set_ylabel('ARI')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/boxplot_{algo}_{data_type}.pdf', bbox_inches='tight')
        plt.close()
    
    print(f"  Boxplots saved for {data_type}")


# ============================================================
# SECTION 9: AGGREGATE ANALYSIS
# ============================================================

def compute_aggregate_tables(df, data_type):
    """Compute win rates and average win/loss ARI changes."""
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    results = {}
    
    for algo in algos:
        algo_results = {}
        
        for dr in dr_methods:
            for lev in levels:
                key = f'{dr}_{lev}'
                
                # Get paired comparisons
                datasets = df['dataset'].unique()
                wins, losses, ties = 0, 0, 0
                win_deltas, loss_deltas = [], []
                
                for ds in datasets:
                    # No reduction ARI
                    mask_nr = (df['dataset'] == ds) & (df['algorithm'] == algo) & (df['dr_method'] == 'No Reduction')
                    nr_vals = df.loc[mask_nr, 'ari'].dropna()
                    
                    # DR ARI
                    mask_dr = (df['dataset'] == ds) & (df['algorithm'] == algo) & \
                              (df['dr_method'] == dr) & (df['dr_level'] == lev)
                    dr_vals = df.loc[mask_dr, 'ari'].dropna()
                    
                    if len(nr_vals) > 0 and len(dr_vals) > 0:
                        nr_ari = nr_vals.mean()
                        dr_ari = dr_vals.mean()
                        delta = dr_ari - nr_ari
                        
                        if delta > 0.001:
                            wins += 1
                            win_deltas.append(delta)
                        elif delta < -0.001:
                            losses += 1
                            loss_deltas.append(delta)
                        else:
                            ties += 1
                
                total = wins + losses + ties
                algo_results[key] = {
                    'win_pct': wins / total * 100 if total > 0 else 0,
                    'loss_pct': losses / total * 100 if total > 0 else 0,
                    'tie_pct': ties / total * 100 if total > 0 else 0,
                    'avg_win': np.mean(win_deltas) if win_deltas else 0,
                    'avg_loss': np.mean(loss_deltas) if loss_deltas else 0,
                }
        
        results[algo] = algo_results
    
    # Save as CSV
    for algo in algos:
        rows = []
        for dr in dr_methods:
            for lev in levels:
                key = f'{dr}_{lev}'
                r = results[algo][key]
                rows.append({
                    'DR_Method': dr,
                    'Level': lev,
                    'Win%': f"{r['win_pct']:.1f}",
                    'Loss%': f"{r['loss_pct']:.1f}",
                    'Tie%': f"{r['tie_pct']:.1f}",
                    'Avg_Win_Delta': f"{r['avg_win']:.3f}",
                    'Avg_Loss_Delta': f"{r['avg_loss']:.3f}",
                })
        
        agg_df = pd.DataFrame(rows)
        agg_df.to_csv(f'{RESULTS_DIR}/aggregate_{algo}_{data_type}.csv', index=False)
    
    return results


# ============================================================
# SECTION 10: WILCOXON TESTS
# ============================================================

def run_wilcoxon_tests(df, data_type):
    """Run Wilcoxon signed-rank tests comparing DR vs No Reduction."""
    from scipy.stats import wilcoxon
    
    algos = ['k-means', 'AHC', 'GMM', 'OPTICS']
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    results = []
    
    for algo in algos:
        for dr in dr_methods:
            for lev in levels:
                datasets = df['dataset'].unique()
                nr_scores, dr_scores = [], []
                
                for ds in datasets:
                    mask_nr = (df['dataset'] == ds) & (df['algorithm'] == algo) & (df['dr_method'] == 'No Reduction')
                    mask_dr = (df['dataset'] == ds) & (df['algorithm'] == algo) & \
                              (df['dr_method'] == dr) & (df['dr_level'] == lev)
                    
                    nr_vals = df.loc[mask_nr, 'ari'].dropna()
                    dr_vals = df.loc[mask_dr, 'ari'].dropna()
                    
                    if len(nr_vals) > 0 and len(dr_vals) > 0:
                        nr_scores.append(nr_vals.mean())
                        dr_scores.append(dr_vals.mean())
                
                if len(nr_scores) >= 5:
                    try:
                        stat, pval = wilcoxon(dr_scores, nr_scores, alternative='two-sided')
                        results.append({
                            'Algorithm': algo,
                            'DR_Method': dr,
                            'Level': lev,
                            'W_statistic': stat,
                            'p_value': pval,
                            'significant': pval < 0.05,
                            'mean_diff': np.mean(np.array(dr_scores) - np.array(nr_scores))
                        })
                    except:
                        pass
    
    wilcoxon_df = pd.DataFrame(results)
    wilcoxon_df.to_csv(f'{RESULTS_DIR}/wilcoxon_{data_type}.csv', index=False)
    print(f"\nWilcoxon test results for {data_type}:")
    print(wilcoxon_df.to_string())
    return wilcoxon_df


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['real', 'circles', 'moons', 'rsg', 'repliclust', 'all'],
                       default='all')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of repetitions per synthetic config')
    args = parser.parse_args()
    
    if args.type in ['real', 'all']:
        print("\n" + "="*60)
        print("REAL-WORLD EXPERIMENTS")
        print("="*60)
        
        real_datasets = load_all_real_world()
        print(f"Loaded {len(real_datasets)} real-world datasets")
        
        df_real, params_real = run_experiments_for_type(real_datasets, 'RealWorld')
        
        # Generate tables
        generate_average_ari_table(df_real, 'RealWorld')
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            generate_per_dataset_table(df_real, 'RealWorld', algo)
        
        # Generate boxplots
        generate_boxplots(df_real, 'RealWorld')
        
        # Aggregate analysis
        compute_aggregate_tables(df_real, 'RealWorld')
        
        # Wilcoxon tests
        run_wilcoxon_tests(df_real, 'RealWorld')
    
    if args.type in ['circles', 'moons', 'rsg', 'repliclust', 'all']:
        print("\n" + "="*60)
        print("SYNTHETIC DATA GENERATION")
        print("="*60)
        
        all_synth = generate_all_synthetic(n_reps=args.n_reps)
        
        type_map = {
            'circles': 'Circles',
            'moons': 'Moons',
            'rsg': 'RSG',
            'repliclust': 'Repliclust'
        }
        
        for key, dtype in type_map.items():
            if args.type in [key, 'all']:
                datasets = all_synth[dtype]
                if len(datasets) == 0:
                    print(f"No datasets for {dtype}, skipping")
                    continue
                
                df_synth, params = run_experiments_for_type(datasets, dtype)
                generate_average_ari_table(df_synth, dtype)
                generate_boxplots(df_synth, dtype)
                compute_aggregate_tables(df_synth, dtype)
                run_wilcoxon_tests(df_synth, dtype)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
