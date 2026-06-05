"""
Main experiment pipeline for DR+clustering paper replication.
Runs 5 DR methods × 4 clustering algorithms × 3 reduction levels on real-world + synthetic data.
"""
import numpy as np
import warnings
import json
import os
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from itertools import product

warnings.filterwarnings('ignore')

# ============================================================
# VAE Implementation
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 64)
        self.enc_bn1 = nn.BatchNorm1d(64)
        self.enc_fc2 = nn.Linear(64, 32)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.enc_drop = nn.Dropout(0.4)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 32)
        self.dec_bn1 = nn.BatchNorm1d(32)
        self.dec_fc2 = nn.Linear(32, 64)
        self.dec_bn2 = nn.BatchNorm1d(64)
        self.dec_drop = nn.Dropout(0.4)
        self.dec_out = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h = torch.relu(self.enc_bn1(self.enc_fc1(x)))
        h = self.enc_drop(h)
        h = torch.relu(self.enc_bn2(self.enc_fc2(h)))
        h = self.enc_drop(h)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.dec_bn1(self.dec_fc1(z)))
        h = self.dec_drop(h)
        h = torch.relu(self.dec_bn2(self.dec_fc2(h)))
        h = self.dec_drop(h)
        return torch.sigmoid(self.dec_out(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def train_vae(X, latent_dim, epochs=100, batch_size=64, device='cpu'):
    """Train VAE and return latent embeddings (z_mean)."""
    n_samples, input_dim = X.shape
    
    # Min-max scale to [0,1] for sigmoid output
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_scaled = (X - X_min) / X_range
    
    # 70/30 split
    n_train = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    
    X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
    X_all = torch.FloatTensor(X_scaled).to(device)
    
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i:i+batch_size]]
            if len(batch) < 2:  # BatchNorm needs at least 2
                continue
            recon, mu, logvar = model(batch)
            # MSE reconstruction loss + KL divergence
            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Extract z_mean for ALL data
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_all)
    return mu.cpu().numpy()


# ============================================================
# Dimensionality Reduction
# ============================================================
def apply_dr(X, method, n_components, random_state=42):
    """Apply dimensionality reduction. Returns reduced X or None on failure."""
    n_samples, n_features = X.shape
    n_components = max(2, min(n_components, n_features - 1))
    
    if n_components >= n_features:
        return None  # No reduction needed/possible
    
    try:
        if method == 'PCA':
            dr = PCA(n_components=n_components)
            return dr.fit_transform(X)
        elif method == 'KernelPCA':
            dr = KernelPCA(n_components=n_components, kernel='rbf')
            result = dr.fit_transform(X)
            # KernelPCA can sometimes return fewer components
            if result.shape[1] < n_components:
                return None
            return result
        elif method == 'VAE':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return train_vae(X, n_components, device=device)
        elif method == 'Isomap':
            dr = Isomap(n_components=n_components)
            return dr.fit_transform(X)
        elif method == 'MDS':
            dr = MDS(n_components=n_components, random_state=10, n_init=50, normalized_stress='auto')
            return dr.fit_transform(X)
        else:
            raise ValueError(f"Unknown DR method: {method}")
    except Exception as e:
        print(f"    DR {method} n_comp={n_components} failed: {e}")
        return None


def get_reduction_dims(n_features, k):
    """Get the 3 reduction levels: k-1, 25%, 50%."""
    km1 = max(2, k - 1)
    pct25 = max(2, round(n_features * 0.25))
    pct50 = max(2, round(n_features * 0.50))
    return {'k-1': km1, '25%': pct25, '50%': pct50}


# ============================================================
# Clustering
# ============================================================
def run_kmeans(X, k):
    """k-means with k-means++ init, n_init=100."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
    return km.fit_predict(X)


def run_ahc(X, k, affinity='euclidean', linkage='ward'):
    """Agglomerative Hierarchical Clustering."""
    # Ward only works with euclidean
    if linkage == 'ward' and affinity != 'euclidean':
        return None
    try:
        ahc = AgglomerativeClustering(n_clusters=k, metric=affinity, linkage=linkage)
        return ahc.fit_predict(X)
    except Exception:
        return None


def run_gmm(X, k, covariance_type='full'):
    """Gaussian Mixture Model."""
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        return gmm.fit_predict(X)
    except Exception:
        return None


def run_optics(X, min_samples=5, min_cluster_size=0.05):
    """OPTICS with xi method."""
    try:
        optics = OPTICS(min_samples=min_samples, xi=0.05, cluster_method='xi',
                        min_cluster_size=min_cluster_size if min_cluster_size > 0 else None)
        labels = optics.fit_predict(X)
        return labels
    except Exception:
        return None


# ============================================================
# Hyperparameter Search for AHC, GMM, OPTICS
# ============================================================
def find_best_ahc_params(datasets_with_labels):
    """Find best (affinity, linkage) across a set of datasets by average ARI."""
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    best_score = -999
    best_params = ('euclidean', 'ward')
    
    for aff, link in product(affinities, linkages):
        if link == 'ward' and aff != 'euclidean':
            continue
        scores = []
        for X, y, k in datasets_with_labels:
            labels = run_ahc(X, k, affinity=aff, linkage=link)
            if labels is not None:
                scores.append(adjusted_rand_score(y, labels))
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_params = (aff, link)
    
    return best_params, best_score


def find_best_gmm_params(datasets_with_labels):
    """Find best covariance_type across a set of datasets by average ARI."""
    cov_types = ['spherical', 'tied', 'diag', 'full']
    
    best_score = -999
    best_cov = 'full'
    
    for cov in cov_types:
        scores = []
        for X, y, k in datasets_with_labels:
            labels = run_gmm(X, k, covariance_type=cov)
            if labels is not None:
                scores.append(adjusted_rand_score(y, labels))
        if scores:
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_cov = cov
    
    return best_cov, best_score


def find_best_optics_params(datasets_with_labels):
    """Find best (min_samples, min_cluster_size) across datasets by average ARI."""
    min_samples_range = range(5, 11)
    min_cluster_size_range = np.arange(0.0, 1.05, 0.05)
    
    best_score = -999
    best_params = (5, 0.05)
    
    for ms in min_samples_range:
        for mcs in min_cluster_size_range:
            scores = []
            for X, y, k in datasets_with_labels:
                labels = run_optics(X, min_samples=ms, min_cluster_size=mcs if mcs > 0 else None)
                if labels is not None:
                    scores.append(adjusted_rand_score(y, labels))
            if scores:
                avg = np.mean(scores)
                if avg > best_score:
                    best_score = avg
                    best_params = (ms, round(mcs, 2))
    
    return best_params, best_score


# ============================================================
# Main Experiment Runner
# ============================================================
def run_real_world_experiments():
    """Run all experiments on 20 UCI real-world datasets."""
    from load_uci import load_all_uci
    
    print("=" * 80)
    print("REAL-WORLD EXPERIMENTS")
    print("=" * 80)
    
    datasets = load_all_uci()
    dr_methods = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
    
    # Step 1: Z-score normalize all datasets
    print("\nStep 1: Z-score normalization...")
    normalized_datasets = []
    for name, X, y, k in datasets:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        normalized_datasets.append((name, X_norm, y, k))
    
    # Step 2: Find best hyperparameters for AHC, GMM, OPTICS on unreduced data
    print("\nStep 2: Finding best hyperparameters for AHC, GMM, OPTICS...")
    datasets_for_search = [(X, y, k) for name, X, y, k in normalized_datasets]
    
    print("  Searching AHC params...")
    best_ahc_params, ahc_score = find_best_ahc_params(datasets_for_search)
    print(f"  Best AHC: affinity={best_ahc_params[0]}, linkage={best_ahc_params[1]}, avg ARI={ahc_score:.3f}")
    
    print("  Searching GMM params...")
    best_gmm_cov, gmm_score = find_best_gmm_params(datasets_for_search)
    print(f"  Best GMM: covariance_type={best_gmm_cov}, avg ARI={gmm_score:.3f}")
    
    print("  Searching OPTICS params...")
    best_optics_params, optics_score = find_best_optics_params(datasets_for_search)
    print(f"  Best OPTICS: min_samples={best_optics_params[0]}, min_cluster_size={best_optics_params[1]}, avg ARI={optics_score:.3f}")
    
    # Step 3: Run all experiments
    print("\nStep 3: Running experiments...")
    
    # Results structure: results[dataset_name][algo][dr_method][reduction_level] = ARI
    results = {}
    
    for name, X_norm, y, k in normalized_datasets:
        print(f"\n  Dataset: {name} (n={X_norm.shape[0]}, d={X_norm.shape[1]}, k={k})")
        results[name] = {}
        
        reduction_dims = get_reduction_dims(X_norm.shape[1], k)
        
        # --- No reduction baseline ---
        print(f"    No reduction...")
        results[name]['kmeans'] = {'None': round(adjusted_rand_score(y, run_kmeans(X_norm, k)), 2)}
        
        ahc_labels = run_ahc(X_norm, k, best_ahc_params[0], best_ahc_params[1])
        results[name]['ahc'] = {'None': round(adjusted_rand_score(y, ahc_labels), 2) if ahc_labels is not None else 0.0}
        
        gmm_labels = run_gmm(X_norm, k, best_gmm_cov)
        results[name]['gmm'] = {'None': round(adjusted_rand_score(y, gmm_labels), 2) if gmm_labels is not None else 0.0}
        
        optics_labels = run_optics(X_norm, best_optics_params[0], best_optics_params[1])
        results[name]['optics'] = {'None': round(adjusted_rand_score(y, optics_labels), 2) if optics_labels is not None else 0.0}
        
        print(f"      Baseline: kmeans={results[name]['kmeans']['None']}, ahc={results[name]['ahc']['None']}, gmm={results[name]['gmm']['None']}, optics={results[name]['optics']['None']}")
        
        # --- With DR ---
        for dr_method in dr_methods:
            for red_name, n_comp in reduction_dims.items():
                if n_comp >= X_norm.shape[1]:
                    # Skip if reduction doesn't actually reduce
                    for algo in ['kmeans', 'ahc', 'gmm', 'optics']:
                        key = f"{dr_method}_{red_name}"
                        results[name][algo][key] = results[name][algo]['None']
                    continue
                
                print(f"    {dr_method} -> {n_comp}d ({red_name})...", end=" ")
                X_red = apply_dr(X_norm, dr_method, n_comp)
                
                if X_red is None:
                    print("FAILED")
                    for algo in ['kmeans', 'ahc', 'gmm', 'optics']:
                        key = f"{dr_method}_{red_name}"
                        results[name][algo][key] = None
                    continue
                
                key = f"{dr_method}_{red_name}"
                
                # k-means
                km_ari = round(adjusted_rand_score(y, run_kmeans(X_red, k)), 2)
                results[name]['kmeans'][key] = km_ari
                
                # AHC
                ahc_labels = run_ahc(X_red, k, best_ahc_params[0], best_ahc_params[1])
                results[name]['ahc'][key] = round(adjusted_rand_score(y, ahc_labels), 2) if ahc_labels is not None else None
                
                # GMM
                gmm_labels = run_gmm(X_red, k, best_gmm_cov)
                results[name]['gmm'][key] = round(adjusted_rand_score(y, gmm_labels), 2) if gmm_labels is not None else None
                
                # OPTICS
                optics_labels = run_optics(X_red, best_optics_params[0], best_optics_params[1])
                results[name]['optics'][key] = round(adjusted_rand_score(y, optics_labels), 2) if optics_labels is not None else None
                
                print(f"km={km_ari}, ahc={results[name]['ahc'][key]}, gmm={results[name]['gmm'][key]}, optics={results[name]['optics'][key]}")
    
    return results, {
        'ahc_params': best_ahc_params,
        'gmm_cov': best_gmm_cov,
        'optics_params': best_optics_params
    }


def run_synthetic_experiments():
    """Run all experiments on synthetic datasets."""
    print("=" * 80)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("=" * 80)
    
    # First generate data
    print("\nGenerating synthetic data...")
    from generate_data import main as generate_main
    generate_main()
    
    DATA_DIR = "/workspace/synthetic_data"
    dr_methods = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
    
    # Load all synthetic datasets grouped by type
    dataset_types = {
        'Circles': [],
        'Moons': [],
        'RSG': [],
        'Repliclust': []
    }
    
    # Load Circles
    for k_type in ['2cluster', '5cluster']:
        for d in [10, 50, 200]:
            config_dir = os.path.join(DATA_DIR, f"circles_{k_type}_d{d}")
            if os.path.exists(config_dir):
                for f in sorted(os.listdir(config_dir)):
                    data = np.load(os.path.join(config_dir, f))
                    k = 2 if k_type == '2cluster' else 5
                    dataset_types['Circles'].append((f"circles_{k_type}_d{d}", data['X'], data['y'], k))
    
    # Load Moons
    for k_type in ['2cluster', '5cluster']:
        for d in [10, 50, 200]:
            config_dir = os.path.join(DATA_DIR, f"moons_{k_type}_d{d}")
            if os.path.exists(config_dir):
                for f in sorted(os.listdir(config_dir)):
                    data = np.load(os.path.join(config_dir, f))
                    k = 2 if k_type == '2cluster' else 5
                    dataset_types['Moons'].append((f"moons_{k_type}_d{d}", data['X'], data['y'], k))
    
    # Load RSG
    rsg_dir = os.path.join(DATA_DIR, "rsg")
    if os.path.exists(rsg_dir):
        for f in sorted(os.listdir(rsg_dir)):
            data = np.load(os.path.join(rsg_dir, f))
            k = len(np.unique(data['y']))
            dataset_types['RSG'].append((f"rsg_{f}", data['X'], data['y'], k))
    
    # Load Repliclust
    for k_type in ['2cluster', '5cluster']:
        for d in [10, 50, 200]:
            config_dir = os.path.join(DATA_DIR, f"repliclust_{k_type}_d{d}")
            if os.path.exists(config_dir):
                for f in sorted(os.listdir(config_dir)):
                    data = np.load(os.path.join(config_dir, f))
                    k = 2 if k_type == '2cluster' else 5
                    dataset_types['Repliclust'].append((f"repliclust_{k_type}_d{d}", data['X'], data['y'], k))
    
    all_synth_results = {}
    all_synth_params = {}
    
    for dtype, dsets in dataset_types.items():
        if not dsets:
            print(f"\n  No {dtype} datasets found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dtype}: {len(dsets)} datasets")
        print(f"{'='*60}")
        
        # Z-score normalize
        norm_dsets = []
        for name, X, y, k in dsets:
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
            norm_dsets.append((name, X_norm, y, k))
        
        # Find best hyperparameters for this type
        print(f"  Finding best AHC params for {dtype}...")
        search_data = [(X, y, k) for _, X, y, k in norm_dsets[:50]]  # Use subset for speed
        best_ahc, _ = find_best_ahc_params(search_data)
        print(f"    Best AHC: {best_ahc}")
        
        print(f"  Finding best GMM params for {dtype}...")
        best_gmm, _ = find_best_gmm_params(search_data)
        print(f"    Best GMM: {best_gmm}")
        
        print(f"  Finding best OPTICS params for {dtype}...")
        best_optics, _ = find_best_optics_params(search_data)
        print(f"    Best OPTICS: {best_optics}")
        
        all_synth_params[dtype] = {
            'ahc': best_ahc,
            'gmm': best_gmm,
            'optics': best_optics
        }
        
        # Run experiments on all datasets of this type
        type_results = {'kmeans': {}, 'ahc': {}, 'gmm': {}, 'optics': {}}
        # Store per-dataset results for aggregate analysis
        per_dataset_results = []
        
        for idx, (name, X_norm, y, k) in enumerate(norm_dsets):
            if idx % 50 == 0:
                print(f"  Processing dataset {idx+1}/{len(norm_dsets)}...")
            
            reduction_dims = get_reduction_dims(X_norm.shape[1], k)
            ds_result = {}
            
            # Baseline
            ds_result['kmeans_None'] = adjusted_rand_score(y, run_kmeans(X_norm, k))
            
            ahc_labels = run_ahc(X_norm, k, best_ahc[0], best_ahc[1])
            ds_result['ahc_None'] = adjusted_rand_score(y, ahc_labels) if ahc_labels is not None else 0.0
            
            gmm_labels = run_gmm(X_norm, k, best_gmm)
            ds_result['gmm_None'] = adjusted_rand_score(y, gmm_labels) if gmm_labels is not None else 0.0
            
            optics_labels = run_optics(X_norm, best_optics[0], best_optics[1])
            ds_result['optics_None'] = adjusted_rand_score(y, optics_labels) if optics_labels is not None else 0.0
            
            # With DR
            for dr_method in dr_methods:
                for red_name, n_comp in reduction_dims.items():
                    if n_comp >= X_norm.shape[1]:
                        for algo in ['kmeans', 'ahc', 'gmm', 'optics']:
                            ds_result[f'{algo}_{dr_method}_{red_name}'] = ds_result[f'{algo}_None']
                        continue
                    
                    X_red = apply_dr(X_norm, dr_method, n_comp)
                    if X_red is None:
                        for algo in ['kmeans', 'ahc', 'gmm', 'optics']:
                            ds_result[f'{algo}_{dr_method}_{red_name}'] = None
                        continue
                    
                    ds_result[f'kmeans_{dr_method}_{red_name}'] = adjusted_rand_score(y, run_kmeans(X_red, k))
                    
                    ahc_labels = run_ahc(X_red, k, best_ahc[0], best_ahc[1])
                    ds_result[f'ahc_{dr_method}_{red_name}'] = adjusted_rand_score(y, ahc_labels) if ahc_labels is not None else None
                    
                    gmm_labels = run_gmm(X_red, k, best_gmm)
                    ds_result[f'gmm_{dr_method}_{red_name}'] = adjusted_rand_score(y, gmm_labels) if gmm_labels is not None else None
                    
                    optics_labels = run_optics(X_red, best_optics[0], best_optics[1])
                    ds_result[f'optics_{dr_method}_{red_name}'] = adjusted_rand_score(y, optics_labels) if optics_labels is not None else None
            
            per_dataset_results.append(ds_result)
        
        # Compute averages
        all_synth_results[dtype] = compute_synthetic_averages(per_dataset_results, dr_methods)
    
    return all_synth_results, all_synth_params


def compute_synthetic_averages(per_dataset_results, dr_methods):
    """Compute average ARI across all datasets for each algo/DR/level combo."""
    algos = ['kmeans', 'ahc', 'gmm', 'optics']
    levels = ['k-1', '25%', '50%']
    
    averages = {}
    for algo in algos:
        averages[algo] = {}
        # Baseline
        vals = [r[f'{algo}_None'] for r in per_dataset_results if r.get(f'{algo}_None') is not None]
        averages[algo]['None'] = round(np.mean(vals), 3) if vals else 0.0
        
        for dr in dr_methods:
            for level in levels:
                key = f'{algo}_{dr}_{level}'
                vals = [r[key] for r in per_dataset_results if r.get(key) is not None]
                averages[algo][f'{dr}_{level}'] = round(np.mean(vals), 3) if vals else None
    
    return averages


# ============================================================
# Results Formatting and Output
# ============================================================
def format_real_world_table(results, algo_key, algo_name):
    """Format results as a table matching paper format."""
    dr_methods = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    header = f"\n{'='*120}\n{algo_name} - Real-World Results\n{'='*120}\n"
    header += f"{'Dataset':<20} {'NoRed':>6}"
    for dr in dr_methods:
        for lv in levels:
            header += f" {dr[:4]}_{lv:>3}".rjust(8)
    header += "\n" + "-" * 120
    
    lines = [header]
    for dname in results:
        algo_results = results[dname].get(algo_key, {})
        baseline = algo_results.get('None', '-')
        line = f"{dname:<20} {baseline:>6}"
        for dr in dr_methods:
            for lv in levels:
                key = f"{dr}_{lv}"
                val = algo_results.get(key, '-')
                if val is None:
                    val = '-'
                line += f" {val:>7}"
        lines.append(line)
    
    return "\n".join(lines)


def compute_aggregate_stats(results, algo_key, is_synthetic=False):
    """Compute % wins and avg win/loss for aggregate tables."""
    dr_methods = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    stats = {}
    for dr in dr_methods:
        stats[dr] = {}
        for lv in levels:
            key = f"{dr}_{lv}"
            wins = 0
            total = 0
            diffs = []
            
            for dname in results:
                algo_results = results[dname].get(algo_key, {})
                baseline = algo_results.get('None')
                reduced = algo_results.get(key)
                
                if baseline is not None and reduced is not None:
                    total += 1
                    diff = reduced - baseline
                    diffs.append(diff)
                    if reduced > baseline:
                        wins += 1
            
            win_pct = (wins / total * 100) if total > 0 else 0
            avg_diff = (np.mean(diffs) * 100) if diffs else 0  # as percentage
            stats[dr][lv] = {
                'win_pct': round(win_pct, 2),
                'avg_diff': round(avg_diff, 2),
                'n_total': total
            }
    
    return stats


def run_wilcoxon_test(results, algo_key):
    """Run Wilcoxon signed-rank test for each DR method/level vs baseline."""
    from scipy.stats import wilcoxon
    
    dr_methods = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
    levels = ['k-1', '25%', '50%']
    
    pvalues = {}
    for dr in dr_methods:
        pvalues[dr] = {}
        for lv in levels:
            key = f"{dr}_{lv}"
            baselines = []
            reduced_vals = []
            
            for dname in results:
                algo_results = results[dname].get(algo_key, {})
                b = algo_results.get('None')
                r = algo_results.get(key)
                if b is not None and r is not None:
                    baselines.append(b)
                    reduced_vals.append(r)
            
            if len(baselines) >= 5:
                try:
                    # One-sided: H1: reduced > baseline
                    stat, p = wilcoxon(reduced_vals, baselines, alternative='greater')
                    pvalues[dr][lv] = round(p, 3)
                except Exception:
                    pvalues[dr][lv] = 1.0
            else:
                pvalues[dr][lv] = None
    
    return pvalues


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.makedirs("/workspace/results", exist_ok=True)
    
    # Run real-world experiments
    t0 = time.time()
    real_results, real_params = run_real_world_experiments()
    t1 = time.time()
    print(f"\nReal-world experiments took {t1-t0:.1f}s")
    
    # Save results
    with open("/workspace/results/real_world_results.json", 'w') as f:
        json.dump({'results': real_results, 'params': {
            'ahc': list(real_params['ahc_params']),
            'gmm': real_params['gmm_cov'],
            'optics': list(real_params['optics_params'])
        }}, f, indent=2, default=str)
    
    print("\nResults saved to /workspace/results/real_world_results.json")
    
    # Print tables
    for algo_key, algo_name in [('kmeans', 'k-means'), ('ahc', 'AHC'), ('gmm', 'GMM'), ('optics', 'OPTICS')]:
        print(format_real_world_table(real_results, algo_key, algo_name))
    
    # Compute aggregate stats
    print("\n\n" + "=" * 80)
    print("AGGREGATE STATISTICS - REAL-WORLD DATA")
    print("=" * 80)
    for algo_key, algo_name in [('kmeans', 'k-means'), ('ahc', 'AHC'), ('gmm', 'GMM'), ('optics', 'OPTICS')]:
        stats = compute_aggregate_stats(real_results, algo_key)
        print(f"\n{algo_name}:")
        for dr in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
            for lv in ['k-1', '25%', '50%']:
                s = stats[dr][lv]
                print(f"  {dr:>10} {lv:>4}: Win%={s['win_pct']:>6.1f}, AvgDiff={s['avg_diff']:>7.2f}%")
    
    # Wilcoxon test
    print("\n\n" + "=" * 80)
    print("WILCOXON SIGNED-RANK TEST - REAL-WORLD DATA")
    print("=" * 80)
    for algo_key, algo_name in [('kmeans', 'k-means'), ('ahc', 'AHC'), ('gmm', 'GMM'), ('optics', 'OPTICS')]:
        pvals = run_wilcoxon_test(real_results, algo_key)
        print(f"\n{algo_name}:")
        for dr in ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']:
            vals = [f"{pvals[dr][lv]}" for lv in ['k-1', '25%', '50%']]
            print(f"  {dr:>10}: {', '.join(vals)}")
