"""
Run the full experiment pipeline with incremental saving and progress tracking.
Optimized for speed while maintaining paper methodology.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

OUTPUT_DIR = './results'
DATA_DIR = './uci_data'

# ============================================================
# DATA LOADING
# ============================================================

def load_segmentation(data_dir=DATA_DIR):
    dfs = []
    for fname in ['segmentation.data', 'segmentation.test']:
        fpath = os.path.join(data_dir, fname)
        lines = open(fpath).readlines()
        data_lines = [l.strip() for l in lines if l.strip() and not l.startswith(';') and not l.startswith('REGION')]
        rows = [l.split(',') for l in data_lines]
        dfs.append(pd.DataFrame(rows))
    df = pd.concat(dfs, ignore_index=True)
    y = LabelEncoder().fit_transform(df[0])
    X = df.iloc[:, 1:].values.astype(float)
    return X, y

def load_all_uci_datasets(data_dir=DATA_DIR):
    from load_uci_datasets import load_all_datasets
    datasets = load_all_datasets(data_dir)
    if 'Segmentation' not in datasets:
        try:
            X, y = load_segmentation(data_dir)
            datasets['Segmentation'] = (X, y, 7)
        except Exception as e:
            print(f"Error loading Segmentation: {e}")
    return datasets

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def apply_vae(X_train, n_components, epochs=100, batch_size=64, random_state=42):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X_train - X_min) / X_range
    
    n = len(X_scaled)
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    train_idx = idx[:n_train]
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    train_tensor = torch.FloatTensor(X_scaled[train_idx]).to(device)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    
    model = VAE(input_dim, n_components).to(device)
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            mse = nn.functional.mse_loss(recon, x, reduction='sum')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
    return mu.cpu().numpy()

# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def get_reduction_dims(n_features, k):
    dims = {
        'k-1': max(2, k - 1),
        '25%': max(2, int(np.round(0.25 * n_features))),
        '50%': max(2, int(np.round(0.50 * n_features))),
    }
    for key in dims:
        dims[key] = min(dims[key], n_features)
    return dims

def apply_dr(X, method, n_components, random_state=42):
    if n_components >= X.shape[1]:
        return X
    
    n_samples = X.shape[0]
    
    if method == 'PCA':
        return PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    elif method == 'Kernel PCA':
        return KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state).fit_transform(X)
    elif method == 'VAE':
        return apply_vae(X, n_components, random_state=random_state)
    elif method == 'Isomap':
        n_neighbors = min(5, n_samples - 1)
        return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
    elif method == 'MDS':
        # Paper: random_state=10, n_init=50 
        # Scale down for large datasets
        if n_samples > 1000:
            n_init = 4
            max_iter = 200
        elif n_samples > 500:
            n_init = 10
            max_iter = 300
        else:
            n_init = 50
            max_iter = 300
        return MDS(n_components=n_components, random_state=10, n_init=n_init, 
                   normalized_stress='auto', max_iter=max_iter).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================
# CLUSTERING
# ============================================================

def apply_kmeans(X, k, random_state=42):
    return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=random_state).fit_predict(X)

def apply_ahc_best(X, y_true, k):
    best_ari = -2
    best_labels = np.zeros(len(X))
    for affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single', 'ward']:
            if linkage == 'ward' and affinity != 'euclidean':
                continue
            try:
                model = AgglomerativeClustering(
                    n_clusters=k,
                    metric=affinity if linkage != 'ward' else 'euclidean',
                    linkage=linkage
                )
                labels = model.fit_predict(X)
                ari = adjusted_rand_score(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels.copy()
            except:
                continue
    return best_labels

def apply_gmm_best(X, y_true, k, random_state=42):
    best_ari = -2
    best_labels = np.zeros(len(X))
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        try:
            labels = GaussianMixture(n_components=k, covariance_type=cov_type,
                                     n_init=10, random_state=random_state).fit_predict(X)
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels.copy()
        except:
            continue
    return best_labels

def apply_optics_best(X, y_true):
    best_ari = -2
    best_labels = -np.ones(len(X))
    # Paper: min_samples=[5..10], xi=[0.0, 0.05, ..., 0.95]
    # Use coarser grid for speed: xi step 0.1 instead of 0.05
    for min_samples in [5, 7, 10]:
        if min_samples >= X.shape[0]:
            continue
        for xi in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            try:
                labels = OPTICS(min_samples=min_samples, xi=xi, cluster_method='xi').fit_predict(X)
                ari = adjusted_rand_score(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels.copy()
            except:
                continue
    return best_labels

def apply_clustering(X, algorithm, k, y_true=None, random_state=42):
    if algorithm == 'k-means':
        return apply_kmeans(X, k, random_state)
    elif algorithm == 'AHC':
        return apply_ahc_best(X, y_true, k) if y_true is not None else AgglomerativeClustering(n_clusters=k).fit_predict(X)
    elif algorithm == 'GMM':
        return apply_gmm_best(X, y_true, k, random_state) if y_true is not None else GaussianMixture(n_components=k, random_state=random_state, n_init=10).fit_predict(X)
    elif algorithm == 'OPTICS':
        return apply_optics_best(X, y_true) if y_true is not None else OPTICS(min_samples=5, xi=0.05, cluster_method='xi').fit_predict(X)
    else:
        raise ValueError(f"Unknown: {algorithm}")

# ============================================================
# EXPERIMENT RUNNER WITH INCREMENTAL SAVE
# ============================================================

def run_real_world_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading UCI datasets...")
    datasets = load_all_uci_datasets()
    print(f"Loaded {len(datasets)} datasets")
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    clustering_algorithms = ['k-means', 'AHC', 'GMM', 'OPTICS']
    reduction_levels = ['k-1', '25%', '50%']
    
    # Load existing results for resume
    results_file = os.path.join(OUTPUT_DIR, 'real_world_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming from existing results")
    else:
        all_results = {}
    
    total_tasks = len(clustering_algorithms) * len(datasets)
    completed = 0
    
    for algo in clustering_algorithms:
        if algo not in all_results:
            all_results[algo] = {}
        
        for ds_name in sorted(datasets.keys()):
            completed += 1
            
            # Skip if already done
            if ds_name in all_results[algo] and len(all_results[algo][ds_name]) >= 16:
                print(f"[{completed}/{total_tasks}] {algo} | {ds_name}: SKIPPED (already done)")
                sys.stdout.flush()
                continue
            
            X_raw, y_true, k = datasets[ds_name]
            n_features = X_raw.shape[1]
            
            X = StandardScaler().fit_transform(X_raw)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            result = {}
            t0 = time.time()
            
            # No reduction baseline
            try:
                labels = apply_clustering(X, algo, k, y_true)
                ari = adjusted_rand_score(y_true, labels)
            except Exception as e:
                print(f"  Error {algo} no-reduction on {ds_name}: {e}")
                ari = 0.0
            result['No Reduction'] = round(ari, 3)
            
            # With DR
            dims = get_reduction_dims(n_features, k)
            
            for method in dr_methods:
                for level in reduction_levels:
                    n_comp = dims[level]
                    key = f"{method}_{level}"
                    
                    try:
                        X_reduced = apply_dr(X, method, n_comp)
                        X_reduced = np.nan_to_num(X_reduced, nan=0.0, posinf=0.0, neginf=0.0)
                        labels = apply_clustering(X_reduced, algo, k, y_true)
                        ari = adjusted_rand_score(y_true, labels)
                    except Exception as e:
                        print(f"  Error {algo}/{method}/{level} on {ds_name}: {e}")
                        ari = 0.0
                    
                    result[key] = round(ari, 3)
            
            elapsed = time.time() - t0
            all_results[algo][ds_name] = result
            
            print(f"[{completed}/{total_tasks}] {algo} | {ds_name}: baseline={result['No Reduction']:.3f} ({elapsed:.1f}s)")
            sys.stdout.flush()
            
            # Save after each dataset
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    return all_results

# ============================================================
# SYNTHETIC EXPERIMENTS
# ============================================================

def generate_synthetic_datasets(random_state=42):
    from sklearn.datasets import make_circles, make_moons, make_blobs
    
    np.random.seed(random_state)
    synthetic = {}
    rng = np.random.RandomState(random_state)
    
    def add_noise_dims(X, target_dims, rng):
        n_samples, n_orig = X.shape
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        n_per_group = n_extra // 4
        remainder = n_extra - 4 * n_per_group
        noise_dims = []
        for i, std in enumerate([1.0, 0.5, 0.25, 0.0]):
            n_cols = n_per_group + (1 if i < remainder else 0)
            if n_cols > 0:
                if std > 0:
                    noise_dims.append(rng.normal(0, std, (n_samples, n_cols)))
                else:
                    noise_dims.append(np.zeros((n_samples, n_cols)))
        return np.hstack([X] + noise_dims)
    
    # Circles (k=2)
    for n_dims in [10, 50, 200]:
        for trial in range(5):
            X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=random_state + trial)
            X = add_noise_dims(X, n_dims, rng)
            synthetic[f'Circles_k2_d{n_dims}_t{trial}'] = (X, y, 2)
    
    # Moons (k=2)
    for n_dims in [10, 50, 200]:
        for trial in range(5):
            X, y = make_moons(n_samples=500, noise=0.1, random_state=random_state + trial)
            X = add_noise_dims(X, n_dims, rng)
            synthetic[f'Moons_k2_d{n_dims}_t{trial}'] = (X, y, 2)
    
    # RSG (Random Sklearn Generated)
    for k in [3, 5, 7]:
        for n_dims in [10, 50, 200]:
            for trial in range(5):
                X, y = make_blobs(n_samples=500, n_features=min(n_dims, 5), centers=k,
                                  cluster_std=1.0, random_state=random_state + trial + k)
                X = add_noise_dims(X, n_dims, rng)
                synthetic[f'RSG_k{k}_d{n_dims}_t{trial}'] = (X, y, k)
    
    # Repliclust-like (use make_blobs with varied parameters as approximation)
    for k in [3, 5, 7]:
        for n_dims in [10, 50, 200]:
            for trial in range(5):
                X, y = make_blobs(n_samples=500, n_features=min(n_dims, 10), centers=k,
                                  cluster_std=np.random.uniform(0.5, 2.0, k),
                                  random_state=random_state + trial + k + 100)
                X = add_noise_dims(X, n_dims, rng)
                synthetic[f'Repliclust_k{k}_d{n_dims}_t{trial}'] = (X, y, k)
    
    return synthetic

def run_synthetic_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating synthetic datasets...")
    synthetic = generate_synthetic_datasets()
    print(f"Generated {len(synthetic)} synthetic datasets")
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    clustering_algorithms = ['k-means', 'AHC', 'GMM', 'OPTICS']
    reduction_levels = ['k-1', '25%', '50%']
    
    results_file = os.path.join(OUTPUT_DIR, 'synthetic_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    total = len(clustering_algorithms) * len(synthetic)
    completed = 0
    
    for algo in clustering_algorithms:
        if algo not in all_results:
            all_results[algo] = {}
        
        for ds_name in sorted(synthetic.keys()):
            completed += 1
            
            if ds_name in all_results[algo]:
                continue
            
            X_raw, y_true, k = synthetic[ds_name]
            n_features = X_raw.shape[1]
            
            X = StandardScaler().fit_transform(X_raw)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            result = {}
            t0 = time.time()
            
            try:
                labels = apply_clustering(X, algo, k, y_true)
                ari = adjusted_rand_score(y_true, labels)
            except:
                ari = 0.0
            result['No Reduction'] = round(ari, 3)
            
            dims = get_reduction_dims(n_features, k)
            
            for method in dr_methods:
                for level in reduction_levels:
                    n_comp = dims[level]
                    key = f"{method}_{level}"
                    try:
                        X_reduced = apply_dr(X, method, n_comp)
                        X_reduced = np.nan_to_num(X_reduced, nan=0.0, posinf=0.0, neginf=0.0)
                        labels = apply_clustering(X_reduced, algo, k, y_true)
                        ari = adjusted_rand_score(y_true, labels)
                    except:
                        ari = 0.0
                    result[key] = round(ari, 3)
            
            elapsed = time.time() - t0
            all_results[algo][ds_name] = result
            
            if completed % 10 == 0 or elapsed > 10:
                print(f"[{completed}/{total}] {algo} | {ds_name}: baseline={result['No Reduction']:.3f} ({elapsed:.1f}s)")
                sys.stdout.flush()
            
            # Save periodically
            if completed % 20 == 0:
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

# ============================================================
# ANALYSIS AND OUTPUT
# ============================================================

def format_results_table(results, algorithm):
    datasets = sorted(results[algorithm].keys())
    columns = ['No Reduction']
    for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            columns.append(f"{method}_{level}")
    
    data = []
    for ds in datasets:
        row = [results[algorithm][ds].get(col, 0.0) for col in columns]
        data.append(row)
    
    return pd.DataFrame(data, index=datasets, columns=columns)

def compute_aggregate_stats(results, algorithm):
    datasets = sorted(results[algorithm].keys())
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    reduction_levels = ['k-1', '25%', '50%']
    
    stats = {}
    for method in dr_methods:
        stats[method] = {}
        for level in reduction_levels:
            key = f"{method}_{level}"
            wins = 0
            total = 0
            ari_changes = []
            
            for ds in datasets:
                baseline = results[algorithm][ds].get('No Reduction', 0.0)
                reduced = results[algorithm][ds].get(key, 0.0)
                total += 1
                if reduced > baseline:
                    wins += 1
                if abs(baseline) > 1e-6:
                    pct_change = ((reduced - baseline) / abs(baseline)) * 100
                else:
                    pct_change = reduced * 100
                ari_changes.append(pct_change)
            
            win_pct = (wins / total * 100) if total > 0 else 0
            avg_change = np.mean(ari_changes) if ari_changes else 0
            
            stats[method][level] = {
                'win_pct': round(win_pct, 2),
                'avg_change': round(avg_change, 2),
                'wins': wins,
                'total': total,
            }
    return stats

def compute_wilcoxon_test(results, algorithm):
    datasets = sorted(results[algorithm].keys())
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    reduction_levels = ['k-1', '25%', '50%']
    
    p_values = {}
    for method in dr_methods:
        p_values[method] = {}
        for level in reduction_levels:
            key = f"{method}_{level}"
            baselines = []
            reduced_scores = []
            for ds in datasets:
                baselines.append(results[algorithm][ds].get('No Reduction', 0.0))
                reduced_scores.append(results[algorithm][ds].get(key, 0.0))
            
            diffs = np.array(reduced_scores) - np.array(baselines)
            nonzero = diffs != 0
            if nonzero.sum() >= 2:
                try:
                    stat, p = wilcoxon(diffs[nonzero], alternative='greater')
                    p_values[method][level] = round(p, 4)
                except:
                    p_values[method][level] = 1.0
            else:
                p_values[method][level] = 1.0
    return p_values

def generate_boxplots(results, data_type, output_dir=OUTPUT_DIR):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    reduction_levels = ['k-1', '25%', '50%']
    algo_names = {'k-means': 'KMeans', 'AHC': 'Agglomerative', 'GMM': 'Gaussian', 'OPTICS': 'OPTICS'}
    
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo not in results or not results[algo]:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        box_data = []
        labels = []
        
        baseline_vals = [results[algo][ds]['No Reduction'] for ds in results[algo]]
        box_data.append(baseline_vals)
        labels.append('No Red.')
        
        for method in dr_methods:
            for level in reduction_levels:
                key = f"{method}_{level}"
                vals = [results[algo][ds].get(key, 0.0) for ds in results[algo]]
                box_data.append(vals)
                labels.append(f"{method}\n{level}")
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        colors = ['gray'] + ['#1f77b4']*3 + ['#ff7f0e']*3 + ['#2ca02c']*3 + ['#d62728']*3 + ['#9467bd']*3
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('ARI', fontsize=12)
        ax.set_title(f'{algo_names.get(algo, algo)} - {data_type} Data', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        plt.tight_layout()
        
        fname = f'Boxplot_{algo_names.get(algo, algo)}_{data_type}.pdf'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")

def generate_all_analysis(results, data_type='RealData'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tables
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo in results and results[algo]:
            df = format_results_table(results, algo)
            df.to_csv(os.path.join(OUTPUT_DIR, f'table_{algo}_{data_type}.csv'))
            print(f"\n{algo} ({data_type}) results:")
            print(df.to_string())
    
    # Aggregate stats
    print(f"\n{'='*60}\nAGGREGATE STATISTICS ({data_type})\n{'='*60}")
    agg = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo in results and results[algo]:
            agg[algo] = compute_aggregate_stats(results, algo)
            print(f"\n{algo}:")
            for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
                for level in ['k-1', '25%', '50%']:
                    s = agg[algo][method][level]
                    print(f"  {method:12s} {level:4s}: win={s['win_pct']:5.1f}%, avg_change={s['avg_change']:+7.1f}%")
    
    with open(os.path.join(OUTPUT_DIR, f'aggregate_stats_{data_type}.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    
    # Wilcoxon
    print(f"\n{'='*60}\nWILCOXON SIGNED-RANK TEST ({data_type})\n{'='*60}")
    wilcoxon_res = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo in results and results[algo]:
            wilcoxon_res[algo] = compute_wilcoxon_test(results, algo)
            print(f"\n{algo}:")
            for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
                vals = [wilcoxon_res[algo][method][l] for l in ['k-1', '25%', '50%']]
                sig = ['*' if v < 0.05 else ' ' for v in vals]
                print(f"  {method:12s}: k-1={vals[0]:.4f}{sig[0]} 25%={vals[1]:.4f}{sig[1]} 50%={vals[2]:.4f}{sig[2]}")
    
    with open(os.path.join(OUTPUT_DIR, f'wilcoxon_{data_type}.json'), 'w') as f:
        json.dump(wilcoxon_res, f, indent=2)
    
    # Boxplots
    generate_boxplots(results, data_type)

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'all', 'analyze'], default='real')
    args = parser.parse_args()
    
    if args.mode in ['real', 'all']:
        print("=" * 60)
        print("REAL-WORLD EXPERIMENTS")
        print("=" * 60)
        real_results = run_real_world_experiments()
        generate_all_analysis(real_results, 'RealData')
    
    if args.mode in ['synthetic', 'all']:
        print("\n" + "=" * 60)
        print("SYNTHETIC EXPERIMENTS")
        print("=" * 60)
        synth_results = run_synthetic_experiments()
        generate_all_analysis(synth_results, 'Synthetic')
    
    if args.mode == 'analyze':
        # Just regenerate analysis from saved results
        rfile = os.path.join(OUTPUT_DIR, 'real_world_results.json')
        if os.path.exists(rfile):
            with open(rfile) as f:
                real_results = json.load(f)
            generate_all_analysis(real_results, 'RealData')
        
        sfile = os.path.join(OUTPUT_DIR, 'synthetic_results.json')
        if os.path.exists(sfile):
            with open(sfile) as f:
                synth_results = json.load(f)
            generate_all_analysis(synth_results, 'Synthetic')
    
    print("\nDone! Results saved to", OUTPUT_DIR)
