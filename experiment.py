"""
Main experiment pipeline: Dimensionality Reduction + Clustering evaluation.
Replicates the paper's systematic study on real-world UCI datasets and synthetic data.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING
# ============================================================

def load_segmentation(data_dir='./uci_data'):
    """Load segmentation dataset with proper parsing."""
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


def load_all_uci_datasets(data_dir='./uci_data'):
    """Load all 20 UCI datasets."""
    from load_uci_datasets import load_all_datasets, DATASET_INFO
    datasets = load_all_datasets(data_dir)
    
    # Fix segmentation if missing
    if 'Segmentation' not in datasets:
        try:
            X, y = load_segmentation(data_dir)
            datasets['Segmentation'] = (X, y, 7)
        except Exception as e:
            print(f"Error loading Segmentation: {e}")
    
    return datasets


# ============================================================
# VAE IMPLEMENTATION
# ============================================================

class VAE(nn.Module):
    """Variational Autoencoder for dimensionality reduction.
    Architecture from paper: encoder [64,32] + BN + Dropout(0.4), 
    latent=n_components, decoder mirrors, sigmoid output.
    """
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


def vae_loss(recon_x, x, mu, logvar):
    """MSE reconstruction + KL divergence."""
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


def apply_vae(X_train, n_components, epochs=100, batch_size=64, random_state=42):
    """Train VAE and return latent representations."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    
    # Min-max scale to [0,1] for sigmoid output
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X_train - X_min) / X_range
    
    # 70/30 split for training
    n = len(X_scaled)
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    train_idx = idx[:n_train]
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    train_tensor = torch.FloatTensor(X_scaled[train_idx]).to(device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = VAE(input_dim, n_components).to(device)
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Get latent representations for ALL data
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
    
    return mu.cpu().numpy()


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def get_reduction_dims(n_features, k):
    """Compute target dimensions for the 3 reduction levels."""
    dims = {
        'k-1': max(2, k - 1),
        '25%': max(2, int(np.round(0.25 * n_features))),
        '50%': max(2, int(np.round(0.50 * n_features))),
    }
    # Ensure dims don't exceed original
    for key in dims:
        dims[key] = min(dims[key], n_features)
    return dims


def apply_dr(X, method, n_components, random_state=42):
    """Apply dimensionality reduction method."""
    if n_components >= X.shape[1]:
        return X  # No reduction needed
    
    if method == 'PCA':
        dr = PCA(n_components=n_components, random_state=random_state)
        return dr.fit_transform(X)
    
    elif method == 'Kernel PCA':
        dr = KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state)
        return dr.fit_transform(X)
    
    elif method == 'VAE':
        return apply_vae(X, n_components, random_state=random_state)
    
    elif method == 'Isomap':
        n_neighbors = min(5, X.shape[0] - 1)
        dr = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        return dr.fit_transform(X)
    
    elif method == 'MDS':
        dr = MDS(n_components=n_components, random_state=10, n_init=50, normalized_stress='auto')
        return dr.fit_transform(X)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# CLUSTERING WITH HYPERPARAMETER SEARCH
# ============================================================

def apply_kmeans(X, k, random_state=42):
    """k-means with k-means++ init, n_init=100."""
    model = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=random_state)
    return model.fit_predict(X)


def apply_ahc_best(X, y_true, k):
    """AHC with hyperparameter search over affinity and linkage.
    Paper: affinity (euclidean, l1, l2, manhattan, cosine) × linkage (complete, average, single, ward).
    Ward only with euclidean. Pick best ARI.
    """
    best_ari = -2
    best_labels = np.zeros(len(X))
    
    affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['complete', 'average', 'single', 'ward']
    
    for affinity in affinities:
        for linkage in linkages:
            # Ward only works with euclidean
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
            except Exception:
                continue
    
    return best_labels


def apply_gmm_best(X, y_true, k, random_state=42):
    """GMM with hyperparameter search over covariance types.
    Paper: covariance types (spherical, tied, diag, full), n_init=10.
    """
    best_ari = -2
    best_labels = np.zeros(len(X))
    
    for cov_type in ['spherical', 'tied', 'diag', 'full']:
        try:
            model = GaussianMixture(
                n_components=k, 
                covariance_type=cov_type,
                n_init=10, 
                random_state=random_state
            )
            labels = model.fit_predict(X)
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels.copy()
        except Exception:
            continue
    
    return best_labels


def apply_optics_best(X, y_true):
    """OPTICS with hyperparameter search.
    Paper: xi method, min_samples=[5,6,7,8,9,10], min_cluster_size=[0.05,0.10,...,1.0].
    Optimized: fit OPTICS once per min_samples, then extract clusters with different min_cluster_size.
    """
    best_ari = -2
    best_labels = -np.ones(len(X), dtype=int)
    
    for min_samples in [5, 6, 7, 8, 9, 10]:
        if min_samples >= X.shape[0]:
            continue
        try:
            model = OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05)
            model.fit(X)
            
            for mcs in np.arange(0.05, 1.01, 0.05):
                mcs = round(mcs, 2)
                try:
                    labels, _ = cluster_optics_xi(
                        reachability=model.reachability_,
                        predecessor=model.predecessor_,
                        ordering=model.ordering_,
                        min_samples=min_samples,
                        xi=0.05,
                        min_cluster_size=mcs
                    )
                    ari = adjusted_rand_score(y_true, labels)
                    if ari > best_ari:
                        best_ari = ari
                        best_labels = labels.copy()
                except Exception:
                    continue
        except Exception:
            continue
    
    return best_labels


def apply_clustering(X, algorithm, k, y_true=None, random_state=42):
    """Apply clustering algorithm and return labels.
    For AHC, GMM, OPTICS: uses hyperparameter search with y_true to pick best.
    """
    if algorithm == 'k-means':
        return apply_kmeans(X, k, random_state)
    
    elif algorithm == 'AHC':
        if y_true is not None:
            return apply_ahc_best(X, y_true, k)
        else:
            model = AgglomerativeClustering(n_clusters=k)
            return model.fit_predict(X)
    
    elif algorithm == 'GMM':
        if y_true is not None:
            return apply_gmm_best(X, y_true, k, random_state)
        else:
            model = GaussianMixture(n_components=k, random_state=random_state, n_init=10)
            return model.fit_predict(X)
    
    elif algorithm == 'OPTICS':
        if y_true is not None:
            return apply_optics_best(X, y_true)
        else:
            model = OPTICS(min_samples=5, xi=0.05, cluster_method='xi')
            return model.fit_predict(X)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def run_real_world_experiments(datasets, output_dir='./results'):
    """Run all experiments on real-world datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    clustering_algorithms = ['k-means', 'AHC', 'GMM', 'OPTICS']
    reduction_levels = ['k-1', '25%', '50%']
    
    # Results storage: {algorithm: {dataset: {method_level: ari}}}
    all_results = {}
    
    for algo in clustering_algorithms:
        all_results[algo] = {}
        
        for ds_name in sorted(datasets.keys()):
            X_raw, y_true, k = datasets[ds_name]
            n_features = X_raw.shape[1]
            
            # Z-score normalization
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            result = {}
            
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
            
            all_results[algo][ds_name] = result
            print(f"  {algo} | {ds_name}: baseline={result['No Reduction']:.3f}")
    
    return all_results


def format_results_table(results, algorithm):
    """Format results as a pandas DataFrame matching paper's table format."""
    datasets = sorted(results[algorithm].keys())
    
    columns = ['No Reduction']
    for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
        for level in ['k-1', '25%', '50%']:
            columns.append(f"{method}_{level}")
    
    data = []
    for ds in datasets:
        row = [results[algorithm][ds].get(col, 0.0) for col in columns]
        data.append(row)
    
    df = pd.DataFrame(data, index=datasets, columns=columns)
    return df


def compute_aggregate_stats(results, algorithm):
    """Compute win rates and average win/loss percentages (Tables 1-4)."""
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
                
                # Percentage change
                if abs(baseline) > 1e-6:
                    pct_change = ((reduced - baseline) / abs(baseline)) * 100
                else:
                    pct_change = reduced * 100  # baseline is ~0
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
    """Compute Wilcoxon signed-rank test (Table A.9)."""
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
            
            baselines = np.array(baselines)
            reduced_scores = np.array(reduced_scores)
            diffs = reduced_scores - baselines
            
            # Remove zeros for Wilcoxon test
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


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_datasets(random_state=42):
    """Generate synthetic datasets: Circles, Moons, RSG, Repliclust.
    Paper uses 1165 total synthetic datasets.
    We generate a representative subset.
    """
    from sklearn.datasets import make_circles, make_moons, make_blobs
    
    np.random.seed(random_state)
    synthetic = {}
    
    def add_noise_dims(X, target_dims, rng):
        """Add noisy dimensions. 25% N(0,1), 25% N(0,0.5), 25% N(0,0.25), 25% no noise."""
        n_samples = X.shape[0]
        n_orig = X.shape[1]
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        
        n_per_group = n_extra // 4
        remainder = n_extra - 4 * n_per_group
        
        noise_dims = []
        noise_dims.append(rng.normal(0, 1, (n_samples, n_per_group + (1 if remainder > 0 else 0))))
        remainder = max(0, remainder - 1)
        noise_dims.append(rng.normal(0, 0.5, (n_samples, n_per_group + (1 if remainder > 0 else 0))))
        remainder = max(0, remainder - 1)
        noise_dims.append(rng.normal(0, 0.25, (n_samples, n_per_group + (1 if remainder > 0 else 0))))
        remainder = max(0, remainder - 1)
        noise_dims.append(np.zeros((n_samples, n_per_group + (1 if remainder > 0 else 0))))
        
        all_noise = np.hstack(noise_dims)[:, :n_extra]
        return np.hstack([X, all_noise])
    
    rng = np.random.RandomState(random_state)
    
    # Circles datasets (k=2)
    for n_dims in [10, 50, 200]:
        for trial in range(5):
            X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=random_state + trial)
            X = add_noise_dims(X, n_dims, rng)
            synthetic[f'Circles_k2_d{n_dims}_t{trial}'] = (X, y, 2)
    
    # Moons datasets (k=2)
    for n_dims in [10, 50, 200]:
        for trial in range(5):
            X, y = make_moons(n_samples=500, noise=0.1, random_state=random_state + trial)
            X = add_noise_dims(X, n_dims, rng)
            synthetic[f'Moons_k2_d{n_dims}_t{trial}'] = (X, y, 2)
    
    # RSG (Random Structured Gaussian) - using make_blobs
    for k in [3, 5, 7]:
        for n_dims in [10, 50, 200]:
            for trial in range(5):
                X, y = make_blobs(n_samples=500, n_features=min(n_dims, 5), centers=k,
                                  cluster_std=1.0, random_state=random_state + trial + k)
                X = add_noise_dims(X, n_dims, rng)
                synthetic[f'RSG_k{k}_d{n_dims}_t{trial}'] = (X, y, k)
    
    # Repliclust datasets
    try:
        import repliclust
        for trial in range(5):
            archetype = repliclust.Archetype(
                n_clusters=5,
                dim=10,
                n_samples=500,
            )
            generator = repliclust.DataGenerator(archetype)
            X, y, _ = generator.synthesize(random_state=random_state + trial)
            for n_dims in [10, 50, 200]:
                X_ext = add_noise_dims(X.copy(), n_dims, rng)
                synthetic[f'Repliclust_k5_d{n_dims}_t{trial}'] = (X_ext, y, 5)
    except Exception as e:
        print(f"Repliclust not available, skipping: {e}")
    
    return synthetic


def run_synthetic_experiments(synthetic_datasets, output_dir='./results'):
    """Run experiments on synthetic datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    clustering_algorithms = ['k-means', 'AHC', 'GMM', 'OPTICS']
    reduction_levels = ['k-1', '25%', '50%']
    
    all_results = {}
    
    for algo in clustering_algorithms:
        all_results[algo] = {}
        
        for ds_name in sorted(synthetic_datasets.keys()):
            X_raw, y_true, k = synthetic_datasets[ds_name]
            n_features = X_raw.shape[1]
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            result = {}
            
            # No reduction baseline
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
            
            all_results[algo][ds_name] = result
            print(f"  {algo} | {ds_name}: baseline={result['No Reduction']:.3f}")
    
    return all_results


# ============================================================
# BOXPLOT GENERATION
# ============================================================

def generate_boxplots(results, data_type, output_dir='./results'):
    """Generate boxplots matching paper's figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    dr_methods = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
    reduction_levels = ['k-1', '25%', '50%']
    algo_names = {'k-means': 'KMeans', 'AHC': 'Agglomerative', 'GMM': 'Gaussian', 'OPTICS': 'OPTICS'}
    
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        if algo not in results:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Collect data for boxplot
        box_data = []
        labels = []
        
        # No reduction baseline
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
        
        # Color by method
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'all'], default='real')
    parser.add_argument('--output_dir', default='./results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ['real', 'all']:
        print("=" * 60)
        print("LOADING UCI DATASETS")
        print("=" * 60)
        datasets = load_all_uci_datasets()
        print(f"Loaded {len(datasets)} datasets")
        
        print("\n" + "=" * 60)
        print("RUNNING REAL-WORLD EXPERIMENTS")
        print("=" * 60)
        real_results = run_real_world_experiments(datasets, args.output_dir)
        
        # Save raw results
        with open(os.path.join(args.output_dir, 'real_world_results.json'), 'w') as f:
            json.dump(real_results, f, indent=2)
        
        # Generate tables
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            df = format_results_table(real_results, algo)
            df.to_csv(os.path.join(args.output_dir, f'table_{algo}_real.csv'))
            print(f"\n{algo} results:")
            print(df.to_string())
        
        # Aggregate stats
        print("\n" + "=" * 60)
        print("AGGREGATE STATISTICS")
        print("=" * 60)
        agg_stats = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            agg_stats[algo] = compute_aggregate_stats(real_results, algo)
            print(f"\n{algo}:")
            for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
                for level in ['k-1', '25%', '50%']:
                    s = agg_stats[algo][method][level]
                    print(f"  {method:12s} {level:4s}: win={s['win_pct']:5.1f}%, avg_change={s['avg_change']:+6.1f}%")
        
        with open(os.path.join(args.output_dir, 'aggregate_stats_real.json'), 'w') as f:
            json.dump(agg_stats, f, indent=2)
        
        # Wilcoxon test
        print("\n" + "=" * 60)
        print("WILCOXON SIGNED-RANK TEST")
        print("=" * 60)
        wilcoxon_results = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            wilcoxon_results[algo] = compute_wilcoxon_test(real_results, algo)
            print(f"\n{algo}:")
            for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
                vals = [wilcoxon_results[algo][method][l] for l in ['k-1', '25%', '50%']]
                sig = ['*' if v < 0.05 else ' ' for v in vals]
                print(f"  {method:12s}: k-1={vals[0]:.4f}{sig[0]} 25%={vals[1]:.4f}{sig[1]} 50%={vals[2]:.4f}{sig[2]}")
        
        with open(os.path.join(args.output_dir, 'wilcoxon_results.json'), 'w') as f:
            json.dump(wilcoxon_results, f, indent=2)
        
        # Generate boxplots
        print("\n" + "=" * 60)
        print("GENERATING BOXPLOTS")
        print("=" * 60)
        generate_boxplots(real_results, 'RealData', args.output_dir)
    
    if args.mode in ['synthetic', 'all']:
        print("\n" + "=" * 60)
        print("GENERATING SYNTHETIC DATASETS")
        print("=" * 60)
        synthetic = generate_synthetic_datasets()
        print(f"Generated {len(synthetic)} synthetic datasets")
        
        print("\n" + "=" * 60)
        print("RUNNING SYNTHETIC EXPERIMENTS")
        print("=" * 60)
        synth_results = run_synthetic_experiments(synthetic, args.output_dir)
        
        with open(os.path.join(args.output_dir, 'synthetic_results.json'), 'w') as f:
            json.dump(synth_results, f, indent=2)
        
        # Generate boxplots for synthetic data types
        for dtype in ['Circles', 'Moons', 'RSG', 'Repliclust']:
            filtered = {}
            for algo in synth_results:
                filtered[algo] = {ds: synth_results[algo][ds] 
                                  for ds in synth_results[algo] if ds.startswith(dtype)}
            if any(filtered[algo] for algo in filtered):
                generate_boxplots(filtered, dtype, args.output_dir)
        
        # Aggregate stats for synthetic
        print("\n" + "=" * 60)
        print("SYNTHETIC AGGREGATE STATISTICS")
        print("=" * 60)
        synth_agg = {}
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            synth_agg[algo] = compute_aggregate_stats(synth_results, algo)
            print(f"\n{algo}:")
            for method in ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']:
                for level in ['k-1', '25%', '50%']:
                    s = synth_agg[algo][method][level]
                    print(f"  {method:12s} {level:4s}: win={s['win_pct']:5.1f}%, avg_change={s['avg_change']:+6.1f}%")
        
        with open(os.path.join(args.output_dir, 'aggregate_stats_synthetic.json'), 'w') as f:
            json.dump(synth_agg, f, indent=2)
    
    print("\nDone! Results saved to", args.output_dir)
