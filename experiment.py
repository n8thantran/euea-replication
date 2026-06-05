"""
Main experiment pipeline: Dimensionality Reduction + Clustering evaluation.
Replicates: "Assessing the impact of dimensionality reduction on clustering performance"

Key design: AHC/GMM/OPTICS hyperparameters are chosen per DATASET TYPE
(one set for all real-world datasets, one per synthetic type).
"""
import os
import sys
import json
import time
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("DR operation timed out")
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
# VAE IMPLEMENTATION
# ============================================================

class VAE(nn.Module):
    """Paper: encoder d→64→32→latent, decoder latent→32→64→d, BN, Dropout=0.4, sigmoid output."""
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


def apply_vae(X, n_components, epochs=50, batch_size=64, random_state=42):
    """Train VAE, return z_mean as embedding for ALL samples."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]

    # Min-max scale to [0,1] for sigmoid output
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_scaled = (X - X_min) / X_range

    # 70/30 train split
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
            loss = mse + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(all_data)
    return mu.cpu().numpy()


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def get_reduction_dims(n_features, k):
    """3 reduction levels: k-1, 25%, 50% of features (min 2, max n_features)."""
    return {
        'k-1': min(n_features, max(2, k - 1)),
        '25%': min(n_features, max(2, int(np.round(0.25 * n_features)))),
        '50%': min(n_features, max(2, int(np.round(0.50 * n_features)))),
    }


def apply_dr(X, method, n_components, random_state=42):
    """Apply one DR method. Returns reduced X or original if n_components >= n_features."""
    if n_components >= X.shape[1]:
        return X.copy()

    if method == 'PCA':
        return PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    elif method == 'Kernel PCA':
        return KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state).fit_transform(X)
    elif method == 'VAE':
        return apply_vae(X, n_components, random_state=random_state)
    elif method == 'Isomap':
        n_neighbors = min(5, X.shape[0] - 1)
        return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
    elif method == 'MDS':
        # Paper: random_state=10, n_init=50. Reduced for computational feasibility.
        n = X.shape[0]
        n_init = 1
        return MDS(n_components=n_components, random_state=10, n_init=n_init,
                   max_iter=50, normalized_stress='auto').fit_transform(X)
    else:
        raise ValueError(f"Unknown DR method: {method}")


DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']


def get_all_conditions():
    """Return list of all conditions: 'No Reduction' + DR combos."""
    conds = ['No Reduction']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            conds.append(f"{m}_{l}")
    return conds


def precompute_all_dr(datasets):
    """Precompute all DR transformations. Returns {ds_name: {condition: X_transformed}}."""
    dr_cache = {}
    total = len(datasets)
    for i, ds_name in enumerate(sorted(datasets.keys())):
        X_raw, y_true, k = datasets[ds_name]
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_features = X.shape[1]
        dims = get_reduction_dims(n_features, k)

        ds_cache = {'No Reduction': X.copy()}
        print(f"  [{i+1}/{total}] {ds_name} (shape={X.shape}, k={k})")

        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                n_comp = dims[level]
                try:
                    t0 = time.time()
                    # Timeout: 60s per DR operation
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                    X_red = apply_dr(X, method, n_comp)
                    signal.alarm(0)  # Cancel alarm
                    X_red = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
                    ds_cache[key] = X_red
                    dt = time.time() - t0
                    if dt > 5:
                        print(f"    {key} ({n_comp}d) [{dt:.1f}s]")
                except TimeoutError:
                    signal.alarm(0)
                    print(f"    TIMEOUT {key} (>30s, skipping)")
                    ds_cache[key] = None
                except Exception as e:
                    signal.alarm(0)
                    print(f"    ERROR {key}: {e}")
                    ds_cache[key] = None

        dr_cache[ds_name] = ds_cache
    return dr_cache


# ============================================================
# CLUSTERING
# ============================================================

def cluster_kmeans(X, k, random_state=42):
    """k-means++, n_init=100."""
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state).fit_predict(X)


def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    """AHC with specific metric and linkage."""
    return AgglomerativeClustering(
        n_clusters=k, metric=metric, linkage=linkage
    ).fit_predict(X)


def cluster_gmm(X, k, covariance_type='full', random_state=42):
    """GMM with specific covariance type."""
    return GaussianMixture(
        n_components=k, covariance_type=covariance_type, n_init=10, random_state=random_state
    ).fit_predict(X)


def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    """OPTICS with xi method."""
    model = OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05,
                   min_cluster_size=min_cluster_size)
    return model.fit_predict(X)


# ============================================================
# HYPERPARAMETER SEARCH (per dataset type)
# ============================================================

def get_ahc_combos():
    """All valid (metric, linkage) combinations for AHC."""
    combos = []
    for metric in ['euclidean', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))  # ward only with euclidean
    return combos


def get_gmm_combos():
    return ['spherical', 'tied', 'diag', 'full']


def find_best_ahc_params(datasets, dr_cache):
    """Find (metric, linkage) with highest average ARI across all datasets × conditions."""
    combos = get_ahc_combos()
    conditions = get_all_conditions()
    best_score = -999
    best_combo = ('euclidean', 'ward')

    print("  AHC hyperparam search...")
    for metric, linkage in combos:
        total_ari = 0.0
        count = 0
        for ds_name in sorted(datasets.keys()):
            _, y_true, k = datasets[ds_name]
            for cond in conditions:
                X = dr_cache[ds_name].get(cond)
                if X is None:
                    continue
                try:
                    labels = cluster_ahc(X, k, metric=metric, linkage=linkage)
                    total_ari += adjusted_rand_score(y_true, labels)
                    count += 1
                except:
                    pass
        avg = total_ari / count if count > 0 else -999
        if avg > best_score:
            best_score = avg
            best_combo = (metric, linkage)

    print(f"    Best AHC: metric={best_combo[0]}, linkage={best_combo[1]}, avg_ari={best_score:.4f}")
    return best_combo


def find_best_gmm_params(datasets, dr_cache):
    """Find covariance_type with highest average ARI."""
    conditions = get_all_conditions()
    best_score = -999
    best_cov = 'full'

    print("  GMM hyperparam search...")
    for cov_type in get_gmm_combos():
        total_ari = 0.0
        count = 0
        for ds_name in sorted(datasets.keys()):
            _, y_true, k = datasets[ds_name]
            for cond in conditions:
                X = dr_cache[ds_name].get(cond)
                if X is None:
                    continue
                try:
                    labels = cluster_gmm(X, k, covariance_type=cov_type)
                    total_ari += adjusted_rand_score(y_true, labels)
                    count += 1
                except:
                    pass
        avg = total_ari / count if count > 0 else -999
        if avg > best_score:
            best_score = avg
            best_cov = cov_type

    print(f"    Best GMM: cov_type={best_cov}, avg_ari={best_score:.4f}")
    return best_cov


def find_best_optics_params(datasets, dr_cache):
    """Find (min_samples, min_cluster_size) with highest average ARI.
    Optimized: fit OPTICS once per (dataset, condition, min_samples), 
    then extract clusters with different min_cluster_size using cluster_optics_xi."""
    conditions = get_all_conditions()
    mcs_values = [round(v, 2) for v in np.arange(0.05, 1.05, 0.2)]  # 0.05 to 1.0 step 0.1

    print("  OPTICS hyperparam search (optimized)...")
    # combo_scores[(ms, mcs)] = list of ARIs
    combo_scores = {}
    for ms in [5, 7, 10]:
        for mcs in mcs_values:
            combo_scores[(ms, mcs)] = []

    n_ds = len(datasets)
    for ds_idx, ds_name in enumerate(sorted(datasets.keys())):
        _, y_true, k = datasets[ds_name]
        if ds_idx % 5 == 0:
            print(f"    OPTICS search: dataset {ds_idx+1}/{n_ds}")
        for cond in conditions:
            X = dr_cache[ds_name].get(cond)
            if X is None:
                continue
            for ms in [5, 7, 10]:
                if ms >= X.shape[0]:
                    continue
                try:
                    # Fit OPTICS once
                    model = OPTICS(min_samples=ms, cluster_method='xi', xi=0.05, min_cluster_size=0.05)
                    model.fit(X)
                    # Extract clusters with different min_cluster_size
                    for mcs in mcs_values:
                        try:
                            labels, _ = cluster_optics_xi(
                                reachability=model.reachability_,
                                predecessor=model.predecessor_,
                                ordering=model.ordering_,
                                min_samples=ms,
                                min_cluster_size=mcs,
                                xi=0.05
                            )
                            ari = adjusted_rand_score(y_true, labels)
                            combo_scores[(ms, mcs)].append(ari)
                        except:
                            pass
                except:
                    pass

    best_score = -999
    best_combo = (5, 0.05)
    for (ms, mcs), aris in combo_scores.items():
        if len(aris) == 0:
            continue
        avg = np.mean(aris)
        if avg > best_score:
            best_score = avg
            best_combo = (ms, mcs)

    print(f"    Best OPTICS: min_samples={best_combo[0]}, min_cluster_size={best_combo[1]}, avg_ari={best_score:.4f}")
    return best_combo


# ============================================================
# MAIN EXPERIMENT RUNNERS
# ============================================================

def run_kmeans_experiments(datasets, dr_cache):
    """k-means: no hyperparameter search, just run."""
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
                labels = cluster_kmeans(X, k)
                row[cond] = round(adjusted_rand_score(y_true, labels), 2)
            except:
                row[cond] = 0.0
        results[ds_name] = row
    return results


def run_ahc_experiments(datasets, dr_cache, metric, linkage):
    """AHC: use fixed (metric, linkage) for all datasets."""
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
                labels = cluster_ahc(X, k, metric=metric, linkage=linkage)
                row[cond] = round(adjusted_rand_score(y_true, labels), 2)
            except:
                row[cond] = 0.0
        results[ds_name] = row
    return results


def run_gmm_experiments(datasets, dr_cache, covariance_type):
    """GMM: use fixed covariance_type for all datasets."""
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
                labels = cluster_gmm(X, k, covariance_type=covariance_type)
                row[cond] = round(adjusted_rand_score(y_true, labels), 2)
            except:
                row[cond] = 0.0
        results[ds_name] = row
    return results


def run_optics_experiments(datasets, dr_cache, min_samples, min_cluster_size):
    """OPTICS: use fixed (min_samples, min_cluster_size) for all datasets."""
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
                labels = cluster_optics(X, min_samples=min_samples, min_cluster_size=min_cluster_size)
                row[cond] = round(adjusted_rand_score(y_true, labels), 2)
            except:
                row[cond] = 0.0
        results[ds_name] = row
    return results


# ============================================================
# RESULTS FORMATTING AND ANALYSIS
# ============================================================

def format_results_table(results_dict, algorithm_name):
    """Format results as DataFrame matching paper table format."""
    conditions = get_all_conditions()
    datasets_list = sorted(results_dict.keys())
    data = []
    for ds in datasets_list:
        row = [results_dict[ds].get(c, 0.0) for c in conditions]
        data.append(row)
    return pd.DataFrame(data, index=datasets_list, columns=conditions)


def compute_aggregate_stats(results_dict):
    """Compute win rates and average win/loss % (Tables 1-4 in paper)."""
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
    """Wilcoxon signed-rank test: DR vs no-reduction."""
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
    """Generate boxplots matching paper's figures (one per clustering algorithm)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    algo_labels = {'k-means': 'K-Means', 'AHC': 'Agglomerative', 'GMM': 'Gaussian Mixture', 'OPTICS': 'OPTICS'}
    conditions = get_all_conditions()

    for algo in ["k-means", "AHC", "GMM", "OPTICS"]:
        results_dict = all_results.get(algo)
        if results_dict is None:
            continue
        if not results_dict:
            continue
        fig, ax = plt.subplots(figsize=(16, 6))
        box_data = []
        xlabels = []
        for cond in conditions:
            vals = [results_dict[ds].get(cond, 0.0) for ds in results_dict]
            box_data.append(vals)
            if cond == 'No Reduction':
                xlabels.append('No Red.')
            else:
                parts = cond.rsplit('_', 1)
                method = parts[0]
                level = parts[1]
                xlabels.append(f"{method}\n{level}")

        bp = ax.boxplot(box_data, patch_artist=True)
        colors = ['gray'] + ['#1f77b4']*3 + ['#ff7f0e']*3 + ['#2ca02c']*3 + ['#d62728']*3 + ['#9467bd']*3
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('ARI')
        ax.set_title(f'{algo_labels.get(algo, algo)} — {data_label}')
        plt.tight_layout()
        fname = f'boxplot_{algo}_{data_label}.pdf'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_datasets(random_state=42):
    """Generate synthetic datasets: Circles, Moons, RSG, Repliclust.
    Returns dict: {name: (X, y, k)}"""
    from sklearn.datasets import make_circles, make_moons
    
    rng = np.random.RandomState(random_state)
    synthetic = {}
    
    def add_noise_dims(X, target_dims, rng_local):
        """Add noisy dimensions. 25% N(0,1), 25% N(0,0.5), 25% N(0,0.25), 25% no noise."""
        n_samples = X.shape[0]
        n_orig = X.shape[1]
        n_extra = target_dims - n_orig
        if n_extra <= 0:
            return X
        n_per = n_extra // 4
        remainder = n_extra - 4 * n_per
        parts = []
        for sigma, extra in zip([1.0, 0.5, 0.25, 0.0], [0]*4):
            n_this = n_per + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            if sigma > 0:
                parts.append(rng_local.normal(0, sigma, (n_samples, n_this)))
            else:
                parts.append(np.zeros((n_samples, n_this)))
        noise = np.hstack(parts)[:, :n_extra]
        return np.hstack([X, noise])
    
    def embed_to_high_dim(X, target_dim, seed):
        """Embed 2D data into higher dimensions via random projection."""
        if target_dim <= X.shape[1]:
            return X
        r = np.random.RandomState(seed)
        proj = r.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        return X @ proj
    
    dims = [10, 50, 200]
    n_per_config = 10  # 10 datasets per config (paper uses 50, we use 10 for speed)
    
    # === CIRCLES ===
    # 2-cluster circles
    for d in dims:
        for i in range(n_per_config):
            X, y = make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=random_state+i)
            X = embed_to_high_dim(X, d, random_state+i+1000)
            X = add_noise_dims(X, d, rng)
            synthetic[f'Circles_k2_d{d}_t{i}'] = (X, y, 2)
    
    # 5-cluster circles
    for d in dims:
        for i in range(n_per_config):
            r = np.random.RandomState(random_state + i + 2000)
            n_per = 400
            X_list, y_list = [], []
            for ci, factor in enumerate([1.0, 2.0, 3.5, 5.0, 7.0]):
                theta = r.uniform(0, 2*np.pi, n_per)
                rad = factor + r.normal(0, 0.05, n_per)
                X_list.append(np.column_stack([rad*np.cos(theta), rad*np.sin(theta)]))
                y_list.append(np.full(n_per, ci))
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            X = embed_to_high_dim(X, d, random_state+i+3000)
            X = add_noise_dims(X, d, rng)
            synthetic[f'Circles_k5_d{d}_t{i}'] = (X, y, 5)
    
    # === MOONS ===
    # 2-cluster moons
    for d in dims:
        for i in range(n_per_config):
            X, y = make_moons(n_samples=2000, noise=0.1, random_state=random_state+i+4000)
            X = embed_to_high_dim(X, d, random_state+i+5000)
            X = add_noise_dims(X, d, rng)
            synthetic[f'Moons_k2_d{d}_t{i}'] = (X, y, 2)
    
    # 5-cluster moons
    for d in dims:
        for i in range(n_per_config):
            r = np.random.RandomState(random_state + i + 6000)
            n_per = 400
            X_list, y_list = [], []
            stretches = [1.0, 1.5, 1.0, 1.5, 1.0]
            rotations = [0, 160, -160, 10, 180]
            x_shifts = [0, 3, -3, 2, -2]
            y_shifts = [0, 1.0, 1.2, 1.5, 1.0]
            for ci in range(5):
                theta = np.linspace(0, np.pi, n_per)
                x = np.cos(theta) * stretches[ci] + r.normal(0, 0.1, n_per)
                yc = np.sin(theta) + r.normal(0, 0.1, n_per)
                angle = np.radians(rotations[ci])
                xr = x*np.cos(angle) - yc*np.sin(angle) + x_shifts[ci]
                yr = x*np.sin(angle) + yc*np.cos(angle) + y_shifts[ci]
                X_list.append(np.column_stack([xr, yr]))
                y_list.append(np.full(n_per, ci))
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            X = embed_to_high_dim(X, d, random_state+i+7000)
            X = add_noise_dims(X, d, rng)
            synthetic[f'Moons_k5_d{d}_t{i}'] = (X, y, 5)
    
    # === RSG ===
    ks_rsg = [2, 10, 50]
    ds_rsg = [10, 50, 200]
    Ncs_rsg = [5, 50, 100]
    n_rsg_per = 3  # fewer per config since there are many configs
    
    for k in ks_rsg:
        for d in ds_rsg:
            for Nc in Ncs_rsg:
                for i in range(n_rsg_per):
                    r = np.random.RandomState(random_state + k*1000 + d*100 + Nc + i)
                    alpha = max(0.1, min(0.9, 1.0 - k*0.01 - d*0.001))
                    centers = r.randn(k, d) * np.sqrt(d) * (1 + alpha)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * alpha
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples)
                        y_list.append(np.full(Nc, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = add_noise_dims(X, d, rng)
                    synthetic[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    
    # === REPLICLUST ===
    try:
        import repliclust
        for k in [2, 5]:
            Nc = 1000 if k == 2 else 400
            for d in dims:
                for i in range(n_per_config):
                    try:
                        archetype = repliclust.Archetype(
                            n_clusters=k, dim=d, n_samples=k*Nc,
                            aspect_ref=3.0, radius=5.0
                        )
                        X, y, _ = repliclust.DataGenerator(archetype).synthesize(random_state=random_state+i)
                        X = add_noise_dims(X, d, rng)
                        synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
                    except:
                        # Fallback to anisotropic Gaussian
                        r = np.random.RandomState(random_state + i + 9000 + k*100 + d)
                        centers = r.randn(k, d) * np.sqrt(d)
                        X_list, y_list = [], []
                        for ci in range(k):
                            A = r.randn(d, d) * 0.5
                            cov = A @ A.T / d + np.eye(d) * 0.1
                            samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                            X_list.append(samples)
                            y_list.append(np.full(Nc, ci))
                        X = np.vstack(X_list)
                        y = np.concatenate(y_list)
                        X = add_noise_dims(X, d, rng)
                        synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    except ImportError:
        print("Repliclust not available, using fallback Gaussian clusters")
        for k in [2, 5]:
            Nc = 1000 if k == 2 else 400
            for d in dims:
                for i in range(n_per_config):
                    r = np.random.RandomState(random_state + i + 9000 + k*100 + d)
                    centers = r.randn(k, d) * np.sqrt(d)
                    X_list, y_list = [], []
                    for ci in range(k):
                        A = r.randn(d, d) * 0.5
                        cov = A @ A.T / d + np.eye(d) * 0.1
                        samples = r.multivariate_normal(centers[ci], cov, size=Nc)
                        X_list.append(samples)
                        y_list.append(np.full(Nc, ci))
                    X = np.vstack(X_list)
                    y = np.concatenate(y_list)
                    X = add_noise_dims(X, d, rng)
                    synthetic[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    
    return synthetic


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_all_real_world(output_dir='./results'):
    """Run full real-world experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    from load_uci import load_all_uci

    print("=" * 60)
    print("LOADING UCI DATASETS")
    print("=" * 60)
    dataset_list = load_all_uci()
    datasets = {}
    for name, X, y, k in dataset_list:
        datasets[name] = (X, y, k)
    print(f"Loaded {len(datasets)} datasets")

    print("\n" + "=" * 60)
    print("PRECOMPUTING DIMENSIONALITY REDUCTIONS")
    print("=" * 60)
    t0 = time.time()
    dr_cache = precompute_all_dr(datasets)
    print(f"DR precomputation took {time.time()-t0:.1f}s")

    all_results = {}

    # --- k-means ---
    print("\n" + "=" * 60)
    print("K-MEANS EXPERIMENTS")
    print("=" * 60)
    t0 = time.time()
    all_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)
    print(f"k-means took {time.time()-t0:.1f}s")
    with open(os.path.join(output_dir, 'real_world_results_partial.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # --- AHC ---
    print("\n" + "=" * 60)
    print("AHC EXPERIMENTS (with hyperparameter search)")
    print("=" * 60)
    t0 = time.time()
    ahc_metric, ahc_linkage = find_best_ahc_params(datasets, dr_cache)
    all_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_metric, ahc_linkage)
    print(f"AHC took {time.time()-t0:.1f}s")
    with open(os.path.join(output_dir, 'real_world_results_partial.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # --- GMM ---
    print("\n" + "=" * 60)
    print("GMM EXPERIMENTS (with hyperparameter search)")
    print("=" * 60)
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    all_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)
    print(f"GMM took {time.time()-t0:.1f}s")
    with open(os.path.join(output_dir, 'real_world_results_partial.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # --- OPTICS ---
    print("\n" + "=" * 60)
    print("OPTICS EXPERIMENTS (with hyperparameter search)")
    print("=" * 60)
    t0 = time.time()
    optics_ms, optics_mcs = find_best_optics_params(datasets, dr_cache)
    all_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, optics_ms, optics_mcs)
    print(f"OPTICS took {time.time()-t0:.1f}s")

    # Save final results
    with open(os.path.join(output_dir, 'real_world_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save chosen hyperparameters
    hyperparams = {
        'AHC': {'metric': ahc_metric, 'linkage': ahc_linkage},
        'GMM': {'covariance_type': gmm_cov},
        'OPTICS': {'min_samples': int(optics_ms), 'min_cluster_size': float(optics_mcs)},
    }
    with open(os.path.join(output_dir, 'chosen_hyperparams_real.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # --- Format tables ---
    print("\n" + "=" * 60)
    print("RESULTS TABLES")
    print("=" * 60)
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        df = format_results_table(all_results[algo], algo)
        df.to_csv(os.path.join(output_dir, f'table_{algo}_real.csv'))
        print(f"\n{algo} results:")
        print(df.to_string())

    # --- Aggregate stats ---
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    agg_stats = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        agg_stats[algo] = compute_aggregate_stats(all_results[algo])
        print(f"\n{algo}:")
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                s = agg_stats[algo][method][level]
                print(f"  {method:12s} {level:4s}: win={s['win_pct']:5.1f}%  loss={s['loss_pct']:5.1f}%  avg_win={s['avg_win_change']:+7.2f}%  avg_loss={s['avg_loss_change']:+7.2f}%")
    with open(os.path.join(output_dir, 'aggregate_stats_real.json'), 'w') as f:
        json.dump(agg_stats, f, indent=2)

    # --- Wilcoxon test ---
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 60)
    wilcoxon_results = {}
    for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
        wilcoxon_results[algo] = compute_wilcoxon_tests(all_results[algo])
        print(f"\n{algo}:")
        for method in DR_METHODS:
            vals = [wilcoxon_results[algo][method][l] for l in REDUCTION_LEVELS]
            sig = ['*' if v < 0.05 else ' ' for v in vals]
            print(f"  {method:12s}: k-1={vals[0]:.4f}{sig[0]}  25%={vals[1]:.4f}{sig[1]}  50%={vals[2]:.4f}{sig[2]}")
    with open(os.path.join(output_dir, 'wilcoxon_results_real.json'), 'w') as f:
        json.dump(wilcoxon_results, f, indent=2)

    # --- Boxplots ---
    print("\n" + "=" * 60)
    print("GENERATING BOXPLOTS")
    print("=" * 60)
    generate_boxplots(all_results, 'RealWorld', output_dir)

    print("\nAll real-world experiments complete!")
    return all_results


def run_all_synthetic(output_dir='./results'):
    """Run full synthetic experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING SYNTHETIC DATASETS")
    print("=" * 60)
    all_synthetic = generate_synthetic_datasets()
    print(f"Generated {len(all_synthetic)} total synthetic datasets")

    # Group by type
    type_names = ['Circles', 'Moons', 'RSG', 'Repliclust']
    all_type_results = {}

    for dtype in type_names:
        datasets = {k: v for k, v in all_synthetic.items() if k.startswith(dtype)}
        if not datasets:
            print(f"No {dtype} datasets, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"PROCESSING {dtype.upper()} ({len(datasets)} datasets)")
        print(f"{'='*60}")

        dr_cache = precompute_all_dr(datasets)

        type_results = {}
        type_results['k-means'] = run_kmeans_experiments(datasets, dr_cache)

        ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache)
        type_results['AHC'] = run_ahc_experiments(datasets, dr_cache, ahc_m, ahc_l)

        gmm_cov = find_best_gmm_params(datasets, dr_cache)
        type_results['GMM'] = run_gmm_experiments(datasets, dr_cache, gmm_cov)

        opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
        type_results['OPTICS'] = run_optics_experiments(datasets, dr_cache, opt_ms, opt_mcs)

        all_type_results[dtype] = type_results

        with open(os.path.join(output_dir, f'synthetic_{dtype}_results.json'), 'w') as f:
            json.dump(type_results, f, indent=2)

        # Print average ARI table
        conditions = get_all_conditions()
        print(f"\nAverage ARI for {dtype}:")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            avg_row = {}
            for cond in conditions:
                vals = [type_results[algo][ds].get(cond, 0.0) for ds in type_results[algo]]
                avg_row[cond] = round(np.mean(vals), 2)
            print(f"  {algo:8s}: No_Red={avg_row['No Reduction']:.2f}", end="")
            for m in DR_METHODS:
                best_l = max(REDUCTION_LEVELS, key=lambda l: avg_row.get(f"{m}_{l}", 0))
                print(f"  {m}_best={avg_row.get(f'{m}_{best_l}', 0):.2f}", end="")
            print()

    # Save all results
    with open(os.path.join(output_dir, 'synthetic_results_all.json'), 'w') as f:
        json.dump(all_type_results, f, indent=2)

    # Generate boxplots per type
    for dtype in type_names:
        if dtype in all_type_results:
            generate_boxplots(all_type_results[dtype], f'Synthetic_{dtype}', output_dir)

    # Aggregate across all synthetic
    print("\n" + "=" * 60)
    print("SYNTHETIC AGGREGATE STATISTICS")
    print("=" * 60)
    for dtype in type_names:
        if dtype not in all_type_results:
            continue
        print(f"\n--- {dtype} ---")
        for algo in ['k-means', 'AHC', 'GMM', 'OPTICS']:
            stats = compute_aggregate_stats(all_type_results[dtype][algo])
            print(f"  {algo}:")
            for method in DR_METHODS:
                for level in REDUCTION_LEVELS:
                    s = stats[method][level]
                    print(f"    {method:12s} {level:4s}: win={s['win_pct']:5.1f}%  loss={s['loss_pct']:5.1f}%")

    print("\nAll synthetic experiments complete!")
    return all_type_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'synthetic', 'all'], default='real')
    parser.add_argument('--output_dir', default='./results')
    args = parser.parse_args()

    if args.mode in ['real', 'all']:
        run_all_real_world(args.output_dir)

    if args.mode in ['synthetic', 'all']:
        run_all_synthetic(args.output_dir)

    print(f"\nAll done! Results in {args.output_dir}")
