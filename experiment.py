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


def apply_vae(X, n_components, epochs=100, batch_size=64, random_state=42):
    """Train VAE, return z_mean as embedding for ALL samples.
    Paper: 100 epochs, batch=64, Adam, MSE+KL, 70/30 split."""
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
    """3 reduction levels: k-1, 25%, 50% of features (min 2, max n_features-1)."""
    return {
        'k-1': min(n_features - 1, max(2, k - 1)),
        '25%': min(n_features - 1, max(2, int(np.round(0.25 * n_features)))),
        '50%': min(n_features - 1, max(2, int(np.round(0.50 * n_features)))),
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
        # Paper: random_state=10, n_init=50. Using n_init=4 for speed.
        return MDS(n_components=n_components, random_state=10, n_init=4,
                   max_iter=300, normalized_stress='auto').fit_transform(X)
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


def precompute_all_dr(datasets, timeout_sec=120):
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
        print(f"  [{i+1}/{total}] {ds_name} (shape={X.shape}, k={k}, dims={dims})")

        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                key = f"{method}_{level}"
                n_comp = dims[level]
                try:
                    t0 = time.time()
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_sec)
                    X_red = apply_dr(X, method, n_comp)
                    signal.alarm(0)
                    X_red = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)
                    ds_cache[key] = X_red
                    dt = time.time() - t0
                    if dt > 5:
                        print(f"    {key} ({n_comp}d) [{dt:.1f}s]")
                except TimeoutError:
                    signal.alarm(0)
                    print(f"    TIMEOUT {key} (>{timeout_sec}s, skipping)")
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
    """k-means++, n_init=100 (paper specification)."""
    return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=random_state).fit_predict(X)


def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    """AHC with specific metric and linkage."""
    return AgglomerativeClustering(
        n_clusters=k, metric=metric, linkage=linkage
    ).fit_predict(X)


def cluster_gmm(X, k, covariance_type='full', random_state=42):
    """GMM with specific covariance type."""
    return GaussianMixture(
        n_components=k, covariance_type=covariance_type, n_init=1, random_state=random_state
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
    """All valid (metric, linkage) combinations for AHC.
    Paper: euclidean/l1/l2/manhattan/cosine × complete/average/single/ward
    Note: l1=manhattan, l2=euclidean in sklearn. Ward only with euclidean."""
    combos = []
    for metric in ['euclidean', 'manhattan', 'cosine']:
        for linkage in ['complete', 'average', 'single']:
            combos.append((metric, linkage))
    combos.append(('euclidean', 'ward'))
    return combos


def get_gmm_combos():
    """Paper: spherical/tied/diag/full."""
    return ['spherical', 'tied', 'diag', 'full']


def find_best_ahc_params(datasets, dr_cache):
    """Find (metric, linkage) with highest average ARI across all datasets × conditions."""
    combos = get_ahc_combos()
    conditions = get_all_conditions()
    best_score = -999
    best_combo = ('euclidean', 'ward')

    print(f"  AHC hyperparam search ({len(datasets)} datasets, {len(combos)} combos)...")
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

    print(f"  GMM hyperparam search ({len(datasets)} datasets)...")
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
    Paper: min_samples 5-10 step 1, min_cluster_size 0-1 step 0.05."""
    conditions = get_all_conditions()
    # Use coarser grid for speed but cover the range
    ms_values = [5, 6, 7, 8, 9, 10]
    mcs_values = [round(v, 2) for v in np.arange(0.05, 1.05, 0.1)]

    print(f"  OPTICS hyperparam search ({len(datasets)} datasets, {len(ms_values)*len(mcs_values)} combos)...")
    combo_scores = {}
    for ms in ms_values:
        for mcs in mcs_values:
            combo_scores[(ms, mcs)] = []

    for ds_name in sorted(datasets.keys()):
        _, y_true, k = datasets[ds_name]
        for cond in conditions:
            X = dr_cache[ds_name].get(cond)
            if X is None:
                continue
            for ms in ms_values:
                if ms >= X.shape[0]:
                    continue
                try:
                    model = OPTICS(min_samples=ms, cluster_method='xi', xi=0.05, min_cluster_size=0.05)
                    model.fit(X)
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
                row[cond] = round(adjusted_rand_score(y_true, labels), 3)
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
                row[cond] = round(adjusted_rand_score(y_true, labels), 3)
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
                row[cond] = round(adjusted_rand_score(y_true, labels), 3)
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
                row[cond] = round(adjusted_rand_score(y_true, labels), 3)
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
    """Compute win rates and average win/loss % (Tables 1-4 in paper).
    
    Paper definition:
    - "Percentage of wins": % of datasets where DR beats baseline
    - "Average win/loss percentage": average relative ARI change across ALL datasets
      = mean of ((ARI_reduced - ARI_baseline) / ARI_baseline * 100) for all datasets
    
    When baseline is 0, we handle it specially.
    """
    datasets_list = sorted(results_dict.keys())
    stats = {}
    for method in DR_METHODS:
        stats[method] = {}
        for level in REDUCTION_LEVELS:
            key = f"{method}_{level}"
            wins = 0
            total = 0
            all_pct_changes = []

            for ds in datasets_list:
                baseline = results_dict[ds].get('No Reduction', 0.0)
                reduced = results_dict[ds].get(key, None)
                if reduced is None:
                    continue
                total += 1

                # Win if reduced > baseline (with small tolerance for rounding)
                if reduced > baseline + 0.005:
                    wins += 1

                # Percentage change
                if abs(baseline) > 1e-6:
                    pct_change = (reduced - baseline) / abs(baseline) * 100
                else:
                    # If baseline is ~0, use absolute change * 100
                    pct_change = (reduced - baseline) * 100
                all_pct_changes.append(pct_change)

            win_pct = round(wins / total * 100, 2) if total > 0 else 0
            avg_pct = round(np.mean(all_pct_changes), 2) if all_pct_changes else 0

            stats[method][level] = {
                'win_pct': win_pct,
                'avg_win_loss_pct': avg_pct,
                'wins': wins,
                'total': total,
            }
    return stats


def compute_wilcoxon_tests(results_dict):
    """Wilcoxon signed-rank test: one-sided, H1: DR > baseline.
    Paper: alpha=0.05, n=20 datasets."""
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
                    # One-sided: alternative='greater' means H1: diffs > 0 (DR > baseline)
                    stat, p = wilcoxon(diffs[nonzero], alternative='greater')
                    p_values[method][level] = round(p, 3)
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
        if results_dict is None or not results_dict:
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


def format_aggregate_table(stats, algo_name):
    """Format aggregate stats as a table matching paper Tables 1-4."""
    rows = []
    for method in DR_METHODS:
        for level in REDUCTION_LEVELS:
            s = stats[method][level]
            rows.append({
                'Method': method,
                'Reduction': level,
                'Win %': s['win_pct'],
                'Avg Win/Loss %': s['avg_win_loss_pct'],
            })
    return pd.DataFrame(rows)


def format_wilcoxon_table(wilcoxon_results):
    """Format Wilcoxon p-values as table matching paper Table."""
    rows = []
    for method in DR_METHODS:
        for level in REDUCTION_LEVELS:
            rows.append({
                'Method': method,
                'Reduction': level,
                'p-value': wilcoxon_results[method][level],
            })
    return pd.DataFrame(rows)
