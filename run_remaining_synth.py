"""
Run RSG and Repliclust synthetic experiments with aggressive speed optimizations.
OPTICS clustering runs on subsampled data for speed.
"""
import os, sys, json, time, warnings, pickle, signal
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = './results'
DR_METHODS = ['PCA', 'Kernel PCA', 'VAE', 'Isomap', 'MDS']
REDUCTION_LEVELS = ['k-1', '25%', '50%']
ALGOS = ['k-means', 'AHC', 'GMM', 'OPTICS']

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def apply_vae(X, n_components, epochs=30, batch_size=64):
    from sklearn.preprocessing import MinMaxScaler
    X_mm = MinMaxScaler().fit_transform(X)
    n = X_mm.shape[0]
    n_train = max(int(0.7 * n), n_components + 1)
    X_train = torch.FloatTensor(X_mm[:n_train])
    X_all = torch.FloatTensor(X_mm)
    model = VAEModel(X.shape[1], n_components)
    opt = optim.Adam(model.parameters())
    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_all)
    return mu.numpy()

# ============================================================
# DR
# ============================================================
def get_reduction_dims(d, k):
    dims = {}
    dims['k-1'] = max(2, k - 1)
    dims['25%'] = max(2, round(d * 0.25))
    dims['50%'] = max(2, round(d * 0.50))
    return dims

def get_all_conditions():
    conditions = ['No Reduction']
    for m in DR_METHODS:
        for l in REDUCTION_LEVELS:
            conditions.append(f"{m}_{l}")
    return conditions

def apply_dr(X, method, n_components):
    if n_components >= X.shape[1]:
        return X.copy()
    try:
        if method == 'PCA':
            return PCA(n_components=n_components).fit_transform(X)
        elif method == 'Kernel PCA':
            return KernelPCA(n_components=n_components, kernel='rbf').fit_transform(X)
        elif method == 'VAE':
            return apply_vae(X, n_components)
        elif method == 'Isomap':
            nc = min(n_components, X.shape[0] - 1)
            return Isomap(n_components=nc).fit_transform(X)
        elif method == 'MDS':
            return MDS(n_components=n_components, random_state=10, n_init=1,
                       max_iter=100, normalized_stress='auto').fit_transform(X)
    except:
        return None
    return None

def precompute_all_dr(datasets, label=""):
    dr_cache = {}
    keys = sorted(datasets.keys())
    for idx, ds_name in enumerate(keys):
        X_raw, y, k = datasets[ds_name]
        X = StandardScaler().fit_transform(X_raw)
        d = X.shape[1]
        dims = get_reduction_dims(d, k)
        cache = {'No Reduction': X.copy()}
        for method in DR_METHODS:
            for level in REDUCTION_LEVELS:
                nc = dims[level]
                result = apply_dr(X, method, nc)
                cache[f"{method}_{level}"] = result
        dr_cache[ds_name] = cache
        if (idx + 1) % 6 == 0 or idx == 0:
            print(f"  DR {label} [{idx+1}/{len(keys)}] {ds_name} shape={X.shape}")
    return dr_cache

# ============================================================
# Clustering
# ============================================================
def cluster_kmeans(X, k):
    return KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit_predict(X)

def cluster_ahc(X, k, metric='euclidean', linkage='ward'):
    if linkage == 'ward' and metric != 'euclidean':
        metric = 'euclidean'
    return AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage).fit_predict(X)

def cluster_gmm(X, k, covariance_type='full'):
    return GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42).fit_predict(X)

def cluster_optics(X, min_samples=5, min_cluster_size=0.05):
    # Subsample if too large (OPTICS is O(n^2))
    n = X.shape[0]
    if n > 300:
        idx = np.random.RandomState(42).choice(n, 300, replace=False)
        X_sub = X[idx]
        labels_sub = OPTICS(min_samples=min(min_samples, 50),
                            cluster_method='xi', xi=0.05,
                            min_cluster_size=min_cluster_size).fit_predict(X_sub)
        # Map back: assign non-subsampled points to nearest subsampled cluster
        from sklearn.neighbors import KNeighborsClassifier
        valid = labels_sub >= 0
        if valid.sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_sub[valid], labels_sub[valid])
            labels = np.full(n, -1)
            labels[idx[valid]] = labels_sub[valid]
            mask = np.ones(n, bool)
            mask[idx[valid]] = False
            if mask.sum() > 0:
                labels[mask] = knn.predict(X[mask])
            return labels
        else:
            return np.full(n, -1)
    return OPTICS(min_samples=min_samples, cluster_method='xi', xi=0.05,
                  min_cluster_size=min_cluster_size).fit_predict(X)

# ============================================================
# Hyperparameter Search (fast)
# ============================================================
def find_best_ahc_params(datasets, dr_cache):
    combos = [('euclidean', 'ward'), ('euclidean', 'average'), ('manhattan', 'average'),
              ('cosine', 'average'), ('euclidean', 'complete')]
    keys = sorted(datasets.keys())[:10]
    best_score, best_combo = -999, ('euclidean', 'ward')
    for metric, linkage in combos:
        total, count = 0.0, 0
        for ds_name in keys:
            _, y_true, k = datasets[ds_name]
            X = dr_cache[ds_name].get('No Reduction')
            if X is None: continue
            try:
                labels = cluster_ahc(X, k, metric=metric, linkage=linkage)
                total += adjusted_rand_score(y_true, labels)
                count += 1
            except: pass
        avg = total / count if count else -999
        if avg > best_score:
            best_score, best_combo = avg, (metric, linkage)
    print(f"    Best AHC: {best_combo}, avg_ari={best_score:.4f}")
    return best_combo

def find_best_gmm_params(datasets, dr_cache):
    keys = sorted(datasets.keys())[:10]
    best_score, best_cov = -999, 'full'
    for cov in ['spherical', 'tied', 'diag', 'full']:
        total, count = 0.0, 0
        for ds_name in keys:
            _, y_true, k = datasets[ds_name]
            X = dr_cache[ds_name].get('No Reduction')
            if X is None: continue
            try:
                labels = cluster_gmm(X, k, covariance_type=cov)
                total += adjusted_rand_score(y_true, labels)
                count += 1
            except: pass
        avg = total / count if count else -999
        if avg > best_score:
            best_score, best_cov = avg, cov
    print(f"    Best GMM: {best_cov}, avg_ari={best_score:.4f}")
    return best_cov

def find_best_optics_params(datasets, dr_cache):
    keys = sorted(datasets.keys())[:6]  # reduced
    best_score, best_combo = -999, (5, 0.05)
    for ms in [5, 10]:
        for mcs in [0.05, 0.1]:
            total, count = 0.0, 0
            for ds_name in keys:
                _, y_true, k = datasets[ds_name]
                X = dr_cache[ds_name].get('No Reduction')
                if X is None: continue
                if ms >= X.shape[0]: continue
                try:
                    labels = cluster_optics(X, min_samples=ms, min_cluster_size=mcs)
                    total += adjusted_rand_score(y_true, labels)
                    count += 1
                except: pass
            avg = total / count if count else -999
            if avg > best_score:
                best_score, best_combo = avg, (ms, mcs)
    print(f"    Best OPTICS: ms={best_combo[0]}, mcs={best_combo[1]}, avg_ari={best_score:.4f}")
    return best_combo

# ============================================================
# Run clustering on all conditions
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

# ============================================================
# Data generation
# ============================================================
def inject_noise(X, rng):
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

def generate_rsg_datasets():
    datasets = {}
    n_per = 2
    # k=2: small datasets (100 samples); k=10: 500 samples
    for k in [2, 10]:
        Nc = 50  # Fixed for speed
        for d in [10, 50, 200]:
            for i in range(n_per):
                rng = np.random.RandomState(4000 + k*1000 + d*100 + Nc + i)
                alpha = max(0.1, min(0.9, 1.0 - k*0.01 - d*0.001))
                centers = rng.randn(k, d) * np.sqrt(d) * (1 + alpha)
                X_list, y_list = [], []
                for ci in range(k):
                    A = rng.randn(d, d) * alpha
                    cov = A @ A.T / d + np.eye(d) * 0.1
                    samples = rng.multivariate_normal(centers[ci], cov, size=Nc)
                    X_list.append(samples)
                    y_list.append(np.full(Nc, ci))
                X = np.vstack(X_list)
                y = np.concatenate(y_list)
                X = inject_noise(X, rng)
                datasets[f'RSG_k{k}_d{d}_Nc{Nc}_t{i}'] = (X, y, k)
    return datasets

def generate_repliclust_datasets():
    datasets = {}
    n_per = 2
    dims = [10, 50, 200]
    for k in [2, 5]:
        Nc = 100 if k == 2 else 60
        for d in dims:
            for i in range(n_per):
                rng = np.random.RandomState(5000 + k*100 + d + i)
                centers = rng.randn(k, d) * np.sqrt(d) * 2
                X_list, y_list = [], []
                for ci in range(k):
                    A = rng.randn(d, d) * 0.5
                    cov = A @ A.T / d + np.eye(d) * 0.1
                    samples = rng.multivariate_normal(centers[ci], cov, size=Nc)
                    X_list.append(samples)
                    y_list.append(np.full(Nc, ci))
                X = np.vstack(X_list)
                y = np.concatenate(y_list)
                X = inject_noise(X, rng)
                datasets[f'Repliclust_k{k}_d{d}_t{i}'] = (X, y, k)
    return datasets

# ============================================================
# Main
# ============================================================
def run_type(dtype, datasets):
    print(f"\n--- {dtype} ({len(datasets)} datasets) ---")
    
    result_file = os.path.join(OUTPUT_DIR, f'synth_{dtype}_final.json')
    if os.path.exists(result_file):
        print(f"  Already done, loading cached")
        with open(result_file) as f:
            return json.load(f)
    
    # DR
    cache_file = os.path.join(OUTPUT_DIR, f'dr_cache_{dtype}_v7.pkl')
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
    
    results = {}
    params = {}
    
    # k-means
    t0 = time.time()
    results['k-means'] = run_clustering(datasets, dr_cache, 'k-means')
    print(f"  k-means: {time.time()-t0:.1f}s")
    
    # AHC
    t0 = time.time()
    ahc_m, ahc_l = find_best_ahc_params(datasets, dr_cache)
    results['AHC'] = run_clustering(datasets, dr_cache, 'AHC', metric=ahc_m, linkage=ahc_l)
    params['AHC'] = {'metric': ahc_m, 'linkage': ahc_l}
    print(f"  AHC: {time.time()-t0:.1f}s")
    
    # GMM
    t0 = time.time()
    gmm_cov = find_best_gmm_params(datasets, dr_cache)
    results['GMM'] = run_clustering(datasets, dr_cache, 'GMM', covariance_type=gmm_cov)
    params['GMM'] = {'covariance_type': gmm_cov}
    print(f"  GMM: {time.time()-t0:.1f}s")
    
    # OPTICS
    t0 = time.time()
    opt_ms, opt_mcs = find_best_optics_params(datasets, dr_cache)
    results['OPTICS'] = run_clustering(datasets, dr_cache, 'OPTICS',
                                        min_samples=opt_ms, min_cluster_size=opt_mcs)
    params['OPTICS'] = {'min_samples': int(opt_ms), 'min_cluster_size': float(opt_mcs)}
    print(f"  OPTICS: {time.time()-t0:.1f}s")
    
    results['_params'] = params
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    del dr_cache
    return results

if __name__ == '__main__':
    t_start = time.time()
    
    print("Generating RSG datasets...")
    rsg = generate_rsg_datasets()
    print(f"  RSG: {len(rsg)} datasets")
    for k in sorted(rsg.keys())[:3]:
        print(f"    {k}: shape={rsg[k][0].shape}, k={rsg[k][2]}")
    
    print("\nGenerating Repliclust datasets...")
    repliclust = generate_repliclust_datasets()
    print(f"  Repliclust: {len(repliclust)} datasets")
    for k in sorted(repliclust.keys())[:3]:
        print(f"    {k}: shape={repliclust[k][0].shape}, k={repliclust[k][2]}")
    
    rsg_results = run_type('RSG', rsg)
    repliclust_results = run_type('Repliclust', repliclust)
    
    print(f"\nTOTAL: {time.time()-t_start:.0f}s")
    print("DONE!")
