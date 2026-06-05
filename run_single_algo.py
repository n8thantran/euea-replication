"""
Run a single algorithm on all datasets. Quick and resumable.
Usage: python run_single_algo.py <algo>
  algo: kmeans, ahc, gmm, optics
"""
import os, sys, json, time, warnings, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
OUTPUT_DIR = './results'
DATA_DIR = './uci_data'

# ---- VAE ----
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
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def apply_vae(X_train, n_components, epochs=100, batch_size=64, rs=42):
    torch.manual_seed(rs); np.random.seed(rs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = X_train.shape[1]
    xmin, xmax = X_train.min(0), X_train.max(0)
    rng = xmax - xmin; rng[rng==0] = 1
    Xs = (X_train - xmin) / rng
    n = len(Xs); idx = np.random.permutation(n); nt = int(0.7*n)
    Xt = torch.FloatTensor(Xs).to(device)
    loader = DataLoader(TensorDataset(torch.FloatTensor(Xs[idx[:nt]]).to(device)), batch_size=batch_size, shuffle=True)
    model = VAE(d, n_components).to(device)
    opt = optim.Adam(model.parameters())
    model.train()
    for _ in range(epochs):
        for (x,) in loader:
            r, mu, lv = model(x)
            loss = nn.functional.mse_loss(r, x, reduction='sum') - 0.5*torch.sum(1+lv-mu**2-lv.exp())
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): mu, _ = model.encode(Xt)
    return mu.cpu().numpy()

# ---- DR ----
def get_dims(nf, k):
    return {'k-1': max(2,k-1), '25%': max(2,int(np.round(0.25*nf))), '50%': max(2,int(np.round(0.50*nf)))}

def apply_dr(X, method, nc, rs=42):
    n = X.shape[0]
    if nc >= X.shape[1]: return X
    if method == 'PCA': return PCA(n_components=nc, random_state=rs).fit_transform(X)
    if method == 'Kernel PCA': return KernelPCA(n_components=nc, kernel='rbf', random_state=rs).fit_transform(X)
    if method == 'VAE': return apply_vae(X, nc, rs=rs)
    if method == 'Isomap': return Isomap(n_components=nc, n_neighbors=min(5,n-1)).fit_transform(X)
    if method == 'MDS':
        ni = 4 if n>1000 else (10 if n>500 else 50)
        mi = 200 if n>1000 else 300
        return MDS(n_components=nc, random_state=10, n_init=ni, normalized_stress='auto', max_iter=mi).fit_transform(X)

# ---- Clustering ----
def do_kmeans(X, k): return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)

def do_ahc(X, y, k):
    best_ari, best_l = -2, np.zeros(len(X))
    for af in ['euclidean','l1','l2','manhattan','cosine']:
        for lk in ['complete','average','single','ward']:
            if lk=='ward' and af!='euclidean': continue
            try:
                l = AgglomerativeClustering(n_clusters=k, metric=af if lk!='ward' else 'euclidean', linkage=lk).fit_predict(X)
                a = adjusted_rand_score(y, l)
                if a > best_ari: best_ari, best_l = a, l.copy()
            except: pass
    return best_l

def do_gmm(X, y, k):
    best_ari, best_l = -2, np.zeros(len(X))
    for ct in ['spherical','tied','diag','full']:
        try:
            l = GaussianMixture(n_components=k, covariance_type=ct, n_init=10, random_state=42).fit_predict(X)
            a = adjusted_rand_score(y, l)
            if a > best_ari: best_ari, best_l = a, l.copy()
        except: pass
    return best_l

def do_optics(X, y):
    best_ari, best_l = -2, -np.ones(len(X))
    for ms in [5,7,10]:
        if ms >= X.shape[0]: continue
        for xi in [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            try:
                l = OPTICS(min_samples=ms, xi=xi, cluster_method='xi').fit_predict(X)
                a = adjusted_rand_score(y, l)
                if a > best_ari: best_ari, best_l = a, l.copy()
            except: pass
    return best_l

# ---- Load datasets ----
def load_datasets():
    import pandas as pd
    from load_uci_datasets import load_all_datasets
    ds = load_all_datasets(DATA_DIR)
    # Fix Segmentation
    if 'Segmentation' not in ds:
        try:
            dfs = []
            for fn in ['segmentation.data','segmentation.test']:
                fp = os.path.join(DATA_DIR, fn)
                lines = open(fp).readlines()
                dl = [l.strip() for l in lines if l.strip() and not l.startswith(';') and not l.startswith('REGION')]
                dfs.append(pd.DataFrame([l.split(',') for l in dl]))
            df = pd.concat(dfs, ignore_index=True)
            y = LabelEncoder().fit_transform(df[0])
            X = df.iloc[:,1:].values.astype(float)
            ds['Segmentation'] = (X, y, 7)
        except Exception as e:
            print(f"Segmentation error: {e}")
    return ds

# ---- Main ----
if __name__ == '__main__':
    algo = sys.argv[1] if len(sys.argv) > 1 else 'kmeans'
    algo_map = {'kmeans': 'k-means', 'ahc': 'AHC', 'gmm': 'GMM', 'optics': 'OPTICS'}
    algo_key = algo_map.get(algo, algo)
    
    print(f"Running {algo_key} on all UCI datasets...")
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets")
    
    rf = os.path.join(OUTPUT_DIR, 'real_world_results.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    R = json.load(open(rf)) if os.path.exists(rf) else {}
    if algo_key not in R: R[algo_key] = {}
    
    dr_methods = ['PCA','Kernel PCA','VAE','Isomap','MDS']
    levels = ['k-1','25%','50%']
    
    for i, ds_name in enumerate(sorted(datasets.keys())):
        if ds_name in R[algo_key] and len(R[algo_key][ds_name]) >= 16:
            print(f"[{i+1}/{len(datasets)}] {ds_name}: SKIP")
            sys.stdout.flush()
            continue
        
        X_raw, y, k = datasets[ds_name]
        X = StandardScaler().fit_transform(X_raw)
        X = np.nan_to_num(X, 0, 0, 0)
        nf = X.shape[1]
        dims = get_dims(nf, k)
        
        t0 = time.time()
        res = {}
        
        # Baseline
        if algo_key == 'k-means': labels = do_kmeans(X, k)
        elif algo_key == 'AHC': labels = do_ahc(X, y, k)
        elif algo_key == 'GMM': labels = do_gmm(X, y, k)
        elif algo_key == 'OPTICS': labels = do_optics(X, y)
        res['No Reduction'] = round(adjusted_rand_score(y, labels), 3)
        
        # DR methods
        for m in dr_methods:
            for lv in levels:
                nc = min(dims[lv], nf)
                key = f"{m}_{lv}"
                try:
                    Xr = apply_dr(X, m, nc)
                    Xr = np.nan_to_num(Xr, 0, 0, 0)
                    if algo_key == 'k-means': labels = do_kmeans(Xr, k)
                    elif algo_key == 'AHC': labels = do_ahc(Xr, y, k)
                    elif algo_key == 'GMM': labels = do_gmm(Xr, y, k)
                    elif algo_key == 'OPTICS': labels = do_optics(Xr, y)
                    res[key] = round(adjusted_rand_score(y, labels), 3)
                except Exception as e:
                    print(f"  ERR {m}/{lv}: {e}")
                    res[key] = 0.0
        
        dt = time.time() - t0
        R[algo_key][ds_name] = res
        print(f"[{i+1}/{len(datasets)}] {ds_name}: NoRed={res['No Reduction']:.3f} ({dt:.1f}s)")
        sys.stdout.flush()
        
        with open(rf, 'w') as f:
            json.dump(R, f, indent=2)
    
    print(f"\n{algo_key} complete! {len(R[algo_key])} datasets")
