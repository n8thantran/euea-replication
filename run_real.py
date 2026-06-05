#!/usr/bin/env python3
"""Fast real-world experiment runner with optimizations."""
import numpy as np, warnings, json, os, time, pickle, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wilcoxon
from itertools import product
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42); torch.manual_seed(42)
RESULTS_DIR = "/workspace/results"; os.makedirs(RESULTS_DIR, exist_ok=True)

DR_METHODS = ['PCA', 'KernelPCA', 'VAE', 'Isomap', 'MDS']
LEVELS = ['k-1', '25%', '50%']

# === VAE ===
class VAE(nn.Module):
    def __init__(self, d, z):
        super().__init__()
        self.e1 = nn.Linear(d, 64); self.bn1 = nn.BatchNorm1d(64)
        self.e2 = nn.Linear(64, 32); self.bn2 = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.4)
        self.mu = nn.Linear(32, z); self.lv = nn.Linear(32, z)
        self.d1 = nn.Linear(z, 32); self.dbn1 = nn.BatchNorm1d(32)
        self.d2 = nn.Linear(32, 64); self.dbn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, d)

    def encode(self, x):
        h = self.drop(torch.relu(self.bn1(self.e1(x))))
        h = self.drop(torch.relu(self.bn2(self.e2(h))))
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(lv) * torch.exp(0.5 * lv)
        h = self.drop(torch.relu(self.dbn1(self.d1(z))))
        h = self.drop(torch.relu(self.dbn2(self.d2(h))))
        return torch.sigmoid(self.out(h)), mu, lv

def train_vae(X, latent_dim, epochs=100, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, d = X.shape
    mn, mx = X.min(0), X.max(0); rng = mx - mn; rng[rng==0] = 1
    Xs = torch.FloatTensor((X - mn) / rng).to(device)
    n_tr = int(0.7*n); idx = np.random.permutation(n)
    Xtr = Xs[idx[:n_tr]]
    m = VAE(d, latent_dim).to(device); opt = torch.optim.Adam(m.parameters())
    m.train()
    for _ in range(epochs):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = Xtr[p[i:i+batch_size]]
            if len(b) < 2: continue
            r, mu, lv = m(b)
            loss = nn.functional.mse_loss(r, b, reduction='sum') - 0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): mu, _ = m.encode(Xs)
    return mu.cpu().numpy()

# === DR ===
def apply_dr(X, method, nc):
    n, d = X.shape; nc = min(nc, d-1)
    if nc < 1: return None
    try:
        if method == 'PCA': return PCA(n_components=nc).fit_transform(X)
        if method == 'KernelPCA':
            r = KernelPCA(n_components=nc, kernel='rbf').fit_transform(X)
            return r if r.shape[1]==nc else None
        if method == 'VAE': return train_vae(X, nc)
        if method == 'Isomap': return Isomap(n_components=nc).fit_transform(X)
        if method == 'MDS':
            ni = min(50, max(2, 50 if n<=300 else (4 if n<=1000 else 2)))
            mi = 300 if n<=500 else 100
            return MDS(n_components=nc, random_state=10, n_init=ni, max_iter=mi, normalized_stress='auto').fit_transform(X)
    except: pass
    return None

def get_dims(d, k):
    return {'k-1': max(2, k-1), '25%': max(1, round(d*0.25)), '50%': max(1, round(d*0.50))}

# === Clustering ===
def run_km(X, k): return KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42).fit_predict(X)
def run_ahc(X, k, m='euclidean', l='ward'):
    if l=='ward' and m!='euclidean': return None
    try: return AgglomerativeClustering(n_clusters=k, metric=m, linkage=l).fit_predict(X)
    except: return None
def run_gmm(X, k, c='full'):
    try: return GaussianMixture(n_components=k, covariance_type=c, random_state=42).fit_predict(X)
    except: return None
def run_optics(X, ms=5, mcs=0.05):
    try: return OPTICS(min_samples=ms, xi=0.05, cluster_method='xi', min_cluster_size=mcs if mcs>0 else None).fit_predict(X)
    except: return None

# === Hyper search ===
def find_ahc(dl):
    best_s, best_p = -999, ('euclidean','ward')
    for a, l in product(['euclidean','l1','l2','manhattan','cosine'], ['complete','average','single','ward']):
        if l=='ward' and a!='euclidean': continue
        s = [adjusted_rand_score(y, lb) for X,y,k in dl for lb in [run_ahc(X,k,a,l)] if lb is not None]
        if s and np.mean(s) > best_s: best_s, best_p = np.mean(s), (a,l)
    return best_p

def find_gmm(dl):
    best_s, best_c = -999, 'full'
    for c in ['spherical','tied','diag','full']:
        s = [adjusted_rand_score(y, lb) for X,y,k in dl for lb in [run_gmm(X,k,c)] if lb is not None]
        if s and np.mean(s) > best_s: best_s, best_c = np.mean(s), c
    return best_c

def find_optics(dl):
    """Reduced grid for speed: 4 combos."""
    best_s, best_p = -999, (5, 0.1)
    for ms in [5, 10]:
        for mcs in [0.1, 0.2]:
            s = [adjusted_rand_score(y, lb) for X,y,k in dl for lb in [run_optics(X,ms,mcs)] if lb is not None]
            if s and np.mean(s) > best_s: best_s, best_p = np.mean(s), (ms,mcs)
    return best_p

# === Main ===
def main():
    from load_uci import load_all_uci
    t0 = time.time()
    
    print("Loading UCI datasets...")
    uci = load_all_uci()
    ds = [(n, StandardScaler().fit_transform(X), y, k) for n,X,y,k in uci]
    
    # Hyper search
    print("Hyperparameter search...")
    dl = [(X,y,k) for _,X,y,k in ds]
    t1 = time.time()
    ahc_p = find_ahc(dl); print(f"  AHC: {ahc_p} ({time.time()-t1:.0f}s)")
    t1 = time.time()
    gmm_c = find_gmm(dl); print(f"  GMM: {gmm_c} ({time.time()-t1:.0f}s)")
    t1 = time.time()
    optics_p = find_optics(dl); print(f"  OPTICS: {optics_p} ({time.time()-t1:.0f}s)")
    
    # Cache DR results
    dr_cache_file = f"{RESULTS_DIR}/dr_cache_real_v3.pkl"
    if os.path.exists(dr_cache_file):
        with open(dr_cache_file, 'rb') as f: dr_cache = pickle.load(f)
        print(f"Loaded DR cache with {len(dr_cache)} entries")
    else:
        dr_cache = {}
    
    results = {}
    
    for di, (name, X, y, k) in enumerate(ds):
        d = X.shape[1]
        dims = get_dims(d, k)
        r = {'kmeans':{}, 'ahc':{}, 'gmm':{}, 'optics':{}}
        
        # Baseline
        r['kmeans']['None'] = round(adjusted_rand_score(y, run_km(X, k)), 2)
        lb = run_ahc(X, k, ahc_p[0], ahc_p[1])
        r['ahc']['None'] = round(adjusted_rand_score(y, lb), 2) if lb is not None else 0.0
        lb = run_gmm(X, k, gmm_c)
        r['gmm']['None'] = round(adjusted_rand_score(y, lb), 2) if lb is not None else 0.0
        lb = run_optics(X, optics_p[0], optics_p[1])
        r['optics']['None'] = round(adjusted_rand_score(y, lb), 2) if lb is not None else 0.0
        
        for dr in DR_METHODS:
            for lvl, nc in dims.items():
                key = f"{dr}_{lvl}"
                if nc >= d:
                    for algo in r: r[algo][key] = r[algo]['None']
                    continue
                
                cache_key = (name, dr, nc)
                if cache_key in dr_cache:
                    X_red = dr_cache[cache_key]
                else:
                    X_red = apply_dr(X, dr, nc)
                    dr_cache[cache_key] = X_red
                
                if X_red is None:
                    for algo in r: r[algo][key] = None
                    continue
                
                r['kmeans'][key] = round(adjusted_rand_score(y, run_km(X_red, k)), 2)
                lb = run_ahc(X_red, k, ahc_p[0], ahc_p[1])
                r['ahc'][key] = round(adjusted_rand_score(y, lb), 2) if lb is not None else None
                lb = run_gmm(X_red, k, gmm_c)
                r['gmm'][key] = round(adjusted_rand_score(y, lb), 2) if lb is not None else None
                lb = run_optics(X_red, optics_p[0], optics_p[1])
                r['optics'][key] = round(adjusted_rand_score(y, lb), 2) if lb is not None else None
        
        results[name] = r
        elapsed = time.time() - t0
        print(f"  [{di+1}/20] {name} ({elapsed:.0f}s) km={r['kmeans']['None']} ahc={r['ahc']['None']} gmm={r['gmm']['None']} opt={r['optics']['None']}")
        
        # Save cache periodically
        if (di+1) % 5 == 0:
            with open(dr_cache_file, 'wb') as f: pickle.dump(dr_cache, f)
    
    # Final cache save
    with open(dr_cache_file, 'wb') as f: pickle.dump(dr_cache, f)
    
    # Save results
    with open(f"{RESULTS_DIR}/real_world_results_v3.json", 'w') as f:
        json.dump({'results': results, 'params': {'ahc': list(ahc_p), 'gmm': gmm_c, 'optics': list(optics_p)}}, f, indent=2, default=str)
    
    # Tables
    algo_map = [('kmeans','k-means'), ('ahc','AHC'), ('gmm','GMM'), ('optics','OPTICS')]
    for ak, an in algo_map:
        header = ['Dataset','No Reduction'] + [f"{dr}_{lv}" for dr in DR_METHODS for lv in LEVELS]
        rows = [','.join(header)]
        for dname in results:
            ar = results[dname].get(ak, {})
            vals = [dname, str(ar.get('None',''))]
            for dr in DR_METHODS:
                for lv in LEVELS:
                    v = ar.get(f"{dr}_{lv}")
                    vals.append(str(v) if v is not None else '')
            rows.append(','.join(vals))
        with open(f"{RESULTS_DIR}/table_{an}_real.csv",'w') as f: f.write('\n'.join(rows))
    
    # Boxplots
    for ak, an in algo_map:
        fig, axes = plt.subplots(1,5,figsize=(20,5),sharey=True)
        fig.suptitle(f'{an} - ARI Change with DR (Real-World)')
        for i, dr in enumerate(DR_METHODS):
            data = []
            for lv in LEVELS:
                key = f"{dr}_{lv}"
                diffs = [results[dn][ak].get(key,0) - results[dn][ak].get('None',0)
                         for dn in results
                         if results[dn][ak].get(key) is not None and results[dn][ak].get('None') is not None]
                data.append(diffs)
            axes[i].boxplot(data, labels=LEVELS); axes[i].set_title(dr)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            if i==0: axes[i].set_ylabel('ARI Change')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/boxplot_{an}_RealWorld.pdf", bbox_inches='tight'); plt.close()
    
    # Aggregate
    agg = {}
    for ak, an in algo_map:
        agg[an] = {}
        for dr in DR_METHODS:
            agg[an][dr] = {}
            for lv in LEVELS:
                key = f"{dr}_{lv}"
                diffs = [(results[dn][ak].get(key,0) - results[dn][ak].get('None',0))
                         for dn in results
                         if results[dn][ak].get(key) is not None and results[dn][ak].get('None') is not None]
                wins = sum(1 for d in diffs if d > 0.005)
                n_total = len(diffs)
                agg[an][dr][lv] = {
                    'win_pct': round(100*wins/n_total,1) if n_total else 0,
                    'avg_diff': round(np.mean(diffs)*100,2) if diffs else 0,
                    'n': n_total
                }
    with open(f"{RESULTS_DIR}/aggregate_stats_real.json",'w') as f: json.dump(agg, f, indent=2)
    
    # Wilcoxon
    wilc = {}
    for ak, an in algo_map:
        wilc[an] = {}
        for dr in DR_METHODS:
            wilc[an][dr] = {}
            for lv in LEVELS:
                key = f"{dr}_{lv}"
                b_list, r_list = [], []
                for dn in results:
                    b = results[dn][ak].get('None')
                    r = results[dn][ak].get(key)
                    if b is not None and r is not None: b_list.append(b); r_list.append(r)
                if len(b_list) >= 5:
                    try: _, p = wilcoxon(r_list, b_list, alternative='greater'); wilc[an][dr][lv] = round(p,3)
                    except: wilc[an][dr][lv] = 1.0
                else: wilc[an][dr][lv] = None
    with open(f"{RESULTS_DIR}/wilcoxon_real.json",'w') as f: json.dump(wilc, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}\nREAL-WORLD RESULTS SUMMARY\n{'='*60}")
    print(f"Params: AHC={ahc_p}, GMM={gmm_c}, OPTICS={optics_p}")
    
    for an in ['k-means','AHC','GMM','OPTICS']:
        print(f"\n{an} - % wins / avg change:")
        for dr in DR_METHODS:
            vals = [f"{agg[an][dr][lv]['win_pct']:5.1f}%/{agg[an][dr][lv]['avg_diff']:+5.1f}%" for lv in LEVELS]
            print(f"  {dr:>12}: {' | '.join(vals)}")
    
    print(f"\nDone in {time.time()-t0:.0f}s")
    return results

if __name__ == "__main__":
    main()
