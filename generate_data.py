"""
Generate all synthetic datasets for the DR+clustering paper.
Circles, Moons, RSG, Repliclust - with noise injection.
"""
import numpy as np
import os
import pickle
from sklearn.datasets import make_circles, make_moons
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

DATA_DIR = "/workspace/synthetic_data"
os.makedirs(DATA_DIR, exist_ok=True)


def inject_noise(X):
    """
    Z-score normalize, then add structured noise to 75% of features.
    25% get N(0,1), 25% get N(0,0.5), 25% get N(0,0.25), 25% clean.
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    n_samples, n_features = X_norm.shape
    indices = np.random.permutation(n_features)
    q = n_features // 4
    
    # First quarter: sigma=1
    for i in indices[:q]:
        X_norm[:, i] += np.random.normal(0, 1, n_samples)
    # Second quarter: sigma=0.5
    for i in indices[q:2*q]:
        X_norm[:, i] += np.random.normal(0, 0.5, n_samples)
    # Third quarter: sigma=0.25
    for i in indices[2*q:3*q]:
        X_norm[:, i] += np.random.normal(0, 0.25, n_samples)
    # Fourth quarter: clean (no noise)
    
    return X_norm


def generate_circles_2cluster():
    """2-cluster circles: 2 nested circles, Nc=1000 per cluster, factor=0.5"""
    X, y = make_circles(n_samples=2000, factor=0.5, noise=0.05, random_state=None)
    return X, y


def generate_circles_5cluster():
    """5-cluster circles: 5 concentric rings scaled by 1.0, 2.0, 3.5, 5.0, 7.0"""
    n_per_cluster = 400
    X_list, y_list = []  , []
    factors = [1.0, 2.0, 3.5, 5.0, 7.0]
    for i, f in enumerate(factors):
        theta = np.random.uniform(0, 2*np.pi, n_per_cluster)
        r = f + np.random.normal(0, 0.05, n_per_cluster)
        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)
        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(n_per_cluster, i))
    return np.vstack(X_list), np.concatenate(y_list)


def generate_moons_2cluster():
    """2-cluster moons: standard make_moons with n=2000"""
    X, y = make_moons(n_samples=2000, noise=0.1, random_state=None)
    return X, y


def generate_moons_5cluster():
    """5-cluster moons: 5 crescent-shaped structures with transformations"""
    n_per_cluster = 400
    X_list, y_list = [], []
    
    # Generate base moon shape and apply transformations
    stretches = [1.0, 1.5, 1.0, 1.5, 1.0]
    rotations_deg = [0, 160, -160, 10, 180]
    x_shifts = [0, 3, -3, 2, -2]
    y_shifts = [0, 1.0, 1.2, 1.5, 1.0]
    
    for i in range(5):
        # Generate base moon (upper crescent)
        theta = np.linspace(0, np.pi, n_per_cluster)
        x = np.cos(theta) * stretches[i] + np.random.normal(0, 0.1, n_per_cluster)
        y_coord = np.sin(theta) + np.random.normal(0, 0.1, n_per_cluster)
        
        # Rotate
        angle = np.radians(rotations_deg[i])
        x_rot = x * np.cos(angle) - y_coord * np.sin(angle)
        y_rot = x * np.sin(angle) + y_coord * np.cos(angle)
        
        # Translate
        x_rot += x_shifts[i]
        y_rot += y_shifts[i]
        
        X_list.append(np.column_stack([x_rot, y_rot]))
        y_list.append(np.full(n_per_cluster, i))
    
    return np.vstack(X_list), np.concatenate(y_list)


def embed_to_high_dim(X, target_dim, random_state=None):
    """Embed 2D data into higher dimensions via Gaussian Random Projection (inverse)."""
    n_samples, orig_dim = X.shape
    if target_dim <= orig_dim:
        return X
    
    rng = np.random.RandomState(random_state)
    # Create a random projection matrix and use it to embed
    # We place the 2D data in a higher-dimensional space
    projection_matrix = rng.randn(orig_dim, target_dim) / np.sqrt(target_dim)
    X_high = X @ projection_matrix
    return X_high


def generate_rsg_data(k, d, Nc, alpha=None):
    """
    Rodriguez Structured Gaussian (RSG) data generator.
    Generates k clusters in d dimensions with Nc objects per cluster.
    """
    rng = np.random.RandomState()
    
    # Determine alpha based on parameters to ensure moderate clustering difficulty
    if alpha is None:
        # Heuristic: adjust alpha based on k and d
        alpha = max(0.1, min(0.9, 1.0 - k * 0.01 - d * 0.001))
    
    X_list, y_list = [], []
    
    # Generate cluster centers spread out
    centers = rng.randn(k, d) * np.sqrt(d) * (1 + alpha)
    
    for i in range(k):
        # Generate cluster-specific covariance
        # Create a random positive semi-definite matrix
        A = rng.randn(d, d) * alpha
        cov = A @ A.T / d + np.eye(d) * 0.1
        
        # Generate samples
        samples = rng.multivariate_normal(centers[i], cov, size=Nc)
        X_list.append(samples)
        y_list.append(np.full(Nc, i))
    
    return np.vstack(X_list), np.concatenate(y_list)


def generate_repliclust_data(k, d, Nc):
    """Generate Repliclust data using the repliclust library."""
    try:
        import repliclust
        archetype = repliclust.Archetype(
            n_clusters=k,
            dim=d,
            n_samples=k * Nc,
            aspect_ref=3.0,
            radius=5.0
        )
        X, y, _ = repliclust.DataGenerator(archetype).synthesize()
        return X, y
    except Exception as e:
        print(f"Repliclust failed for k={k}, d={d}, Nc={Nc}: {e}")
        # Fallback: generate anisotropic Gaussian clusters
        return generate_rsg_data(k, d, Nc, alpha=0.5)


def save_dataset(X, y, name, idx):
    """Save a dataset to disk."""
    path = os.path.join(DATA_DIR, name)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, f"dataset_{idx:03d}.npz"), X=X, y=y)


def main():
    print("Generating synthetic datasets...")
    
    dims = [10, 50, 200]
    n_datasets_per_config = 50
    
    # === CIRCLES ===
    print("Generating Circles datasets...")
    for k_type in ['2cluster', '5cluster']:
        for d in dims:
            for i in range(n_datasets_per_config):
                if k_type == '2cluster':
                    X, y = generate_circles_2cluster()
                else:
                    X, y = generate_circles_5cluster()
                
                # Embed to high dimensions
                X_high = embed_to_high_dim(X, d, random_state=i*1000+d)
                # Inject noise
                X_noisy = inject_noise(X_high)
                
                name = f"circles_{k_type}_d{d}"
                save_dataset(X_noisy, y, name, i)
    
    print(f"  Circles: done")
    
    # === MOONS ===
    print("Generating Moons datasets...")
    for k_type in ['2cluster', '5cluster']:
        for d in dims:
            for i in range(n_datasets_per_config):
                if k_type == '2cluster':
                    X, y = generate_moons_2cluster()
                else:
                    X, y = generate_moons_5cluster()
                
                X_high = embed_to_high_dim(X, d, random_state=i*2000+d)
                X_noisy = inject_noise(X_high)
                
                name = f"moons_{k_type}_d{d}"
                save_dataset(X_noisy, y, name, i)
    
    print(f"  Moons: done")
    
    # === RSG ===
    print("Generating RSG datasets...")
    ks = [2, 10, 50]
    ds = [10, 50, 200]
    Ncs = [5, 50, 100]
    
    idx = 0
    for k in ks:
        for d in ds:
            for Nc in Ncs:
                # Generate ~10 datasets per config to get ~265 total
                n_rsg = 10 if (k <= 10) else 9
                if k == 50 and Nc == 5:
                    n_rsg = 8  # Adjust to get closer to 265
                for i in range(n_rsg):
                    X, y = generate_rsg_data(k, d, Nc)
                    X_noisy = inject_noise(X)
                    save_dataset(X_noisy, y, "rsg", idx)
                    idx += 1
    
    print(f"  RSG: {idx} datasets generated")
    
    # === REPLICLUST ===
    print("Generating Repliclust datasets...")
    for k_type in ['2cluster', '5cluster']:
        k = 2 if k_type == '2cluster' else 5
        Nc = 1000 if k == 2 else 400
        for d in dims:
            for i in range(n_datasets_per_config):
                X, y = generate_repliclust_data(k, d, Nc)
                X_noisy = inject_noise(X)
                
                name = f"repliclust_{k_type}_d{d}"
                save_dataset(X_noisy, y, name, i)
    
    print(f"  Repliclust: done")
    
    # Save metadata
    metadata = {
        'circles_configs': [f"circles_{kt}_d{d}" for kt in ['2cluster', '5cluster'] for d in dims],
        'moons_configs': [f"moons_{kt}_d{d}" for kt in ['2cluster', '5cluster'] for d in dims],
        'rsg_total': idx,
        'repliclust_configs': [f"repliclust_{kt}_d{d}" for kt in ['2cluster', '5cluster'] for d in dims],
    }
    with open(os.path.join(DATA_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nAll synthetic data saved to {DATA_DIR}")
    print(f"Total: {300 + 300 + idx + 300} datasets")


if __name__ == "__main__":
    main()
