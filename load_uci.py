"""
Load 20 UCI datasets used in the paper.
Returns (X, y, n_clusters) for each dataset.
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

UCI_DIR = "/workspace/uci_data"


def load_breast_tissue():
    df = pd.read_excel(os.path.join(UCI_DIR, "BreastTissue.xls"), sheet_name="Data")
    y = pd.Categorical(df["Class"]).codes
    X = df.drop(columns=["Class", "Case #"]).values.astype(float)
    return X, y, 6

def load_breast_wisconsin():
    df = pd.read_csv(os.path.join(UCI_DIR, "wdbc.data"), header=None)
    y = pd.Categorical(df[1]).codes
    X = df.iloc[:, 2:].values.astype(float)
    return X, y, 2

def load_ecoli():
    df = pd.read_csv(os.path.join(UCI_DIR, "ecoli.data"), sep=r'\s+', header=None)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, 1:-1].values.astype(float)
    return X, y, 8

def load_glass():
    df = pd.read_csv(os.path.join(UCI_DIR, "glass.data"), header=None)
    y = df.iloc[:, -1].values
    unique_labels = np.unique(y)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    X = df.iloc[:, 1:-1].values.astype(float)
    return X, y, 7  # Paper says 7

def load_haberman():
    df = pd.read_csv(os.path.join(UCI_DIR, "haberman.data"), header=None)
    y = df.iloc[:, -1].values - 1
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 2

def load_ionosphere():
    df = pd.read_csv(os.path.join(UCI_DIR, "ionosphere.data"), header=None)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 2

def load_iris():
    from sklearn.datasets import load_iris as sk_load_iris
    data = sk_load_iris()
    return data.data, data.target, 3

def load_movement_libras():
    df = pd.read_csv(os.path.join(UCI_DIR, "movement_libras.data"), header=None)
    y = df.iloc[:, -1].values.astype(int) - 1
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 15

def load_musk():
    df = pd.read_csv(os.path.join(UCI_DIR, "clean1.data"), header=None)
    y = df.iloc[:, -1].values.astype(int)
    X = df.iloc[:, 2:-1].values.astype(float)
    return X, y, 2

def load_parkinsons():
    df = pd.read_csv(os.path.join(UCI_DIR, "parkinsons.data"))
    y = df["status"].values.astype(int)
    X = df.drop(columns=["name", "status"]).values.astype(float)
    return X, y, 2

def load_segmentation():
    # Combine train + test to get 2310 objects as per paper
    def parse_seg(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';') or line.startswith('REGION'):
                    continue
                parts = line.split(',')
                rows.append(parts)
        return pd.DataFrame(rows)
    
    train = parse_seg(os.path.join(UCI_DIR, "segmentation.data"))
    test = parse_seg(os.path.join(UCI_DIR, "segmentation.test"))
    df = pd.concat([train, test], ignore_index=True)
    y = pd.Categorical(df.iloc[:, 0]).codes
    X = df.iloc[:, 1:].values.astype(float)
    return X, y, 7

def load_sonar():
    df = pd.read_csv(os.path.join(UCI_DIR, "sonar.all-data"), header=None)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 2

def load_spectf():
    train = pd.read_csv(os.path.join(UCI_DIR, "SPECTF.train"), header=None)
    test = pd.read_csv(os.path.join(UCI_DIR, "SPECTF.test"), header=None)
    df = pd.concat([train, test], ignore_index=True)
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y, 2

def load_transfusion():
    df = pd.read_csv(os.path.join(UCI_DIR, "transfusion.data"))
    y = df.iloc[:, -1].values.astype(int)
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 2

def load_vehicle():
    dfs = []
    for c in 'abcdefghi':
        fpath = os.path.join(UCI_DIR, f"vehicle_xa{c}.dat")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep=r'\s+', header=None)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 4

def load_vertebral_column():
    df = pd.read_csv(os.path.join(UCI_DIR, "column_3C.dat"), sep=r'\s+', header=None)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, 3

def load_vowel_context():
    df = pd.read_csv(os.path.join(UCI_DIR, "vowel-context.data"), sep=r'\s+', header=None)
    y = df.iloc[:, -1].values.astype(int)
    X = df.iloc[:, 3:-1].values.astype(float)
    return X, y, 11

def load_wine():
    from sklearn.datasets import load_wine as sk_load_wine
    data = sk_load_wine()
    return data.data, data.target, 3

def load_wine_quality_red():
    df = pd.read_csv(os.path.join(UCI_DIR, "winequality-red.csv"), sep=";")
    y = df["quality"].values.astype(int)
    unique_labels = np.unique(y)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    X = df.drop(columns=["quality"]).values.astype(float)
    return X, y, 6

def load_yeast():
    df = pd.read_csv(os.path.join(UCI_DIR, "yeast.data"), sep=r'\s+', header=None)
    y = pd.Categorical(df.iloc[:, -1]).codes
    X = df.iloc[:, 1:-1].values.astype(float)
    return X, y, 10


def load_all_uci():
    """Return ordered list of (name, X, y, k) tuples."""
    loaders = [
        ("Breast tissue", load_breast_tissue),
        ("Breast Wisconsin", load_breast_wisconsin),
        ("Ecoli", load_ecoli),
        ("Glass", load_glass),
        ("Haberman", load_haberman),
        ("Ionosphere", load_ionosphere),
        ("Iris", load_iris),
        ("Movement libras", load_movement_libras),
        ("Musk", load_musk),
        ("Parkinsons", load_parkinsons),
        ("Segmentation", load_segmentation),
        ("Sonar all", load_sonar),
        ("Spectf", load_spectf),
        ("Transfusion", load_transfusion),
        ("Vehicle", load_vehicle),
        ("Vertebral column", load_vertebral_column),
        ("Vowel context", load_vowel_context),
        ("Wine", load_wine),
        ("Wine quality red", load_wine_quality_red),
        ("Yeast", load_yeast),
    ]
    datasets = []
    for name, loader in loaders:
        try:
            X, y, k = loader()
            print(f"  {name}: {X.shape[0]} objects, {X.shape[1]} features, {k} clusters")
            datasets.append((name, X, y, k))
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    return datasets


if __name__ == "__main__":
    print("Loading all UCI datasets...")
    datasets = load_all_uci()
    print(f"\nLoaded {len(datasets)} datasets")
