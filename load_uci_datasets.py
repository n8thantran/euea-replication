"""
Download and prepare the 20 UCI real-world datasets used in the paper.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_wine

# Dataset info from Table 1 of the paper
DATASET_INFO = {
    'Breast tissue':      {'objects': 106,  'features': 9,   'clusters': 6},
    'Breast Wisconsin':   {'objects': 569,  'features': 30,  'clusters': 2},
    'Ecoli':              {'objects': 336,  'features': 7,   'clusters': 8},
    'Glass':              {'objects': 214,  'features': 9,   'clusters': 7},  # Note: actually 6 classes in data
    'Haberman':           {'objects': 306,  'features': 3,   'clusters': 2},
    'Ionosphere':         {'objects': 351,  'features': 34,  'clusters': 2},
    'Iris':               {'objects': 150,  'features': 4,   'clusters': 3},
    'Movement libras':    {'objects': 360,  'features': 90,  'clusters': 15},
    'Musk':               {'objects': 476,  'features': 166, 'clusters': 2},
    'Parkinsons':         {'objects': 195,  'features': 22,  'clusters': 2},
    'Segmentation':       {'objects': 2310, 'features': 19,  'clusters': 7},
    'Sonar all':          {'objects': 208,  'features': 60,  'clusters': 2},
    'Spectf':             {'objects': 267,  'features': 44,  'clusters': 2},
    'Transfusion':        {'objects': 748,  'features': 4,   'clusters': 2},
    'Vehicle':            {'objects': 846,  'features': 18,  'clusters': 4},
    'Vertebral column':   {'objects': 310,  'features': 6,   'clusters': 3},
    'Vowel context':      {'objects': 990,  'features': 10,  'clusters': 11},
    'Wine':               {'objects': 178,  'features': 13,  'clusters': 3},
    'Wine quality red':   {'objects': 1599, 'features': 11,  'clusters': 6},
    'Yeast':              {'objects': 1484, 'features': 8,   'clusters': 10},
}


def download_from_url(url, save_path):
    """Download a file from URL."""
    import urllib.request
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)


def load_all_datasets(data_dir='./uci_data'):
    """Load all 20 UCI datasets. Returns dict of {name: (X, y, k)}."""
    os.makedirs(data_dir, exist_ok=True)
    datasets = {}
    
    # 1. Breast tissue
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls'
        path = os.path.join(data_dir, 'BreastTissue.xls')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_excel(path, sheet_name='Data')
        y = LabelEncoder().fit_transform(df['Class'])
        X = df.drop(['Case #', 'Class'], axis=1).values.astype(float)
        datasets['Breast tissue'] = (X, y, DATASET_INFO['Breast tissue']['clusters'])
    except Exception as e:
        print(f"Error loading Breast tissue: {e}")
    
    # 2. Breast Wisconsin (Diagnostic)
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        path = os.path.join(data_dir, 'wdbc.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df[1])
        X = df.iloc[:, 2:].values.astype(float)
        datasets['Breast Wisconsin'] = (X, y, DATASET_INFO['Breast Wisconsin']['clusters'])
    except Exception as e:
        print(f"Error loading Breast Wisconsin: {e}")
    
    # 3. Ecoli
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        path = os.path.join(data_dir, 'ecoli.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 1:-1].values.astype(float)
        datasets['Ecoli'] = (X, y, DATASET_INFO['Ecoli']['clusters'])
    except Exception as e:
        print(f"Error loading Ecoli: {e}")
    
    # 4. Glass
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        path = os.path.join(data_dir, 'glass.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 1:-1].values.astype(float)
        datasets['Glass'] = (X, y, DATASET_INFO['Glass']['clusters'])
    except Exception as e:
        print(f"Error loading Glass: {e}")
    
    # 5. Haberman
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
        path = os.path.join(data_dir, 'haberman.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values.astype(float)
        datasets['Haberman'] = (X, y, DATASET_INFO['Haberman']['clusters'])
    except Exception as e:
        print(f"Error loading Haberman: {e}")
    
    # 6. Ionosphere
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        path = os.path.join(data_dir, 'ionosphere.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values.astype(float)
        datasets['Ionosphere'] = (X, y, DATASET_INFO['Ionosphere']['clusters'])
    except Exception as e:
        print(f"Error loading Ionosphere: {e}")
    
    # 7. Iris
    try:
        iris = load_iris()
        datasets['Iris'] = (iris.data, iris.target, DATASET_INFO['Iris']['clusters'])
    except Exception as e:
        print(f"Error loading Iris: {e}")
    
    # 8. Movement libras
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        path = os.path.join(data_dir, 'movement_libras.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values.astype(float)
        datasets['Movement libras'] = (X, y, DATASET_INFO['Movement libras']['clusters'])
    except Exception as e:
        print(f"Error loading Movement libras: {e}")
    
    # 9. Musk (Version 1)
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean1.data.Z'
        path_z = os.path.join(data_dir, 'clean1.data.Z')
        path = os.path.join(data_dir, 'clean1.data')
        if not os.path.exists(path):
            if not os.path.exists(path_z):
                download_from_url(url, path_z)
            os.system(f'uncompress {path_z} 2>/dev/null || gzip -d {path_z} 2>/dev/null || true')
            if not os.path.exists(path):
                # Try alternative approach
                import subprocess
                subprocess.run(['uncompress', '-f', path_z], capture_output=True)
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            y = LabelEncoder().fit_transform(df.iloc[:, -1])
            X = df.iloc[:, 2:-1].values.astype(float)
            datasets['Musk'] = (X, y, DATASET_INFO['Musk']['clusters'])
        else:
            # Alternative: try to load from ucimlrepo
            from ucimlrepo import fetch_ucirepo
            musk = fetch_ucirepo(id=74)
            X = musk.data.features.values.astype(float)
            y = LabelEncoder().fit_transform(musk.data.targets.values.ravel())
            datasets['Musk'] = (X, y, DATASET_INFO['Musk']['clusters'])
    except Exception as e:
        print(f"Error loading Musk: {e}")
    
    # 10. Parkinsons
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        path = os.path.join(data_dir, 'parkinsons.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path)
        y = df['status'].values
        X = df.drop(['name', 'status'], axis=1).values.astype(float)
        datasets['Parkinsons'] = (X, y, DATASET_INFO['Parkinsons']['clusters'])
    except Exception as e:
        print(f"Error loading Parkinsons: {e}")
    
    # 11. Segmentation
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test'
        path = os.path.join(data_dir, 'segmentation.data')
        path_test = os.path.join(data_dir, 'segmentation.test')
        if not os.path.exists(path):
            download_from_url(url, path)
        if not os.path.exists(path_test):
            download_from_url(url_test, path_test)
        # Read both train and test
        df_train = pd.read_csv(path, skiprows=5)
        df_test = pd.read_csv(path_test, skiprows=5)
        df = pd.concat([df_train, df_test], ignore_index=True)
        y = LabelEncoder().fit_transform(df.iloc[:, 0])
        X = df.iloc[:, 1:].values.astype(float)
        datasets['Segmentation'] = (X, y, DATASET_INFO['Segmentation']['clusters'])
    except Exception as e:
        print(f"Error loading Segmentation: {e}")
    
    # 12. Sonar
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        path = os.path.join(data_dir, 'sonar.all-data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values.astype(float)
        datasets['Sonar all'] = (X, y, DATASET_INFO['Sonar all']['clusters'])
    except Exception as e:
        print(f"Error loading Sonar all: {e}")
    
    # 13. SPECTF
    try:
        url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train'
        url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test'
        path_train = os.path.join(data_dir, 'SPECTF.train')
        path_test = os.path.join(data_dir, 'SPECTF.test')
        if not os.path.exists(path_train):
            download_from_url(url_train, path_train)
        if not os.path.exists(path_test):
            download_from_url(url_test, path_test)
        df_train = pd.read_csv(path_train, header=None)
        df_test = pd.read_csv(path_test, header=None)
        df = pd.concat([df_train, df_test], ignore_index=True)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values.astype(float)
        datasets['Spectf'] = (X, y, DATASET_INFO['Spectf']['clusters'])
    except Exception as e:
        print(f"Error loading Spectf: {e}")
    
    # 14. Transfusion
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        path = os.path.join(data_dir, 'transfusion.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values.astype(float)
        datasets['Transfusion'] = (X, y, DATASET_INFO['Transfusion']['clusters'])
    except Exception as e:
        print(f"Error loading Transfusion: {e}")
    
    # 15. Vehicle
    try:
        base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/'
        dfs = []
        for fname in ['xaa.dat', 'xab.dat', 'xac.dat', 'xad.dat', 'xae.dat', 'xaf.dat', 'xag.dat', 'xah.dat', 'xai.dat']:
            fpath = os.path.join(data_dir, f'vehicle_{fname}')
            if not os.path.exists(fpath):
                try:
                    download_from_url(base_url + fname, fpath)
                    dfs.append(pd.read_csv(fpath, delim_whitespace=True, header=None))
                except:
                    pass
            else:
                dfs.append(pd.read_csv(fpath, delim_whitespace=True, header=None))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            y = LabelEncoder().fit_transform(df.iloc[:, -1])
            X = df.iloc[:, :-1].values.astype(float)
            datasets['Vehicle'] = (X, y, DATASET_INFO['Vehicle']['clusters'])
    except Exception as e:
        print(f"Error loading Vehicle: {e}")
    
    # 16. Vertebral column (3 classes)
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
        zip_path = os.path.join(data_dir, 'vertebral_column_data.zip')
        if not os.path.exists(zip_path):
            download_from_url(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        # 3-class version
        vert_path = os.path.join(data_dir, 'column_3C.dat')
        if os.path.exists(vert_path):
            df = pd.read_csv(vert_path, delim_whitespace=True, header=None)
            y = LabelEncoder().fit_transform(df.iloc[:, -1])
            X = df.iloc[:, :-1].values.astype(float)
            datasets['Vertebral column'] = (X, y, DATASET_INFO['Vertebral column']['clusters'])
    except Exception as e:
        print(f"Error loading Vertebral column: {e}")
    
    # 17. Vowel context
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        path = os.path.join(data_dir, 'vowel-context.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 3:-1].values.astype(float)  # Skip first 3 columns (train/test, speaker, sex)
        datasets['Vowel context'] = (X, y, DATASET_INFO['Vowel context']['clusters'])
    except Exception as e:
        print(f"Error loading Vowel context: {e}")
    
    # 18. Wine
    try:
        wine = load_wine()
        datasets['Wine'] = (wine.data, wine.target, DATASET_INFO['Wine']['clusters'])
    except Exception as e:
        print(f"Error loading Wine: {e}")
    
    # 19. Wine quality red
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        path = os.path.join(data_dir, 'winequality-red.csv')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, sep=';')
        y = LabelEncoder().fit_transform(df['quality'])
        X = df.drop('quality', axis=1).values.astype(float)
        datasets['Wine quality red'] = (X, y, DATASET_INFO['Wine quality red']['clusters'])
    except Exception as e:
        print(f"Error loading Wine quality red: {e}")
    
    # 20. Yeast
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        path = os.path.join(data_dir, 'yeast.data')
        if not os.path.exists(path):
            download_from_url(url, path)
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, 1:-1].values.astype(float)
        datasets['Yeast'] = (X, y, DATASET_INFO['Yeast']['clusters'])
    except Exception as e:
        print(f"Error loading Yeast: {e}")
    
    return datasets


if __name__ == '__main__':
    datasets = load_all_datasets()
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, (X, y, k) in sorted(datasets.items()):
        print(f"  {name}: X={X.shape}, classes={len(np.unique(y))}, k={k}")
