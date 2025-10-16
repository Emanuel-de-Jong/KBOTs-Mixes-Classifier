import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from pathlib import Path
from enum import Enum

class SamplingType(Enum):
    RAW = 0
    UNDERSAMPLING = 1
    OVERSAMPLING = 2

SAMPLING = SamplingType.UNDERSAMPLING

cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))
X_train = joblib.load(cache_dir / "embs_train.joblib")
X_test = joblib.load(cache_dir / "embs_test.joblib")
y_train = joblib.load(cache_dir / "labels_train.joblib")
y_test = joblib.load(cache_dir / "labels_test.joblib")

if SAMPLING == SamplingType.UNDERSAMPLING:
    df = pd.DataFrame({'label': y_train.values})
    df['idx'] = df.index
    min_n = df['label'].value_counts().min()

    selected_idxs = (
        df.groupby('label')['idx']
        .apply(lambda idxs: resample(idxs, replace=False, n_samples=min_n, random_state=1))
        .explode()
        .astype(int)
        .values
    )

    X_train = X_train[selected_idxs]
    y_train = y_train.iloc[selected_idxs].reset_index(drop=True)
elif SAMPLING == SamplingType.OVERSAMPLING:
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

train_distribution = y_train.value_counts()
for label_num, count in train_distribution.items():
    print(f"{labels[label_num]}: {count}")

X_train_norm = torch.nn.functional.normalize(torch.from_numpy(X_train), p=2, dim=1)
X_test_norm = torch.nn.functional.normalize(torch.from_numpy(X_test), p=2, dim=1)
X_train_norm = X_train_norm.numpy()
X_test_norm = X_test_norm.numpy()

scaler = StandardScaler()
X_train_scale = torch.tensor(scaler.fit_transform(X_train))
X_test_scale = torch.tensor(scaler.transform(X_test))

joblib.dump(X_train, cache_dir / f'X_train.joblib')
joblib.dump(X_test, cache_dir / f'X_test.joblib')
joblib.dump(X_train_norm, cache_dir / f'X_train_norm.joblib')
joblib.dump(X_test_norm, cache_dir / f'X_test_norm.joblib')
joblib.dump(X_train_scale, cache_dir / f'X_train_scale.joblib')
joblib.dump(X_test_scale, cache_dir / f'X_test_scale.joblib')
joblib.dump(y_train, cache_dir / f'y_train.joblib')
joblib.dump(y_test, cache_dir / f'y_test.joblib')
