import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from pathlib import Path
from enum import Enum

class SamplingType(Enum):
    RAW = 0
    UNDERSAMPLING = 1
    PARTIAL_UNDERSAMPLING = 2
    OVERSAMPLING = 3

SAMPLING = SamplingType.OVERSAMPLING
PARTIAL_UNDERSAMPLING_TRES = 100
OVERSAMPLING_TRES = 130

cache_dir = Path("cache")
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))
X_train = joblib.load(cache_dir / "embs_train.joblib")
X_test = joblib.load(cache_dir / "embs_test.joblib")
y_train = joblib.load(cache_dir / "labels_train.joblib")
y_test = joblib.load(cache_dir / "labels_test.joblib")

# print(f'X_train shape: {X_train.shape}')
# print(f'X_train type: {type(X_train)}')
# print(f'X_train[0][0] value type: {type(X_train[0][0])}')
# print(f'y_train shape: {y_train.shape}')
# print(f'y_train type: {type(y_train)}')
# print(f'y_train[0] value type: {type(y_train[0])}')

def undersample(label_num, sample_target):
    global X_train, y_train

    label_idxs = y_train[y_train == label_num].index.to_numpy()
    if len(label_idxs) <= sample_target:
        return

    sampled_label_idxs = resample(label_idxs, replace=False, n_samples=sample_target, random_state=1)

    other_idxs = y_train[y_train != label_num].index.to_numpy()

    final_idxs = np.concatenate([other_idxs, sampled_label_idxs]).astype(int)
    final_idxs.sort()

    X_train = X_train[final_idxs]
    y_train = y_train.iloc[final_idxs].reset_index(drop=True)

if SAMPLING == SamplingType.UNDERSAMPLING:
    min_sample_count = y_train.value_counts().min()
    for label_num in range(len(labels)):
        undersample(label_num, min_sample_count)
elif SAMPLING == SamplingType.PARTIAL_UNDERSAMPLING:
    for label_num in range(len(labels)):
        label_count = len(y_train[y_train == label_num])
        if label_count > PARTIAL_UNDERSAMPLING_TRES:
            undersample(label_num, PARTIAL_UNDERSAMPLING_TRES)
elif SAMPLING == SamplingType.OVERSAMPLING:
    for label_num in range(len(labels)):
        label_count = len(y_train[y_train == label_num])
        if label_count > OVERSAMPLING_TRES:
            undersample(label_num, OVERSAMPLING_TRES)
    
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

train_distribution = y_train.value_counts()
for label_num, count in train_distribution.items():
    print(f"{labels[label_num]}: {count}")

X_train_norm = torch.nn.functional.normalize(torch.from_numpy(X_train), p=2, dim=1)
X_test_norm = torch.nn.functional.normalize(torch.from_numpy(X_test), p=2, dim=1)
X_train_norm = X_train_norm.numpy()
X_test_norm = X_test_norm.numpy()

# scaler = StandardScaler()
scaler = RobustScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

print(f"\nTrain stats - Mean: {np.mean(X_train_scale):.4f}, Std: {np.std(X_train_scale):.4f}")
print(f"Test stats - Mean: {np.mean(X_test_scale):.4f}, Std: {np.std(X_test_scale):.4f}")
print(f"NaN in train: {np.isnan(X_train_scale).sum()}, test: {np.isnan(X_test_scale).sum()}")
print(f"Inf in train: {np.isinf(X_train_scale).sum()}, test: {np.isinf(X_test_scale).sum()}")

print(f"Train lengths: {len(X_train)} | {len(X_train_norm)} | {len(X_train_scale)}")
print(f"Test lengths: {len(X_test)} | {len(X_test_norm)} | {len(X_test_scale)}")
print(f"Label lengths: {len(y_train)} | {len(y_test)}")

joblib.dump(X_train, cache_dir / f'X_train.joblib')
joblib.dump(X_test, cache_dir / f'X_test.joblib')
joblib.dump(X_train_norm, cache_dir / f'X_train_norm.joblib')
joblib.dump(X_test_norm, cache_dir / f'X_test_norm.joblib')
joblib.dump(X_train_scale, cache_dir / f'X_train_scale.joblib')
joblib.dump(X_test_scale, cache_dir / f'X_test_scale.joblib')
joblib.dump(y_train, cache_dir / f'y_train.joblib')
joblib.dump(y_test, cache_dir / f'y_test.joblib')
