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
    OVERSAMPLING = 2

SAMPLING = SamplingType.UNDERSAMPLING
# -1 means no treshold
UNDERSAMPLING_TRES = 175
# -1 means no treshold
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

# TEST START
label_counts = y_train.value_counts()
min_label = label_counts.idxmin()
min_label_count = label_counts.min()
min_label_indices = np.where(y_train == min_label)[0]
test_idx = min_label_indices[len(min_label_indices) // 2]
original_label = y_train.iloc[test_idx]
original_embedding = X_train[test_idx, :10].copy()
print(f"\nBefore sampling:")
print(f"Test index: {test_idx}")
print(f"Label: {original_label}")
# TEST END

if SAMPLING == SamplingType.UNDERSAMPLING:
    undersampling_tres = UNDERSAMPLING_TRES if UNDERSAMPLING_TRES != -1 else y_train.value_counts().min()
    for label_num in range(len(labels)):
        label_count = (y_train == label_num).sum()
        if label_count > undersampling_tres:
            undersample(label_num, undersampling_tres)
elif SAMPLING == SamplingType.OVERSAMPLING:
    if OVERSAMPLING_TRES != -1:
        for label_num in range(len(labels)):
            label_count = (y_train == label_num).sum()
            if label_count > OVERSAMPLING_TRES:
                undersample(label_num, OVERSAMPLING_TRES)
    
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# TEST START
matches = []
for i in range(len(X_train)):
    if np.allclose(X_train[i, :10], original_embedding, rtol=1e-5, atol=1e-8):
        matches.append(i)
if len(matches) > 0:
    for match_idx in matches:
        found_label = y_train.iloc[match_idx] if isinstance(y_train, pd.Series) else y_train[match_idx]
        embedding_match = X_train[match_idx, :10]
        print(f"\nMatch at index: {match_idx}")
        print(f"Label: {found_label}")
        if found_label == original_label:
            print("X_train and y_train are SYNCHRONIZED!")
        else:
            print("X_train and y_train are OUT OF SYNC!")
# TEST END

print()
train_distribution = y_train.value_counts()
for label_num, count in train_distribution.items():
    print(f"{labels[label_num]}: {count}")

# print(f"Train lengths: {len(X_train)} | {len(X_train_norm)} | {len(X_train_scale)}")
# print(f"Test lengths: {len(X_test)} | {len(X_test_norm)} | {len(X_test_scale)}")
# print(f"Label lengths: {len(y_train)} | {len(y_test)}")

joblib.dump(X_train, cache_dir / f'X_train.joblib')
joblib.dump(X_test, cache_dir / f'X_test.joblib')
joblib.dump(y_train, cache_dir / f'y_train.joblib')
joblib.dump(y_test, cache_dir / f'y_test.joblib')
