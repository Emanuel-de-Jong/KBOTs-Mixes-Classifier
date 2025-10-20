import pandas as pd
import numpy as np
import global_params as g
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from enum import Enum

class SamplingType(Enum):
    raw = 0
    undersample = 1
    oversample = 2

SAMPLING = SamplingType.undersample
# -1 means no treshold
UNDERSAMPLE_TRES = 175
# -1 means no treshold
OVERSAMPLE_TRES = 130

g.load_data(3)

def undersample(label, sample_target):
    label_idxs = g.data[g.data['label'] == label].index.to_numpy()

    sampled_label_idxs = resample(label_idxs, replace=False, n_samples=sample_target, random_state=1)
    other_idxs = g.data[g.data['label'] != label].index.to_numpy()

    final_idxs = np.concatenate([other_idxs, sampled_label_idxs]).astype(int)
    final_idxs.sort()

    g.data = g.data.iloc[final_idxs].reset_index(drop=True)

label_counts = g.data['label'].value_counts()
if SAMPLING == SamplingType.undersample:
    undersampling_tres = UNDERSAMPLE_TRES if UNDERSAMPLE_TRES != -1 else label_counts.min()
    for label, count in label_counts.items():
        if count > undersampling_tres:
            undersample(label, undersampling_tres)
elif SAMPLING == SamplingType.oversample:
    if OVERSAMPLE_TRES != -1:
        for label, count in label_counts.items():
            if count > OVERSAMPLE_TRES:
                undersample(label, OVERSAMPLE_TRES)
    
    smote = SMOTE(random_state=1)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

label_counts = g.data['label'].value_counts()
for label, count in label_counts.items():
    print(f"{g.labels[label]}: {count}")

g.save_data(4)
