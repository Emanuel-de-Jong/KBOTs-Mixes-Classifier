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

VALIDATE_PERC = 0.15

SAMPLING = SamplingType.undersample
# -1 means no treshold
UNDERSAMPLE_TRES = 175
# -1 means no treshold
OVERSAMPLE_TRES = 130

g.load_data(3)

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data['label'].value_counts()

validate_target = label_counts.max() * VALIDATE_PERC
for label in range(g.label_count):
    label_train_data = train_data[train_data["label"] == label]
    organic_validate_target = int(VALIDATE_PERC * len(label_train_data))
    label_validate_idxs = label_train_data.index
    # TODO: Shuffle label_validate_idxs

    for i in range(organic_validate_target):
        g.data[label_validate_idxs[i]]["data_set"] = g.DataSetType.validate
    
    remaining_validate_target = validate_target - organic_validate_target
    validate_data = g.data[g.data["data_set"] == g.DataSetType.validate]
    label_validate_data = validate_data[validate_data["label"] == label]
    # TODO: Go over and duplicate validate data untill remaining_validate_target is reached

    train_data = g.data[g.data["data_set"] == g.DataSetType.train]

label_counts = train_data['label'].value_counts()
def undersample(data, label, sample_target):
    label_idxs = data[data['label'] == label].index.to_numpy()

    sampled_label_idxs = resample(label_idxs, replace=False, n_samples=sample_target, random_state=1)
    other_idxs = data[data['label'] != label].index.to_numpy()

    final_idxs = np.concatenate([other_idxs, sampled_label_idxs]).astype(int)
    final_idxs.sort()

    g.data = g.data.iloc[final_idxs].reset_index(drop=True)

if SAMPLING == SamplingType.undersample:
    undersampling_tres = UNDERSAMPLE_TRES if UNDERSAMPLE_TRES != -1 else label_counts.min()
    for label, count in label_counts.items():
        if count > undersampling_tres:
            undersample(train_data, label, undersampling_tres)
            train_data = g.data[g.data["data_set"] == g.DataSetType.train]
elif SAMPLING == SamplingType.oversample:
    if OVERSAMPLE_TRES != -1:
        for label, count in label_counts.items():
            if count > OVERSAMPLE_TRES:
                undersample(train_data, label, OVERSAMPLE_TRES)
                train_data = g.data[g.data["data_set"] == g.DataSetType.train]
    
    smote = SMOTE(random_state=1)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

label_counts = train_data.value_counts()
for label, count in label_counts.items():
    print(f"{g.labels[label]}: {count}")

g.save_data(4)
