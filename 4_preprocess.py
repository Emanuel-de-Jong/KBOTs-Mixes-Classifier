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
UNDERSAMPLE_TRES = 150
# -1 means no treshold
OVERSAMPLE_TRES = 130

g.load_data(3)

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data['label'].value_counts()

all_new_rows = []
validate_target = label_counts.max() * VALIDATE_PERC
for label in range(g.label_count):
    label_train_data = train_data[train_data["label"] == label]

    organic_validate_target = int(VALIDATE_PERC * len(label_train_data))
    label_validate_idxs = label_train_data.index.values
    np.random.shuffle(label_validate_idxs)
    
    g.data.loc[label_validate_idxs[:organic_validate_target], "data_set"] = g.DataSetType.validate
    
    remaining_validate_target = int(validate_target - organic_validate_target)
    if remaining_validate_target > 0:
        validate_sample_idxs = g.data[(g.data["data_set"] == g.DataSetType.validate) & 
            (g.data["label"] == label)].index.values
        
        if len(validate_sample_idxs) > 0:
            num_repeats = (remaining_validate_target + len(validate_sample_idxs) - 1) // len(validate_sample_idxs)
            repeated_idxs = np.tile(validate_sample_idxs, num_repeats)
            selected_idxs = repeated_idxs[:remaining_validate_target]
            
            new_rows = g.data.loc[selected_idxs].copy()
            new_rows["data_set"] = g.DataSetType.validate
            all_new_rows.append(new_rows)

if all_new_rows:
    g.data = pd.concat([g.data] + all_new_rows, ignore_index=False)

validate_data = g.data[g.data["data_set"] == g.DataSetType.validate]
print("== Validate label counts ==")
for label, count in validate_data["label"].value_counts().items():
    print(f"{g.labels[label]}: {count}")

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data['label'].value_counts()
def undersample(label, sample_target):
    train_data = g.data[g.data["data_set"] == g.DataSetType.train]
    
    label_idxs = train_data[train_data['label'] == label].index.to_numpy()
    sampled_label_idxs = resample(label_idxs, replace=False, n_samples=sample_target, random_state=1)
    other_train_idxs = train_data[train_data['label'] != label].index.to_numpy()
    
    new_train_idxs = np.concatenate([other_train_idxs, sampled_label_idxs]).astype(int)
    non_train_data = g.data[g.data["data_set"] != g.DataSetType.train]
    new_train_data = g.data.loc[new_train_idxs].copy()
    
    g.data = pd.concat([new_train_data, non_train_data], ignore_index=False)

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

train_data = g.data[g.data["data_set"] == g.DataSetType.train]
label_counts = train_data["label"].value_counts()
print("\n== Train label counts after resample ==")
for label, count in label_counts.items():
    print(f"{g.labels[label]}: {count}")

g.save_data(4)
