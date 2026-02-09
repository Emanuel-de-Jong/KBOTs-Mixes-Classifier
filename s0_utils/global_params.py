import joblib
import os
import zarr
import numpy as np
from pathlib import Path
from enum import Enum

class DataSetType(Enum):
    train = 0
    validate = 1
    test = 2

class DataSectionType(Enum):
    time = 0
    feature = 1
    layer = 2

# 0-24. But maybe don't use 24?
#     clip_min   clip_max
# 0  -6.267006  16.034060
# 1  -9.345669  14.299644
# 2  -9.852882  14.021239
# 22 -5.666114   9.925220
# 23 -5.706382   9.931857
# 24 -0.257531   0.291407
DATA_LAYER_INDEXES = [2, 5, 7] + list(range(9, 14+1)) + [16, 18, 21]
DATA_COUNTS = {
    DataSectionType.time: 6,
    DataSectionType.feature: 1024,
    DataSectionType.layer: len(DATA_LAYER_INDEXES)
}
DATA_ORDER = [DataSectionType.time, DataSectionType.feature, DataSectionType.layer]
DATA_SHAPE = tuple(DATA_COUNTS[section] for section in DATA_ORDER)

DATA_BATCH_SIZE = 7_000
MODEL_BATCH_SIZE = 256

NAME = "global"
MODELS = {
    "global": [],
    # "general_pop": [],
    # "rock": [],
    # "edm_hard": [],
    # "edm_easy": [
    #     "Bassline",
    #     "Breakcore",
    #     "Chillstep",
    #     "Future_Funk",
    #     "Groovy_UK_Garage",
    #     "Hardstyle",
    #     "Lofi_Hip_Hop",
    #     "Melodic_Extratone",
    #     "Pioneer_Glitch_Hop",
    #     "Rawstyle",
    #     "Synthwave",
    # ],
}

TRAIN_DIR = Path("data_sets") / "train"
TRAIN_PLAYLISTS_DIR = TRAIN_DIR / "playlists"
TRAIN_PUBLIC_PLAYLISTS_DIR = TRAIN_DIR / "public_playlists"
TEST_DIR = Path("data_sets") / "test"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DLS_DIR = Path("s1_prep") / "dls"
DLS_DIR.mkdir(exist_ok=True)

LABELS = None
LABEL_COUNT = 0
if os.path.exists(MODELS_DIR / f"labels_{NAME}.joblib"):
    LABELS = joblib.load(MODELS_DIR / f"labels_{NAME}.joblib")
    LABEL_COUNT = len(LABELS)

def get_data_path(step, data_set_type=DataSetType.train, idx=0):
    return CACHE_DIR / f"data_{step}_{data_set_type.name}_{idx}.joblib"

def save_data(data, step, data_set_type=DataSetType.train, idx=0):
    joblib.dump(data, get_data_path(step, data_set_type, idx))

def save_data_batched(data, step, data_set_type=DataSetType.train, batch_size=DATA_BATCH_SIZE):
    for idx, start in enumerate(range(0, len(data), batch_size)):
        batch = data.iloc[start:start + batch_size]
        save_data(batch, step, data_set_type, idx)

def load_data(path):
    return joblib.load(path)

def get_data_count(step, data_set_type=DataSetType.train):
    idx = 0
    while idx < 100_000:
        if not get_data_path(step, data_set_type, idx).exists():
            return idx
        idx += 1

def iter_data_paths(step, data_set_type=DataSetType.train):
    count = get_data_count(step, data_set_type)
    for idx in range(count):
        yield get_data_path(step, data_set_type, idx)

def iter_all_data_paths(step):
    for data_set_type in DataSetType:
        for path in iter_data_paths(step, data_set_type):
            yield path

def get_all_data_paths(step):
    data_paths = {}
    for data_set_type in DataSetType:
        data_paths[data_set_type] = []
        for path in iter_data_paths(step, data_set_type):
            data_paths[data_set_type].append(path)
    return data_paths

def save_zarr(data, step, data_set_type, batch_idx):
    root = zarr.open_group(
        CACHE_DIR / f"data_{step}_{data_set_type.name}_{batch_idx}.zarr",
        mode="w"
    )

    X = np.stack(data["data"].to_numpy())
    y = data["label"].to_numpy(dtype=np.int64)

    root.create_array(
        name="data",
        data=X,
        chunks=(min(MODEL_BATCH_SIZE, len(X)),) + X.shape[1:],
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")],
    )

    root.create_array(
        name="label",
        data=y,
        chunks=(min(MODEL_BATCH_SIZE, len(y)),),
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")],
    )

def iter_zarr_data_paths(step, data_set_type=DataSetType.train):
    idx = 0
    while idx < 100_000:
        data_path = CACHE_DIR / f"data_{step}_{data_set_type.name}_{idx}.zarr"
        if not data_path.exists():
            break
        
        yield data_path
        idx += 1
