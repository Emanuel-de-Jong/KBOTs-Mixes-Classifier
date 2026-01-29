import joblib
import os
from pathlib import Path
from enum import Enum

class DataSetType(Enum):
    train = 0
    validate = 1
    test = 2

DATA_BATCH_SIZE = 7_000

NAME = "global"
MODELS = {
    "global": [],
    "general_pop": [],
    "rock": [],
    "edm_hard": [],
    "edm_easy": [
        "Bassline",
        "Breakcore",
        "Chillstep",
        "Future Funk",
        "Groovy UK Garage",
        "Hardstyle",
        "Lofi Hip Hop",
        "Melodic Extratone",
        "Pioneer Glitch Hop",
        "Rawstyle",
        "Synthwave",
    ],
}

TRAIN_DIR = Path("train")
TRAIN_PLAYLISTS_DIR = TRAIN_DIR / "playlists"
TRAIN_PUBLIC_PLAYLISTS_DIR = TRAIN_DIR / "public_playlists"
TEST_DIR = Path("test")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
BATCH_DIR = Path("batch")

PLAYLIST_COUNTS = {}
for folder in TRAIN_PLAYLISTS_DIR.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.glob("**/*.mp3")))
        PLAYLIST_COUNTS[folder.name] = mp3_count
for folder in TRAIN_PUBLIC_PLAYLISTS_DIR.iterdir():
    if folder.is_dir():
        mp3_count = len(list(folder.glob("**/*.mp3")))
        PLAYLIST_COUNTS[folder.name] += mp3_count

MIN_SONG_COUNT = min(PLAYLIST_COUNTS.values())

LABELS = None
LABEL_COUNT = 0
if os.path.exists(CACHE_DIR / f"labels_{NAME}.joblib"):
    LABELS = joblib.load(CACHE_DIR / f"labels_{NAME}.joblib")
    LABEL_COUNT = len(LABELS)

def get_song_name(song_path):
    return os.path.splitext(os.path.basename(song_path))[0]

def get_data_path(step, data_set_type=DataSetType.train, idx=0):
    return CACHE_DIR / f"data_{step}_{data_set_type}_{idx}.joblib"

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
