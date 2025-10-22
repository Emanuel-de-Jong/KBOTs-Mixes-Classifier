import joblib
import os
from pathlib import Path
from enum import Enum

class DataSetType(Enum):
    train = 0
    validate = 1
    test = 2

NAME = "global"

TRAIN_DIR = Path("train")
TEST_DIR = Path("test")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
BATCH_DIR = Path("batch")

MIN_SONG_COUNT = min(len(list(f.glob("*.mp3"))) for f in TRAIN_DIR.iterdir() if f.is_dir())

labels = None
label_count = 0
if os.path.exists(CACHE_DIR / f"labels_{NAME}.joblib"):
    labels = joblib.load(CACHE_DIR / f"labels_{NAME}.joblib")
    label_count = len(labels)

def get_song_name(song_path):
    return os.path.splitext(os.path.basename(song_path))[0]

data = None
def save_data(name):
    joblib.dump(data, CACHE_DIR / f"data_{name}.joblib")

def load_data(name):
    global data
    data = joblib.load(CACHE_DIR / f"data_{name}.joblib")
