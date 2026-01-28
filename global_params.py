import joblib
import os
from pathlib import Path
from enum import Enum

class DataSetType(Enum):
    train = 0
    validate = 1
    test = 2

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

data = None
def save_data(name):
    joblib.dump(data, CACHE_DIR / f"data_{name}.joblib")

def load_data(name):
    global data
    data = joblib.load(CACHE_DIR / f"data_{name}.joblib")
