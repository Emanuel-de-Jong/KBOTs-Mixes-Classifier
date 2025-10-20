import joblib
import os
from pathlib import Path

TRAIN_DIR = Path("train")
TEST_DIR = Path("test")
TEST_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
BATCH_DIR = Path("batch")

labels = None
label_count = 0
if os.path.exists(CACHE_DIR / "labels.joblib"):
    labels = joblib.load(CACHE_DIR / "labels.joblib")
    label_count = len(labels)
