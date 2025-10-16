import joblib
import json
import sys
import os
from Classifier import Classifier
from pathlib import Path
from Utils import Logger
from tqdm import tqdm
from joblib import load

batch_dir = Path("batch")
cache_path = batch_dir / "cache.joblib"
logger = Logger(batch_dir / "log.log")
classifier = Classifier()

cache = {}
if cache_path.exists():
    print("Loading cache...")
    cache = joblib.load(cache_path)

results = {}
song_paths = list(batch_dir.glob("*.mp3"))
for song_path in tqdm(song_paths, total=len(song_paths)):
    chunk_data = cache[song_path] if song_path in cache else None
    top, chunk_data = classifier.infer(song_path, chunk_data)
    if top is None or len(top) == 0:
        logger.writeln(f'[ERROR]: Inference failed on "{song_path}"!')
        sys.exit(1)
    
    cache[song_path] = chunk_data
    
    song_name, _ = os.path.splitext(os.path.basename(song_path))
    results[song_name] = top

joblib.dump(cache, cache_path)

with open(batch_dir / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
