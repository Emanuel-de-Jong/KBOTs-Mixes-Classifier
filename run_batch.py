import joblib
import json
import sys
import os
import global_params as g
from Classifier import Classifier
from Utils import Logger
from tqdm import tqdm
from Mert import Mert

cache_path = g.BATCH_DIR / "cache.joblib"
logger = Logger(g.BATCH_DIR / "log.log")
mert = Mert()
models = {
    "global": None,
    "general_pop": None,
    "rock": None,
    "edm": None,
}

cache = {}
if cache_path.exists():
    print("Loading cache...")
    cache = joblib.load(cache_path)

for name in models.keys():
    models[name] = Classifier(name, mert)

results = {}
song_paths = list(g.BATCH_DIR.glob("*.mp3"))
for song_path in tqdm(song_paths, total=len(song_paths)):
    embs = cache[song_path] if song_path in cache else None

    tops = {}
    for model_name, model in models.items():
        top, embs = model.infer(song_path, embs)
        if top is None or len(top) == 0:
            logger.writeln(f'[ERROR]: Inference failed on model: "{model_name}", song: "{song_path}"!')
            sys.exit(1)
        
        tops[model_name] = top
    
    cache[song_path] = embs
    
    song_name, _ = os.path.splitext(os.path.basename(song_path))
    results[song_name] = tops

joblib.dump(cache, cache_path)

with open(g.BATCH_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
