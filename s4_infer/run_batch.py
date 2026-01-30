import joblib
import yaml
import sys
import os
import s0_utils.global_params as g
from pathlib import Path
from s0_utils.Classifier import Classifier
from s0_utils.Logger import Logger
from tqdm import tqdm
from s0_utils.Mert import Mert

BATCH_DIR = Path("s4_infer/batch")
CACHE_PATH = BATCH_DIR / "cache.joblib"
RESULTS_PATH = Path("s4_infer/batch_results.yaml")

logger = Logger(BATCH_DIR / "batch.log")
mert = Mert()

models = {}
for name in g.MODELS.keys():
    models[name] = None

cache = {}
if CACHE_PATH.exists():
    print("Loading cache...")
    cache = joblib.load(CACHE_PATH)

for name in models.keys():
    models[name] = Classifier(name, mert)

results = {}
song_paths = list(BATCH_DIR.glob("*.mp3"))
for song_path in tqdm(song_paths, total=len(song_paths)):
    embs = None
    if song_path in cache:
        embs = cache[song_path]
    else:
        embs = mert.run(song_path)
        cache[song_path] = embs

    tops = []
    for model_name, model in models.items():
        top, _ = model.infer(song_path, model.scale_embs(embs))
        if top is None or len(top) == 0:
            logger.writeln(f'[ERROR]: Inference failed on model: "{model_name}", song: "{song_path}"!')
            sys.exit(1)

        top_count = 3 if model_name == "global" else 2
        top = top[:top_count]
        top = [f"{item[0]}: {item[1]}" for item in top]
        
        print(f"{song_path}: {top}")
        tops.append([model_name, top])
    
    song_name, _ = os.path.splitext(os.path.basename(song_path))
    results[song_name] = tops

joblib.dump(cache, CACHE_PATH)

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    yaml.dump(results, f)
