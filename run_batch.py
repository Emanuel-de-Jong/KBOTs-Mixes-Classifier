import joblib
import yaml
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

models = {}
for name in g.MODELS.keys():
    models[name] = None

cache = {}
if cache_path.exists():
    print("Loading cache...")
    cache = joblib.load(cache_path)

for name in models.keys():
    models[name] = Classifier(name, mert)

results = {}
song_paths = list(g.BATCH_DIR.glob("*.mp3"))
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

joblib.dump(cache, cache_path)

with open(g.BATCH_DIR / "results.yaml", "w", encoding="utf-8") as f:
    yaml.dump(results, f)
