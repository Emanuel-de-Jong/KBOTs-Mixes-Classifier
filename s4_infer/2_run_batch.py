import joblib
import sys
import os
import json
import s0_utils.global_params as g
from pathlib import Path
from s0_utils.Classifier import Classifier
from s0_utils.Logger import Logger
from tqdm import tqdm
from s0_utils.Mert import Mert

TOP_COUNT = 3

BATCH_DIR = Path("s4_infer/batch")
CACHE_PATH = BATCH_DIR / "cache.joblib"
RESULTS_PATH = Path("s4_infer/results.js")

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
song_paths.sort()
for song_path in tqdm(song_paths, total=len(song_paths)):
    embs = None
    if song_path in cache:
        embs = cache[song_path]
    else:
        embs = mert.run(song_path)
        cache[song_path] = embs

    tops = {}
    for model_name, model in models.items():
        top, _ = model.infer(song_path, model.scale_embs(embs))
        if top is None or len(top) == 0:
            logger.writeln(f'[ERROR]: Inference failed on model: "{model_name}", song: "{song_path}"!')
            sys.exit(1)

        top = top[:TOP_COUNT]

        top_dicts = []
        for genre, prob in top:
            top_dicts.append({
                "genre": genre,
                "prob": prob
            })
        
        print(f"{song_path}: {top}")
        tops[model_name] = top_dicts
    
    song_name, _ = os.path.splitext(os.path.basename(song_path))
    results[song_name] = tops

joblib.dump(cache, CACHE_PATH)

# js_list = ["let results = {"]
# for song_name, tops in results.items():
#     js_list.append(f"\t'{song_name}': {{")
#     for model_name, top in tops.items():
#         js_list.append(f"\t\t{model_name}: [")
#         for dict in top:
#             js_list.append(f"\t\t\t{{ genre: '{dict["genre"]}', prob: {dict["prob"]} }},")
#         js_list.append("\t\t],")
#     js_list.append("\t},")
# js_list.append("};")

# js_str = "\n".join(js_list)

js_str = f"let results = {json.dumps(results, indent=4)};"

RESULTS_PATH.write_text(js_str, encoding="utf-8")
