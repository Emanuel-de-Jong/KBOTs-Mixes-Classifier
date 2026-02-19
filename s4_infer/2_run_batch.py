import joblib
import sys
import os
import json
import s0_utils.global_params as g
from pathlib import Path
from s0_utils.Classifier import Classifier
from s0_utils.EssDiscogs import EssDiscogs
from s0_utils.Logger import Logger
from tqdm import tqdm
from s0_utils.Mert import Mert

TOP_COUNT = 3

MODEL_POSITIONS = {
    "pop_rock_edm": 0
}

BATCH_DIR = Path("s4_infer/batch")
CACHE_PATH = BATCH_DIR / "cache.joblib"
ESS_CACHE_PATH = BATCH_DIR / "ess_cache.joblib"
RESULTS_PATH = Path("s4_infer/results.js")

logger = Logger(BATCH_DIR / "batch.log")
mert = Mert()
ess = EssDiscogs()

models = {}
for path in g.MODELS_DIR.rglob("*.keras"):
    if path.suffix == ".keras":
        name = path.stem.replace("model_", "")
        models[name] = None

cache = {}
if CACHE_PATH.exists():
    print("Loading cache...")
    cache = joblib.load(CACHE_PATH)

for name in models.keys():
    models[name] = Classifier(name, mert)

ordered_model_names = sorted(models.keys())
for custom_name, custom_position in MODEL_POSITIONS.items():
    if custom_name in ordered_model_names:
        ordered_model_names.remove(custom_name)

for custom_name, custom_position in sorted(MODEL_POSITIONS.items(), key=lambda item: item[1]):
    if custom_name in models:
        ordered_model_names.insert(custom_position, custom_name)

results = {}
def append_top(song_name, model_name, top):
    top = top[:TOP_COUNT]

    top_dicts = []
    for genre, prob in top:
        top_dicts.append({
            "genre": genre,
            "prob": prob
        })
    
    print(f"{song_name}: {top}")

    if not song_name in results:
        results[song_name] = {}
    
    results[song_name][model_name] = top_dicts

song_paths = list(BATCH_DIR.rglob("*.mp3"))
song_paths.sort()
for song_path in tqdm(song_paths, total=len(song_paths)):
    song_name = song_path.stem

    embs, ess_embs = None, None
    if song_path in cache:
        embs, ess_embs = cache[song_path]
    else:
        embs = mert.run(song_path)
        ess_embs = ess.get_embs(song_path)
        cache[song_path] = (embs, ess_embs)
    
    top, _ = ess.infer(song_path, ess_embs)
    if top is None or len(top) == 0:
        logger.writeln(f'[ERROR]: Inference failed on model: "{ess.NAME}", song: "{song_path}"!')
        sys.exit(1)
    
    append_top(song_name, ess.NAME, top)

    for model_name in ordered_model_names:
        model = models[model_name]

        model_embs = model.scale_embs(embs)
        model_embs = model.reshape_data(model_embs)

        top, _ = model.infer(song_path, model_embs)
        if top is None or len(top) == 0:
            logger.writeln(f'[ERROR]: Inference failed on model: "{model_name}", song: "{song_path}"!')
            sys.exit(1)

        append_top(song_name, model_name, top)

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
