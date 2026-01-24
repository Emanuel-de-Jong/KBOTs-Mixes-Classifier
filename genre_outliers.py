import os
import numpy as np
import pandas as pd
import joblib
import global_params as g
from tqdm import tqdm
from Mert import Mert

GENRE_TO_TEST = "Breakbeat"
MAX_CHUNKS = 5
Z_TRES = 1.0

OUTLIERS_PATH = g.CACHE_DIR / f"outliers_{GENRE_TO_TEST}.joblib"

def extract_embeddings():
    genre_dir = g.TRAIN_DIR / GENRE_TO_TEST
    if not genre_dir.exists() or not genre_dir.is_dir():
        raise Exception(f"Genre dir not found: {genre_dir}")

    mp3s = sorted(list(genre_dir.glob("*.mp3")))
    if len(mp3s) == 0:
        raise Exception(f"No mp3 files found in: {genre_dir}")

    mert = Mert()
    data = []

    for path in tqdm(mp3s, total=len(mp3s)):
        song_name = g.get_song_name(str(path))
        song_embs = mert.run(str(path), MAX_CHUNKS)
        if song_embs is None:
            continue

        if not isinstance(song_embs, np.ndarray) or song_embs.ndim != 4:
            continue

        for emb in song_embs:
            if not isinstance(emb, np.ndarray):
                continue
            if emb.shape != (Mert.TIME_STEPS, 1024, 25):
                continue

            data.append({
                "data_set": g.DataSetType.train,
                "label": -1,
                "song": song_name,
                "filepath": str(path),
                "data": emb
            })

    df = pd.DataFrame(data)
    return df

def compute_outliers(df):
    if df is None or len(df) == 0:
        print("No data.")
        return None

    song_tensors = {}
    for song, group in df.groupby("song"):
        chunks = np.stack(group["data"].values, axis=0)
        song_tensors[song] = chunks.mean(axis=0)

    songs = list(song_tensors.keys())
    if len(songs) < 3:
        print("Not enough songs to compute outliers reliably.")
        return None

    all_song_stack = np.stack([song_tensors[s] for s in songs], axis=0)
    centroid = all_song_stack.mean(axis=0)

    distances = []
    for s in songs:
        d = layerwise_cosine_distance(song_tensors[s], centroid)
        distances.append(d)

    z, med, mad, scale = robust_zscores(distances)

    results = pd.DataFrame({
        "song": songs,
        "distance": distances,
        "robust_z": z
    }).sort_values("distance", ascending=False)

    outliers = results[results["robust_z"] >= Z_TRES].copy()

    print(f"Genre: {GENRE_TO_TEST}")
    print(f"Songs: {len(results)}")
    print(f"Median distance: {med:.6f}")
    print(f"MAD: {mad:.6f}")
    print(f"Z threshold: {Z_TRES}")
    print("")

    if len(outliers) == 0:
        print("No clear outliers found.")
    else:
        for _, row in outliers.iterrows():
            print(f"{row['song']}: distance={row['distance']:.6f} z={row['robust_z']:.3f}")

    return {
        "results": results,
        "centroid": centroid
    }

def layerwise_cosine_distance(a, b):
    dists = []
    for l in range(a.shape[-1]):
        va = a[:, :, l].reshape(-1)
        vb = b[:, :, l].reshape(-1)
        va = l2_normalize(va)
        vb = l2_normalize(vb)
        dists.append(1.0 - float(np.dot(va, vb)))
    return float(np.mean(dists))

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x)
    return x / (n + eps)

def robust_zscores(values):
    values = np.asarray(values, dtype=np.float64)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = (mad * 1.4826) if mad > 0 else (np.std(values) + 1e-12)
    z = (values - med) / (scale + 1e-12)
    return z, med, mad, scale

if os.path.exists(OUTLIERS_PATH):
    saved = joblib.load(OUTLIERS_PATH)
    df = saved.get("data", None)
else:
    df = extract_embeddings()
    saved = {"data": df}
    joblib.dump(saved, OUTLIERS_PATH)

out = compute_outliers(df)
if out is not None:
    saved["outliers"] = out["results"]
    joblib.dump(saved, OUTLIERS_PATH)
