import os
import time
import numpy as np
import pandas as pd
import joblib
import global_params as g
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Mert import Mert

class GenreOutliers():
    MAX_CHUNKS = 15
    CONTRAST_FACTOR = 1.1

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.mert = Mert()

    def run(self, genre):
        outliers_path = g.CACHE_DIR / f"outliers_{genre}.joblib"

        if self.use_cache and os.path.exists(outliers_path):
            saved = joblib.load(outliers_path)
            song_embeddings = saved.get("song_embeddings", None)
        else:
            song_embeddings = self.extract_embeddings(genre)
            joblib.dump({"song_embeddings": song_embeddings}, outliers_path)

        compute_outliers_start_time = time.perf_counter()
        out = self.compute_outliers(song_embeddings)
        compute_outliers_time = time.perf_counter() - compute_outliers_start_time

        return out, compute_outliers_time

    def extract_embeddings(self, genre):
        print(f"\n= Finding outliers in {genre} =")

        genre_dir = g.TRAIN_DIR / genre
        if not genre_dir.exists() or not genre_dir.is_dir():
            raise Exception(f"Genre dir not found: {genre_dir}")

        mp3s = sorted(list(genre_dir.glob("*.mp3")))
        if len(mp3s) == 0:
            raise Exception(f"No mp3 files found in: {genre_dir}")

        song_embeddings = []
        for path in tqdm(mp3s, total=len(mp3s)):
            song_name = g.get_song_name(str(path))
            song_embs = self.mert.run(str(path), self.MAX_CHUNKS)
            if song_embs is None or not isinstance(song_embs, np.ndarray):
                continue
            if song_embs.ndim != 4:
                continue

            for emb in song_embs:
                if not isinstance(emb, np.ndarray):
                    continue
                if emb.shape != (Mert.TIME_STEPS, 1024, 25):
                    continue

                song_embeddings.append({
                    "song": song_name,
                    "data": emb
                })

        df = pd.DataFrame(song_embeddings)
        if len(df) == 0:
            return df

        all_vals = np.concatenate([x.reshape(-1, x.shape[-1]) for x in df["data"]], axis=0)
        clip_min = np.percentile(all_vals, 1, axis=0)
        clip_max = np.percentile(all_vals, 99, axis=0)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(np.clip(all_vals, clip_min, clip_max))

        scaled = []
        for arr in df["data"]:
            flat = arr.reshape(-1, arr.shape[-1])
            flat = np.clip(flat, clip_min, clip_max)
            flat = scaler.transform(flat)
            scaled.append(flat.reshape(arr.shape))

        df["data"] = scaled
        return df

    def compute_outliers(self, song_embeddings):
        if song_embeddings is None or len(song_embeddings) == 0:
            return None

        song_tensors = {}
        for song, group in song_embeddings.groupby("song"):
            chunks = np.stack(group["data"].values, axis=0)
            song_tensors[song] = np.median(chunks, axis=0)

        songs = list(song_tensors.keys())
        n = len(songs)
        if n < 3:
            return None

        X = np.stack([song_tensors[s] for s in songs], axis=0)

        layer_vars = []
        for l in range(X.shape[-1]):
            flat = X[..., l].reshape(n, -1)
            layer_vars.append(np.var(flat, axis=0).mean())

        layer_vars = np.asarray(layer_vars)
        weights = layer_vars / (layer_vars.sum() + 1e-12)

        distances = []
        for i in range(n):
            centroid = (X.sum(axis=0) - X[i]) / (n - 1)
            distances.append(self.weighted_layerwise_cosine(X[i], centroid, weights))

        distances = np.asarray(distances)
        z, med, mad, scale = self.robust_zscores(distances)

        results = pd.DataFrame({
            "song": songs,
            "distance": distances,
            "robust_z": z
        }).sort_values("distance", ascending=False).reset_index(drop=True)

        if n <= 9:
            k = 1
        elif n <= 15:
            k = 2
        else:
            k = max(2, int(0.15 * n))

        gate = self.CONTRAST_FACTOR * np.median(distances)
        outliers = results.head(k)
        outliers = outliers[outliers["distance"] >= gate]

        return {
            "results": results,
            "outliers": outliers,
            "median": med,
            "mad": mad,
            "layer_weights": weights
        }

    def weighted_layerwise_cosine(self, a, b, weights):
        d = 0.0
        for l in range(a.shape[-1]):
            va = self.l2_normalize(a[..., l].reshape(-1))
            vb = self.l2_normalize(b[..., l].reshape(-1))
            d += weights[l] * (1.0 - float(np.dot(va, vb)))
        return float(d)

    def l2_normalize(self, x, eps=1e-12):
        n = np.linalg.norm(x)
        return x / (n + eps)

    def robust_zscores(self, values):
        values = np.asarray(values, dtype=np.float64)
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        scale = (mad * 1.4826) if mad > 0 else (np.std(values) + 1e-12)
        z = (values - med) / (scale + 1e-12)
        return z, med, mad, scale

    def results_to_string(self, genre, out):
        if out is None:
            print("No data.")
            return None

        results = out["results"]
        outliers = out["outliers"]

        lines = []
        lines.append(f"= {genre} =")
        lines.append(f"Songs: {len(results)}")
        lines.append(f"Median distance: {out['median']:.6f}")
        lines.append(f"MAD: {out['mad']:.6f}")

        if len(outliers) == 0:
            lines.append("No clear outliers found.")
        else:
            for _, row in outliers.iterrows():
                lines.append(row["song"])
                lines.append(f"\tdistance: {row['distance']:.6f}")
                lines.append(f"\tz: {row['robust_z']:.3f}")

            lines.append("")

        return "\n".join(lines)

if __name__ == "__main__":
    genre = "Acid Trance"
    genre_outliers = GenreOutliers(use_cache=False)
    out, _ = genre_outliers.run(genre)
    print(genre_outliers.results_to_string(genre, out))
