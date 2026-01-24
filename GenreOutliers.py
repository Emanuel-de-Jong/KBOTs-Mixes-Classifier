import os
import time
import numpy as np
import pandas as pd
import joblib
import global_params as g
from tqdm import tqdm
from Mert import Mert

class GenreOutliers():
    MAX_CHUNKS = 5
    Z_TRES = 1.2

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.mert = Mert()

    def run(self, genre):
        outliers_path = g.CACHE_DIR / f"outliers_{genre}.joblib"

        if self.use_cache and os.path.exists(outliers_path):
            saved = joblib.load(outliers_path)
            df = saved.get("data", None)
        else:
            df = self.extract_embeddings(genre)
            saved = {"data": df}
            joblib.dump(saved, outliers_path)

        compute_outliers_start_time = time.perf_counter()
        out = self.compute_outliers(df)
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

        data = []

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

                data.append({
                    "data_set": g.DataSetType.train,
                    "label": -1,
                    "song": song_name,
                    "filepath": str(path),
                    "data": emb
                })

        return pd.DataFrame(data)

    def compute_outliers(self, df):
        if df is None or len(df) == 0:
            return None

        song_tensors = {}
        for song, group in df.groupby("song"):
            chunks = np.stack(group["data"].values, axis=0)
            song_tensors[song] = chunks.mean(axis=0)

        songs = list(song_tensors.keys())
        if len(songs) < 3:
            return None

        all_song_stack = np.stack([song_tensors[s] for s in songs], axis=0)
        centroid = all_song_stack.mean(axis=0)

        distances = []
        for s in songs:
            distances.append(self.layerwise_cosine_distance(song_tensors[s], centroid))

        z, med, mad, scale = self.robust_zscores(distances)

        results = pd.DataFrame({
            "song": songs,
            "distance": distances,
            "robust_z": z
        }).sort_values("distance", ascending=False)

        return {
            "results": results,
            "centroid": centroid,
            "median": med,
            "mad": mad,
            "scale": scale
        }

    def layerwise_cosine_distance(self, a, b):
        dists = []
        for l in range(a.shape[-1]):
            va = a[:, :, l].reshape(-1)
            vb = b[:, :, l].reshape(-1)
            va = self.l2_normalize(va)
            vb = self.l2_normalize(vb)
            dists.append(1.0 - float(np.dot(va, vb)))
        return float(np.mean(dists))

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
        med = out["median"]
        mad = out["mad"]

        outliers = results[results["robust_z"] >= self.Z_TRES]

        lines = []
        lines.append(f"= {genre} =")
        lines.append(f"Songs: {len(results)}")
        lines.append(f"Median distance: {med:.6f}")
        lines.append(f"MAD: {mad:.6f}")
        lines.append(f"Z threshold: {self.Z_TRES}")

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
    genre = "Breakbeat"
    genre_outliers = GenreOutliers()
    out, _ = genre_outliers.run(genre)
    print(genre_outliers.results_to_string(genre, out))
