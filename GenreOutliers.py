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
    RANK_THRESHOLD = 0.75 # Lower = more sensitive
    MEDIAN_MULT = 1.15 # Lower = more sensitive
    SMALL_N_STRICT_MULT = 2.0

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

        song_embeddings = {}
        for path in tqdm(mp3s, total=len(mp3s)):
            song_name = g.get_song_name(str(path))
            song_embs = self.mert.run(str(path), self.MAX_CHUNKS)
            if song_embs is None or not isinstance(song_embs, np.ndarray):
                continue
            if song_embs.ndim != 4:
                continue

            pooled_chunks = []
            for emb in song_embs:
                if not isinstance(emb, np.ndarray):
                    continue
                if emb.shape != (Mert.TIME_STEPS, 1024, 25):
                    continue
                pooled_chunks.append(emb.mean(axis=0))

            if len(pooled_chunks) == 0:
                continue

            song_embeddings[song_name] = np.mean(pooled_chunks, axis=0)

        return song_embeddings

    def compute_outliers(self, song_embeddings):
        if song_embeddings is None or len(song_embeddings) < 2:
            return None

        songs = list(song_embeddings.keys())
        N = len(songs)

        X = np.stack([
            self.flatten_layers(song_embeddings[s]) for s in songs
        ], axis=0)

        X = self.l2_normalize_batch(X)

        centroid = X.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        distances = 1.0 - np.dot(X, centroid)

        order = np.argsort(distances)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(N)

        rank_score = ranks / max(1, N - 1)

        med = np.median(distances)

        results = pd.DataFrame({
            "song": songs,
            "distance": distances,
            "rank_score": rank_score
        }).sort_values("distance", ascending=False)

        if N < 8:
            outliers = results[
                results["distance"] >= self.SMALL_N_STRICT_MULT * med
            ]
        else:
            outliers = results[
                (results["rank_score"] >= self.RANK_THRESHOLD) &
                (results["distance"] >= self.MEDIAN_MULT * med)
            ]

        return {
            "results": results,
            "outliers": outliers,
            "median": med
        }

    def flatten_layers(self, x):
        layers = []
        for l in range(x.shape[-1]):
            layers.append(x[:, l])
        return np.concatenate(layers)

    def l2_normalize_batch(self, X, eps=1e-12):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + eps)

    def results_to_string(self, genre, out):
        if out is None:
            print("No data.")
            return None

        results = out["results"]
        outliers = out["outliers"]

        lines = []
        lines.append(f"= {genre} =")
        lines.append(f"Songs: {len(results)}")
        lines.append(f"Rank threshold: {self.RANK_THRESHOLD:.2f}")
        lines.append(f"Median multiplier: {self.MEDIAN_MULT:.2f}")
        lines.append(f"Median distance: {out['median']:.6f}")

        if len(outliers) == 0:
            lines.append("No clear outliers found.")
        else:
            for _, row in outliers.iterrows():
                lines.append(row["song"])
                lines.append(f"\tdistance: {row['distance']:.6f}")
                lines.append(f"\trank: {row['rank_score']:.3f}")

            lines.append("")

        return "\n".join(lines)

if __name__ == "__main__":
    genre = "Acid Trance"
    genre_outliers = GenreOutliers(use_cache=True)
    out, _ = genre_outliers.run(genre)
    print(genre_outliers.results_to_string(genre, out))
