import s0_utils.global_params as g
from s1_prep.GenreOutliers import GenreOutliers
from pathlib import Path

GENRE_MIN_SONG_COUNT = -1
GENRE_MAX_SONG_COUNT = 50

OUTLIERS_FILE = Path("s1_prep/5_outliers.log")

genre_counts = {}
for path in g.TRAIN_PLAYLISTS_DIR.iterdir():
    if path.is_dir():
        mp3_count = sum(1 for f in path.iterdir() if f.suffix.lower() == ".mp3")
        genre_counts[path.name] = mp3_count

out_by_genre = {}
genre_outliers = GenreOutliers(use_cache=True)

compute_outliers_sum_time = 0
for genre, count in sorted(genre_counts.items()):
    if GENRE_MIN_SONG_COUNT != -1 and count < GENRE_MIN_SONG_COUNT:
        continue
    if GENRE_MAX_SONG_COUNT != -1 and count > GENRE_MAX_SONG_COUNT:
        continue

    out, compute_outliers_time = genre_outliers.run(genre)
    out_by_genre[genre] = out

    compute_outliers_sum_time += compute_outliers_time

with open(OUTLIERS_FILE, "w") as f:
    for genre, out in sorted(out_by_genre.items()):
        results_str = genre_outliers.results_to_string(genre, out)
        if results_str:
            print(results_str)
            f.write(results_str + "\n")

# print(f"Computing the outliers took {compute_outliers_sum_time:.2f} seconds")
