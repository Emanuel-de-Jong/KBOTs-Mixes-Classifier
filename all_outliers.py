import global_params as g
from GenreOutliers import GenreOutliers

GENRE_MIN_SONG_COUNT = 5
GENRE_MAX_SONG_COUNT = 50

genre_counts = {}
for path in g.TRAIN_DIR.iterdir():
    if path.is_dir():
        mp3_count = sum(1 for f in path.iterdir() if f.suffix.lower() == ".mp3")
        genre_counts[path.name] = mp3_count

out_by_genre = {}
genre_outliers = GenreOutliers()
for genre, count in sorted(genre_counts.items()):
    if GENRE_MIN_SONG_COUNT != -1 and count < GENRE_MIN_SONG_COUNT:
        continue
    if GENRE_MAX_SONG_COUNT != -1 and count > GENRE_MAX_SONG_COUNT:
        continue

    out = genre_outliers.run(genre)
    out_by_genre[genre] = out

for genre, out in sorted(out_by_genre.items()):
    genre_outliers.print_results(genre, out)
