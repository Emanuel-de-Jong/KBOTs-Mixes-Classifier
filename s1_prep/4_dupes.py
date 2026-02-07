import re
import s0_utils.global_params as g
from pathlib import Path
from itertools import combinations

MIN_WORD_MATCHES = 5

SAME_GENRE_OUTPUT_FILE = Path("s1_prep/4_dupes_same_genre.log")
DIFFERENT_GENRE_OUTPUT_FILE = Path("s1_prep/4_dupes_different_genre.log")

def sanitize(name):
    sanitized = name.lower()
    sanitized = re.sub(r"[.\-:]", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9 ]+", " ", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()

def collect_songs(root):
    songs = {}
    for path in root.rglob("*.mp3"):
        filename = sanitize(path.stem)
        trimmed_path = Path(*path.parts[1:])
        songs[trimmed_path] = filename.split(" ")
    
    return songs

def compare_songs(songs):
    results = []
    for (path1, words1), (path2, words2) in combinations(songs.items(), 2):
        matches = len(set(words1) & set(words2))
        if matches >= MIN_WORD_MATCHES:
            p1, p2 = sorted((str(path1), str(path2)))
            results.append((matches, p1, p2))
    
    return results

def write_group(f, group):
    for matches, p1, p2 in group:
        spacing = max(len(p1.stem), len(p2.stem)) + 6
        f.write(f"{matches} words:\n")
        f.write(p1.stem.replace("_", " ").ljust(spacing) + f"[{p1.parent}]\n")
        f.write(p2.stem.replace("_", " ").ljust(spacing) + f"[{p2.parent}]\n\n")

def write_results(results):
    results.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    same_genre = []
    different_genre = []
    
    for matches, p1, p2 in results:
        p1 = Path(p1)
        p2 = Path(p2)

        p1_genre = p1.parent.parent.name if "watchv" in str(p1) else p1.parent.name
        p2_genre = p2.parent.parent.name if "watchv" in str(p2) else p2.parent.name
        if p1_genre == p2_genre:
            same_genre.append((matches, p1, p2))
        else:
            different_genre.append((matches, p1, p2))
    
    with SAME_GENRE_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        write_group(f, same_genre)
    
    with DIFFERENT_GENRE_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        write_group(f, different_genre)

songs = collect_songs(g.TRAIN_DIR)
results = compare_songs(songs)
write_results(results)
print("Done!")
