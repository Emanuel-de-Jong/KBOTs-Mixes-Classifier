import re
import s0_utils.global_params as g
from pathlib import Path
from itertools import combinations

MIN_WORD_MATCHES = 5

OUTPUT_FILE = Path("s1_prep/2_dupes.log")

def sanitize(name):
    sanitized = re.sub(r"[^A-Za-z0-9]+", " ", name)
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

def write_results(results):
    results.sort(key=lambda x: (-x[0], x[1], x[2]))
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for matches, p1, p2 in results:
            path1 = Path(p1)
            path2 = Path(p2)
            spacing = max(len(path1.stem), len(path2.stem)) + 6
            f.write(f"{matches} words:\n")
            f.write(path1.stem.replace("_", " ").ljust(spacing) + f"[{path1.parent}]\n")
            f.write(path2.stem.replace("_", " ").ljust(spacing) + f"[{path2.parent}]\n\n")

songs = collect_songs(g.TRAIN_DIR)
results = compare_songs(songs)
write_results(results)
print("Done!")
