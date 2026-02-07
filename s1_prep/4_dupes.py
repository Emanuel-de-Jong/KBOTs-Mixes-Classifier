import re
import s0_utils.global_params as g
from pathlib import Path
from itertools import combinations

MIN_WORD_MATCHES = 5

SAME_GENRE_OUTPUT_FILE = Path("s1_prep/4_dupes_same_genre.log")
DIFFERENT_GENRE_OUTPUT_FILE = Path("s1_prep/4_dupes_different_genre.log")

BLACKLIST = [
    "As Played on Uplifting Only",
    "Deep House Antidote Music",
    "FREE DOWNLOAD",
    "Emotional EDM Slap House",
    "Emotional Slap House",
    "Emotional EDM",
    "2025 Dance Your Feelings",
    "2025 Deep Slap House Feelings",
    "2025 Deep Vibes",
    "2025 Feel the Vibe",
    "2025 Feel the Rush",
    "2025 Night Vibes",
    "Boris Brejcha Style Minimal Techno Song",
    "Official Lyric Video",
    "Official Music Video",
    "Lyrics",
    "Lyric",
    "Music Video",
    "Elektroshok Records",
    "Deep House Atjazz Record Company",
    "Deep House South Africa Records",
    "Deep House South Africa",
    "Antidote Music",
    "4K",
]
BLACKLIST = list(map(lambda x: x.lower(), BLACKLIST))

def sanitize(name):
    sanitized = name.lower()
    sanitized = re.sub(r"[.':\-]", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9 ]+", " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    for phrase in BLACKLIST:
        sanitized = re.sub(rf"\b{re.escape(phrase)}\b", "", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized

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

def strip_blacklist(name):
    sanitized = name.replace("_", " ")
    for phrase in BLACKLIST:
        sanitized = re.sub(rf"\b{re.escape(phrase)}\b", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized

def write_group(f, group):
    for matches, p1, p2 in group:
        s1 = strip_blacklist(p1.stem)
        s2 = strip_blacklist(p2.stem)
        spacing = max(len(s1), len(s2)) + 6
        f.write(f"{matches} words:\n")
        f.write(s1.ljust(spacing) + f"[{p1.parent}]\n")
        f.write(s2.ljust(spacing) + f"[{p2.parent}]\n\n")

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
