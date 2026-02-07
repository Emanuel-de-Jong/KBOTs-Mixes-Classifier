import re
import s0_utils.global_params as g
from pathlib import Path
from itertools import combinations

MIN_WORD_MATCHES = 5
MAX_MB_DIFF = 0.005

WORDS_SAME_GENRE_FILE = Path("s1_prep/4_dupes_words_same_genre.log")
WORDS_DIFFERENT_GENRE_FILE = Path("s1_prep/4_dupes_words_different_genre.log")
SIZE_SAME_GENRE_FILE = Path("s1_prep/4_dupes_size_same_genre.log")
SIZE_DIFFERENT_GENRE_FILE = Path("s1_prep/4_dupes_size_different_genre.log")

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
    "2025 Sad Beautiful Vibes",
    "2025 Let the Music Speak",
    "2025 Feel Every Beat",
    "Boris Brejcha Minimal Techno Style with Future Minimal",
    "Boris Brejcha Style Minimal Techno Song",
    "Boris Brejcha Style Minimal Techno",
    "Official Lyric Video",
    "Official Music Video",
    "Lyrics",
    "Lyric",
    "Music Video",
    "Extended Mix",
    "ASOT 800 Anthem",
    "ASOT 900 Anthem",
    "ASOT 2023 Anthem",
    "FSOE 550 Anthem",
    "Anthem",
    "FSOE",
    "Elektroshok Records",
    "Deep House Atjazz Record Company",
    "Deep House South Africa Records",
    "Deep House South Africa",
    "Antidote Music",
    "4K",
]
BLACKLIST = list(map(lambda x: x.lower(), BLACKLIST))

class SongInfo:
    def __init__(self, words, size_mb):
        self.words = words
        self.size_mb = size_mb

def sanitize(name):
    sanitized = name.lower()
    for phrase in BLACKLIST:
        sanitized = re.sub(rf"{re.escape(phrase)}", "", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
    
    sanitized = re.sub(r"[.':\-]", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9 ]+", " ", sanitized)
    return re.sub(r"\s+", " ", sanitized).strip()

def collect_songs(root):
    songs = {}
    for path in root.rglob("*.mp3"):
        trimmed_path = Path(*path.parts[1:])
        filename = sanitize(path.stem).split(" ")
        size_mb = path.stat().st_size / (1024 * 1024)
        songs[trimmed_path] = SongInfo(filename, size_mb)
    
    return songs

def compare_songs_words(songs):
    results = []
    for (path1, s1), (path2, s2) in combinations(songs.items(), 2):
        matches = len(set(s1.words) & set(s2.words))
        if matches >= MIN_WORD_MATCHES:
            p1, p2 = sorted((str(path1), str(path2)))
            results.append((matches, p1, p2))
    
    return results

def compare_songs_size(songs):
    results = []
    for (path1, s1), (path2, s2) in combinations(songs.items(), 2):
        if abs(s1.size_mb - s2.size_mb) <= MAX_MB_DIFF:
            p1, p2 = sorted((str(path1), str(path2)))
            results.append((abs(s1.size_mb - s2.size_mb), p1, p2))
    
    return results

def strip_blacklist(name):
    sanitized = name.replace("_", " ")
    for phrase in BLACKLIST:
        sanitized = re.sub(rf"{re.escape(phrase)}", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized

def write_group(f, group, is_size=False):
    for metric, p1, p2 in group:
        s1 = strip_blacklist(p1.stem)
        s2 = strip_blacklist(p2.stem)
        
        label = f"{metric:.6f} MB diff" if is_size else f"{metric} words"
        spacing = max(len(s1), len(s2)) + 6

        f.write(f"{label}:\n")
        f.write(s1.ljust(spacing) + f"[{p1.parent}]\n")
        f.write(s2.ljust(spacing) + f"[{p2.parent}]\n\n")

def write_results(results, out_same, out_diff, is_size=False):
    same_genre = []
    different_genre = []
    for metric, p1, p2 in results:
        p1 = Path(p1)
        p2 = Path(p2)

        p1_genre = p1.parent.parent.name if "watchv" in str(p1) else p1.parent.name
        p2_genre = p2.parent.parent.name if "watchv" in str(p2) else p2.parent.name
        if p1_genre == p2_genre:
            same_genre.append((metric, p1, p2))
        else:
            different_genre.append((metric, p1, p2))
    
    with out_same.open("w", encoding="utf-8") as f:
        write_group(f, same_genre, is_size)
    
    with out_diff.open("w", encoding="utf-8") as f:
        write_group(f, different_genre, is_size)

songs = collect_songs(g.TRAIN_DIR)

word_results = compare_songs_words(songs)
write_results(word_results, WORDS_SAME_GENRE_FILE, WORDS_DIFFERENT_GENRE_FILE)

size_results = compare_songs_size(songs)
write_results(size_results, SIZE_SAME_GENRE_FILE, SIZE_DIFFERENT_GENRE_FILE, is_size=True,)

print("Done!")
