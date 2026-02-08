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
    "As_Played_on_Uplifting_Only",
    "Deep_House_Antidote_Music",
    "FREE_DOWNLOAD",
    "Emotional_EDM_Slap_House",
    "Emotional_Slap_House",
    "Emotional_EDM",
    "2025_Dance_Your_Feelings",
    "2025_Deep_Slap_House_Feelings",
    "2025_Deep_Vibes",
    "2025_Feel_the_Vibe",
    "2025_Feel_the_Rush",
    "2025_Night_Vibes",
    "2025_Sad_Beautiful_Vibes",
    "2025_Let_the_Music_Speak",
    "2025_Feel_Every_Beat",
    "Boris_Brejcha_Minimal_Techno_Style_with_Future_Minimal",
    "Boris_Brejcha_Style_Minimal_Techno_Song",
    "Boris_Brejcha_Style_Minimal_Techno",
    "Official_Lyric_Video",
    "Official_Music_Video",
    "Lyrics",
    "Lyric",
    "Music_Video",
    "Extended_Mix",
    "ASOT_800_Anthem",
    "ASOT_900_Anthem",
    "ASOT_2023_Anthem",
    "FSOE_550_Anthem",
    "Anthem",
    "FSOE",
    "Elektroshok_Records",
    "Deep_House_Atjazz_Record_Company",
    "Deep_House_South_Africa_Records",
    "Deep_House_South_Africa",
    "Antidote_Music",
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
    
    sanitized = re.sub(r"[.':\-]", "", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", sanitized)
    return re.sub(r"_+", "_", sanitized).strip()

def collect_songs(root):
    songs = {}
    for path in root.rglob("*.mp3"):
        new_path = Path(path).resolve()
        words = set(sanitize(path.stem).split("_"))
        size_mb = path.stat().st_size / (1024 * 1024)
        songs[new_path] = SongInfo(words, size_mb)
    
    return songs

def compare_songs_words(songs):
    results = []
    for (path1, s1), (path2, s2) in combinations(songs.items(), 2):
        matches = len(s1.words & s2.words)
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
    sanitized = name
    for phrase in BLACKLIST:
        sanitized = re.sub(rf"{re.escape(phrase)}", "", sanitized, flags=re.IGNORECASE)
    
    return sanitized.replace("_", " ").replace("-", "█")

def write_group(f, group, is_size=False):
    for metric, p1, p2 in group:
        s1 = strip_blacklist(p1.stem)
        s2 = strip_blacklist(p2.stem)
        
        label = f"{metric:.6f} MB diff" if is_size else f"{metric} words"
        spacing = max(len(s1), len(s2)) + 6

        f.write(f"{label}:\n")
        if is_size:
            f.write(f"{p1}\n")
            f.write(f"{p2}\n\n")
        else:
            f.write(s1.ljust(spacing) + f"{p1}\n")
            f.write(s2.ljust(spacing) + f"{p2}\n\n")

def write_results(results, out_same, out_diff, is_size=False):
    if is_size:
        results.sort(key=lambda x: (x[0], x[1], x[2]))
    else:
        results.sort(key=lambda x: (-x[0], x[1], x[2]))
    
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
