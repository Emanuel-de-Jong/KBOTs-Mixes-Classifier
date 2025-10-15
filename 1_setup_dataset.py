import unidecode
import shutil
import re
import os
from pathlib import Path

PLAYLISTS_TO_REMOVE = [
    "Chill EDM",
    "Gamer",
    "Groovy EDM",
    "Pioneer EDM",
    "Nature Vibe",
    "Pioneer",
    "Romantic",
    "Jazz",
    "Ambient Techno",
    "Chill Alternate Rock",
    "DnB",
    "Funk",
    "Grunge",
    "Blues",
    "Folk",
    "IDM",
    "Punk",
    "Acoustic",
    "Groovy House",
]
PLAYLISTS_TO_MERGE = {
    # New
    "Vocal Trance": ["Downbeat Vocal Trance", "Upbeat Vocal Trance"],
    "Dark and Psych Pop": ["Dark Pop", "Psych Pop"],
    "Dark and Psyc Rock": ["Dark Rock", "Psyc Rock"],
    # Combined
    "Progressive House": ["Mainstream Progressive House"],
    "Slap House": ["Chill Slap House"],
    "Swing": ["Electro Swing"],
    "Synthwave": ["Synth Funk"],
}

music_dir = Path("music")
test_dir = Path("test")

# Clean folder names
for folder in os.listdir(music_dir):
    folder_path = music_dir / folder
    if folder_path.is_dir():
        new_name = folder
        if new_name.lower().startswith("kbot's "):
            new_name = new_name[7:]
        if new_name.lower().endswith(" mix"):
            new_name = new_name[:-4]
        
        new_name = new_name.strip()
        new_path = music_dir / new_name

        if os.path.basename(new_path) in PLAYLISTS_TO_REMOVE:
            shutil.rmtree(folder_path)
            continue

        if new_path != folder_path:
            folder_path.rename(new_path)

# Merge playlists
for target, sources in PLAYLISTS_TO_MERGE.items():
    target_dir = music_dir / target
    target_dir.mkdir(exist_ok=True)
    for src in sources:
        src_dir = music_dir / src
        if src_dir.exists() and src_dir.is_dir():
            for mp3_file in src_dir.glob("*.mp3"):
                dest_file = target_dir / mp3_file.name
                shutil.copy2(mp3_file, dest_file)
            
            shutil.rmtree(src_dir)

# Clean mp3 names
for p in music_dir.glob("*/*.mp3"):
    old_stem = p.stem
    new_stem = unidecode.unidecode(old_stem)
    new_stem = re.sub(r'[^a-zA-Z0-9\s\.\-\_\,]', '', new_stem)
    new_stem = re.sub(r'\s+', ' ', new_stem).strip()
    
    if new_stem and new_stem != old_stem:
        new_filename = new_stem + p.suffix
        new_path = p.with_name(new_filename)
        
        if not new_path.exists():
            p.rename(new_path)

# Fill test dir
if not test_dir.exists():
    for subdir in music_dir.iterdir():
        if subdir.is_dir():
            mp3_files = sorted(subdir.glob("*.mp3"))
            if mp3_files:
                test_subdir = test_dir / subdir.name
                test_subdir.mkdir(parents=True, exist_ok=True)

                src_file = mp3_files[0]
                dest_file = test_subdir / src_file.name
                shutil.move(str(src_file), str(dest_file))
