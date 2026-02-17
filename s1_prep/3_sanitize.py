import unidecode
import shutil
import re
import os
import s0_utils.global_params as g

PLAYLISTS_TO_REMOVE = [
    # Too small
    "Chill Alternate Rock",
    # Too vague
    "Gamer",
    "Pioneer_EDM",
    "Pioneer",
    "Romantic",
    "Chill_EDM",
    "Groovy_EDM",
    # Pending
    "Pending",
    "Pending_Genre",
    "Pending_Second_Chance",
    "Pending_Trance",
]
PLAYLISTS_TO_MERGE = {
    # New
    "Dark_and_Psych_Pop": ["Dark_Pop", "Psych_Pop"],
    "Dark_and_Psych_Rock": ["Dark_Rock", "Psych_Rock"],
    "Vocal_Trance": ["Downbeat_Vocal_Trance", "Upbeat_Vocal_Trance"],
    # Combined
    "Hardstyle": ["Mainstream_Hardstyle"],
    "Mainstream_Vocal_Psytrance": ["Pending_Mainstream_Vocal_Psytrance"],
}

# Clean folder names
for folder in os.listdir(g.TRAIN_PLAYLISTS_DIR):
    folder_path = g.TRAIN_PLAYLISTS_DIR / folder
    if folder_path.is_dir():
        new_name = folder
        if new_name.lower().startswith("kbot's "):
            new_name = new_name[7:]
        if new_name.lower().endswith(" mix"):
            new_name = new_name[:-4]
        
        new_name = new_name.strip().replace(" ", "_")
        new_path = g.TRAIN_PLAYLISTS_DIR / new_name

        if os.path.basename(new_path) in PLAYLISTS_TO_REMOVE:
            shutil.rmtree(folder_path)
            continue

        if new_path != folder_path:
            folder_path.rename(new_path)

# Merge playlists
for target, sources in PLAYLISTS_TO_MERGE.items():
    is_src_dir_missing = False
    for src in sources:
        src_dir = g.TRAIN_PLAYLISTS_DIR / src
        if not os.path.exists(src_dir):
            is_src_dir_missing = True
            break
    
    if is_src_dir_missing:
        continue
    
    target_dir = g.TRAIN_PLAYLISTS_DIR / target
    target_dir.mkdir(exist_ok=True)
    for src in sources:
        src_dir = g.TRAIN_PLAYLISTS_DIR / src
        if src_dir.exists() and src_dir.is_dir():
            for mp3_file in src_dir.glob("*.mp3"):
                dest_file = target_dir / mp3_file.name
                shutil.copy2(mp3_file, dest_file)
            
            shutil.rmtree(src_dir)

# Clean mp3 names
for p in g.TRAIN_DIR.rglob("*.mp3"):
    old_stem = p.stem
    new_stem = unidecode.unidecode(old_stem)
    new_stem = re.sub(r'[^a-zA-Z0-9\s\.\-\_\,]', '', new_stem)
    new_stem = re.sub(r'\s+', ' ', new_stem).strip()
    new_stem = new_stem.replace(" ", "_")
    
    if new_stem and new_stem != old_stem:
        new_filename = new_stem + p.suffix
        new_path = p.with_name(new_filename)
        
        if not new_path.exists():
            p.rename(new_path)
