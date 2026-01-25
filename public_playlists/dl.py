import sys
import subprocess
import yaml
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import global_params as g

PUBLIC_PLAYLISTS_DIR = BASE_DIR / "public_playlists"
DLS_DIR = PUBLIC_PLAYLISTS_DIR / "dls"
DLS_DIR.mkdir(exist_ok=True)

with open(PUBLIC_PLAYLISTS_DIR / "public_playlists.yaml", "r") as f:
    categories_playlists = yaml.safe_load(f)
    del categories_playlists["requirements"]

for category, genres_playlists in categories_playlists.items():
    for genre, genre_playlists in genres_playlists.items():
        for playlist in genre_playlists:
            if not playlist:
                continue

            genre_dir = DLS_DIR / genre
            genre_dir.mkdir(exist_ok=True)
            cmd = [
                "yt-dlp",
                "-t", "mp3",
                "--playlist-random",
                "--max-downloads", "60",
                "--match-filters", "duration<600",
                "-o", f"{genre_dir}/%(playlist)s/%(playlist_index)s_-_%(title)s.%(ext)s",
                playlist]
            subprocess.run(cmd)
