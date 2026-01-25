import re
import sys
import yaml
import math
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import yt_dlp
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import global_params as g

SONGS_PER_GENRE = 60

PUBLIC_PLAYLISTS_DIR = BASE_DIR / "public_playlists"
DLS_DIR = PUBLIC_PLAYLISTS_DIR / "dls"
DLS_DIR.mkdir(exist_ok=True)

class YtDlpLogger:
    def debug(self, msg):
        if msg.startswith('[debug] '):
            print(msg)
        else:
            self.info(msg)

    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)

def yt_dlp_hook(d):
    if d['status'] == 'finished':
        pass

yt_dlp_config_base = {
    "logger": YtDlpLogger(),
    "progress_hooks": [yt_dlp_hook],
    "download_archive": DLS_DIR / "archive.txt",

    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
    }],

    "noplaylist": False,
    "playlist_random": True,
    "match_filter": yt_dlp.utils.match_filter_func("duration<600"),
}

with open(PUBLIC_PLAYLISTS_DIR / "public_playlists.yaml", "r") as f:
    categories_playlists = yaml.safe_load(f)
    categories_playlists.pop("requirements", None)

for category, genres_playlists in categories_playlists.items():
    for genre, genre_playlists in genres_playlists.items():
        valid_playlists = [p for p in genre_playlists if p]
        if not valid_playlists:
            continue

        per_playlist_target = math.ceil(SONGS_PER_GENRE / len(valid_playlists))

        genre_dir = DLS_DIR / genre
        genre_dir.mkdir(exist_ok=True)

        for playlist_url in valid_playlists:
            url = urlparse(playlist_url)
            
            if "://music.y" in playlist_url:
                query = parse_qs(url.query)
                query.pop('v', None)
                playlist_url = urlunparse(url._replace(query=urlencode(query, doseq=True)))

            unsanitized_playlist_name = url.path + url.query
            playlist_name = re.sub(r'[^a-zA-Z0-9]', '', unsanitized_playlist_name)
            playlist_dir = genre_dir / playlist_name

            existing = 0
            if playlist_dir.exists():
                existing = len(list(playlist_dir.rglob("*.mp3")))

            remaining = max(per_playlist_target - existing, 0)
            if remaining == 0:
                continue

            yt_dlp_config = {
                **yt_dlp_config_base,
                "max_downloads": remaining,
                "outtmpl": str(playlist_dir / "%(playlist_index)s_-_%(title)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(yt_dlp_config) as ydl:
                try:
                    error_code = ydl.download([playlist_url])
                except yt_dlp.utils.MaxDownloadsReached:
                    pass
