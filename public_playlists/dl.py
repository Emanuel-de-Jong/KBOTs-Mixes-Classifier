import sys
import yaml
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import yt_dlp
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import global_params as g

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
    "download_archive": PUBLIC_PLAYLISTS_DIR / "archive.txt",

    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
    }],

    "noplaylist": False,
    "playlist_random": True,
    "max_downloads": 2,
    "match_filter": yt_dlp.utils.match_filter_func("duration<600"),
}

with open(PUBLIC_PLAYLISTS_DIR / "public_playlists.yaml", "r") as f:
    categories_playlists = yaml.safe_load(f)
    categories_playlists.pop("requirements", None)

for category, genres_playlists in categories_playlists.items():
    for genre, genre_playlists in genres_playlists.items():
        for playlist_url in genre_playlists:
            if not playlist_url:
                continue

            if "://music.y" in playlist_url:
                url = urlparse(playlist_url)
                query = parse_qs(url.query)
                query.pop('v', None)
                playlist_url = urlunparse(url._replace(query=urlencode(query, doseq=True)))

            genre_dir = DLS_DIR / genre
            genre_dir.mkdir(exist_ok=True)

            yt_dlp_config = {
                **yt_dlp_config_base,
                "outtmpl": str(genre_dir / "%(playlist)s/%(playlist_index)s_-_%(title)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(yt_dlp_config) as ydl:
                try:
                    error_code = ydl.download([playlist_url])
                except yt_dlp.utils.MaxDownloadsReached:
                    pass
