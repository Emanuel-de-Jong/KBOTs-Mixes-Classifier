import sys
import yaml
from pathlib import Path
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
            pass
        else:
            self.info(msg)

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

def yt_dlp_hook(d):
    if d['status'] == 'finished':
        pass

yt_dlp_config_base = {
    "logger": YtDlpLogger(),
    "progress_hooks": [yt_dlp_hook],
    "format": "ba[acodec^=mp3]/ba/b",
    "extractaudio": True,
    "audioformat": "mp3",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
    }],
    "playlist_random": True,
    "max_downloads": 60,
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

            genre_dir = DLS_DIR / genre
            genre_dir.mkdir(exist_ok=True)

            yt_dlp_config = {
                **yt_dlp_config_base,
                "outtmpl": str(
                    genre_dir / "%(playlist)s/%(playlist_index)s_-_%(title)s.%(ext)s"
                ),
            }
            with yt_dlp.YoutubeDL(yt_dlp_config) as ydl:
                try:
                    error_code = ydl.download([playlist_url])
                except yt_dlp.utils.MaxDownloadsReached:
                    pass
                
                if error_code != 0:
                    raise Exception(f"yt_dlp error code {error_code} on {genre} {playlist_url} .")
