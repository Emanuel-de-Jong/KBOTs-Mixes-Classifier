import re
import sys
import yaml
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import yt_dlp
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import global_params as g

SONGS_PER_GENRE = 60
MAX_DURATION_SECONDS = 600
AVAILABILITY_BLACKLIST = ["private", "premium_only", "subscriber_only"]

PUBLIC_PLAYLISTS_DIR = BASE_DIR / "public_playlists"
DLS_DIR = PUBLIC_PLAYLISTS_DIR / "dls"
DLS_DIR.mkdir(exist_ok=True)

class YtDlpLogger:
    def debug(self, msg):
        if msg.startswith("[debug] "):
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
    if d["status"] == "finished":
        pass

availability_expr = " & ".join(f"availability!='{a}'" for a in AVAILABILITY_BLACKLIST)
yt_dlp_config_base = {
    "logger": YtDlpLogger(),
    "progress_hooks": [yt_dlp_hook],
    "js_runtimes": {
        "deno": {
            "path": "/home/graviton/.deno/bin/deno",
        },
    },
    
    "download_archive": str(DLS_DIR / "archive.txt"),
    "restrictfilenames": True,
    "windowsfilenames": True,

    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
    }],

    "noplaylist": False,
    "playlistrandom": True,
    "match_filter": yt_dlp.utils.match_filter_func(f"duration<={MAX_DURATION_SECONDS} & {availability_expr}"),
}

with open(PUBLIC_PLAYLISTS_DIR / "public_playlists.yaml", "r") as f:
    categories_playlists = yaml.safe_load(f)
    categories_playlists.pop("requirements", None)

def get_playlists_songs_needed(valid_playlists):
    playlist_caps = []

    yt_dlp_config = {
        **yt_dlp_config_base,
        "simulate": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(yt_dlp_config) as ydl:
        for playlist_url in valid_playlists:
            info = ydl.extract_info(playlist_url, download=False)

            usable = 0
            entries = info.get("entries") or []
            for e in entries:
                if not e:
                    continue
                if e.get("duration") and e["duration"] > MAX_DURATION_SECONDS:
                    continue
                if e.get("availability") and e["availability"] in AVAILABILITY_BLACKLIST:
                    continue

                usable += 1
            
            playlist_caps.append(usable)

    remaining_songs = SONGS_PER_GENRE
    playlists_songs_needed = [0] * len(valid_playlists)
    while remaining_songs > 0:
        progressed = False
        for i, cap in enumerate(playlist_caps):
            if playlists_songs_needed[i] < cap and remaining_songs > 0:
                playlists_songs_needed[i] += 1
                remaining_songs -= 1
                progressed = True
        
        if not progressed:
            break
    
    return playlists_songs_needed

def dl_playlists(genre_dir, valid_playlists, playlists_songs_needed):
    genre_dir.mkdir(exist_ok=True)

    for playlist_url, songs_needed in zip(valid_playlists, playlists_songs_needed):
        url = urlparse(playlist_url)

        if "://music.y" in playlist_url:
            query = parse_qs(url.query)
            query.pop('v', None)
            playlist_url = urlunparse(url._replace(query=urlencode(query, doseq=True)))

        unsanitized_playlist_name = url.path + url.query
        playlist_name = re.sub(r"[^a-zA-Z0-9]", "", unsanitized_playlist_name)
        playlist_name = playlist_name[:32]
        playlist_dir = genre_dir / playlist_name

        existing_songs = 0
        if playlist_dir.exists():
            existing_songs = len(list(playlist_dir.rglob("*.mp3")))

        remaining_dl = max(songs_needed - existing_songs, 0)
        if remaining_dl == 0:
            continue

        yt_dlp_config = {
            **yt_dlp_config_base,
            "max_downloads": remaining_dl,
            "outtmpl": str(playlist_dir / "%(title)s.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(yt_dlp_config) as ydl:
            try:
                error_code = ydl.download([playlist_url])
            except yt_dlp.utils.DownloadError as e:
                if "private video." not in e.msg.lower():
                    raise e
            except yt_dlp.utils.MaxDownloadsReached:
                pass

for category, genres_playlists in categories_playlists.items():
    for genre, genre_playlists in genres_playlists.items():
        valid_playlists = [p for p in genre_playlists if p]
        if not valid_playlists:
            continue

        genre_dir = DLS_DIR / genre
        if genre_dir.exists():
            existing_songs = len(list(genre_dir.rglob("*.mp3")))
            if existing_songs >= SONGS_PER_GENRE:
                continue

        playlists_songs_needed = get_playlists_songs_needed(valid_playlists)
        dl_playlists(genre_dir, valid_playlists, playlists_songs_needed)
