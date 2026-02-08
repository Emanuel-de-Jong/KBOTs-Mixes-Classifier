# KBOT's Mixes Classifier
Finds the right playlist for a new song.

**Active Development:** 2025-10-13 - 2026-02-01<br>
**Last Change:** 2026-02-01<br>
**Highlights:** Machine Learning<br>

| | |
| :---: | :---: |
| ![](/Screenshots/1-Stats-Matrix.png) | ![](/Screenshots/2-Stats-Graph.png) |

## Requirements
- python (tested on `3.11`)
- ffmpeg (tested on `2025-08-04-git-9a32b86307-full_build-www.gyan.dev`)

## Setup
1. `pip install poetry` (tested on `2.2.1`).
2. `poetry install`.
3. `poetry env activate` and run the script displayed.
4. Put playlist directories with MP3 files in `data_sets/train/playlists`.
5. The rest can be done in 2 ways:
    - Manual:
        - Go through `s1_prep`, `s2_preprocess` and `s3_train`. In each, running the scripts in numerical order.
        - Some scripts in `s1_prep` just generate a log file and require manual action afterwards.
        - Scripts can be run with `python -m STEP.SCRIPT_NAME`. So for example: `python -m s1_prep.1_dl`.
    - Automatic:
        1. Follow `Manual` as described above, **except you only need to go through `s1_prep`**!
        1. Run `train.bat` for Windows or `train.sh` for Linux or MAC. This will take a while...

## Usage
For a single MP3 file anywhere:
1. `poetry run python -m s4_infer.run PATH_TO_SONG.mp3`.

For a directory full of MP3 files:
1. Put the MP3 files in the `s4_infer/batch` directory.
2. `poetry run python -m s4_infer.run_batch`.
3. Check out `s4_infer/batch_results.yaml`.

If you want to get the results of multiple models like the `global`, `general_pop`, `rock`, `edm_hard` and `edm_easy` in `run_batch.py`, you'll have to run the full pipeline multiple times. After each run, delete the `test` directory, replace the songs in the `train` directory and change the NAME variable in `global_params.py`.

Don't forget to add the source playlists for merging, even if they fall outside the model's scope. Example: `Electro Swing` for `Swing` in the `general_pop` model.

## Public playlists
Lets you download playlists from YouTube for more training data.

# Setup
You'll have to install deno and let yt-dlp know about it:
1. Install deno with `curl -fsSL https://deno.land/install.sh | sh`.
2. Find the path to deno with `which deno`.
3. Put the path in the `js_runtimes` part of the yt-dlp config in `dl.py`.

## Roadmap
- Normalize volume before MERT step
- Remove outliers after MERT step
- 2x weight on non public songs
