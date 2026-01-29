# KBOT's Mixes Classifier
Finds the right playlist for a new song.

**Active Development:** 2025-10-13 - 2025-10-23<br>
**Last Change:** 2025-10-23<br>
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
4. Put playlist directories with MP3 files in the `train` directory.
5. The rest can be done in 2 ways:
    - Manual:
        1. `python 1_setup_dataset.py`.
        2. `python 2_gen_labels.py`.
        3. `python 3_extract_embs.py`. This will take a while.
        4. `python 4_scale.py`. This will take a while.
        5. `python 5_balance.py`.
        6. `python 6_shuffle.py`.
        7. `python 7_train.py`.
        8. Optionally run `python 8_test.py` for a more realistic inference simulation test.
    - Automatic:
        1. `train.bat/sh`. This will take a while.

## Usage
For a single MP3 file anywhere:
1. `poetry run python run.py PATH_TO_SONG.mp3`.

For a directory full of MP3 files:
1. Put the MP3 files in the `batch` directory.
2. `poetry run python run_batch.py`.

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
- Remove duplicate songs
- Look for outliers in training data
- 2x weight on non public songs
