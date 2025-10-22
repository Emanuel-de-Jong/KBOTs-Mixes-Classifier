# KBOT's Mixes Classifier
Finds the right playlist for a new song.

**Active Development:** 2025-10-13 - 2025-10-21<br>
**Last Change:** 2025-10-21<br>

| | |
| :---: | :---: |
| ![](/Screenshots/1-Stats-Matrix.png) | ![](/Screenshots/2-Stats-Graph.png) |

## Requirements
- python (tested on 3.11)
- ffmpeg

## Setup
1. `pip install poetry`.
2. `poetry install`.
3. `poetry env activate` and run the script displayed.
4. Put playlist directories with MP3 files in the `music` directory.
5. The rest can be done in 2 ways:
    - Manual:
        1. `python 1_setup_dataset.py`.
        2. `python 2_gen_labels.py`.
        3. `python 3_extract_embs.py`. This will take a while.
        4. `python 4_preprocess.py`.
        5. `python 5_train.py`.
        6. Optionally run `python 6_test.py` for a more realistic inference simulation test.
    - Automatic:
        1. `train.bat`. This will take a while.

## Usage
For a single MP3 file anywhere:
1. `poetry run python run.py PATH_TO_SONG.mp3`.

For a directory full of MP3 files:
1. Put the MP3 files in the `batch` directory.
2. `poetry run python run_batch.py`.

If you want to get the results of multiple models like the `global`, `general_pop`, `rock` and `edm` in `run_batch.py`, you'll have to run the full pipeline multiple times. Rename the `cache/model_global.joblib` file before running again!

## Roadmap
- Reevaluate removing/merging playlists
- Remove unneeded packages

- Balanced batching
- Duplicates to balance songs?
    - Up to max songs per label
    - Make sure no songs removed during downsample
    - Remove dupes before others?
- Don't use unique songs in validate if not enough data?
