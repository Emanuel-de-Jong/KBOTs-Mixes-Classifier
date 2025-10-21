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
4. Put playlists in `music` directly. Not with the main categories like `EDM`.
5. `python 1_setup_dataset.py`.
6. `python 2_gen_labels.py`.
7. `python 3_extract_embs.py`. This will take a while.
8. `python 4_preprocess.py`
9. `python 5_train.py`.

## Usage
For a single MP3 file anywhere:
1. `poetry run python run.py PATH_TO_SONG.mp3`.

For a directory full of MP3 files:
1. Put the MP3 files in the `batch` directory.
2. `poetry run python run_batch.py`.

If you want to get the results of multiple models like the `global`, `general_pop`, `rock` and `edm` in `run_batch.py`, you'll have to run the full pipeline multiple times. Rename the `cache/model.joblib` file before running again!

## Roadmap
- Test acc on non scaled data
- Reevaluate removing/merging playlists
- Remove unneeded packages
- Models for playlist subsets

- Balanced batching
- Duplicates to balance songs?
    - up to max songs per label
    - make sure no songs removed during downsample
    - remove dupes before others?
- Don't use unique songs in validate if not enough data?
