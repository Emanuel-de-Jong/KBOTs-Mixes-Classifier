# KBOT's Mixes Classifier
Finds the right playlist for a new song.

## Requirements
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
1. `poetry run python run.py PATH_TO_SONG.mp3`.
