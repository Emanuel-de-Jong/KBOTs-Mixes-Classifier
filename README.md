# KBOT's Mixes Classifier
Finds the right playlist for a new song.

## Setup
1. `pip install poetry`.
2. `poetry install`.
3. Put playlists in `music` directly. Not with the main categories like `EDM`.
    - Remove:
        - Chill EDM
        - Gamer
        - Groovy EDM
        - Pioneer EDM
        - Nature Vibe
        - Pioneer
        - Romantic
    - Merge:
        - Downbeat Vocal Trance + Upbeat Vocal Trance = Vocal Trance
        - Dark Pop + Psych Pop = Dark and Psych Pop
        - Dark Rock + Psych Rock = Dark and Psych Rock
4. `python 1_setup_dataset.py`.
5. `python 2_extract_embeddings.py`. This will take a while.
6. `python 3_train.py`.

## Usage
1. `python 4_inference.py PATH_TO_SONG.mp3`.
