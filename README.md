# KBOT's Mixes Classifier
Pipeline to train a model that can find the right playlist/genre for a new song.

**Active Development:** 2025-10-13 - 2026-02-19<br>
**Last Change:** 2026-02-19<br>
**Highlights:** Machine Learning<br>

| | |
| :---: | :---: |
| ![](/Screenshots/1-Batch-Inference-Site.png) | ![](/Screenshots/2-Stats-Matrix.png) |
| ![](/Screenshots/3-Stats-Graph.png) | ![](/Screenshots/.png) |

## Requirements
- conda (tested on `Miniconda 25.11.1`)
- ffmpeg (tested on `2025-08-04-git-9a32b86307-full_build-www.gyan.dev`)

## Setup
Sorry in advance for making you install both PyTorch and TensorFlow with each their own CUDA version. The different models used force me to.
1. `conda create -n kbotsmixesclassifier python=3.11`
2. `conda activate kbotsmixesclassifier`
3. `conda install -c conda-forge -y cudatoolkit=11.2 cudnn=8.1 poetry=2.3.1`
4. `poetry config virtualenvs.create false`
5. `poetry install`.
6. Download the Essentia Discogs model from [their site](https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.pb). Or from this repos [release](https://github.com/Emanuel-de-Jong/KBOTs-Mixes-Classifier/releases/download/essentia-discogs-519/discogs-maest-30s-pw-519l-2.pb) if Essentia (re)moves it for some reason.
7. Place the `.pb` file in the `models` directory.
8. Put playlist directories with MP3 files in `data_sets/train/playlists`.
9. The rest can be done in 2 ways:
    - Manual:
        - Go through `s1_prep`, `s2_preprocess` and `s3_train`. In each, running the scripts in numerical order.
        - Some scripts in `s1_prep` just generate a log file and require manual action afterwards.
        - Scripts can be run with `python -m STEP.SCRIPT_NAME`. So for example: `python -m s1_prep.1_dl`.
    - Automatic:
        1. Follow `Manual` as described above, **except you only need to go through `s1_prep`**!
        1. Run `train.bat` for Windows or `train.sh` for Linux or MAC. This will take a while...

### Optional
#### Public playlists
Lets you download playlists from YouTube for more training data.

If you're going to use `s1_prep/1_dl.py` to download public playlists, you'll also want deno:
1. Install deno with `curl -fsSL https://deno.land/install.sh | sh`.
2. Find the path to deno with `which deno`.
3. Put the path in the `js_runtimes` part of the yt-dlp config in `dl.py`.

## Usage
For a single MP3 file anywhere:
1. `python -m s4_infer.run PATH_TO_SONG.mp3`.

For a directory full of MP3 files:
1. Put the MP3 files in the `s4_infer/batch` directory.
2. `python -m s4_infer.run_batch`.
3. Check out `s4_infer/batch_results.yaml`.

If you want to get the results of multiple models like the `global`, `general_pop`, `rock`, `edm_hard` and `edm_easy` in `run_batch.py`, you'll have to run the full pipeline multiple times. After each run, delete the `test` directory, replace the songs in the `train` directory and change the NAME variable in `global_params.py`.

Don't forget to add the source playlists for merging, even if they fall outside the model's scope. Example: `Electro Swing` for `Swing` in the `general_pop` model.

## Roadmap
- Finish comments.js
- Release all experiment models
- Generalize codebase (template files)
- Update README

### Far future
- Rock and pop model with just a few renamed EDM genres
- Copy Mert script into models or use model yaml file for values to batch inference models with different settings?
- Include Essentia Discogs genre predictions in training model input
- Experiments:
    - Input shape order
    - Model window asymmetrical shape
    - Model window bigger size
	- ResNet
    - Batch size
	- Different scaling algorithm
	- 1.5x weight on non public songs
	- Mert chunk or window overlap
- Inference NN decision tree:
    - Is edm?
        - Yes: edm model (50 songs)
        - No:
            - Is non_general? (9 songs)
                - Yes: non_general model
                - No: general model (5 songs)
- Public playlists for non EDM
