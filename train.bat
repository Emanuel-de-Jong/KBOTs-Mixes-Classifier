@echo off
python 1_setup_dataset.py
python 2_gen_labels.py
python 3_extract_embs.py
python 4_preprocess.py
python 5_train.py
python 6_test.py
