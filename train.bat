@echo off
python 1_setup_dataset.py
python 2_gen_labels.py
python 3_extract_embs.py
python 4_scale.py
python 5_balance.py
python 6_train.py
python 7_test.py
