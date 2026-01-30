@echo off
python -m s2_preprocess.1_gen_labels
python -m s2_preprocess.2_extract_embs
python -m s2_preprocess.3_scale
del "cache\data_2*"
python -m s2_preprocess.4_balance
del "cache\data_3_train*"
python -m s2_preprocess.5_shuffle
del "cache\data_3_test*"
del "cache\data_4*"
python -m s3_train.1_train
python -m s3_train.2_test
