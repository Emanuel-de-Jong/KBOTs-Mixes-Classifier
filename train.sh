#!/usr/bin/env bash
set -e
python -m s2_preprocess.1_gen_labels
python -m s2_preprocess.2_extract_embs
python -m s2_preprocess.3_scale
rm cache/data_2*
python -m s2_preprocess.4_balance
rm cache/data_3_train*
python -m s2_preprocess.5_shuffle
rm cache/data_3_test*
rm cache/data_4*
python -m s3_train.1_train
python -m s3_train.2_test
