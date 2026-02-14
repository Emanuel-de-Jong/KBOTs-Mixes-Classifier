#!/usr/bin/env bash
set -e
python -m s2_preprocess.1_gen_labels
python -m s2_preprocess.2_extract_embs
python -m s2_preprocess.3_outliers
# rm cache/data_2*
python -m s2_preprocess.4_scale
rm cache/data_3*
python -m s2_preprocess.5_balance
rm cache/data_4_train*
python -m s2_preprocess.6_shuffle
rm cache/data_4_test*
rm cache/data_5*
python -m s2_preprocess.7_reshape
python -m s3_train.1_train
python -m s3_train.2_test
