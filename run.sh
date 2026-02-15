#!/usr/bin/env bash
set -e

category="$1"
step="$2"

shift 2

declare -A CATEGORIES=(
  [1]="s1_prep"
  [2]="s2_preprocess"
  [3]="s3_train"
  [4]="s4_infer"
)

declare -A STEPS_1=(
  [1]="1_dl"
  [2]="2_copy_public"
  [3]="3_sanitize"
  [4]="4_dupes"
  [5]="5_outliers"
  [6]="6_test_data_set"
)

declare -A STEPS_2=(
  [1]="1_gen_labels"
  [2]="2_extract_embs"
  [3]="3_outliers"
  [4]="4_scale"
  [5]="5_balance"
  [6]="6_shuffle"
  [7]="7_reshape"
)

declare -A STEPS_3=(
  [1]="1_train"
  [2]="2_test"
)

declare -A STEPS_4=(
  [1]="1_run"
  [2]="2_run_batch"
)

steps_var="STEPS_${category}"
module="${CATEGORIES[$category]}"
stepmod="$(eval "echo \${${steps_var}[$step]}")"

python -m "${module}.${stepmod}" "$@"
