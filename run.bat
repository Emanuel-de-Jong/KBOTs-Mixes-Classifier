@echo off
setlocal enabledelayedexpansion

set "category=%~1"
set "step=%~2"

shift
shift

if "%category%"=="1" set "module=s1_prep"
if "%category%"=="2" set "module=s2_preprocess"
if "%category%"=="3" set "module=s3_train"
if "%category%"=="4" set "module=s4_infer"

set "stepmod="

if "%category%"=="1" (
    if "%step%"=="1" set "stepmod=1_dl"
    if "%step%"=="2" set "stepmod=2_copy_public"
    if "%step%"=="3" set "stepmod=3_sanitize"
    if "%step%"=="4" set "stepmod=4_dupes"
    if "%step%"=="5" set "stepmod=5_outliers"
    if "%step%"=="6" set "stepmod=6_test_data_set"
)

if "%category%"=="2" (
    if "%step%"=="1" set "stepmod=1_gen_labels"
    if "%step%"=="2" set "stepmod=2_extract_embs"
    if "%step%"=="3" set "stepmod=3_outliers"
    if "%step%"=="4" set "stepmod=4_scale"
    if "%step%"=="5" set "stepmod=5_balance"
    if "%step%"=="6" set "stepmod=6_shuffle"
    if "%step%"=="7" set "stepmod=7_reshape"
)

if "%category%"=="3" (
    if "%step%"=="1" set "stepmod=1_train"
    if "%step%"=="2" set "stepmod=2_test"
)

if "%category%"=="4" (
    if "%step%"=="1" set "stepmod=1_run"
    if "%step%"=="2" set "stepmod=2_run_batch"
)

if not defined module (
    echo Invalid category: %category%
    exit /b 1
)

if not defined stepmod (
    echo Invalid step: %step%
    exit /b 1
)

python -m "%module%.%stepmod%" %*

endlocal
