#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


config_model=$1
# seed=${2:-42}
seed=$2
gpu=$3

. $config_model    # $config should include its path; to get $MODEL_FOLDER

dir=$(dirname $0)

# formal run: send to background
echo -e "\n[Run decoding on every absolute length validation prefix data ---]"
echo $MODEL_FOLDER

# maintain the previous evaluation log files in case of rerunning due to some issues
log_suffix=decode_abs
if [[ -f $MODEL_FOLDER/log.$log_suffix ]]; then
    n=0
    while [[ -f $MODEL_FOLDER/log.${log_suffix}$n ]]; do
        n=$(( $n + 1 ))
    done
    mv $MODEL_FOLDER/log.$log_suffix $MODEL_FOLDER/log.${log_suffix}$n
fi

# save the job submission logs
if [[ ! -d .job_logs ]]; then
    mkdir .job_logs
fi

# this is necessary for output redirected to file
# mkdir -p $MODEL_FOLDER    # NOTE here at decoding time, the $MODEL_FOLDER should already exist
CUDA_VISIBLE_DEVICES=$gpu /bin/bash $dir/run_decode_multi_abs.sh $config_model $seed &> $MODEL_FOLDER/log.$log_suffix &

now=$(date +"[%T - %D]")
echo "$now decode abs - PID - $! - GPU$gpu: $MODEL_FOLDER" >> .job_logs/pid_model-folder.history

# Get pid by "$!"

echo -e "\n[Log for decoding absolute length validation prefix data written at:]"
echo "$MODEL_FOLDER/log.${log_suffix}"
echo
