#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config=$1
    . $config    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"

dir=$(dirname $0)

##### script specific config
if [ -z "$2" ]; then
    data_split="valid"
else
    data_split=$2
fi


model_epoch=${model_epoch:-_last}    # 15
beam_size=${beam_size:-1}
batch_size=${batch_size:-256}

RESULTS_FOLDER=$MODEL_FOLDER/beam${beam_size}
results_prefix=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}
model=$MODEL_FOLDER/checkpoint${model_epoch}.pt

TASK=${TASK:-translation}


##### DECODING
# rm -Rf $RESULTS_FOLDER
mkdir -p $RESULTS_FOLDER

if [[ ! -s $results_prefix.hypo ]]; then

# --nbest 3 \
# --quiet
python $dir/generate.py \
    $PREFIX_ALL_BIN_FOLDER  \
    --task $TASK \
    --gen-subset $data_split \
    --bpe gpt2 \
    --beam $beam_size \
    --nbest $beam_size \
    --batch-size $batch_size \
    --remove-bpe \
    --path $model \
    --results-path $results_prefix \

fi

# exit 0
