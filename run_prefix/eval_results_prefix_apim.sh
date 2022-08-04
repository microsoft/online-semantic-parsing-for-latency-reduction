#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

# usage: bash $0 <config-model> <seed>

##############################################################

# seed=${2:-42}        # NOTE do not specify seed -> this should be from the $config_model

##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config_model=$1
    . $config_model    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts


set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"

# from data config part, now we should have
# environment set
# DATA_ROOT_DIR
#   |- DATA_FOLDER
#        |- ORACLE_FOLDER
#        |- BIN_FOLDER
#        |- EMB_FOLDER
# from model config part, now we should have
# MODEL_FOLDER

echo "[Model configuration file:]"
echo $config_model
echo

dir=$(dirname $0)

###############################################################

echo
echo "[Computing evaluation scores:]"
echo

##### decoding configuration
model_epoch=_last
beam_size=1

# for beam_size in 1 5 10
len_percs=(100 90 80 70 60 50 40 30 20 10 0)
# len_percs=(100 50 0)     # for debugging

for len_perc in "${len_percs[@]}"
do
    validset=valid-prefix${len_perc}p
    echo "beam size: $beam_size"
    echo "valid set: $validset"
    echo

    if [[ -s $MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.actions ]]; then

        if [[ ! -s $MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.apim ]]; then
            . $dir/ag_test_api.sh "" $validset
        else
            echo "$MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.apim already exists --- do not recompute"
            cat "$MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.apim"
            echo
            echo "$MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.mod-graph.apim already exists --- do not recompute"
            cat "$MODEL_FOLDER/beam${beam_size}/${validset}_checkpoint${model_epoch}.mod-graph.apim"
        fi

    else
        echo "Decoding results not existing --- skipped"
    fi

    echo
done
