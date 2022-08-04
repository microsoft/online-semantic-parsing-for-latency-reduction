#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

# usage: bash $0 <config-model> <seed>

##############################################################

seed=${2:-42}        # NOTE this must be before ". $config_model" to feed in seed value

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

dir=run_prefix

##############################################################


echo
echo "[Decoding and computing evaluation scores:]"
echo

##### record some setups in the upper level config
beam_size_dlm=$beam_size

##### decoding configuration

model_epoch=_last
beam_size=1
batch_size=256


# subset=valid
# max_len_valid=60
# len_abss=$(seq 0 $max_len_valid)

subset=valid
bss=$(seq 1 $beam_size_dlm)


# for beam_size in 1 5 10; do
# for beam_size in 1 5; do
# for beam_size in 1; do

# for len_abs in $len_abss
# do
#     validset=${subset}-prefix${len_abs}t
#     echo "beam size: $beam_size"
#     echo "valid set: $validset"
#     echo
#     . $dir/bg_test.sh "" $validset
#     echo
# done

# done

# cp $dir/bg_test.sh $MODEL_FOLDER/test_multi_abs.sh

for beam_size in 1; do

for bs in $bss
do
    validset=${subset}-prefixallt-beam${beam_size_dlm}-$bs
    echo "beam size: $beam_size"
    echo "valid set: $validset"
    echo
    . $dir/bg_test.sh "" $validset
    echo
done

done

# cp $dir/bg_test.sh $MODEL_FOLDER/test_multi_abs.sh    # do not copy which might cover the previous copied one
