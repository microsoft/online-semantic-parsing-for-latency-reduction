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

dir=$(dirname $0)

##############################################################

#### data

echo "[Training Data directories:]"
echo $PREFIX_DATA_FOLDER
echo $PREFIX_ORACLE_FOLDER
echo $PREFIX_BIN_FOLDER
echo $PREFIX_EMB_FOLDER
echo

echo "[Validation Data directories:]"
echo $PREFIX_ALL_DATA_FOLDER
echo $PREFIX_ALL_ORACLE_FOLDER
echo $PREFIX_ALL_BIN_FOLDER
echo $PREFIX_ALL_EMB_FOLDER
echo

##### preprocess data (will do nothing if data already processed and exists)
echo "[Create all prefix data:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. $dir/aa_create_prefix.sh ""

echo "[Build dictionary:]"
if [[ ! -s $PREFIX_ALL_BIN_FOLDER/dict.actions_nopos.txt ]]; then
    . $dir/ab_preprocess_dict.sh ""
fi

cp $config_data $DATA_FOLDER/config_data.sh

##### preprocess the validation data
echo "[Preprocess validation data:]"
bash $dir/run_data_valid.sh $config_data

##### preprocess the training data
echo "[Preprocess training data:]"
bash $dir/run_data_train.sh $config_data



###############################################################

##### train model (will do nothing if desired last checkpoint in $MODEL_FOLDER exists)

echo "[Training:]"
echo

mkdir -p $MODEL_FOLDER

cp $config_data $SAVEDIR/$expdir/ || true
# to skip cp error (e.g. when $config_model already exists and cp the same file)
cp $config_model $MODEL_FOLDER/ || true

# change the seed name in the particular model configuration copied
sed -i "s/seed:-42/seed:-${seed}/g" $MODEL_FOLDER/$(basename $config_model)

cp $0 $MODEL_FOLDER/
cp $dir/af_train.sh $MODEL_FOLDER/train.sh

. $dir/af_train.sh

# exit 0
###############################################################

echo
echo "[Decoding and computing evaluation scores:]"
echo

##### decoding configuration
model_epoch=_last
beam_size=1
batch_size=256

# for beam_size in 1 5 10
len_percs=(100 90 80 70 60 50 40 30 20 10 0)
# len_percs=(100 50 0)     # for debugging

for len_perc in "${len_percs[@]}"
do
    validset=valid-prefix${len_perc}p
    echo "beam size: $beam_size"
    echo "valid set: $validset"
    echo
    . $dir/ag_test.sh "" $validset
    echo
done

cp $dir/ag_test.sh $MODEL_FOLDER/test.sh
