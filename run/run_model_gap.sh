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

echo "[Data directories:]"
echo $DATA_FOLDER
echo $ORACLE_FOLDER
echo $BIN_FOLDER
echo $EMB_FOLDER
echo

##### preprocess data (will do nothing if data already processed and exists)
echo "[Building graph oracle actions:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. $dir/aa_calflow_actions.sh ""

echo "[Preprocessing data:]"
. $dir/ab_preprocess.sh ""

cp $config_data $DATA_FOLDER/config_data.sh

##### optional: test the oracle smatch for dev and test set
echo "[Testing oracle Exact Match:]"
echo "valid data:"
. $dir/calflow_oracle_em.sh "" valid
echo


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
cp $dir/ac_train.sh $MODEL_FOLDER/train.sh

. $dir/ac_train.sh

# exit 0
###############################################################

echo
echo "[Decoding and computing evaluation scores:]"
echo

##### decoding configuration
model_epoch=_last
# beam_size=1
batch_size=256

# for beam_size in 1 5 10
for beam_size in 1 5
# for beam_size in 1
do
    echo "checkpoint: checkpoint${model_epoch}.pt"
    echo "beam size: $beam_size:"
    . $dir/ad_test.sh "" valid
    echo
done

##### decoding configuration
model_epoch=_best
# beam_size=1
batch_size=256

for beam_size in 1
do
    echo "checkpoint: checkpoint${model_epoch}.pt"
    echo "beam size $beam_size:"
    . $dir/ad_test.sh "" valid
    echo
done

cp $dir/ad_test.sh $MODEL_FOLDER/test.sh
