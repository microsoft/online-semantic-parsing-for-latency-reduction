#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config_data=$1
    . $config_data    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"

echo "[Prefix configuration file:]"
echo "$config_data"
echo

echo "[Prefix Training Data directories:]"
echo $PREFIX_DATA_FOLDER
echo $PREFIX_ORACLE_FOLDER
echo $PREFIX_BIN_FOLDER
echo $PREFIX_EMB_FOLDER
echo

dir=$(dirname $0)


##### preprocess the training data

# mix the prefix
echo "[Mixing prefix data for training:]"
bash $dir/ad_mix_prefix_train.sh $config_data

# process the training data
echo "[Preprocess the mixed prefix training data:]"
bash $dir/ae_preprocess_train.sh $config_data

cp $config_data $PREFIX_DATA_FOLDER/config_data_prefix.sh
