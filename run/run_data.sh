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


##### print out configurations
echo "[Data configuration file:]"
echo $config_data
echo

echo "[Data directories:]"
echo $DATA_FOLDER
echo $ORACLE_FOLDER
echo $BIN_FOLDER
echo $EMB_FOLDER
echo

dir=$(dirname $0)

##### preprocess data (will do nothing if data already processed and exists)
echo "[Building graph oracle actions:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. $dir/aa_calflow_actions.sh ""

echo "[Preprocessing data:]"
. $dir/ab_preprocess.sh ""

cp $config_data $DATA_FOLDER/config_data.sh

# exit 0

##### optional: test the oracle smatch for dev and test set
echo "[Testing oracle Exact Match:]"
echo "valid data:"
. $dir/calflow_oracle_em.sh "" valid

# echo "train data:"
# . $dir/calflow_oracle_em.sh "" train
echo
