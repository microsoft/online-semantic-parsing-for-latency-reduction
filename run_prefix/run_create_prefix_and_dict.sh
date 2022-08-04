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

echo "[Prefix basic data directories:]"
echo $PREFIX_ALL_DATA_FOLDER
echo $PREFIX_ALL_ORACLE_FOLDER
echo $PREFIX_ALL_BIN_FOLDER
echo $PREFIX_ALL_EMB_FOLDER
echo


dir=$(dirname $0)


##### create prefix

echo "[Create all prefix data:]"
# use sourcing instead of call bash, otherwise the variables will not be recognized
. $dir/aa_create_prefix.sh ""

echo "[Build dictionary:]"
if [[ ! -s $PREFIX_ALL_BIN_FOLDER/dict.actions_nopos.txt ]]; then
    . $dir/ab_preprocess_dict.sh ""
fi

cp $config_data $PREFIX_ALL_DATA_FOLDER/config_data_prefix.sh
