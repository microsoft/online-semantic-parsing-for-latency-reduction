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


dir=run_prefix


##### preprocess valid data
subset=valid

bss=$(seq 1 $beam_size)

for bs in $bss; do

    validpref=${subset}-prefixallt-beam${beam_size}-$bs

    echo -e "\n[Preprocessing valid dataset: $validpref]\n"

    bash $dir/ac_preprocess_valid.sh $config_data $validpref

done
