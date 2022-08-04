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


##### preprocess valid data

len_percs=(0 10 20 30 40 50 60 70 80 90 100)
# len_percs=(0 10)

for len_perc in "${len_percs[@]}"; do

    validpref=valid-prefix${len_perc}p

    echo -e "\n[Preprocessing valid dataset: $validpref]\n"

    bash $dir/ac_preprocess_valid.sh $config_data $validpref

done
