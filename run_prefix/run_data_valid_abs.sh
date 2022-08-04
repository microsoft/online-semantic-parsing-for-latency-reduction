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
subset=valid
max_len_valid=60

max_len_valid_treedst=43
if [[ $config_data ==  *"treedst"* ]]; then
    max_len_valid=$max_len_valid_treedst
fi

len_abss=$(seq 0 $max_len_valid)

# subset=validfirst50
# max_len_validfirst50=17
# len_abss=$(seq 0 $max_len_validfirst50)


# len_abss="0 10"    # debug

for len_abs in $len_abss; do

    validpref=${subset}-prefix${len_abs}t

    echo -e "\n[Preprocessing valid dataset: $validpref]\n"

    bash $dir/ac_preprocess_valid.sh $config_data $validpref

done
