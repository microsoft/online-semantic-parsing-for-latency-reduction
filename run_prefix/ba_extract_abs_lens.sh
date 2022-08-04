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

##### create prefix by truncating the source and modifying the target with copy

for subset in "train" "valid"; do
# for subset in "validfirst50"; do

    python -m calflow_prefix.stats_len \
        --src_utterances $ORACLE_FOLDER/${subset}.utters \
        --out_len_stats $PREFIX_ALL_ORACLE_FOLDER/${subset}.lens.json

done
