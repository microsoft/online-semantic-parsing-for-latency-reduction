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

##### build dictionary for a specific training data (to be used for different combinations of the prefix)

# use the dictionary built from the percentage prefix data (with the addition of CPANY)
# NOTE we could also copy the future all to <mask>, where the dictionary can be exactly the same as the full parser

mkdir -p $PREFIX_ALL_BIN_FOLDER
cp ${DATA_FOLDER}_prefix-all/fairseq_bin/dict.utters.txt $PREFIX_ALL_BIN_FOLDER/
cp ${DATA_FOLDER}_prefix-all/fairseq_bin/dict.actions_nopos.txt $PREFIX_ALL_BIN_FOLDER/
