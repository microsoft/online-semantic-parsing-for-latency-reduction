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

dir=$(dirname $0)

##### binary the src and tgt data with GPT-2 bpe vocab and bpe data

# training data
if [[ ! -s $PREFIX_ALL_BIN_FOLDER/train.utters-fullutters.fullutters.bin ]]; then

python $dir/preprocess.py \
  --source-lang "utters" \
  --target-lang "fullutters" \
  --trainpref $PREFIX_ALL_BIN_FOLDER/train.bpe \
  --destdir $PREFIX_ALL_BIN_FOLDER/ \
  --workers 60 \
  --srcdict $PREFIX_ALL_BIN_FOLDER/dict.txt \
  --tgtdict $PREFIX_ALL_BIN_FOLDER/dict.txt

fi


# validation
# split=valid-prefix10t    # debug
split=valid

if [[ ! -s $PREFIX_ALL_BIN_FOLDER/${split}.utters-fullutters.fullutters.bin ]]; then

python $dir/preprocess.py \
  --source-lang "utters" \
  --target-lang "fullutters" \
  --validpref $PREFIX_ALL_BIN_FOLDER/${split}.bpe \
  --destdir $PREFIX_ALL_BIN_FOLDER/ \
  --workers 60 \
  --srcdict $PREFIX_ALL_BIN_FOLDER/dict.txt \
  --tgtdict $PREFIX_ALL_BIN_FOLDER/dict.txt

fi