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

##### download BART checkpoints and GPT-2 byte-level BPE vocabulary
if [[ ! -s $PREFIX_ALL_BIN_FOLDER/dict.txt ]]; then
# TODO this is a mistake?
# original was
# wget -P $PREFIX_ALL_BIN_FOLDER -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
../DATA/processed/treedst/src-ct1-npwa_utter-dlm_abs-prefix-all/ $PREFIX_ALL_BIN_FOLDER -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -P $PREFIX_ALL_BIN_FOLDER -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -P $PREFIX_ALL_BIN_FOLDER -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
fi


##### bpe preprocessing

dir=$(dirname $0)

# training data
split=train

for lang in utters fullutters; do

    echo "conversion to GPT-2 byte bpe tokens: $PREFIX_ALL_ORACLE_FOLDER/${split}.$lang -> $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang"

    if [[ -s $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang ]]; then
        echo "done already"
    else
        python $dir/multiprocessing_bpe_encoder.py \
            --encoder-json $PREFIX_ALL_BIN_FOLDER/encoder.json \
            --vocab-bpe $PREFIX_ALL_BIN_FOLDER/vocab.bpe \
            --inputs $PREFIX_ALL_ORACLE_FOLDER/${split}.$lang \
            --outputs $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang \
            --workers 60 \
            --keep-empty
    fi

done


# validation data

# split=valid-prefix10t    # debug
split=valid

for lang in utters fullutters; do

    echo "conversion to GPT-2 byte bpe tokens: $PREFIX_ALL_ORACLE_FOLDER/${split}.$lang -> $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang"

    if [[ -s $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang ]]; then
        echo "done already"
    else
        python $dir/multiprocessing_bpe_encoder.py \
            --encoder-json $PREFIX_ALL_BIN_FOLDER/encoder.json \
            --vocab-bpe $PREFIX_ALL_BIN_FOLDER/vocab.bpe \
            --inputs $PREFIX_ALL_ORACLE_FOLDER/${split}.$lang \
            --outputs $PREFIX_ALL_BIN_FOLDER/${split}.bpe.$lang \
            --workers 60 \
            --keep-empty
    fi

done
