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

python -m fairseq_gap.preprocess \
        --user-dir src/fairseq_gap \
        --task $TASK \
        --source-lang utters \
        --target-lang actions \
        --trainpref $PREFIX_ALL_ORACLE_FOLDER/train \
        --destdir $PREFIX_ALL_BIN_FOLDER \
        --embdir $PREFIX_ALL_EMB_FOLDER \
        --workers 1 \
        \
        --dict-only \
        \
        $PREPROCESS_ARGS \
        \
        --pretrained-embed $PRETRAINED_EMBED \
        --bert-layers $BERT_LAYERS \
        --remove-be $REMOVE_BE \
        --avg-word $AVG_WORD
        # --dict-only
        # --joined-dictionary \