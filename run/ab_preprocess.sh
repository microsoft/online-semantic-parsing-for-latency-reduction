#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config=$1
    . $config    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"


##### preprocess to binarize data into fairseq format
PREPROCESS_ARGS=${PREPROCESS_ARGS:-""}

if [[ (-f $BIN_FOLDER/.done) && (-f $EMB_FOLDER/.done) ]]; then

    echo "[$BIN_FOLDER] fairseq preprocessed data already exist --- do nothing"
    echo "[$EMB_FOLDER] extracted and binarized pretrained features data already exist --- do nothing"
    echo

else

    python -m fairseq_gap.preprocess \
        --user-dir src/fairseq_gap \
        --task $TASK \
        --source-lang utters \
        --target-lang actions \
        --trainpref $ORACLE_FOLDER/train \
        --validpref $ORACLE_FOLDER/valid \
        --destdir $BIN_FOLDER \
        --embdir $EMB_FOLDER \
        --workers 1 \
        \
        $PREPROCESS_ARGS \
        \
        --pretrained-embed $PRETRAINED_EMBED \
        --bert-layers $BERT_LAYERS \
        --remove-be $REMOVE_BE \
        --avg-word $AVG_WORD
        # --dict-only
        # --joined-dictionary \

    touch $BIN_FOLDER/.done
    touch $EMB_FOLDER/.done
    echo -e "\nData preprocessed and saved at $BIN_FOLDER\n"

fi