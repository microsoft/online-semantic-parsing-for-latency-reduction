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

[[ -z ${2+x} ]] && echo "Please provide the validation subset prefix to proceed" && exit 1
validpref=$2    # e.g. valid-prefix0p, valid-prefix5t

##### preprocess the prefix data for validation

# validation data

# remove the .done flag so that the new validation data can be put under the same folder
if [[ ! -s $PREFIX_ALL_BIN_FOLDER/$validpref.utters-actions.actions.pos.bin ]]; then
    if [[ -f $PREFIX_ALL_BIN_FOLDER/.done ]]; then
        rm $PREFIX_ALL_BIN_FOLDER/.done
    fi
fi

# remove the .done flag so that the new validation data can be put under the same folder
if [[ ! -s $PREFIX_ALL_EMB_FOLDER/$validpref.utters-actions.utters.wordpieces.bin ]]; then
    if [[ -f $PREFIX_ALL_EMB_FOLDER/.done ]]; then
        rm $PREFIX_ALL_EMB_FOLDER/.done
    fi
fi


# preprocess the valid data: given a vocabulary, build action states and pretrained embedding features
python -m fairseq_gap.preprocess \
        --user-dir src/fairseq_gap \
        --task $TASK \
        --source-lang utters \
        --target-lang actions \
        --srcdict $PREFIX_ALL_BIN_FOLDER/dict.utters.txt \
        --tgtdict $PREFIX_ALL_BIN_FOLDER/dict.actions_nopos.txt \
        --validpref $PREFIX_ALL_ORACLE_FOLDER/$validpref \
        --destdir $PREFIX_ALL_BIN_FOLDER \
        --embdir $PREFIX_ALL_EMB_FOLDER \
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
