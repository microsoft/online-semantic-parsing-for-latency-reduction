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

##### create prefix by truncating the source (with context) and use the full utterance as target

max_len_train=79
max_len_valid=60
max_len_validfirst50=17

max_len_train_treedst=49
max_len_valid_treedst=43
if [[ $config_data ==  *"treedst"* ]]; then
    max_len_train=$max_len_train_treedst
    max_len_valid=$max_len_valid_treedst
fi

# to avoid rerunning
if [ -f $PREFIX_ALL_ORACLE_FOLDER/.done ]; then

    echo -e "\n[$PREFIX_ALL_ORACLE_FOLDER] src prefix and tgt sequences already exist --- do nothing\n"

else

    subset=train
    for len_abs in $(seq 0 $max_len_train); do

        echo -e "\nSubset [$subset]: prefix utterance length: $len_abs"

        python -m calflow_prefix.abs_truncate_src_utter_dlm \
            --src_utterances $ORACLE_FOLDER/${subset}.utters \
            --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
            --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
            --out_tgt_utterance $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.fullutters \
            --len_abs $len_abs

    done

    subset=valid
    for len_abs in $(seq 0 $max_len_valid); do

        echo -e "\nSubset [$subset]: prefix utterance length: $len_abs"

        python -m calflow_prefix.abs_truncate_src_utter_dlm \
            --src_utterances $ORACLE_FOLDER/${subset}.utters \
            --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
            --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
            --out_tgt_utterance $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.fullutters \
            --len_abs $len_abs

    done

    # ===== concatenate all the training data into a single file
    echo "concatenating all the training prefix data into a single file $PREFIX_ALL_ORACLE_FOLDER/train.utters and $PREFIX_ALL_ORACLE_FOLDER/train.fullutters ---"

    if [[ -s $PREFIX_ALL_ORACLE_FOLDER/train.utters ]] || [[ -s $PREFIX_ALL_ORACLE_FOLDER/train.fullutters ]]; then
        rm $PREFIX_ALL_ORACLE_FOLDER/train.utters
        rm $PREFIX_ALL_ORACLE_FOLDER/train.fullutters
    fi

    touch $PREFIX_ALL_ORACLE_FOLDER/train.utters
    touch $PREFIX_ALL_ORACLE_FOLDER/train.fullutters

    for len_abs in $(seq 0 $max_len_train); do
    # for len_abs in $(seq 0 3); do    # for debugging
        echo -e "\nconcatenate train-prefix${len_abs}t.utters/fullutters"

        cat $PREFIX_ALL_ORACLE_FOLDER/train.utters $PREFIX_ALL_ORACLE_FOLDER/train-prefix${len_abs}t.utters \
           > $PREFIX_ALL_ORACLE_FOLDER/train.utters.tmp

        cat $PREFIX_ALL_ORACLE_FOLDER/train.fullutters $PREFIX_ALL_ORACLE_FOLDER/train-prefix${len_abs}t.fullutters \
           > $PREFIX_ALL_ORACLE_FOLDER/train.fullutters.tmp

        mv $PREFIX_ALL_ORACLE_FOLDER/train.utters.tmp $PREFIX_ALL_ORACLE_FOLDER/train.utters

        mv $PREFIX_ALL_ORACLE_FOLDER/train.fullutters.tmp $PREFIX_ALL_ORACLE_FOLDER/train.fullutters

    done

    echo "number of lines in src train.utters: $(wc -l $PREFIX_ALL_ORACLE_FOLDER/train.utters)"
    echo "number of lines in tgt train.fullutters: $(wc -l $PREFIX_ALL_ORACLE_FOLDER/train.fullutters)"


    # ===== concatenate all the validation data into a single file
    echo "concatenating all the validation prefix data into a single file $PREFIX_ALL_ORACLE_FOLDER/valid.utters and $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters ---"

    if [[ -s $PREFIX_ALL_ORACLE_FOLDER/valid.utters ]] || [[ -s $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters ]]; then
        rm $PREFIX_ALL_ORACLE_FOLDER/valid.utters
        rm $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters
    fi

    touch $PREFIX_ALL_ORACLE_FOLDER/valid.utters
    touch $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters

    for len_abs in $(seq 0 $max_len_valid); do
    # for len_abs in $(seq 0 3); do    # for debugging
        echo -e "\nconcatenate valid-prefix${len_abs}t.utters/fullutters"

        cat $PREFIX_ALL_ORACLE_FOLDER/valid.utters $PREFIX_ALL_ORACLE_FOLDER/valid-prefix${len_abs}t.utters \
           > $PREFIX_ALL_ORACLE_FOLDER/valid.utters.tmp

        cat $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters $PREFIX_ALL_ORACLE_FOLDER/valid-prefix${len_abs}t.fullutters \
           > $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters.tmp

        mv $PREFIX_ALL_ORACLE_FOLDER/valid.utters.tmp $PREFIX_ALL_ORACLE_FOLDER/valid.utters

        mv $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters.tmp $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters

    done

    echo "number of lines in src valid.utters: $(wc -l $PREFIX_ALL_ORACLE_FOLDER/valid.utters)"
    echo "number of lines in tgt valid.fullutters: $(wc -l $PREFIX_ALL_ORACLE_FOLDER/valid.fullutters)"


    # subset=validfirst50
    # for len_abs in $(seq 0 $max_len_validfirst50); do

    #     echo -e "\nSubset [$subset]: prefix utterance length: $len_abs"

    #     python -m calflow_prefix.abs_truncate_src_utter_dlm \
    #         --src_utterances $ORACLE_FOLDER/${subset}.utters \
    #         --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
    #         --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
    #         --out_tgt_utterance $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.fullutters \
    #         --len_abs $len_abs

    # done

    # create .done flag
    touch $PREFIX_ALL_ORACLE_FOLDER/.done
    echo -e "\nPrefix data created and saved at ${PREFIX_ALL_ORACLE_FOLDER}\n"

fi
