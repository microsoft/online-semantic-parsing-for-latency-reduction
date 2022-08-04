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

        python -m calflow_prefix.abs_truncate_src_modify_tgt_copy \
            --src_utterances $ORACLE_FOLDER/${subset}.utters \
            --tgt_actions $ORACLE_FOLDER/${subset}.actions.src_copy \
            --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
            --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
            --out_actions $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.actions.src_copy \
            --len_abs $len_abs

    done

    subset=valid
    for len_abs in $(seq 0 $max_len_valid); do

        echo -e "\nSubset [$subset]: prefix utterance length: $len_abs"

        python -m calflow_prefix.abs_truncate_src_modify_tgt_copy \
            --src_utterances $ORACLE_FOLDER/${subset}.utters \
            --tgt_actions $ORACLE_FOLDER/${subset}.actions.src_copy \
            --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
            --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
            --out_actions $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.actions.src_copy \
            --len_abs $len_abs

    done

    # subset=validfirst50
    # for len_abs in $(seq 0 $max_len_validfirst50); do

    #     echo -e "\nSubset [$subset]: prefix utterance length: $len_abs"

    #     python -m calflow_prefix.abs_truncate_src_modify_tgt_copy \
    #         --src_utterances $ORACLE_FOLDER/${subset}.utters \
    #         --tgt_actions $ORACLE_FOLDER/${subset}.actions.src_copy \
    #         --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.utters \
    #         --out_indices $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.idxs \
    #         --out_actions $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_abs}t.actions.src_copy \
    #         --len_abs $len_abs

    # done

    # create .done flag
    touch $PREFIX_ALL_ORACLE_FOLDER/.done
    echo -e "\nPrefix data created and saved at ${PREFIX_ALL_ORACLE_FOLDER}\n"

fi
