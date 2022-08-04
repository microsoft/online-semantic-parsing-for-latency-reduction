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

len_percs=(0 10 20 30 40 50 60 70 80 90 100)

# to avoid rerunning
if [ -f $PREFIX_ALL_ORACLE_FOLDER/.done ]; then

    echo -e "\n[$PREFIX_ALL_ORACLE_FOLDER] src prefix and tgt sequences already exist --- do nothing\n"

else

    for len_perc in "${len_percs[@]}"; do

        echo "prefix of lengh ${len_perc}%"

        for subset in "train" "valid"; do
            python -m calflow_prefix.truncate_src_modify_tgt_copy \
                --src_utterances $ORACLE_FOLDER/${subset}.utters \
                --tgt_actions $ORACLE_FOLDER/${subset}.actions.src_copy \
                --out_src_prefix $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_perc}p.utters \
                --out_actions $PREFIX_ALL_ORACLE_FOLDER/${subset}-prefix${len_perc}p.actions.src_copy \
                --len_percentage ${len_perc}
        done

    done

    # # mix the training prefix with 0%, 50%, 100% length, for the purpose of constructing dictionary
    # # 3 percentages: 0 50 100
    # # NOTE this is needed when we have '_CPANY_' action for the prefix target
    # echo -e "\nConcatenating 3 prefix files (0%, 50%, 100%) for the training data for the purpose of building vocabulary"

    # cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.utters \
    #     $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
    #     $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
    # > $PREFIX_ALL_ORACLE_FOLDER/train.utters

    # cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.actions.src_copy \
    #     $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
    #     $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
    # > $PREFIX_ALL_ORACLE_FOLDER/train.actions.src_copy

    # for the purpose of constructing dictionary
    # 1 percentage: 100
    # NOTE this is when we have '<mask>' for prefix future, where we can copy from -> so dict does not change
    echo -e "\nConcatenating 1 prefix files (100%) for the training data for the purpose of building vocabulary"

    cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
    > $PREFIX_ALL_ORACLE_FOLDER/train.utters

    cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
    > $PREFIX_ALL_ORACLE_FOLDER/train.actions.src_copy

    echo -e "\nResulted training files:"
    echo "$PREFIX_ALL_ORACLE_FOLDER/train.utters"
    echo "$PREFIX_ALL_ORACLE_FOLDER/train.actions.src_copy"
    echo

    # create .done flag
    touch $PREFIX_ALL_ORACLE_FOLDER/.done
    echo -e "\nPrefix data created and saved at ${PREFIX_ALL_ORACLE_FOLDER}\n"

fi
