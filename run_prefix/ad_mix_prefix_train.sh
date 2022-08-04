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

##### mix the training prefix data

mkdir -p $PREFIX_ORACLE_FOLDER

# to avoid rerunning
if [ -f $PREFIX_ORACLE_FOLDER/.done ]; then

    echo -e "\n[$PREFIX_ORACLE_FOLDER] mixed src prefix and tgt sequences already exist --- do nothing\n"

else

    if [[ $MIX_TRAIN == "5ps" ]]; then
        # 5 percentages: 20 40 60 80 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters. \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "11ps" ]]; then
        # 11 percentages: 0 10 20 30 40 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "2ps" ]]; then
        # 2 percentages: 50 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "3ps" ]]; then
        # 3 percentages: 0 50 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "3ps-nop0" ]]; then
        # 3 percentages: 10 50 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-10ps" ]]; then
        # last 10 percentages: 10 20 30 40 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-9ps" ]]; then
        # last 9 percentages: 20 30 40 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-8ps" ]]; then
        # last 8 percentages: 30 40 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-7ps" ]]; then
        # last 7 percentages: 40 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-6ps" ]]; then
        # last 6 percentages: 50 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-5ps" ]]; then
        # last 5 percentages: 60 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-4ps" ]]; then
        # last 4 percentages: 70 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-3ps" ]]; then
        # last 3 percentages: 80 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "last-2ps" ]]; then
        # last 2 percentages: 90 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
            $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p0" ]]; then
        # 1 percentage: 0
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix0p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p10" ]]; then
        # 1 percentage: 10
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix10p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p20" ]]; then
        # 1 percentage: 20
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix20p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p30" ]]; then
        # 1 percentage: 30
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix30p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p40" ]]; then
        # 1 percentage: 40
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix40p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p50" ]]; then
        # 1 percentage: 50
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix50p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p60" ]]; then
        # 1 percentage: 60
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix60p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p70" ]]; then
        # 1 percentage: 70
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix70p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p80" ]]; then
        # 1 percentage: 80
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix80p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p90" ]]; then
        # 1 percentage: 90
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix90p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    elif [[ $MIX_TRAIN == "p100" ]]; then
        # 1 percentage: 100
        echo -e "\nConcatenating prefix files for the training data"

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.utters \
        > $PREFIX_ORACLE_FOLDER/train.utters

        cat $PREFIX_ALL_ORACLE_FOLDER/train-prefix100p.actions.src_copy \
        > $PREFIX_ORACLE_FOLDER/train.actions.src_copy

        echo -e "\nResulted training files:"
        echo "$PREFIX_ORACLE_FOLDER/train.utters"
        echo "$PREFIX_ORACLE_FOLDER/train.actions.src_copy"
        echo

    fi


    # create .done flag
    touch $PREFIX_ORACLE_FOLDER/.done
    echo -e "\nPrefix mixed training data created and saved at ${PREFIX_ORACLE_FOLDER}\n"

fi
