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

echo "[Prefix completion data configuration file:]"
echo "$config_data"
echo

##### create src to be input to the parser
# original src file: $results_src (with context and utterance prefix)
# model prediction utterance: $results_prefix.hypo, $results_prefix.1.hypo, etc.

mkdir -p $PREFIX_ALL_ORACLE_FOLDER

if [[ $beam_size == 5 ]]; then

    python -m calflow_prefix.recombine_utter_to_src_dlm \
        --src_utterances $results_src \
        --src_completions $results_prefix.hypo \
        --out_src $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-1.utters

    nlines=$(wc -l < $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-1.utters)
    echo "number of lines: $nlines"
    echo | awk "{ for (counter = $nlines; counter >= 1; counter--) print \"Empty\"; }" > $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-1.actions.src_copy
    echo -e "create empty (place holder) gold action files $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-1.actions.src_copy\n"

    for bs in 1 2 3 4
    do
        num=$(($bs+1))
        python -m calflow_prefix.recombine_utter_to_src_dlm \
            --src_utterances $results_src \
            --src_completions $results_prefix.${bs}.hypo \
            --out_src $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-$num.utters

        nlines=$(wc -l < $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-$num.utters)
        echo "number of lines: $nlines"
        echo | awk "{ for (counter = $nlines; counter >= 1; counter--) print \"Empty\"; }" > $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-$num.actions.src_copy
        echo -e "create empty (place holder) gold action files $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam5-$num.actions.src_copy\n"

    done
fi

if [[ $beam_size == 1 ]]; then
    python -m calflow_prefix.recombine_utter_to_src_dlm \
        --src_utterances $results_src \
        --src_completions $results_prefix.hypo \
        --out_src $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam1-1.utters

    nlines=$(wc -l < $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam1-1.utters)
    echo "number of lines: $nlines"
    echo | awk "{ for (counter = $nlines; counter >= 1; counter--) print \"Empty\"; }" > $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam1-1.actions.src_copy
    echo -e "create empty (place holder) gold action files $PREFIX_ALL_ORACLE_FOLDER/valid-prefixallt-beam1-1.actions.src_copy\n"

fi
