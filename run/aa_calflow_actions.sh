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


##### generate the graph actions following certain order
# linearize the data to be (src, tgt) sequences to be used by model
dataflow_dialogues_dir=$RAW_DIR
graph_actions_dir=$ORACLE_FOLDER

# dataset source
dataset=${DATASET:-smcalflow}

# to avoid rerunning
if [ -f $ORACLE_FOLDER/.done ]; then

    echo -e "\n[$ORACLE_FOLDER] src and tgt sequences already exist --- do nothing\n"

else

    for subset in "train" "valid"; do
        if [[ $subset == "train" ]]; then
            raw_file=$TRAIN_RAW_FILE
        elif [[ $subset == "valid" ]]; then
            raw_file=$VAL_RAW_FILE
        fi
        python -m calflow_parsing.calflow_graph \
            --dialogues_jsonl $raw_file \
            --out_actions $graph_actions_dir/$subset.actions \
            --out_source $graph_actions_dir/$subset.utters \
            --out_lispress $graph_actions_dir/$subset.lispress \
            --graph_order $ORACLE_ORDER \
            --dataset $dataset \
            --num_context_turns $NUM_CONTEXT_TURNS \
            $LIN_ARGS
    done

    touch $ORACLE_FOLDER/.done
    echo -e "\nSource and target sequences created and saved at $graph_actions_dir\n"

fi
