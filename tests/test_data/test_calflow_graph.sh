#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# set -e
# set -o pipefail

# ##### config
# if [ -z "$1" ]; then
#     :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
# else
#     config=$1
#     . $config    # $config should include its path
# fi
# # NOTE: when the first configuration argument is not provided, this script must
# #       be called from other scripts

# [[ -z $2 ]] && echo "No data split provided; default to valid set" && subset="valid" || subset=$2

# set -o nounset
# # NOTE this should be set after "set_environment.sh", particularly for the line
# #      eval "$(conda shell.bash hook)"

[[ -z $1 ]] && echo "No data split provided; default to valid set" && subset="valid" || subset=$1


##### linearize the data to be linearized (src, tgt) sequences to be used by model
# dataflow_dialogues_dir=$RAW_FOLDER
# graph_actions_dir=$DATADIR/tests

dataflow_dialogues_dir=../DATA/smcalflow2.0
graph_actions_dir=tests/data/calflow_graph

graph_order="top-down"
# graph_order="bottom-up"

python -m calflow_parsing.calflow_graph \
    --dialogues_jsonl ${dataflow_dialogues_dir}/${subset}.dataflow_dialogues.jsonl \
    --out_actions $graph_actions_dir/$subset.actions \
    --out_source $graph_actions_dir/$subset.utters \
    --out_lispress $graph_actions_dir/$subset.lispress \
    --graph_order $graph_order \
    --num_context_turns 1 \
    --include_agent_utterance
