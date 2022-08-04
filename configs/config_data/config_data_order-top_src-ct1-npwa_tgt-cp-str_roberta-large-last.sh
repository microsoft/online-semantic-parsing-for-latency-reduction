#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -o errexit
set -o pipefail

##### set environment and default directories for data and saving
. set_environment.sh
set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"
. set_default_dirs.sh
# this defines $DATADIR and $SAVEDIR

##### path to the raw data
RAW_DIR=$DATADIR/smcalflow2.0
TRAIN_RAW_FILE=$RAW_DIR/train.dataflow_dialogues.jsonl
VAL_RAW_FILE=$RAW_DIR/valid.dataflow_dialogues.jsonl

##### path to save processed data
DATA_ROOT_DIR=$DATADIR/processed/smcalflow2.0


# ==========================================================

##### setup: data src sequence linearization
NUM_CONTEXT_TURNS=1
# LIN_ARGS="
#     --include_program
#     --include_agent_utterance
#     --include_described_entities
# "
LIN_ARGS="--include_agent_utterance"

##### setup: graph actions oracle
ORACLE_ORDER="top-down"

COPY_ONLY_STRING=1
# COPY_ONLY_FROM_CURRENT_UTTER=1


##### setup: fairseq task
TASK=action_pointer_scp


##### setup: fairseq preprocess
PREPROCESS_ARGS=""


##### automatically create data folder name based on setup
order_tag=""
if [[ $ORACLE_ORDER == "top-down" ]]; then
    order_tag="-top"
elif [[ $ORACLE_ORDER == "bottom-up" ]]; then
    order_tag="-bot"
else
    echo -e "\nUnrecognized oracle graph node generation order $ORACLE_ORDER"
    exit 1
fi

src_context_tag="-ct${NUM_CONTEXT_TURNS}"    # context turn
if [[ $LIN_ARGS == *"--include_program"* ]]; then
    src_context_tag=${src_context_tag}-wp    # with program
else
    src_context_tag=${src_context_tag}-np    # no program
fi
if [[ $LIN_ARGS == *"--include_agent_utterance"* ]]; then
    src_context_tag=${src_context_tag}wa     # with agent utterance
else
    src_context_tag=${src_context_tag}na     # no agent utterance
fi
# default is not including entities
if [[ $LIN_ARGS == *"--include_described_entities"* ]]; then
    src_context_tag=${src_context_tag}we     # with entities
# else
#     src_context_tag=${src_context_tag}ne
fi

tgt_copy_tag="_tgt-cp"
if [[ $COPY_ONLY_STRING == 1 ]]; then
    tgt_copy_tag=${tgt_copy_tag}-str
else
    tgt_copy_tag=${tgt_copy_tag}-any
fi

DATA_FOLDER=$DATA_ROOT_DIR/act${order_tag}_src${src_context_tag}${tgt_copy_tag}
ORACLE_FOLDER=$DATA_FOLDER/oracle                             # oracle actions, etc.
BIN_FOLDER=$DATA_FOLDER/fairseq_bin                           # preprocessed actions states information, etc.
EMB_FOLDER=$DATA_FOLDER/embeddings/roberta_large_last        # pre-stored pretrained en embeddings (not changing with oracle)

##### pretrained embeddings
PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="24"
REMOVE_BE=1
AVG_WORD=1
