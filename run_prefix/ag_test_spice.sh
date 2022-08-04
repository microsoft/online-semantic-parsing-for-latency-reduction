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


##### script specific config
if [ -z "$2" ]; then
    data_split="valid"
else
    data_split=$2
fi

if [ $data_split == "valid" ]; then
    reference_file=$VAL_RAW_FILE
elif [ $data_split == "test" ]; then
    reference_file=$TEST_RAW_FILE
else
    # echo "$2 is invalid; must be valid or test"
    :
fi


# ===== NOTE here we hard code the gold files; be careful!
gold_lispress=$ORACLE_FOLDER/valid.lispress
gold_actions=$ORACLE_FOLDER/valid.actions
# gold_actions=$PREFIX_ALL_ORACLE_FOLDER/${data_split}.actions
# ===========================================


model_epoch=${model_epoch:-_last}
beam_size=${beam_size:-1}
batch_size=${batch_size:-256}

RESULTS_FOLDER=$MODEL_FOLDER/beam${beam_size}
# results_prefix=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}.nopos-score
results_prefix=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}
model=$MODEL_FOLDER/checkpoint${model_epoch}.pt

TASK=${TASK:-action_pointer}

src_fix_emb_use=${src_pretrained_emb:-0}


# ##### DECODING
# # rm -Rf $RESULTS_FOLDER
# mkdir -p $RESULTS_FOLDER
# # --nbest 3 \
# # --quiet
# python -m fairseq_gap.generate \
#     $PREFIX_ALL_BIN_FOLDER  \
#     --emb-dir $PREFIX_ALL_EMB_FOLDER \
#     --user-dir src/fairseq_gap \
#     --task $TASK \
#     --gen-subset $data_split \
#     --src-fix-emb-use $src_fix_emb_use \
#     --src-fix-emb-dim $src_pretrained_emb_dim \
#     --modify-arcact-score 1 \
#     --beam $beam_size \
#     --batch-size $batch_size \
#     --remove-bpe \
#     --path $model  \
#     --quiet \
#     --results-path $results_prefix \

# # exit 0


# ##### Recover the original graph, and to specific data format
# python -m calflow_parsing.calflow_graph_lispress \
#     --in_actions $results_prefix.actions \
#     --out_lispress $results_prefix.lispress

# # exit 0


# ##### Evaluation: exact match
# python -m calflow_parsing.exact_match \
#     --gold_lispress $gold_lispress \
#     --test_actions $results_prefix.actions \
#     --out_file $results_prefix.em \
#     --gold_actions $gold_actions \
#     --test_lispress $results_prefix.lispress
# # NOTE the last two files are optional - but they reduce recomputation

# echo -e "\n[Exact match results with full graph:]"
# cat $results_prefix.em

# # exact match for the graph with anonymous copy action tailored for the prefix
# python -m calflow_parsing.exact_match \
#     --gold_lispress $gold_lispress \
#     --test_actions $results_prefix.actions \
#     --out_file $results_prefix.mod-graph.em \
#     --gold_actions $PREFIX_ALL_ORACLE_FOLDER/${data_split}.actions \
#     --test_lispress $results_prefix.lispress
# # NOTE the last two files are optional - but they reduce recomputation

# echo -e "\n[Exact match results with graph copy modified with prefix:]"
# cat $results_prefix.mod-graph.em


##### Evaluation: graph tuple match, precision recall and f1
python -m calflow_parsing.tuple_match \
    --test_actions $results_prefix.actions \
    --out_file $results_prefix.tm \
    --gold_actions $gold_actions

echo -e "\n[Tuple match results with full graph:]"
cat $results_prefix.tm

# exact match for the graph with anonymous copy action tailored for the prefix
python -m calflow_parsing.tuple_match \
    --test_actions $results_prefix.actions \
    --out_file $results_prefix.mod-graph.tm \
    --gold_actions $PREFIX_ALL_ORACLE_FOLDER/${data_split}.actions

echo -e "\n[Tuple match results with graph copy modified with prefix:]"
cat $results_prefix.mod-graph.tm
