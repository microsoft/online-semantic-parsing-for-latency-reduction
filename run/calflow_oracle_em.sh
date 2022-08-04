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
    data_split=$2    # could be "train"
fi

# dataset source
dataset=${DATASET:-smcalflow}


##### Recover the original graph, and to specific data format; and then evaluate exact match scores
# to avoid rerunning
if [[ -s $ORACLE_FOLDER/oracle_${data_split}.em ]]; then

    echo -e "\nOracle exact match already evaluated -- do nothing\n"

else

    # Recover the original graph, and to specific data format
    python -m calflow_parsing.calflow_graph_lispress \
        --dataset $dataset \
        --in_actions $ORACLE_FOLDER/${data_split}.actions \
        --out_lispress $ORACLE_FOLDER/oracle_${data_split}.lispress

    # exit 0


    ##### Evaluation: exact match
    python -m calflow_parsing.exact_match \
        --dataset $dataset \
        --gold_lispress $ORACLE_FOLDER/${data_split}.lispress \
        --test_actions $ORACLE_FOLDER/${data_split}.actions \
        --out_file $ORACLE_FOLDER/oracle_${data_split}.em \
        --gold_actions $ORACLE_FOLDER/${data_split}.actions \
        --test_lispress $ORACLE_FOLDER/oracle_${data_split}.lispress
    # NOTE the last two files are optional - but they reduce recomputation

fi

cat $ORACLE_FOLDER/oracle_${data_split}.em
