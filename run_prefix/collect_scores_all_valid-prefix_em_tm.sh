#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -o errexit
set -o pipefail
# setup environment
# . set_environment.sh

checkpoints_folder=$1
[ -z "$checkpoints_folder" ] && \
    echo -e "\n$0 <checkpoints_folder>\n" && \
    exit 1
set -o nounset

score_names="em tm"
score_names="em tm mod-graph.em mod-graph.tm"
# score_items="exact_match_graphs exact_match_lispress tuple_match_average_f1"

# data_sets=("dev")
# data_sets=("test")
data_sets=("dev" "test")

### both of below work
# data_sets=("valid-prefix100p" "valid-prefix90p" "valid-prefix80p" "valid-prefix70p" "valid-prefix60p" "valid-prefix50p"
            # "valid-prefix40p" "valid-prefix30p" "valid-prefix20p" "valid-prefix10p" "valid-prefix0p")
data_sets=("valid-prefix100p valid-prefix90p valid-prefix80p valid-prefix70p valid-prefix60p valid-prefix50p
            valid-prefix40p valid-prefix30p valid-prefix20p valid-prefix10p valid-prefix0p")

# or (loop over a list of strings)
# data_sets="dev test"
# for data in $data_sets; do
#     echo $data
# done

# for data in ${data_sets[@]}; do

#     if [[ $data == "dev" ]]; then
#         data=valid
#     fi

# done

for ((i=0;i<${#data_sets[@]};++i)); do
    if [[ ${data_sets[i]} == "dev" ]]; then
        data_sets[$i]=valid
    fi
done



# the following will join the string list to a single string;
# this is in constrast with: ${data_sets[*]}, or ${data_sets[@]}, or "${data_sets[@]}"
for data in "${data_sets[*]}"; do

    python run/collect_scores_all.py \
           $checkpoints_folder \
           --score_names $score_names \
           --score_items ${score_items:-""} \
           --data_sets $data \

done
