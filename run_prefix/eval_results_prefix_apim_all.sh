#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -o errexit
set -o pipefail


if [[ -z $1 ]]; then
    exp_prefix="exp_prefix"
else
    exp_prefix=$1
fi

if [[ -z $2 ]]; then
    save_file="results_collection/compute_all_apim_results_from_${exp_prefix}.txt"
else
    save_file=$2
fi


echo "Computed apim results to be saved at: [$save_file]"
mkdir -p $(dirname $save_file)

rootdir=/mnt/container_amulet/SAVE2

exp_dirs=($rootdir/${exp_prefix}*)


eval_apim_all () {

# iterate over all exp directories
for exp_dir in "${exp_dirs[@]}"; do

# if [[ $exp_dir == *"order-top"* ]]; then
if [[ $exp_dir == *"order-bot"* ]]; then

    echo "-------------------------------------------------------------------------------"
    echo -e "\n[Compute evaluation scores for all model checkpoints under experiments:]"
    echo "$exp_dir"
    echo

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do

        echo "----- $checkpoints_folder"

        config_model=($checkpoints_folder/config_model_*)

        echo "----- $config_model"

        bash run_prefix/eval_results_prefix_apim.sh $config_model

    done

fi

done

}

eval_apim_all |& tee $save_file    # dump to file and print to console
# eval_apim_all &> $save_file        # only dump to file
# less $save_file
