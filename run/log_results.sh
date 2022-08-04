#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -o errexit
set -o pipefail


if [[ -z $1 ]]; then
    exp_prefix="exp_"
else
    exp_prefix=$1
fi

score_suffix=em
# score_suffix=tm
# score_suffix=apim

# whether to include the full path for each result file in the collection log
include_full_path=0

if [[ -z $2 ]]; then
    save_file="results_collection/all_${score_suffix}_results_from_${exp_prefix}.txt"
    if [[ $include_full_path == 1 ]]; then
    save_file="results_collection/all_${score_suffix}_results_from_${exp_prefix}_full-path.txt"
    fi
else
    save_file=$2
fi


echo "All results collected to be saved at: [$save_file]"
mkdir -p $(dirname $save_file)


rootdir=/mnt/container_amulet/SAVE

exp_dirs=($rootdir/${exp_prefix}*)


cat_all_results() {

# iterate over all exp directories
for exp_dir in "${exp_dirs[@]}"; do

    echo "-------------------------------------------------------------------------------"
    echo -e "\n[Results for all model checkpoints under experiments:]"
    echo "$exp_dir"
    echo

    model_folders=($exp_dir/models*)

    for checkpoints_folder in "${model_folders[@]}"; do

        if [[ $include_full_path == 1 ]]; then
        echo "----- $checkpoints_folder"
        else
        echo "-----> $(basename -- "$checkpoints_folder")"
        fi
        echo

        beam_folders=($checkpoints_folder/beam*)

        for beam_folder in "${beam_folders[@]}"; do

            if [[ $include_full_path == 1 ]]; then
            echo "---------- $beam_folder"
            else
            echo "----------> $(basename -- "$beam_folder") <----------"
            fi
            echo

            score_files=($beam_folder/*.${score_suffix})

            for score_file in "${score_files[@]}"; do

                if [[ $include_full_path == 1 ]]; then
                echo "<$score_file>"
                else
                echo "<$(basename -- "$score_file")>"
                fi
                echo
                if [[ -s $score_file ]]; then
                    cat $score_file
                else
                    echo "score file not existing --- skipped"
                fi
                echo

            done

        done

    done

done

}


cat_all_results > $save_file
less $save_file
