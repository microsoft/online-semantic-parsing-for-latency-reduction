#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e

##### modify files in a folder with the same patterns
[[ -z $1 ]] && echo "Usage: $0 <files_dir>" && exit 1
files_dir=$1


old_pattern="order-top"
new_pattern="order-bot"

# old_pattern="act"
# new_pattern="old_act"

old_pattern="exp_prefix"
new_pattern="old_exp_prefix"

echo -e "\nIn folder [${files_dir}]:"
echo "Subsituting '${old_pattern}' -> '${new_pattern}' in file names"

##### one solution
files=($files_dir/*)
echo "(number of files: ${#files[@]})"
echo

for file in ${files[@]}; do

    if [[ $file != *"abs-prefix"* ]] && [[ $file == *"prefix"* ]]; then

    echo $file

    new_file=$(echo $file | sed "s/${old_pattern}/${new_pattern}/g")

    echo "----------> $new_file"

    mv $file $new_file

    fi

done
