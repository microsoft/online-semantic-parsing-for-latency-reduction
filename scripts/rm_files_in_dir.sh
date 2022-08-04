#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e

##### rm files in a folder with the same patterns
[[ -z $1 ]] && echo "Usage: $0 <files_dir>" && exit 1
files_dir=$1

rm_pattern="act-top_src-ct1-npwa_tgt-cp-str_prefix"
exclude_pattern="old"

echo -e "\nIn folder [${files_dir}]:"
echo "Removing files/folders with pattern '${rm_pattern}' (excluding '${exclude_pattern}') in file names"

##### one solution
files=($files_dir/*)
echo "(number of files: ${#files[@]})"
echo

for file in ${files[@]}; do

    if [[ $file != *"${exclude_pattern}"* ]] && [[ $file == *"${rm_pattern}"* ]]; then

    echo "remove $file"

    rm -r $file

    fi

done
