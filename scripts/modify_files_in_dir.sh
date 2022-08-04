#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e

##### modify files in a folder with the same patterns
[[ -z $1 ]] && echo "Usage: $0 <files_dir>" && exit 1
files_dir=$1

old_pattern="share_decoder_embed=0"
new_pattern="share_decoder_embed=1"

old_pattern="max_epoch=100"
new_pattern="max_epoch=200"

old_pattern="_6x6x4hx128x512"
new_pattern="_6x6x8hx512x2048"

old_pattern="order-top"
new_pattern="order-bot"

old_pattern="1k"
new_pattern="10k"

old_pattern="128"
new_pattern="256"

old_pattern="80"
new_pattern="50"

old_pattern="order-top"
new_pattern="order-bot"

old_pattern="prefix/"
new_pattern="prefix_order-bot/"

echo -e "\nIn folder [${files_dir}]:"
echo "Subsituting '${old_pattern}' -> '${new_pattern}'"

##### one solution
files=($files_dir/*)
echo "(number of files: ${#files[@]})"
echo

for file in ${files[@]}; do
    echo $file
    sed -i "s@${old_pattern}@${new_pattern}@g" $file
done


# ##### another solution (not working)
# find $files_dir/ -name '*' -exec sed -i -e "s/${old_pattern}/${new_pattern}/g" {} \;

# ##### another solution (not working)
# find $files_dir/ -name 'config_*' -print0 | xargs -0 sed -i "s/${old_pattern}/${new_pattern}/g" $file
