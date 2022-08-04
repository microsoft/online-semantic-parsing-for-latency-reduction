#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail
set -o nounset


function add_ms_header {
    local file=$1

    header='# Copyright (c) Microsoft Corporation.\n# Licensed under the MIT license.'

    echo $file

    if [[ $(head -1 $file) == "#!"* ]]; then
    if [[  $(head -2 $file | tail -1) != *"Copyright"* ]]; then
        echo "     starting with #!: insert from the 2nd line"
        sed -i "1a $header" $file    # sed: a text | Append text after a line (alternative syntax).
        # Or equivalently
        # sed -i "2i $header" $file    # sed: i text | Insert text before a line (alternative syntax).
    fi
    else
    if [[  $(head -1 $file) != *"Copyright"* ]]; then
        echo "     not starting with #!: insert from the 1st line"
        sed -i "1i $header" $file    # NOTE this does not work for empty file (no 1st line)
        # https://unix.stackexchange.com/questions/600241/why-does-this-sed-construct-not-insert-anything-into-an-empty-file
        if [[ ! -s $file ]]; then
            echo "     empty file: insert from the 1st line"
            echo -e $header >> $file
        fi
    fi
    fi

}


# ============================================================================
# iterate over a directory recurively, and process all files to add the header

[[ -z $1 ]] && echo "usage: bash $0 directory_name" && exit 1

dir=$1

# === Approach 1: for loop with find
# for file in $(find $dir -type f \( -name "*.py" -or -name "*.sh" \) ! -path "*cenv/*" ! -path "*/__pycache__/*" ); do
#     # echo $file
#     add_ms_header $file
# done

# === Approach 2: directly use find -exec expression
# https://www.baeldung.com/linux/find-exec-command#the-command
export -f add_ms_header
find $dir -type f \( -name "*.py" -or -name "*.sh" \) ! -path "*cenv/*" ! -path "*/__pycache__/*" -exec bash -c "add_ms_header \"{}\"" \;


# ============================================================================
# example usage: to add ms header to all files in the repo
# bash scripts/add_file_header.sh ./
