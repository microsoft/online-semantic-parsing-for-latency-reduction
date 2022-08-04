#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

### set up conda environment for running the code

# set -e
# set -o pipefail

# NOTE do not include the above!!! They would change the shell behavior as the scipt is to be sourced directly


### activate conda environment
cenv_name=cenv

# [ ! -d $cenv_name ] && echo "local environment $cenv_name does not exist --- please install first" && exit 1
if [[ -d $cenv_name ]]; then
    echo
    echo "$(which conda); version: $(conda --version)"
    # eval "$(conda shell.bash hook)"
    # echo "conda activate ./$cenv_name"
    # conda activate ./$cenv_name
    echo "source activate ./$cenv_name"
    source activate ./$cenv_name
    echo
else
    echo
    echo "local environment $cenv_name does not exist"
    echo "assuming running from a docker container (e.g. on the cloud compute), where environment requirements are already satisfied"
    echo
fi
