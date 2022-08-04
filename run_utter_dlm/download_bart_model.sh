#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

##### config
if [ -z "$1" ]; then
    :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
else
    config_data=$1
    . $config_data    # $config should include its path
fi
# NOTE: when the first configuration argument is not provided, this script must
#       be called from other scripts

set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"

echo "[Prefix configuration file:]"
echo "$config_data"
echo

##### download the BART model checkpoints
# Download bart.large model
if [[ ! -s $SAVEDIR/bart.large/model.pt ]]; then
wget -P $SAVEDIR $MODEL_FOLDER https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf $SAVEDIR/bart.large.tar.gz -C $SAVEDIR
fi
echo -e "\nbart.large downloaded at $SAVEDIR/bart.large/model.pt"

# Download bart.base model
if [[ ! -s $SAVEDIR/bart.base/model.pt ]]; then
wget -P $SAVEDIR $MODEL_FOLDER https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xzvf $SAVEDIR/bart.base.tar.gz -C $SAVEDIR
fi
echo -e "\nbart.base downloaded at $SAVEDIR/bart.base/model.pt"
