#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -o errexit
set -o pipefail


### local conda environment
cenv_name=cenv
[ ! -d $cenv_name ] && conda create -y -p ./$cenv_name

# eval "$(conda shell.bash hook)"
# echo "conda activate ./$cenv_name"
# conda activate ./$cenv_name

echo "source activate ./$cenv_name"
source activate ./$cenv_name

set -o nounset

### install packages
conda install python=3.8 -y
conda install numpy -y

# conda install pytorch==1.8.0 cudatoolkit=10.1 -c pytorch -y
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch -y

# pip install fairseq==0.10.2
pip install git+https://github.com/pytorch/fairseq.git@7ca8bc12c09d91187d95117094f6b31b3342cd17    # commit at 6/22/2021

# tensorboard [for fairseq logging; optional]
pip install tensorboard

# install dataflow package
pip install git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git@5012802f7db5b31a908927c510a1288cef659f06    # commit at 2/22/2022
# Download the spaCy model for tokenization
python -m spacy download en_core_web_md-2.2.0 --direct

# install osp package in development mode
pip install --editable .

# install pytorch_scatter
# pip install torch-scatter==1.3.2
# pip install torch-scatter==2.0.2
# pip install torch-scatter==2.0.9    # latest version that goes with PyTorch 1.10 updates
# pip install torch-scatter==2.0.8
# pip install torch-scatter==2.0.6    # highest version that works with PyTorch 1.9
pip install torch-scatter==2.0.2


echo "Finished installation!"
