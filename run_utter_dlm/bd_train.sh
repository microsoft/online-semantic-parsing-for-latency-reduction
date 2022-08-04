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

##### train the denoising LM by finetuning BART
TOTAL_NUM_UPDATES=${TOTAL_NUM_UPDATES:-60000}

BART_PATH=$SAVEDIR/bart.${BART_SIZE}/model.pt

if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

CUDA_VISIBLE_DEVICES=0 fairseq-train $PREFIX_ALL_BIN_FOLDER \
    --restore-file $BART_PATH \
    --task translation \
    --source-lang utters --target-lang fullutters \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_${BART_SIZE} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout $dropout --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm $clip_norm \
    $LR_ARGS \
    --lr $lr --warmup-updates $warmup \
    --max-epoch $max_epoch \
    --keep-last-epochs 5 \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --log-format json \
    --seed $seed \
    --save-dir $MODEL_FOLDER \
    --tensorboard-logdir $MODEL_FOLDER $fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters

fi