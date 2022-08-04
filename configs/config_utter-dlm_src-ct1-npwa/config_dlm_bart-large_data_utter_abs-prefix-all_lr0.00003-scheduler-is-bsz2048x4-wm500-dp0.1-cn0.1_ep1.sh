# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
##############################################################

##### load data config
config_data_base=configs/config_data/config_data_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24.sh

data_base_tag="$(basename $config_data_base | sed 's@config_data_\(.*\)\.sh@\1@g')"

. $config_data_base   # $config_data should include its path
# now we should have
# environment set
# DATA_ROOT_DIR
#   |- DATA_FOLDER
#        |- ORACLE_FOLDER
#        |- BIN_FOLDER
#        |- EMB_FOLDER

echo "[Data Base configuration file:]"
echo $config_data_base
echo

###############################################################

##### setup: fairseq task
TASK=translation

##### new data directory for the prefix: all individual prefixes
DATA_FOLDER_DLM=$DATA_ROOT_DIR/src${src_context_tag}
PREFIX_ALL_DATA_FOLDER=${DATA_FOLDER_DLM}_utter-dlm_abs-prefix-all
PREFIX_ALL_ORACLE_FOLDER=${PREFIX_ALL_DATA_FOLDER}/oracle
PREFIX_ALL_BIN_FOLDER=${PREFIX_ALL_DATA_FOLDER}/fairseq_bin
PREFIX_ALL_EMB_FOLDER=${PREFIX_ALL_DATA_FOLDER}/embeddings/bart



###############################################################

##### setup: train and model
BART_SIZE=large

expdir=exp_utter_abs-prefix-all_dlm_bart-${BART_SIZE}


##### optimization
seed=${seed:-42}
max_epoch=1
# max_epoch=3

lr=3e-05    # 0.0001
lr_scheduler="inverse_sqrt"
max_tokens=2048
update_freq=4
warmup=500    # 4000
dropout=0.1    # 0.1
clip_norm=0.1    # default is 0.0 (DO NOT use 0)
# fp16=""    # or "--fp16"
fp16="--fp16"

LR_ARGS="--lr-scheduler ${lr_scheduler}"


# specific model directory name with a set random seed
fp16_tag=""
[[ $fp16 ]] && fp16_tag="fp16-"

if [[ $clip_norm != "0.0" ]]; then
    cn_tag=-cn${clip_norm}
else
    cn_tag=""
fi

optim_tag=_${fp16_tag}lr${lr}-scheduler${lr_scheduler}-mt${max_tokens}x${update_freq}-wm${warmup}-dp${dropout}${cn_tag}
MODEL_FOLDER=$SAVEDIR/$expdir/models_ep${max_epoch}_seed${seed}${optim_tag}
