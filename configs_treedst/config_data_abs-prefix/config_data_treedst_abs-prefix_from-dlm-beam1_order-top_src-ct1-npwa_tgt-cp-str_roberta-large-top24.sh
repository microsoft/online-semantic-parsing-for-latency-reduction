# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
##############################################################

##### load utter-dlm config (including the config_data_base)
config_model_dlm=configs_treedst/config_utter-dlm_src-ct1-npwa/config_dlm_bart-large_data_utter_abs-prefix-all_lr0.00005-scheduler-pd-bsz2048x4-wm500-dp0.1-cn0.1_ep12.sh

. $config_model_dlm   # should include its path
# now we should have
# environment set
# DATA_ROOT_DIR
#   |- DATA_FOLDER
#        |- ORACLE_FOLDER
#        |- BIN_FOLDER
#        |- EMB_FOLDER
# should also have the path for the trained utter-dlm
# MODEL_FOLDER

echo "[Data utter-dlm model configuration file:]"
echo $config_model_dlm
echo

###############################################################

##### the utter-dlm model results folder: $MODEL_FOLDER
beam_size=1
data_split=valid
model_epoch=_best
RESULTS_FOLDER=$MODEL_FOLDER/beam${beam_size}

results_src=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}.src
results_prefix=$RESULTS_FOLDER/${data_split}_checkpoint${model_epoch}    # .hypo, .1.hypo, ..., .4.hypo


##### setup: fairseq task
TASK=action_pointer_scp_prefix

##### new data directory for the prefix: all individual prefixes
PREFIX_ALL_DATA_FOLDER=${DATA_FOLDER}_from-utter-dlm-beam${beam_size}_abs-prefix-all
PREFIX_ALL_ORACLE_FOLDER=${PREFIX_ALL_DATA_FOLDER}/oracle
PREFIX_ALL_BIN_FOLDER=${PREFIX_ALL_DATA_FOLDER}/fairseq_bin
PREFIX_ALL_EMB_FOLDER=${PREFIX_ALL_DATA_FOLDER}/embeddings/roberta_large_top24


##### specific mixing of the prefix data for training

# MIX_TRAIN=all     # all prefix data with absolute length

# PREFIX_DATA_FOLDER=${DATA_FOLDER}_abs-prefix-${MIX_TRAIN}

# PREFIX_ORACLE_FOLDER=$PREFIX_DATA_FOLDER/oracle                             # oracle actions, etc.
# PREFIX_BIN_FOLDER=$PREFIX_DATA_FOLDER/fairseq_bin                           # preprocessed actions states information, etc.
# PREFIX_EMB_FOLDER=$PREFIX_DATA_FOLDER/embeddings/roberta_large_top24        # pre-stored pretrained en embeddings (not changing with oracle)

##### pretrained embeddings
PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
REMOVE_BE=1
AVG_WORD=1

# for decoding setup
src_pretrained_emb=1
# src_fix_emb_use=1
src_pretrained_emb_dim=$PRETRAINED_EMBED_DIM
