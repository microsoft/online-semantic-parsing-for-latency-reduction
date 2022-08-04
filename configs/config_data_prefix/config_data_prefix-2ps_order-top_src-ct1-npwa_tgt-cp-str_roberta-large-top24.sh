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
TASK=action_pointer_scp_prefix

##### new data directory for the prefix: all individual prefixes
PREFIX_ALL_DATA_FOLDER=${DATA_FOLDER}_prefix-all
PREFIX_ALL_ORACLE_FOLDER=${PREFIX_ALL_DATA_FOLDER}/oracle
PREFIX_ALL_BIN_FOLDER=${PREFIX_ALL_DATA_FOLDER}/fairseq_bin
PREFIX_ALL_EMB_FOLDER=${PREFIX_ALL_DATA_FOLDER}/embeddings/roberta_large_top24


##### specific mixing of the prefix data for training

MIX_TRAIN=2ps     # 2 percentages: 50 100

PREFIX_DATA_FOLDER=${DATA_FOLDER}_prefix-${MIX_TRAIN}

PREFIX_ORACLE_FOLDER=$PREFIX_DATA_FOLDER/oracle                             # oracle actions, etc.
PREFIX_BIN_FOLDER=$PREFIX_DATA_FOLDER/fairseq_bin                           # preprocessed actions states information, etc.
PREFIX_EMB_FOLDER=$PREFIX_DATA_FOLDER/embeddings/roberta_large_top24        # pre-stored pretrained en embeddings (not changing with oracle)

##### pretrained embeddings
PRETRAINED_EMBED=roberta.large
PRETRAINED_EMBED_DIM=1024
BERT_LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
REMOVE_BE=1
AVG_WORD=1
