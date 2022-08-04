# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
##############################################################

##### load data config
config_data=configs/config_data_prefix/config_data_prefix-last-6ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24.sh

data_tag="$(basename $config_data | sed 's@config_data_\(.*\)\.sh@\1@g')"

. $config_data   # $config_data should include its path
# now we should have
# environment set
# DATA_ROOT_DIR
#   |- DATA_FOLDER
#        |- ORACLE_FOLDER
#        |- BIN_FOLDER
#        |- EMB_FOLDER

echo "[Data configuration file:]"
echo $config_data
echo

###############################################################

##### model configuration

shift_pointer_value=1
apply_tgt_actnode_masks=0
tgt_vocab_masks=1
share_all_embed=0
share_decoder_embed=0


arch=transformer_tgt_pointer_scp
# NOTE when "arch" is changed (including model layers etc.), REMEMBER to
#      check and update the setup below for pointer layers
criterion=label_smoothed_cross_entropy_pointer_scp
validset=valid-prefix100p    # validation subset during training time


pointer_dist_decoder_selfattn_layers="5"
pointer_dist_decoder_selfattn_heads=1
pointer_dist_decoder_selfattn_avg=0
pointer_dist_decoder_selfattn_infer=5

tgt_src_copy_layers="4"
tgt_src_copy_heads=1
TRAIN_ARGS="
    --tgt-src-copy-layers $tgt_src_copy_layers
    --tgt-src-copy-heads $tgt_src_copy_heads
    "

##### pretrained features
src_pretrained_emb=1
src_pretrained_emb_dim=$PRETRAINED_EMBED_DIM
src_pool_wp2w=bot

##### optimization hyper-parameters

seed=${seed:-42}
max_epoch=50
eval_init_epoch=31
# max_epoch=5
# eval_init_epoch=1

lr=0.0001
max_tokens=4096
update_freq=4
warmup=4000
dropout=0.2
clip_norm=0.0    # default is 0.0 (DO NOT use 0)
# fp16=""    # or "--fp16"
fp16="--fp16"


##### set the experiment dir name based on model configurations

if [[ $pointer_dist_decoder_selfattn_layers == "0 1 2 3 4 5" ]]; then
    lay="all"
else
    lay=""
    for n in $pointer_dist_decoder_selfattn_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'pointer_dist_decoder_selfattn_layers' input: $pointer_dist_decoder_selfattn_layers" && exit 1
        lay=$lay$(( $n + 1 ))
    done
fi

if [[ $tgt_src_copy_layers == "0 1 2 3 4 5" ]]; then
    scp_lay="all"
else
    scp_lay=""
    for n in $tgt_src_copy_layers; do
        [[ $n < 0 || $n > 5 ]] && echo "Invalid 'tgt_src_copy_layers' input: $tgt_src_copy_layers" && exit 1
        scp_lay=$scp_lay$(( $n + 1 ))
    done
fi

# set the experiment directory name
expdir=exp_${data_tag}_gap_vmask${tgt_vocab_masks}_shiftpos${shift_pointer_value}

# pointer distribution
ptr_tag=_ptr-lay${lay}-h${pointer_dist_decoder_selfattn_heads}    # action-pointer

# source copy distribution
scp_tag=_scp-lay${scp_lay}-h${tgt_src_copy_heads}

# # initialize with bart
# if [[ $initialize_with_bart == 0 ]]; then
#     init_tag=_bart-init${initialize_with_bart}
# else
#     if [[ $initialize_with_bart_enc == 0 ]]; then
#         [[ $initialize_with_bart_dec == 0 ]] && echo "initialize_with_bart_dec should be 1 here" && exit 1
#         init_tag=_bart-init-enc0
#     fi
#     if [[ $initialize_with_bart_dec == 0 ]]; then
#         [[ $initialize_with_bart_enc == 0 ]] && echo "initialize_with_bart_enc should be 1 here" && exit 1
#         init_tag=_bart-init-dec0
#     fi
#     if [[ $initialize_with_bart_enc == 1 ]] && [[ $initialize_with_bart_dec == 1 ]]; then
#         init_tag=""
#     fi
# fi

# # fix bart encoder
# if [[ $bart_encoder_backprop == 0 ]]; then
#     [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
#     enc_fix_tag=_bart-enc-fix
# else
#     enc_fix_tag=""
# fi

# # fix bart embedding
# if [[ $bart_emb_backprop == 0 ]]; then
#     [[ $initialize_with_bart == 0 ]] && echo "must initialize with bart to fix encoder" && exit 1
#     emb_fix_tag=_bart-emb-fix
# else
#     emb_fix_tag=""
# fi

# share or separate decoder input and output embedding
dec_emb_tag=_dec-emb-sha${share_decoder_embed}

# share all embeddings
# share_emb_tag=_all-emb-sha${share_all_embed}
if [[ $share_all_embed == 1 ]]; then
    share_emb_tag=_all-emb-sha${share_all_embed}
    share_decoder_embed=1
    dec_emb_tag=""
elif [[ $share_all_embed == 0 ]]; then
    share_emb_tag=""
fi


# # bart decoder input embedding
# if [[ $bart_emb_decoder_input == 0 ]]; then
#     [[ $bart_emb_decoder == 1 ]] && echo "bart_emb_decoder should be 0" && exit 1
#     dec_emb_in_tag=""
# else
#     if [[ $bart_emb_decoder == 1 ]]; then
#         dec_emb_in_tag=""
#     else
#         # decoder input BART embeddings, output customized embeddings
#         dec_emb_in_tag="_bart-dec-emb-in"
#     fi
# fi
# # initialize target embedding with compositional sub-token embeddings
# if [[ $bart_emb_init_composition == 1 ]]; then
#     dec_emb_init_tag="_bart-init-dec-emb"
# else
#     dec_emb_init_tag=""
# fi

# # combine different model configuration tags to the name
# expdir=${expdir}${ptr_tag}${cam_tag}${tis_tag}${dec_emb_tag}${dec_emb_in_tag}${dec_emb_init_tag}${init_tag}${enc_fix_tag}${emb_fix_tag}

# combine different model configuration tags to the name
expdir=${expdir}${ptr_tag}${scp_tag}${share_emb_tag}${dec_emb_tag}

# specific model directory name with a set random seed
fp16_tag=""
[[ $fp16 ]] && fp16_tag="fp16-"

if [[ $clip_norm != "0.0" ]]; then
    cn_tag=-cn${clip_norm}
else
    cn_tag=""
fi

optim_tag=_${fp16_tag}lr${lr}-mt${max_tokens}x${update_freq}-wm${warmup}-dp${dropout}${cn_tag}
MODEL_FOLDER=$SAVEDIR/$expdir/models_ep${max_epoch}_seed${seed}${optim_tag}
