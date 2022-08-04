#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e
set -o pipefail

# ##### config
# if [ -z "$1" ]; then
#     :        # in this case, must provide $1 as "" empty; otherwise put "set -o nounset" below
# else
#     config=$1
#     . $config    # $config should include its path
# fi
# # NOTE: when the first configuration argument is not provided, this script must
# #       be called from other scripts

# set -o nounset
# # NOTE this should be set after "set_environment.sh", particularly for the line
# #      eval "$(conda shell.bash hook)"


##### check if the script is being sourced from other script or directly called
(return 0 2>/dev/null) && sourced=1 || sourced=0
# [[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "script ${BASH_SOURCE[0]} is being sourced ..." || echo "script ${BASH_SOURCE[0]} is NOT being sourced ..."
# [[ "${BASH_SOURCE[0]}" != "${0}" ]] && sourced=1 || sourced=0
# echo $sourced


##### CONFIG
if [[ $sourced == 0 ]]; then
    dir=$(dirname $0)
    if [ ! -z "${1+x}" ]; then
        config=$1
        . $config    # $config_model should always include its path
    fi
    # NOTE: when the first configuration argument is not provided, this script must
    #       be called from other scripts
fi

set -o nounset
# NOTE this should be set after "set_environment.sh", particularly for the line
#      eval "$(conda shell.bash hook)"


##### script specific config (default values)
if [ -z ${max_epoch+x} ]; then
    max_epoch=100
fi
eval_init_epoch=${eval_init_epoch:-61}
seed=${seed:-42}

# $TASK is defined in the data configuration file
# $arch is defined in the model configuration file
TASK=${TASK:-action_pointer}
arch=${arch:-transformer_tgt_pointer}

apply_tgt_actnode_masks=${apply_tgt_actnode_masks:-0}

# initialize_with_bart=${initialize_with_bart:-1}
# initialize_with_bart_enc=${initialize_with_bart_enc:-1}
# initialize_with_bart_dec=${initialize_with_bart_dec:-1}
# bart_encoder_backprop=${bart_encoder_backprop:-1}
# bart_emb_backprop=${bart_emb_backprop:-1}
# bart_emb_decoder=${bart_emb_decoder:-1}
# bart_emb_decoder_input=${bart_emb_decoder_input:-1}
# bart_emb_init_composition=${bart_emb_init_composition:-0}
# bart_emb_composition_pred=${bart_emb_composition_pred:-0}

src_pretrained_emb=${src_pretrained_emb:-0}
src_pretrained_emb_dim=${src_pretrained_emb_dim:-768}
src_fix_emb_use=$src_pretrained_emb
src_pool_wp2w=${src_pool_wp2w:-bot}
# src_avg_layers=${src_avg_layers:-""}
# src_roberta_enc=${src_roberta_enc:-0}

TRAIN_ARGS=${TRAIN_ARGS:-""}

lr=${lr:-0.0005}
max_tokens=${max_tokens:-3584}
update_freq=${update_freq:-1}
warmup=${warmup:-4000}
dropout=${dropout:-0.3}
clip_norm=${clip_norm:-0.0}

weight_decay=${weight_decay:-0.0}
loss_coef=${loss_coef:-1}


##### TRAINING

if [ -f $MODEL_FOLDER/checkpoint_last.pt ] && [ -f $MODEL_FOLDER/checkpoint${max_epoch}.pt ]; then

    echo "Model checkpoint $MODEL_FOLDER/checkpoint_last.pt && $MODEL_FOLDER/checkpoint${max_epoch}.pt already exist --- do nothing."

else

    # python -m ipdb fairseq_gap.train \
    python -m fairseq_gap.train \
        $BIN_FOLDER \
        --emb-dir $EMB_FOLDER \
        --user-dir src/fairseq_gap \
        --task $TASK \
        --append-eos-to-target 0 \
        --collate-tgt-states 1 \
        --src-fix-emb-use $src_fix_emb_use \
        --shift-pointer-value $shift_pointer_value \
        --apply-tgt-vocab-masks $tgt_vocab_masks \
        --share-all-embeddings ${share_all_embed:-0} \
        --share-decoder-input-output-embed $share_decoder_embed \
        \
        --src-pretrained-emb $src_pretrained_emb \
        --src-pretrained-emb-dim $src_pretrained_emb_dim \
        --src-pool-wp2w $src_pool_wp2w \
        \
        --pointer-dist-decoder-selfattn-layers $pointer_dist_decoder_selfattn_layers \
        --pointer-dist-decoder-selfattn-heads $pointer_dist_decoder_selfattn_heads \
        --pointer-dist-decoder-selfattn-avg $pointer_dist_decoder_selfattn_avg \
        --pointer-dist-decoder-selfattn-infer $pointer_dist_decoder_selfattn_infer \
        \
        --apply-tgt-actnode-masks $apply_tgt_actnode_masks \
        \
        $TRAIN_ARGS \
        \
        --max-epoch $max_epoch \
        --arch $arch \
        --optimizer adam \
        --adam-betas '(0.9,0.98)' \
        --clip-norm $clip_norm \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 \
        --warmup-updates $warmup \
        --lr $lr \
        --stop-min-lr 1e-09 \
        --dropout $dropout \
        --weight-decay $weight_decay \
        --criterion ${criterion:-label_smoothed_cross_entropy_pointer} \
        --label-smoothing 0.01 \
        --loss-coef $loss_coef \
        --keep-last-epochs $(( $max_epoch - $eval_init_epoch + 1 )) \
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --log-format json \
        --seed $seed \
        --save-dir $MODEL_FOLDER \
        --tensorboard-logdir $MODEL_FOLDER $fp16

fi
