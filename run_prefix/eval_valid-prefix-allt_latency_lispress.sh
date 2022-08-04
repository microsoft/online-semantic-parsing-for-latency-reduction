#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


# prefix prediction file
SAVEDIR=../SAVE

# exp_dir=exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

# exp_dir=exp_prefix-11ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

exp_dir=exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2

# bottom-up generation
# exp_dir=exp_prefix-last-8ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

beam_size=1

results_prefix=$SAVEDIR/$exp_dir/$model_dir/beam${beam_size}/valid-prefixallt
prefix_results_path=$results_prefix.hypos.json


# reference lispress file
DATA_ROOT_DIR=../DATA/processed/smcalflow2.0
reference_lispress_path=$DATA_ROOT_DIR/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.lispress


# ===== debug
# output file
output_file=tmp.lay
threshold=-1.0
# threshold=-0.5
# threshold=-0

python -m calflow_parsing.eval_latency_lispress \
    --predictions-path $prefix_results_path \
    --gold-lispress-path $reference_lispress_path \
    --output-file $output_file \
    --threshold $threshold \
    > /dev/null
    # --max-turns 100 \
    # > /dev/null

cat $output_file.summary

# =====


# for threshold in -100 -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3
# # for threshold in -1.2 -1 -0.8 -0.5 -0.3 -0.1 -0
# # for threshold in -1.1 -0.9 -0.7 -0.6 -0.4 -0.2
# do

# echo -e "\npolicy threshold: $threshold"
# # output_file=${results_prefix}5000.threshold${threshold}.lay
# output_file=${results_prefix}.threshold${threshold}.lay


# python -m calflow_parsing.eval_latency_lispress \
#     --predictions-path $prefix_results_path \
#     --gold-lispress-path $reference_lispress_path \
#     --output-file $output_file \
#     --threshold $threshold \
#     > /dev/null
#     #  --max-turns 5000 \
#     # > /dev/null
#     # --max-turns 10

# echo "latency results written to: $output_file"
# cat $output_file.summary

# done
