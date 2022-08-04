#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


# prefix prediction file
SAVEDIR=../SAVE

# top-down generation
# exp_dir_default=exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir_default=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

# exp_dir_default=exp_prefix-11ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir_default=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

exp_dir_default=exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir_default=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2

# bottom-up generation
# exp_dir_default=exp_prefix-last-8ps_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir_default=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2

# exp_dir_default=exp_prefix-p100_order-bot_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir_default=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2

# from input, otherwise default here
exp_dir=${1:-$exp_dir_default}
model_dir=${2:-$model_dir_default}


beam_size=1

results_prefix=$SAVEDIR/$exp_dir/$model_dir/beam${beam_size}/valid-prefixallt
prefix_results_path=$results_prefix.hypos.json


# reference lispress file
DATA_ROOT_DIR=../DATA/processed/smcalflow2.0

if [[ $results_prefix == *"treedst"* ]]; then
    DATA_ROOT_DIR=../DATA/processed/treedst
fi

reference_lispress_path=$DATA_ROOT_DIR/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.lispress

dir=$(dirname $0)


dataset="smcalflow"
if [[ $results_prefix == *"treedst"* ]]; then
    dataset="treedst"
fi


# # ===== debug
# # output file
# output_file=tmp.lay

# word_rate_model="char-linear"
# execution_time=10
# # word_rate_model="constant"
# # execution_time=2000

# threshold=-1.0
# # threshold=-0.5
# # threshold=-0

# python -m calflow_parsing.eval_latency \
#     --dataset $dataset \
#     --predictions-path $prefix_results_path \
#     --gold-lispress-path $reference_lispress_path \
#     --word-rate-model $word_rate_model \
#     --execution-time $execution_time \
#     --output-file $output_file \
#     --threshold $threshold \
#     > /dev/null
#     # --max-turns 5000 \
#     # > /dev/null

# cat $output_file.summary

# # =====
# exit 0


word_rate_model="constant"
# execution_time=100

setting_tag_default=""
# setting_tag_default=".score-sum"
setting_tag=${3:-$setting_tag_default}


# NOTE for all intergers we do not write .0 in the file names

# for execution_time in 200 1000 3000 2500 # 500 1200 1500 2000 3500 4000 # 0
for execution_time in 500 1200 1500 2000 3500 4000
do

    # for threshold in -100 -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3
    # for threshold in -1.2 -1 -0.8 -0.5 -0.3 -0.1 -0
    # for threshold in -1.1 -0.9 -0.7 -0.6 -0.4 -0.2

    # for threshold in -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 -0.08 -0.06 -0.04 -0.02 -0
    for threshold in -10 -5 -3 -2 -1.8 -1.6 -1.4 -1.2 -1 -0.8 -0.5 -0.3 -0.1 -0.08 -0.06 -0.04 -0.02 -0
    do

    echo -e "\nword rate model: $word_rate_model"
    echo -e "\nexecution time: $execution_time"
    echo -e "\npolicy threshold: $threshold"
    output_file=$results_prefix${setting_tag}.wordrate-${word_rate_model}.exec${execution_time}.threshold${threshold}.lay

    # if [[ -s $output_file.summary ]]; then

    # echo "latency results already summarized in: $output_file.summary"
    # cat $output_file.summary

    # else

    python -m calflow_parsing.eval_latency \
        --dataset $dataset \
        --predictions-path $prefix_results_path \
        --gold-lispress-path $reference_lispress_path \
        --word-rate-model $word_rate_model \
        --execution-time $execution_time \
        --output-file $output_file \
        --threshold $threshold \
        > /dev/null
        # --max-turns 5000 \
        # > /dev/null

    # python -m calflow_parsing.eval_latency_lispress \
    #     --dataset $dataset \
    #     --predictions-path $prefix_results_path \
    #     --gold-lispress-path $reference_lispress_path \
    #     --output-file $output_file \
    #     --threshold $threshold \
    #     --max-turns 5000 \
    #     > /dev/null
    #     # --max-turns 10

    echo "latency results written to: $output_file"
    cat $output_file.summary

    # fi

    done

    # collect all threshold results for the latency-cost curve
    python $dir/collect_threshold_latency.py \
        --latency-results-dir $SAVEDIR/$exp_dir/$model_dir/beam${beam_size} \
        --word-rate-model $word_rate_model \
        --execution-time $execution_time \
        --output-tag "valid-prefixallt${setting_tag}"

done
