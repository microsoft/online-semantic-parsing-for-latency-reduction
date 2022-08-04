#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


dir=$(dirname $0)


# ===== top-down generation: p100 (full parser) + DLM beam1 completion
# policy scoring setting
setting_tag=""
# setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# testing model
exp_dir=exp_treedst_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2
log_model_tag="treedst_order-top_dlm-beam1-to-prefix-p100"

# word-rate: constant (intrinsic setting)
# for execution_time_list in 200 1000 3000 2500
for execution_time_list in 500 1200 1500 2000 3500 4000
do
# NOTE must be "$setting_tag" instead of $setting_tag, otherwise it will be skipped and mess up with the next argument when setting_tag="" empty
bash $dir/eval_valid-prefix-allt_dlm-beam1-to-parser_latency-b.sh $exp_dir $model_dir "$setting_tag" $execution_time_list &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.exec${execution_time_list}.search &
done

# word-rate: char-linear
# for execution_time_list in 50 100 500 20
for execution_time_list in 200 800 1000
do
# NOTE must be "$setting_tag" instead of $setting_tag, otherwise it will be skipped and mess up with the next argument when setting_tag="" empty
bash $dir/eval_valid-prefix-allt_dlm-beam1-to-parser_latency_wr-char-linear-b.sh $exp_dir $model_dir "$setting_tag" "$execution_time_list" &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.exec${execution_time_list}.search &
done

# check if running correctly
# tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.exec1000.search
echo "check log with"
echo "tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.exec1000.search"
