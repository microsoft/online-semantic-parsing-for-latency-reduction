#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


dir=$(dirname $0)


# ===== top-down generation: last-8ps
# policy scoring setting
setting_tag=""
# setting_tag=".score-sum"    # NOTE make sure the policy code is matching!
# setting_tag=".score-beam"    # this is using the score for the whole beam for every subgraph; need to change the code

# testing model
exp_dir=exp_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
log_model_tag="order-top_prefix-last-8ps"

# word-rate: real voice acted utterance timing
bash $dir/eval_valid-prefix-allt_latency_wr-real-voice.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_300voice_${log_model_tag}.search &

# check if running correctly
# tail -f logs_lay_search/lay${setting_tag}_300voice_${log_model_tag}.search


# ===== top-down generation: p100 (full parser)
# policy scoring setting
setting_tag=""
# setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# testing model
exp_dir=exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2
log_model_tag="order-top_prefix-p100"

# word-rate: real voice acted utterance timing
bash $dir/eval_valid-prefix-allt_latency_wr-real-voice.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_300voice_${log_model_tag}.search &

# check if running correctly
# tail -f logs_lay_search/lay${setting_tag}_300voice_${log_model_tag}.search



# ===== NOTE below not updated !!!

# # ===== top-down generation: 11ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-11ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-11ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search


# # ===== top-down generation: last-10ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-last-10ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-last-10ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search


# # ===== top-down generation: last-9ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-last-9ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-last-9ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search


# # ===== top-down generation: last-7ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-last-7ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-last-7ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search


# # ===== top-down generation: last-6ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-last-6ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-last-6ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search


# # ===== top-down generation: last-5ps
# # policy scoring setting
# setting_tag=""
# # setting_tag=".score-sum"    # NOTE make sure the policy code is matching!

# # testing model
# exp_dir=exp_prefix-last-5ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
# model_dir=models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2
# log_model_tag="order-top_prefix-last-5ps"

# # word-rate: constant (intrinsic setting)
# bash $dir/eval_valid-prefix-allt_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# # word-rate: char-linear
# bash $dir/eval_valid-prefix-allt_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# # check if running correctly
# # tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search
