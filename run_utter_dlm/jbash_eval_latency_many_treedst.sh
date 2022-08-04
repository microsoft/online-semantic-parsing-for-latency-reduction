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
bash $dir/eval_valid-prefix-allt_dlm-beam1-to-parser_latency.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search &

# word-rate: char-linear
bash $dir/eval_valid-prefix-allt_dlm-beam1-to-parser_latency_wr-char-linear.sh $exp_dir $model_dir $setting_tag &> logs_lay_search/lay${setting_tag}_char-linear_${log_model_tag}.search &

# check if running correctly
tail -f logs_lay_search/lay${setting_tag}_intrinsic_${log_model_tag}.search
