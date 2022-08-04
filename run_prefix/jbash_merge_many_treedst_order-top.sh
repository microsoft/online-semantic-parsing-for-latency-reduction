#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


# # merge prefixes
# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-11ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-10ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-9ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --max_len 43 \
#     --data_folder "/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle" \
#     --model_folder "exp_treedst_prefix-last-8ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-7ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-6ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --max_len 43 \
#     --data_folder "/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle" \
#     --model_folder "exp_treedst_prefix-last-5ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-4ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

python run_prefix/merge_results_all_prefix.py \
    --max_len 43 \
    --data_folder "/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle" \
    --model_folder "exp_treedst_prefix-last-3ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --max_len 43 \
#     --data_folder "/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle" \
#     --model_folder "exp_treedst_prefix-last-3ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-2ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0001-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-last-2ps_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"


# single prefix subset
# python run_prefix/merge_results_all_prefix.py \
#     --max_len 43 \
#     --data_folder "/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst/act-top_src-ct1-npwa_tgt-cp-str_abs-prefix-all/oracle" \
#     --model_folder "exp_treedst_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p90_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p80_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p70_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p60_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p50_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p40_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p30_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p20_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p10_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"

# python run_prefix/merge_results_all_prefix.py \
#     --model_folder "exp_treedst_prefix-p0_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0/models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2"
