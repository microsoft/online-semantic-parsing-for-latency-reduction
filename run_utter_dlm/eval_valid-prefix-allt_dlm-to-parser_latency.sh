#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


# prefix prediction file
SAVEDIR=/mnt/container_amulet/SAVE2
SAVEDIR=/n/tata_ddos_ceph/jzhou/incremental-interpretation/SAVE

exp_dir=exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2


beam_size=1

results_prefix=$SAVEDIR/$exp_dir/$model_dir/beam${beam_size}/valid-prefixallt.parsing-from-dlm_checkpoint_best-beam5
prefix_results_path=$results_prefix.hypos.json


# reference lispress file
DATA_ROOT_DIR=/mnt/container_amulet/DATA2/processed/smcalflow2.0
DATA_ROOT_DIR=/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/smcalflow2.0
reference_lispress_path=$DATA_ROOT_DIR/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.lispress

dir=$(dirname $0)
dir=run_prefix


# ===== debug
# output file
output_file=tmp.lay

word_rate_model="char-linear"
execution_time=100
# word_rate_model="constant"
# execution_time=2000

threshold=-1.0
# threshold=-0.5
# threshold=-0

merge_subgraph=0
num_beams=4
threshold=-10


python -m calflow_parsing.eval_latency_dlm-to-parser \
    --predictions-path $prefix_results_path \
    --gold-lispress-path $reference_lispress_path \
    --word-rate-model $word_rate_model \
    --execution-time $execution_time \
    --merge-subgraph $merge_subgraph \
    --num-beams $num_beams \
    --output-file $output_file \
    --threshold $threshold \
    > /dev/null
    # --max-turns 100 \
    # > /dev/null

cat $output_file.summary

# =====
exit 0


word_rate_model="constant"
# execution_time=100

# NOTE for all intergers we do not write .0 in the file names

for execution_time in 200 500 1000 1200 1500 2000 2500 3000 3500 4000 # 0
do

    # for threshold in -100 -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3
    # for threshold in -1.2 -1 -0.8 -0.5 -0.3 -0.1 -0
    # for threshold in -1.1 -0.9 -0.7 -0.6 -0.4 -0.2

    for threshold in -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3 -1.2 -1.1 -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 -0.08 -0.06 -0.04 -0.02 -0
    do

    echo -e "\nword rate model: $word_rate_model"
    echo -e "\nexecution time: $execution_time"
    echo -e "\npolicy threshold: $threshold"
    output_file=$results_prefix.wordrate-${word_rate_model}.exec${execution_time}.threshold${threshold}.lay

    if [[ -s $output_file.summary ]]; then

    echo "latency results already summarized in: $output_file.summary"
    cat $output_file.summary

    else

    python -m calflow_parsing.eval_latency_dlm-to-parser \
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
    #     --predictions-path $prefix_results_path \
    #     --gold-lispress-path $reference_lispress_path \
    #     --output-file $output_file \
    #     --threshold $threshold \
    #     --max-turns 5000 \
    #     > /dev/null
    #     # --max-turns 10

    echo "latency results written to: $output_file"
    cat $output_file.summary

    fi

    done

    # collect all threshold results for the latency-cost curve
    python $dir/collect_threshold_latency.py \
        --latency-results-dir $SAVEDIR/$exp_dir/$model_dir/beam${beam_size} \
        --word-rate-model $word_rate_model \
        --execution-time $execution_time

done
