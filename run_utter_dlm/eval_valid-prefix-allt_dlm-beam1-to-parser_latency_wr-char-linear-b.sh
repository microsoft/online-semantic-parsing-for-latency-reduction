#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -e


# prefix prediction file
SAVEDIR=/mnt/container_amulet/SAVE2
SAVEDIR=/n/tata_ddos_ceph/jzhou/incremental-interpretation/SAVE

exp_dir_default=exp_prefix-p100_order-top_src-ct1-npwa_tgt-cp-str_roberta-large-top24_gap_vmask1_shiftpos1_ptr-lay6-h1_scp-lay5-h1_dec-emb-sha0
model_dir_default=models_ep50_seed42_fp16-lr0.0005-mt4096x4-wm4000-dp0.2

# from input, otherwise default here
exp_dir=${1:-$exp_dir_default}
model_dir=${2:-$model_dir_default}


beam_size=1

# results_prefix=$SAVEDIR/$exp_dir/$model_dir/beam${beam_size}/valid-prefixallt.parsing-from-dlm_checkpoint_best-beam5
results_prefix=$SAVEDIR/$exp_dir/$model_dir/beam${beam_size}/valid-prefixallt.parsing-from-dlm_checkpoint_best-beam1
prefix_results_path=$results_prefix.hypos.json


# reference lispress file
DATA_ROOT_DIR=/mnt/container_amulet/DATA2/processed/smcalflow2.0
DATA_ROOT_DIR=/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/smcalflow2.0

if [[ $results_prefix == *"treedst"* ]]; then
    DATA_ROOT_DIR=/n/tata_ddos_ceph/jzhou/incremental-interpretation/DATA/processed/treedst
fi

reference_lispress_path=$DATA_ROOT_DIR/act-top_src-ct1-npwa_tgt-cp-str/oracle/valid.lispress

dir=$(dirname $0)
dir=run_prefix


dataset="smcalflow"
if [[ $results_prefix == *"treedst"* ]]; then
    dataset="treedst"
fi


# # ===== debug
# # output file
# output_file=tmp.lay

# word_rate_model="char-linear"
# execution_time=100
# # word_rate_model="constant"
# # execution_time=2000

# threshold=-1.0
# # threshold=-0.5
# # threshold=-0

# merge_subgraph=0
# num_beams=1
# threshold=-3    # min is around -154, but -100 is enough, and -50 and -10 not so much different


# python -m calflow_parsing.eval_latency_dlm-to-parser \
#     --dataset $dataset \
#     --predictions-path $prefix_results_path \
#     --gold-lispress-path $reference_lispress_path \
#     --word-rate-model $word_rate_model \
#     --execution-time $execution_time \
#     --merge-subgraph $merge_subgraph \
#     --num-beams $num_beams \
#     --output-file $output_file \
#     --threshold $threshold \
#     > /dev/null
#     # --max-turns 100 \
#     # > /dev/null

# cat $output_file.summary

# # =====
# exit 0


merge_subgraph=0
num_beams=1

word_rate_model="char-linear"
# execution_time=100

setting_tag_default=""
# setting_tag_default=".score-sum"
setting_tag=${3:-$setting_tag_default}

# NOTE for all intergers we do not write .0 in the file names
execution_time_list_default="50 100 500 20 200 800 1000"
execution_time_list=${4:-$execution_time_list_default}

for execution_time in $execution_time_list
do

    # for threshold in -100 -10 -5 -3 -2 -1.8 -1.5 -1.4 -1.3
    # for threshold in -1.2 -1 -0.8 -0.5 -0.3 -0.1 -0
    # for threshold in -1.1 -0.9 -0.7 -0.6 -0.4 -0.2

    for threshold in -200 -100 -50 -10 -8 -6 -5 -4 -3 -2 -1.8 -1.5 -1.4 -1.3 -1.2 -1.1 -1 -0.5 -0
    do

    echo -e "\nword rate model: $word_rate_model"
    echo -e "\nexecution time: $execution_time"
    echo -e "\npolicy threshold: $threshold"
    output_file=$results_prefix${setting_tag}.wordrate-${word_rate_model}.exec${execution_time}.threshold${threshold}.lay

    # if [[ -s $output_file.summary ]]; then

    # echo "latency results already summarized in: $output_file.summary"
    # cat $output_file.summary

    # else

    python -m calflow_parsing.eval_latency_dlm-to-parser \
        --dataset $dataset \
        --predictions-path $prefix_results_path \
        --gold-lispress-path $reference_lispress_path \
        --word-rate-model $word_rate_model \
        --execution-time $execution_time \
        --merge-subgraph $merge_subgraph \
        --num-beams $num_beams \
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
        --output-tag "valid-prefixallt.parsing-from-dlm_checkpoint_best-beam1${setting_tag}"

done
